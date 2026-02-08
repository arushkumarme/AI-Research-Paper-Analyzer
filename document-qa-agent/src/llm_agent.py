"""
LLM Agent for Document Q&A using Google Gemini

Implements intelligent document question-answering with:
- Query classification (lookup, summarization, extraction)
- RAG-based context retrieval
- Specialized prompts for each query type
- Response caching for performance (with TTL and invalidation)
- Gemini function calling for tool integration
- Enterprise-grade security and monitoring

Usage:
    from src.llm_agent import DocumentQAAgent
    
    agent = DocumentQAAgent()
    
    # Ask questions about documents
    response = agent.query("What is the conclusion of Paper X?")
    print(response.answer)
    print(response.sources)
    
    # With specific document filter
    response = agent.query(
        "Summarize the methodology",
        document_filter="research_paper.pdf"
    )
"""

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import google.generativeai as genai
from langchain_core.documents import Document
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_settings
from src.vector_store import VectorStoreManager, get_vector_store

# Import enterprise features for enhanced functionality
try:
    from src.enterprise import (
        EnhancedResponseCache,
        performance_tracker,
        security_validator,
        validate_query,
        with_fallback,
        CircuitBreaker,
    )
    ENTERPRISE_FEATURES_AVAILABLE = True
except ImportError:
    ENTERPRISE_FEATURES_AVAILABLE = False

# Configure module logger
logger = logging.getLogger("document_qa.agent")


class QueryType(Enum):
    """Classification of query types for specialized handling."""
    
    LOOKUP = "lookup"           # Direct content lookup: "What does Paper X say about Y?"
    SUMMARIZATION = "summarize" # Summarization: "Summarize the methodology"
    EXTRACTION = "extract"      # Metric extraction: "What are the F1 scores?"
    COMPARISON = "compare"      # Compare across documents
    GENERAL = "general"         # General questions


@dataclass
class QueryResult:
    """Container for query response with metadata."""
    
    answer: str
    query_type: QueryType
    sources: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    cached: bool = False
    tokens_used: int = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "answer": self.answer,
            "query_type": self.query_type.value,
            "sources": self.sources,
            "confidence": self.confidence,
            "cached": self.cached,
            "tokens_used": self.tokens_used,
        }


class ResponseCache:
    """
    LRU cache for query responses to avoid duplicate API calls.
    
    Caches responses based on query + context hash.
    Now supports document-based invalidation for enterprise use.
    """
    
    def __init__(self, max_size: int = 100):
        self._cache: dict[str, QueryResult] = {}
        self._max_size = max_size
        self._access_order: list[str] = []
        # Track which documents are associated with each cache entry
        self._document_index: dict[str, set] = {}  # doc_id -> set of cache keys
        self._key_documents: dict[str, set] = {}   # cache_key -> set of doc_ids
        # Statistics for monitoring
        self._hits = 0
        self._misses = 0
    
    def _generate_key(self, query: str, context_hash: str) -> str:
        """Generate cache key from query and context."""
        combined = f"{query.lower().strip()}:{context_hash}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, query: str, context_hash: str) -> Optional[QueryResult]:
        """Retrieve cached response if available."""
        key = self._generate_key(query, context_hash)
        if key in self._cache:
            # Update access order
            self._access_order.remove(key)
            self._access_order.append(key)
            result = self._cache[key]
            result.cached = True
            self._hits += 1
            logger.debug(f"Cache hit for query: {query[:50]}...")
            return result
        self._misses += 1
        return None
    
    def set(
        self,
        query: str,
        context_hash: str,
        result: QueryResult,
        document_ids: Optional[set] = None,
    ) -> None:
        """
        Cache a query result with optional document tracking.
        
        Args:
            query: The query string
            context_hash: Hash of context
            result: Query result to cache
            document_ids: Set of document IDs for invalidation tracking
        """
        key = self._generate_key(query, context_hash)
        
        # Evict oldest entries if at capacity
        while len(self._cache) >= self._max_size:
            oldest_key = self._access_order.pop(0)
            self._remove_key(oldest_key)
        
        self._cache[key] = result
        self._access_order.append(key)
        
        # Track document associations for invalidation
        if document_ids:
            self._key_documents[key] = document_ids
            for doc_id in document_ids:
                if doc_id not in self._document_index:
                    self._document_index[doc_id] = set()
                self._document_index[doc_id].add(key)
    
    def _remove_key(self, key: str) -> None:
        """Remove a key and clean up indices."""
        if key in self._cache:
            del self._cache[key]
        
        # Clean up document index
        if key in self._key_documents:
            for doc_id in self._key_documents[key]:
                if doc_id in self._document_index:
                    self._document_index[doc_id].discard(key)
            del self._key_documents[key]
    
    def invalidate_by_document(self, document_id: str) -> int:
        """
        Invalidate all cache entries associated with a document.
        
        Call this when a document is updated or deleted.
        
        Args:
            document_id: Document identifier (filename)
            
        Returns:
            Number of entries invalidated
        """
        if document_id not in self._document_index:
            return 0
        
        keys_to_remove = self._document_index[document_id].copy()
        count = 0
        
        for key in keys_to_remove:
            if key in self._access_order:
                self._access_order.remove(key)
            self._remove_key(key)
            count += 1
        
        del self._document_index[document_id]
        logger.info(f"Invalidated {count} cache entries for document: {document_id}")
        return count
    
    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()
        self._document_index.clear()
        self._key_documents.clear()
    
    @property
    def size(self) -> int:
        """Return current cache size."""
        return len(self._cache)
    
    def get_stats(self) -> dict:
        """Get cache statistics for monitoring."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.1%}",
            "documents_tracked": len(self._document_index),
        }


class PromptTemplates:
    """Specialized prompts for different query types."""
    
    SYSTEM_PROMPT = """You are an expert academic research assistant specializing in analyzing scientific papers and documents. Your role is to provide accurate, well-sourced answers based on the provided document context.

Guidelines:
- Always base your answers on the provided context
- If the context doesn't contain relevant information, clearly state that
- Cite specific sections or pages when referencing information
- Be precise with technical terms and metrics
- Maintain academic rigor in your responses"""

    QUERY_CLASSIFIER_PROMPT = """Classify the following query into one of these categories:
- LOOKUP: Direct content lookup (e.g., "What does the paper say about X?", "What is the conclusion?")
- SUMMARIZE: Summarization requests (e.g., "Summarize the methodology", "Give an overview of...")
- EXTRACT: Metric/data extraction (e.g., "What are the accuracy scores?", "List the F1 scores")
- COMPARE: Comparison queries (e.g., "Compare the methods in Paper A and B")
- GENERAL: General questions about the documents

Query: {query}

Respond with ONLY the category name (LOOKUP, SUMMARIZE, EXTRACT, COMPARE, or GENERAL)."""

    LOOKUP_PROMPT = """Based on the following document excerpts, answer the question directly and precisely.

CONTEXT:
{context}

QUESTION: {query}

Instructions:
- Provide a direct, factual answer based on the context
- Quote relevant passages when appropriate
- If the answer is not in the context, say "The provided documents do not contain information about this."
- Cite the source document and page/section when possible

ANSWER:"""

    SUMMARIZATION_PROMPT = """Based on the following document excerpts, provide a comprehensive summary.

CONTEXT:
{context}

REQUEST: {query}

Instructions:
- Provide a structured, well-organized summary
- Include key points, findings, and important details
- Use bullet points for clarity when appropriate
- Maintain the technical accuracy of the original content
- Note which document(s) the information comes from

SUMMARY:"""

    EXTRACTION_PROMPT = """Extract specific metrics, data, or structured information from the following document excerpts.

CONTEXT:
{context}

EXTRACTION REQUEST: {query}

Instructions:
- Extract all relevant metrics, numbers, and data points
- Present data in a clear, structured format (tables, lists)
- Include units and context for each metric
- Note which document/section each metric comes from
- If metrics are not found, clearly state that

EXTRACTED DATA:"""

    COMPARISON_PROMPT = """Compare and contrast information across the following document excerpts.

CONTEXT:
{context}

COMPARISON REQUEST: {query}

Instructions:
- Identify similarities and differences
- Create a structured comparison (consider using a comparison format)
- Highlight key distinguishing features
- Note the source of each piece of information
- Provide a balanced analysis

COMPARISON:"""

    GENERAL_PROMPT = """Based on the following document excerpts, answer the question thoughtfully.

CONTEXT:
{context}

QUESTION: {query}

Instructions:
- Provide a comprehensive answer based on the context
- Structure your response clearly
- Cite sources when making specific claims
- If information is not available, acknowledge the limitation

ANSWER:"""


class DocumentQAAgent:
    """
    Intelligent Document Q&A Agent using Google Gemini.
    
    Features:
    - Query classification for specialized handling
    - RAG-based context retrieval from vector store
    - Response caching for performance
    - Tool/function calling integration
    - Comprehensive error handling
    
    Example:
        agent = DocumentQAAgent()
        
        # Basic query
        result = agent.query("What are the main findings?")
        print(result.answer)
        
        # Query specific document
        result = agent.query(
            "Summarize the experiments",
            document_filter="paper.pdf"
        )
        
        # Query with custom context
        result = agent.query(
            "Explain this methodology",
            context_docs=my_documents
        )
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        vector_store: Optional[VectorStoreManager] = None,
        cache_enabled: bool = True,
        temperature: float = 0.3,
        max_output_tokens: int = 2048,
    ):
        """
        Initialize the Document Q&A Agent.
        
        Args:
            model_name: Gemini model to use (default from settings)
            vector_store: Vector store instance (creates new if not provided)
            cache_enabled: Whether to cache responses
            temperature: Model temperature (lower = more focused)
            max_output_tokens: Maximum response length
        """
        settings = get_settings()
        
        self.model_name = model_name or settings.gemini_model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        
        # Configure Gemini API
        genai.configure(api_key=settings.google_api_key)
        
        # Initialize model
        self._model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=genai.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            ),
            system_instruction=PromptTemplates.SYSTEM_PROMPT,
        )
        
        # Initialize classifier model (faster, more deterministic)
        self._classifier = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=genai.GenerationConfig(
                temperature=0.0,
                max_output_tokens=20,
            ),
        )
        
        # Vector store for document retrieval
        self._vector_store = vector_store or get_vector_store()
        
        # Response cache
        self._cache = ResponseCache() if cache_enabled else None
        self._cache_enabled = cache_enabled
        
        # Registered tools for function calling
        self._tools: dict[str, Callable] = {}
        
        logger.info(
            f"DocumentQAAgent initialized with model: {self.model_name}, "
            f"cache={'enabled' if cache_enabled else 'disabled'}"
        )
    
    def register_tool(self, name: str, func: Callable, description: str) -> None:
        """
        Register a tool for function calling.
        
        Args:
            name: Tool name
            func: Callable function
            description: Tool description for the model
        """
        self._tools[name] = {
            "function": func,
            "description": description,
        }
        logger.info(f"Registered tool: {name}")
    
    def query(
        self,
        query: str,
        document_filter: Optional[str] = None,
        context_docs: Optional[list[Document]] = None,
        num_context: int = 5,
        use_tools: bool = True,
        user_id: Optional[str] = None,
        skip_validation: bool = False,
    ) -> QueryResult:
        """
        Process a query and generate a response.
        
        Args:
            query: User's question
            document_filter: Optional filename to restrict search
            context_docs: Optional pre-retrieved documents to use
            num_context: Number of context documents to retrieve
            use_tools: Whether to allow tool/function calling
            user_id: Optional user ID for rate limiting
            skip_validation: Skip input validation (for internal calls)
            
        Returns:
            QueryResult with answer, sources, and metadata
        """
        start_time = time.perf_counter()
        
        try:
            # Step 0: Input validation (enterprise security)
            if not skip_validation and ENTERPRISE_FEATURES_AVAILABLE:
                is_valid, sanitized_query, issues = validate_query(query, user_id)
                if not is_valid:
                    logger.warning(f"Query validation failed: {issues}")
                    return QueryResult(
                        answer=f"Invalid query: {'; '.join(issues)}",
                        query_type=QueryType.GENERAL,
                        confidence=0.0,
                    )
                query = sanitized_query
            
            # Step 1: Classify the query
            query_type = self._classify_query(query)
            logger.debug(f"Query classified as: {query_type.value}")
            
            # Step 2: Retrieve relevant context
            if context_docs:
                context = context_docs
            else:
                context = self._retrieve_context(
                    query,
                    document_filter=document_filter,
                    k=num_context,
                )
            
            # Step 3: Check cache
            context_hash = self._hash_context(context)
            if self._cache_enabled and self._cache:
                cached_result = self._cache.get(query, context_hash)
                if cached_result:
                    # Track cache hit in performance metrics
                    if ENTERPRISE_FEATURES_AVAILABLE:
                        performance_tracker.increment("cache_hits")
                    return cached_result
            
            # Step 4: Generate response based on query type
            result = self._generate_response(query, query_type, context)
            
            # Step 5: Cache result with document tracking
            if self._cache_enabled and self._cache:
                # Extract document IDs for cache invalidation
                doc_ids = set(
                    doc.metadata.get('filename', '')
                    for doc in context
                    if doc.metadata.get('filename')
                )
                self._cache.set(query, context_hash, result, document_ids=doc_ids)
            
            # Track performance metrics
            if ENTERPRISE_FEATURES_AVAILABLE:
                duration = time.perf_counter() - start_time
                performance_tracker.record_time(
                    "query_processing",
                    duration,
                    tags={"query_type": query_type.value}
                )
                performance_tracker.increment("queries_processed")
                performance_tracker.increment("api_calls")
            
            return result
            
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            
            # Track error in performance metrics
            if ENTERPRISE_FEATURES_AVAILABLE:
                performance_tracker.increment("query_errors")
            
            # Provide fallback response
            return self._create_error_response(query, e)
    
    def _classify_query(self, query: str) -> QueryType:
        """Classify query type using the model."""
        
        # Quick keyword-based classification first
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["summarize", "summary", "overview", "briefly describe"]):
            return QueryType.SUMMARIZATION
        
        if any(word in query_lower for word in ["accuracy", "f1", "score", "metric", "percentage", "precision", "recall", "performance"]):
            return QueryType.EXTRACTION
        
        if any(word in query_lower for word in ["compare", "difference", "versus", "vs", "contrast"]):
            return QueryType.COMPARISON
        
        if any(word in query_lower for word in ["what is", "what are", "what does", "explain", "describe", "conclusion", "finding"]):
            return QueryType.LOOKUP
        
        # Use model for ambiguous cases
        try:
            prompt = PromptTemplates.QUERY_CLASSIFIER_PROMPT.format(query=query)
            response = self._classifier.generate_content(prompt)
            
            classification = response.text.strip().upper()
            
            type_map = {
                "LOOKUP": QueryType.LOOKUP,
                "SUMMARIZE": QueryType.SUMMARIZATION,
                "EXTRACT": QueryType.EXTRACTION,
                "COMPARE": QueryType.COMPARISON,
                "GENERAL": QueryType.GENERAL,
            }
            
            return type_map.get(classification, QueryType.GENERAL)
            
        except Exception as e:
            logger.warning(f"Classification failed, defaulting to GENERAL: {e}")
            return QueryType.GENERAL
    
    def _retrieve_context(
        self,
        query: str,
        document_filter: Optional[str] = None,
        k: int = 5,
    ) -> list[Document]:
        """Retrieve relevant documents from vector store."""
        
        filter_dict = None
        if document_filter:
            filter_dict = {"filename": document_filter}
        
        try:
            documents = self._vector_store.similarity_search(
                query=query,
                k=k,
                filter_dict=filter_dict,
                score_threshold=0.3,  # Minimum relevance threshold
            )
            
            logger.debug(f"Retrieved {len(documents)} context documents")
            return documents
            
        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            return []
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,)),
        before_sleep=lambda retry_state: logger.warning(
            f"Generation retry {retry_state.attempt_number}/3"
        ),
    )
    def _generate_response(
        self,
        query: str,
        query_type: QueryType,
        context: list[Document],
    ) -> QueryResult:
        """Generate response using appropriate prompt template."""
        
        # Format context
        context_text = self._format_context(context)
        
        # Select prompt template
        prompt_templates = {
            QueryType.LOOKUP: PromptTemplates.LOOKUP_PROMPT,
            QueryType.SUMMARIZATION: PromptTemplates.SUMMARIZATION_PROMPT,
            QueryType.EXTRACTION: PromptTemplates.EXTRACTION_PROMPT,
            QueryType.COMPARISON: PromptTemplates.COMPARISON_PROMPT,
            QueryType.GENERAL: PromptTemplates.GENERAL_PROMPT,
        }
        
        template = prompt_templates.get(query_type, PromptTemplates.GENERAL_PROMPT)
        prompt = template.format(context=context_text, query=query)
        
        # Generate response
        response = self._model.generate_content(prompt)
        
        # Extract sources from context
        sources = self._extract_sources(context)
        
        # Calculate confidence based on context relevance
        confidence = self._calculate_confidence(context)
        
        # Get token usage if available
        tokens_used = 0
        if hasattr(response, 'usage_metadata'):
            tokens_used = getattr(response.usage_metadata, 'total_token_count', 0)
        
        return QueryResult(
            answer=response.text,
            query_type=query_type,
            sources=sources,
            confidence=confidence,
            tokens_used=tokens_used,
        )
    
    def _format_context(self, documents: list[Document]) -> str:
        """Format documents into context string."""
        
        if not documents:
            return "No relevant documents found."
        
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            metadata = doc.metadata
            source_info = f"[Source {i}: {metadata.get('filename', 'Unknown')}]"
            
            if metadata.get('page_number'):
                source_info += f" (Page {metadata['page_number']})"
            
            if metadata.get('section') and metadata['section'] != 'Unknown':
                source_info += f" - {metadata['section']}"
            
            context_parts.append(f"{source_info}\n{doc.page_content}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def _extract_sources(self, documents: list[Document]) -> list[dict]:
        """Extract source information from documents."""
        
        sources = []
        seen = set()
        
        for doc in documents:
            metadata = doc.metadata
            filename = metadata.get('filename', 'Unknown')
            page = metadata.get('page_number', 'N/A')
            section = metadata.get('section', 'Unknown')
            score = metadata.get('similarity_score', 0.0)
            
            # Avoid duplicates
            key = f"{filename}:{page}:{section}"
            if key not in seen:
                seen.add(key)
                sources.append({
                    "filename": filename,
                    "page": page,
                    "section": section,
                    "relevance_score": score,
                })
        
        return sources
    
    def _calculate_confidence(self, documents: list[Document]) -> float:
        """Calculate confidence score based on context quality."""
        
        if not documents:
            return 0.0
        
        # Average similarity scores
        scores = [
            doc.metadata.get('similarity_score', 0.5)
            for doc in documents
        ]
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        # Adjust based on number of relevant documents
        doc_factor = min(len(documents) / 3, 1.0)  # Optimal: 3+ documents
        
        confidence = avg_score * 0.7 + doc_factor * 0.3
        
        return round(min(confidence, 1.0), 2)
    
    def _hash_context(self, documents: list[Document]) -> str:
        """Generate hash for context documents."""
        
        content = "".join(doc.page_content[:100] for doc in documents)
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def get_document_summary(self, filename: str) -> QueryResult:
        """
        Get a summary of a specific document.
        
        Args:
            filename: Name of the document to summarize
            
        Returns:
            QueryResult with document summary
        """
        return self.query(
            f"Provide a comprehensive summary of this document, including its main topic, "
            f"key findings, methodology (if applicable), and conclusions.",
            document_filter=filename,
            num_context=10,
        )
    
    def extract_metrics(self, filename: Optional[str] = None) -> QueryResult:
        """
        Extract all metrics and quantitative data from document(s).
        
        Args:
            filename: Optional specific document to extract from
            
        Returns:
            QueryResult with extracted metrics
        """
        return self.query(
            "Extract all numerical metrics, scores, percentages, and quantitative "
            "results mentioned in the document. Include accuracy, F1-scores, "
            "precision, recall, and any other performance metrics.",
            document_filter=filename,
            num_context=8,
        )
    
    def answer_with_citations(
        self,
        query: str,
        document_filter: Optional[str] = None,
    ) -> QueryResult:
        """
        Answer query with inline citations.
        
        Args:
            query: User's question
            document_filter: Optional filename filter
            
        Returns:
            QueryResult with citations in the answer
        """
        # Modify query to request citations
        citation_query = f"{query} Please include inline citations [Source X] for each claim."
        
        return self.query(
            citation_query,
            document_filter=document_filter,
            num_context=6,
        )
    
    def clear_cache(self) -> None:
        """Clear the response cache."""
        if self._cache:
            self._cache.clear()
            logger.info("Response cache cleared")
    
    def invalidate_document_cache(self, document_id: str) -> int:
        """
        Invalidate cache entries for a specific document.
        
        Call this when a document is updated or deleted.
        
        Args:
            document_id: Document filename
            
        Returns:
            Number of cache entries invalidated
        """
        if self._cache:
            return self._cache.invalidate_by_document(document_id)
        return 0
    
    def _create_error_response(self, query: str, error: Exception) -> QueryResult:
        """
        Create a user-friendly error response with fallback message.
        
        Args:
            query: Original query
            error: The exception that occurred
            
        Returns:
            QueryResult with error information
        """
        # Determine appropriate error message based on error type
        error_type = type(error).__name__
        
        fallback_messages = {
            "RateLimitError": "The AI service is currently busy. Please try again in a moment.",
            "APIError": "There was an issue connecting to the AI service. Please try again.",
            "TimeoutError": "The request timed out. Please try a simpler query.",
            "ResourceExhausted": "API quota exceeded. Please try again later.",
        }
        
        message = fallback_messages.get(
            error_type,
            f"I encountered an error processing your query. Please try rephrasing or simplifying your question."
        )
        
        logger.error(f"Query error ({error_type}): {error}")
        
        return QueryResult(
            answer=message,
            query_type=QueryType.GENERAL,
            confidence=0.0,
        )
    
    def get_stats(self) -> dict:
        """Get agent statistics including performance metrics."""
        stats = {
            "model": self.model_name,
            "temperature": self.temperature,
            "cache_enabled": self._cache_enabled,
            "cache_size": self._cache.size if self._cache else 0,
            "registered_tools": list(self._tools.keys()),
        }
        
        # Include cache statistics if available
        if self._cache and hasattr(self._cache, 'get_stats'):
            stats["cache_stats"] = self._cache.get_stats()
        
        # Include performance metrics if enterprise features are available
        if ENTERPRISE_FEATURES_AVAILABLE:
            stats["performance"] = performance_tracker.get_report()
        
        return stats


class AgentWithTools(DocumentQAAgent):
    """
    Extended agent with function calling capabilities.
    
    Supports registering and executing tools via Gemini function calling.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tool_definitions = []
    
    def register_function_tool(
        self,
        func: Callable,
        name: str,
        description: str,
        parameters: dict,
    ) -> None:
        """
        Register a function as a tool for Gemini function calling.
        
        Args:
            func: The function to call
            name: Function name
            description: Description for the model
            parameters: Parameter schema (JSON Schema format)
        """
        self._tools[name] = func
        
        # Create Gemini function declaration
        self._tool_definitions.append(
            genai.protos.FunctionDeclaration(
                name=name,
                description=description,
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        param_name: genai.protos.Schema(
                            type=self._get_proto_type(param_info.get("type", "string")),
                            description=param_info.get("description", ""),
                        )
                        for param_name, param_info in parameters.get("properties", {}).items()
                    },
                    required=parameters.get("required", []),
                ),
            )
        )
        
        logger.info(f"Registered function tool: {name}")
    
    def _get_proto_type(self, type_str: str) -> genai.protos.Type:
        """Convert string type to Gemini proto type."""
        type_map = {
            "string": genai.protos.Type.STRING,
            "number": genai.protos.Type.NUMBER,
            "integer": genai.protos.Type.INTEGER,
            "boolean": genai.protos.Type.BOOLEAN,
            "array": genai.protos.Type.ARRAY,
            "object": genai.protos.Type.OBJECT,
        }
        return type_map.get(type_str, genai.protos.Type.STRING)
    
    def query_with_tools(
        self,
        query: str,
        document_filter: Optional[str] = None,
    ) -> QueryResult:
        """
        Process query with tool calling enabled.
        
        Args:
            query: User's question
            document_filter: Optional filename filter
            
        Returns:
            QueryResult potentially using tool results
        """
        if not self._tool_definitions:
            return self.query(query, document_filter)
        
        try:
            # Create model with tools
            model_with_tools = genai.GenerativeModel(
                model_name=self.model_name,
                tools=[genai.protos.Tool(function_declarations=self._tool_definitions)],
                generation_config=genai.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                ),
            )
            
            # Initial query
            chat = model_with_tools.start_chat()
            response = chat.send_message(query)
            
            # Check for function calls
            while response.candidates[0].content.parts:
                function_call = None
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        function_call = part.function_call
                        break
                
                if not function_call:
                    break
                
                # Execute function
                func_name = function_call.name
                func_args = dict(function_call.args)
                
                logger.info(f"Executing tool: {func_name} with args: {func_args}")
                
                if func_name in self._tools:
                    try:
                        result = self._tools[func_name](**func_args)
                        result_str = str(result)
                    except Exception as e:
                        result_str = f"Error executing {func_name}: {e}"
                        logger.error(result_str)
                else:
                    result_str = f"Unknown function: {func_name}"
                
                # Send function result back
                response = chat.send_message(
                    genai.protos.Content(
                        parts=[
                            genai.protos.Part(
                                function_response=genai.protos.FunctionResponse(
                                    name=func_name,
                                    response={"result": result_str},
                                )
                            )
                        ]
                    )
                )
            
            # Get final text response
            answer = response.text if hasattr(response, 'text') else str(response)
            
            return QueryResult(
                answer=answer,
                query_type=QueryType.GENERAL,
                confidence=0.8,
            )
            
        except Exception as e:
            logger.error(f"Tool query failed: {e}")
            # Fallback to regular query
            return self.query(query, document_filter)


# Convenience function
def get_agent() -> DocumentQAAgent:
    """Get a default configured agent instance."""
    return DocumentQAAgent()


if __name__ == "__main__":
    # Test the agent
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    print("Document Q&A Agent - Test")
    print("-" * 50)
    
    try:
        agent = DocumentQAAgent()
        stats = agent.get_stats()
        print(f"Model: {stats['model']}")
        print(f"Cache enabled: {stats['cache_enabled']}")
        
        # Test query classification
        test_queries = [
            "What is the conclusion of the paper?",
            "Summarize the methodology section",
            "What are the accuracy and F1 scores?",
            "Compare the approaches in papers A and B",
        ]
        
        print("\nQuery Classification Tests:")
        for query in test_queries:
            query_type = agent._classify_query(query)
            print(f"  '{query[:40]}...' -> {query_type.value}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
