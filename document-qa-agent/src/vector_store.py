"""
Vector Store Manager using ChromaDB and Google Gemini Embeddings

Provides vector storage and retrieval for document Q&A:
- Embedding generation using Gemini models/embedding-001
- ChromaDB persistence and collection management
- Similarity search with metadata filtering
- Caching for improved performance

Usage:
    from src.vector_store import VectorStoreManager
    
    # Initialize
    store = VectorStoreManager()
    
    # Add documents
    store.add_documents(documents)
    
    # Search
    results = store.similarity_search("What is the main finding?", k=5)
    
    # Search within specific document
    results = store.similarity_search(
        "methodology",
        k=3,
        filter_dict={"filename": "paper.pdf"}
    )
"""

import hashlib
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_settings

# Configure module logger
logger = logging.getLogger("document_qa.vector_store")


class EmbeddingCache:
    """
    Simple in-memory cache for embeddings to reduce API calls.
    
    Caches embeddings by content hash to avoid re-embedding identical text.
    """
    
    def __init__(self, max_size: int = 1000):
        self._cache: dict[str, list[float]] = {}
        self._max_size = max_size
    
    def _hash_text(self, text: str) -> str:
        """Generate hash for text content."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def get(self, text: str) -> Optional[list[float]]:
        """Get cached embedding if available."""
        key = self._hash_text(text)
        return self._cache.get(key)
    
    def set(self, text: str, embedding: list[float]) -> None:
        """Cache an embedding."""
        if len(self._cache) >= self._max_size:
            # Simple eviction: remove oldest entries
            keys_to_remove = list(self._cache.keys())[:100]
            for key in keys_to_remove:
                del self._cache[key]
        
        key = self._hash_text(text)
        self._cache[key] = embedding
    
    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
    
    @property
    def size(self) -> int:
        """Return current cache size."""
        return len(self._cache)


class GeminiEmbeddings:
    """
    Wrapper for Google Gemini embeddings with caching and retry logic.
    
    Uses models/embedding-001 for generating text embeddings.
    Implements caching to reduce API calls for repeated text.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "models/embedding-001",
        cache_enabled: bool = True,
    ):
        """
        Initialize Gemini embeddings.
        
        Args:
            api_key: Google API key (uses settings if not provided)
            model: Embedding model name
            cache_enabled: Whether to cache embeddings
        """
        settings = get_settings()
        self.api_key = api_key or settings.google_api_key
        self.model = model
        
        # Initialize LangChain Gemini embeddings
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model=self.model,
            google_api_key=self.api_key,
        )
        
        # Initialize cache
        self._cache = EmbeddingCache() if cache_enabled else None
        self._cache_enabled = cache_enabled
        
        logger.info(f"GeminiEmbeddings initialized with model: {model}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,)),
        before_sleep=lambda retry_state: logger.warning(
            f"Embedding retry {retry_state.attempt_number}/3"
        ),
    )
    def embed_query(self, text: str) -> list[float]:
        """
        Generate embedding for a query text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        # Check cache first
        if self._cache_enabled and self._cache:
            cached = self._cache.get(text)
            if cached is not None:
                logger.debug("Embedding cache hit")
                return cached
        
        # Generate embedding
        embedding = self._embeddings.embed_query(text)
        
        # Cache result
        if self._cache_enabled and self._cache:
            self._cache.set(text, embedding)
        
        return embedding
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,)),
    )
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple documents.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        texts_to_embed = []
        cached_indices = {}
        
        # Check cache for each text
        if self._cache_enabled and self._cache:
            for i, text in enumerate(texts):
                cached = self._cache.get(text)
                if cached is not None:
                    cached_indices[i] = cached
                else:
                    texts_to_embed.append((i, text))
        else:
            texts_to_embed = list(enumerate(texts))
        
        # Embed non-cached texts in batches
        if texts_to_embed:
            batch_texts = [t[1] for t in texts_to_embed]
            new_embeddings = self._embeddings.embed_documents(batch_texts)
            
            # Cache new embeddings
            for (idx, text), embedding in zip(texts_to_embed, new_embeddings):
                if self._cache_enabled and self._cache:
                    self._cache.set(text, embedding)
                cached_indices[idx] = embedding
        
        # Reconstruct in original order
        embeddings = [cached_indices[i] for i in range(len(texts))]
        
        logger.debug(
            f"Embedded {len(texts)} documents "
            f"({len(texts) - len(texts_to_embed)} from cache)"
        )
        
        return embeddings
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        if self._cache:
            self._cache.clear()
            logger.info("Embedding cache cleared")


class VectorStoreManager:
    """
    ChromaDB vector store manager for document storage and retrieval.
    
    Features:
    - Persistent storage with ChromaDB
    - Google Gemini embeddings
    - Similarity search with filtering
    - Document deduplication
    - Collection management
    
    Example:
        # Initialize manager
        manager = VectorStoreManager()
        
        # Add documents from processor
        manager.add_documents(processed_docs)
        
        # Search for relevant content
        results = manager.similarity_search(
            query="What are the main results?",
            k=5,
            filter_dict={"filename": "paper.pdf"}
        )
        
        # Get all documents from a file
        docs = manager.get_documents_by_filename("paper.pdf")
    """
    
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        """
        Initialize the vector store manager.
        
        Args:
            persist_directory: Directory for ChromaDB persistence
            collection_name: Name of the ChromaDB collection
            embedding_model: Gemini embedding model to use
        """
        settings = get_settings()
        
        self.persist_directory = persist_directory or settings.chroma_persist_dir
        self.collection_name = collection_name or settings.chroma_collection_name
        self.embedding_model = embedding_model or settings.embedding_model
        
        # Ensure persist directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        self._client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )
        
        # Initialize embeddings
        self._embeddings = GeminiEmbeddings(model=self.embedding_model)
        
        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )
        
        logger.info(
            f"VectorStoreManager initialized: "
            f"collection='{self.collection_name}', "
            f"persist_dir='{self.persist_directory}'"
        )
    
    def add_documents(
        self,
        documents: list[Document],
        batch_size: int = 50,
    ) -> list[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of LangChain Document objects
            batch_size: Number of documents to process per batch
            
        Returns:
            List of document IDs that were added
        """
        if not documents:
            logger.warning("No documents to add")
            return []
        
        added_ids = []
        
        # Process in batches to avoid memory issues
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_ids = self._add_batch(batch)
            added_ids.extend(batch_ids)
            
            logger.debug(f"Added batch {i//batch_size + 1}, total: {len(added_ids)}")
        
        logger.info(f"Added {len(added_ids)} documents to vector store")
        return added_ids
    
    def _add_batch(self, documents: list[Document]) -> list[str]:
        """Add a batch of documents to the collection."""
        
        ids = []
        texts = []
        metadatas = []
        
        for doc in documents:
            # Generate unique ID based on content and metadata
            doc_id = self._generate_doc_id(doc)
            
            # Check if document already exists
            if self._document_exists(doc_id):
                logger.debug(f"Document {doc_id} already exists, skipping")
                continue
            
            ids.append(doc_id)
            texts.append(doc.page_content)
            
            # Ensure metadata values are valid types for ChromaDB
            clean_metadata = self._clean_metadata(doc.metadata)
            metadatas.append(clean_metadata)
        
        if not ids:
            return []
        
        # Generate embeddings
        embeddings = self._embeddings.embed_documents(texts)
        
        # Add to collection
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        
        return ids
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> list[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Search query text
            k: Number of results to return
            filter_dict: Metadata filters (e.g., {"filename": "paper.pdf"})
            score_threshold: Minimum similarity score (0-1, higher is more similar)
            
        Returns:
            List of Document objects sorted by relevance
            
        Example:
            # Basic search
            results = store.similarity_search("neural networks", k=5)
            
            # Search within specific document
            results = store.similarity_search(
                "methodology",
                k=3,
                filter_dict={"filename": "research.pdf"}
            )
        """
        try:
            # Generate query embedding
            query_embedding = self._embeddings.embed_query(query)
            
            # Build filter for ChromaDB
            where_filter = self._build_where_filter(filter_dict) if filter_dict else None
            
            # Query collection
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
            
            # Convert to Document objects
            documents = []
            
            if results["documents"] and results["documents"][0]:
                for i, doc_text in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    distance = results["distances"][0][i] if results["distances"] else 0
                    
                    # Convert distance to similarity score (cosine distance to similarity)
                    similarity = 1 - distance
                    
                    # Apply score threshold if specified
                    if score_threshold and similarity < score_threshold:
                        continue
                    
                    # Add similarity score to metadata
                    metadata["similarity_score"] = round(similarity, 4)
                    
                    documents.append(Document(
                        page_content=doc_text,
                        metadata=metadata,
                    ))
            
            logger.debug(f"Found {len(documents)} results for query: '{query[:50]}...'")
            return documents
            
        except Exception as e:
            logger.error(f"Similarity search error: {e}")
            raise
    
    def similarity_search_with_scores(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[dict[str, Any]] = None,
    ) -> list[tuple[Document, float]]:
        """
        Search with explicit similarity scores.
        
        Args:
            query: Search query
            k: Number of results
            filter_dict: Metadata filters
            
        Returns:
            List of (Document, score) tuples
        """
        documents = self.similarity_search(query, k, filter_dict)
        return [
            (doc, doc.metadata.get("similarity_score", 0.0))
            for doc in documents
        ]
    
    def get_documents_by_filename(self, filename: str) -> list[Document]:
        """
        Get all documents from a specific file.
        
        Args:
            filename: Name of the file
            
        Returns:
            List of Document objects from that file
        """
        results = self._collection.get(
            where={"filename": filename},
            include=["documents", "metadatas"],
        )
        
        documents = []
        if results["documents"]:
            for i, doc_text in enumerate(results["documents"]):
                metadata = results["metadatas"][i] if results["metadatas"] else {}
                documents.append(Document(
                    page_content=doc_text,
                    metadata=metadata,
                ))
        
        # Sort by chunk index if available
        documents.sort(key=lambda d: d.metadata.get("chunk_index", 0))
        
        return documents
    
    def list_documents(self) -> list[dict]:
        """
        List all unique documents in the store.
        
        Returns:
            List of document info dictionaries
        """
        # Get all items
        results = self._collection.get(include=["metadatas"])
        
        # Extract unique documents
        seen = set()
        documents = []
        
        if results["metadatas"]:
            for metadata in results["metadatas"]:
                filename = metadata.get("filename", "Unknown")
                if filename not in seen:
                    seen.add(filename)
                    documents.append({
                        "filename": filename,
                        "title": metadata.get("title", "Unknown"),
                        "authors": metadata.get("authors", "Unknown"),
                        "total_pages": metadata.get("total_pages", 0),
                        "chunk_total": metadata.get("chunk_total", 0),
                    })
        
        return documents
    
    def delete_document(self, filename: str) -> int:
        """
        Delete all chunks from a specific document.
        
        Args:
            filename: Name of the file to delete
            
        Returns:
            Number of chunks deleted
        """
        # Get IDs of documents to delete
        results = self._collection.get(
            where={"filename": filename},
            include=[],
        )
        
        if results["ids"]:
            self._collection.delete(ids=results["ids"])
            logger.info(f"Deleted {len(results['ids'])} chunks for '{filename}'")
            return len(results["ids"])
        
        return 0
    
    def clear_collection(self) -> None:
        """Delete all documents from the collection."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Collection '{self.collection_name}' cleared")
    
    def get_stats(self) -> dict:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with store statistics
        """
        count = self._collection.count()
        documents = self.list_documents()
        
        return {
            "total_chunks": count,
            "total_documents": len(documents),
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "embedding_model": self.embedding_model,
            "documents": documents,
        }
    
    def _generate_doc_id(self, doc: Document) -> str:
        """Generate unique ID for a document chunk."""
        # Combine filename, chunk index, and content hash
        filename = doc.metadata.get("filename", "unknown")
        chunk_idx = doc.metadata.get("chunk_index", 0)
        content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()[:8]
        
        return f"{filename}_{chunk_idx}_{content_hash}"
    
    def _document_exists(self, doc_id: str) -> bool:
        """Check if a document ID already exists."""
        results = self._collection.get(ids=[doc_id], include=[])
        return len(results["ids"]) > 0
    
    def _clean_metadata(self, metadata: dict) -> dict:
        """Ensure metadata values are valid for ChromaDB."""
        cleaned = {}
        for key, value in metadata.items():
            if value is None:
                cleaned[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            elif isinstance(value, list):
                cleaned[key] = ", ".join(str(v) for v in value)
            else:
                cleaned[key] = str(value)
        return cleaned
    
    def _build_where_filter(self, filter_dict: dict[str, Any]) -> dict:
        """Build ChromaDB where filter from dictionary."""
        if len(filter_dict) == 1:
            key, value = next(iter(filter_dict.items()))
            return {key: value}
        
        # Multiple conditions with AND
        return {
            "$and": [{k: v} for k, v in filter_dict.items()]
        }


# Convenience function for quick setup
@lru_cache()
def get_vector_store() -> VectorStoreManager:
    """
    Get a cached vector store instance.
    
    Returns:
        VectorStoreManager: Singleton instance
        
    Example:
        store = get_vector_store()
        results = store.similarity_search("query")
    """
    return VectorStoreManager()


if __name__ == "__main__":
    # Example usage and testing
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    print("Vector Store Manager - Test")
    print("-" * 50)
    
    try:
        # Initialize store
        store = VectorStoreManager()
        
        # Get stats
        stats = store.get_stats()
        print(f"Collection: {stats['collection_name']}")
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Total documents: {stats['total_documents']}")
        
        if stats['documents']:
            print("\nStored documents:")
            for doc in stats['documents']:
                print(f"  - {doc['filename']}: {doc['title']}")
        
        # Test search if documents exist
        if stats['total_chunks'] > 0:
            print("\nTest search for 'introduction':")
            results = store.similarity_search("introduction", k=2)
            for i, doc in enumerate(results):
                print(f"\n  Result {i+1} (score: {doc.metadata.get('similarity_score', 'N/A')}):")
                print(f"    Source: {doc.metadata.get('filename')}")
                print(f"    Content: {doc.page_content[:100]}...")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
