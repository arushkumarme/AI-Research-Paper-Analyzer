"""
Document Q&A AI Agent

A RAG-based document question-answering system using Google Gemini API.

Modules:
    - document_processor: PDF ingestion and text extraction
    - vector_store: ChromaDB vector storage and retrieval
    - llm_agent: Gemini-powered Q&A agent
    - arxiv_tool: ArXiv paper search and download
    - utils: Helper functions and utilities

Usage:
    from src import DocumentProcessor, VectorStoreManager, DocumentQAAgent
    
    # Process a PDF
    processor = DocumentProcessor()
    docs = processor.process_pdf("paper.pdf")
    
    # Store in vector database
    store = VectorStoreManager()
    store.add_documents(docs)
    
    # Query with agent
    agent = DocumentQAAgent(vector_store=store)
    result = agent.query("What is the main finding?")
    print(result.answer)
"""

__version__ = "0.1.0"
__author__ = "Document QA Team"
__all__ = [
    "DocumentProcessor",
    "VectorStoreManager",
    "DocumentQAAgent",
    "AgentWithTools",
    "ArxivTool",
    "QueryType",
    "QueryResult",
]

# Lazy imports to avoid circular dependencies and speed up initial import
def __getattr__(name: str):
    """Lazy import of submodules."""
    if name == "DocumentProcessor":
        from src.document_processor import DocumentProcessor
        return DocumentProcessor
    elif name == "VectorStoreManager":
        from src.vector_store import VectorStoreManager
        return VectorStoreManager
    elif name == "DocumentQAAgent":
        from src.llm_agent import DocumentQAAgent
        return DocumentQAAgent
    elif name == "AgentWithTools":
        from src.llm_agent import AgentWithTools
        return AgentWithTools
    elif name == "ArxivTool":
        from src.arxiv_tool import ArxivTool
        return ArxivTool
    elif name == "QueryType":
        from src.llm_agent import QueryType
        return QueryType
    elif name == "QueryResult":
        from src.llm_agent import QueryResult
        return QueryResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
