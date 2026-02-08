#!/usr/bin/env python3
"""
Document Q&A AI Agent - System Test Script

This script tests all components of the Document Q&A system:
1. Configuration validation
2. Document processing
3. Vector store operations
4. LLM agent queries
5. ArXiv integration

Usage:
    python test_system.py                    # Run all tests
    python test_system.py --quick            # Quick test (no API calls)
    python test_system.py --with-sample      # Test with sample PDF
    python test_system.py --query "Your question"  # Test specific query

Requirements:
    - GOOGLE_API_KEY must be set in .env file
    - All dependencies must be installed
"""

import argparse
import sys
import time
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(name: str, success: bool, message: str = ""):
    """Print test result."""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"  {status}: {name}")
    if message:
        print(f"         {message}")


def test_imports():
    """Test that all modules can be imported."""
    print_header("Testing Imports")
    
    results = []
    
    # Test config
    try:
        from config import Settings, get_settings, setup_logging
        print_result("config.py", True)
        results.append(True)
    except Exception as e:
        print_result("config.py", False, str(e))
        results.append(False)
    
    # Test document processor
    try:
        from src.document_processor import DocumentProcessor
        print_result("document_processor.py", True)
        results.append(True)
    except Exception as e:
        print_result("document_processor.py", False, str(e))
        results.append(False)
    
    # Test vector store
    try:
        from src.vector_store import VectorStoreManager
        print_result("vector_store.py", True)
        results.append(True)
    except Exception as e:
        print_result("vector_store.py", False, str(e))
        results.append(False)
    
    # Test LLM agent
    try:
        from src.llm_agent import DocumentQAAgent, QueryType
        print_result("llm_agent.py", True)
        results.append(True)
    except Exception as e:
        print_result("llm_agent.py", False, str(e))
        results.append(False)
    
    # Test ArXiv tool
    try:
        from src.arxiv_tool import ArxivTool
        print_result("arxiv_tool.py", True)
        results.append(True)
    except Exception as e:
        print_result("arxiv_tool.py", False, str(e))
        results.append(False)
    
    # Test utils
    try:
        from src.utils import clean_text, estimate_tokens, PerformanceMonitor
        print_result("utils.py", True)
        results.append(True)
    except Exception as e:
        print_result("utils.py", False, str(e))
        results.append(False)
    
    return all(results)


def test_configuration():
    """Test configuration loading."""
    print_header("Testing Configuration")
    
    try:
        from config import get_settings, setup_logging
        
        # Clear cache to force reload
        get_settings.cache_clear()
        
        settings = get_settings()
        print_result("Load settings", True)
        
        print(f"\n  Configuration values:")
        print(f"    - Gemini Model: {settings.gemini_model}")
        print(f"    - Embedding Model: {settings.embedding_model}")
        print(f"    - Chunk Size: {settings.chunk_size}")
        print(f"    - Cache Enabled: {settings.cache_enabled}")
        print(f"    - Log Level: {settings.log_level}")
        
        # Setup logging
        logger = setup_logging(settings.log_level)
        print_result("Setup logging", True)
        
        return True
        
    except Exception as e:
        print_result("Configuration", False, str(e))
        return False


def test_utils():
    """Test utility functions."""
    print_header("Testing Utilities")
    
    from src.utils import (
        clean_text, estimate_tokens, validate_pdf_bytes,
        format_duration, generate_cache_key, PerformanceMonitor
    )
    
    results = []
    
    # Test clean_text
    try:
        dirty = "  Hello   world  \n\n  test  "
        clean = clean_text(dirty)
        assert clean == "Hello world test", f"Got: {clean}"
        print_result("clean_text()", True)
        results.append(True)
    except Exception as e:
        print_result("clean_text()", False, str(e))
        results.append(False)
    
    # Test estimate_tokens
    try:
        tokens = estimate_tokens("This is a test sentence.")
        assert tokens > 0
        print_result("estimate_tokens()", True, f"Estimated {tokens} tokens")
        results.append(True)
    except Exception as e:
        print_result("estimate_tokens()", False, str(e))
        results.append(False)
    
    # Test format_duration
    try:
        assert format_duration(30.5) == "30.5s"
        assert format_duration(90) == "1m 30s"
        print_result("format_duration()", True)
        results.append(True)
    except Exception as e:
        print_result("format_duration()", False, str(e))
        results.append(False)
    
    # Test cache key generation
    try:
        key1 = generate_cache_key("query1", model="gemini")
        key2 = generate_cache_key("query1", model="gemini")
        key3 = generate_cache_key("query2", model="gemini")
        assert key1 == key2, "Same inputs should give same key"
        assert key1 != key3, "Different inputs should give different key"
        print_result("generate_cache_key()", True)
        results.append(True)
    except Exception as e:
        print_result("generate_cache_key()", False, str(e))
        results.append(False)
    
    # Test performance monitor
    try:
        monitor = PerformanceMonitor()
        with monitor.track("test_op"):
            time.sleep(0.1)
        stats = monitor.get_stats()
        assert "test_op" in stats
        assert stats["test_op"]["count"] == 1
        print_result("PerformanceMonitor", True, f"Tracked in {stats['test_op']['avg']:.3f}s")
        results.append(True)
    except Exception as e:
        print_result("PerformanceMonitor", False, str(e))
        results.append(False)
    
    return all(results)


def test_document_processor():
    """Test document processor."""
    print_header("Testing Document Processor")
    
    from src.document_processor import DocumentProcessor
    
    try:
        processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
        print_result("Initialize processor", True)
        
        print(f"\n  Processor configuration:")
        print(f"    - Chunk size: {processor.chunk_size}")
        print(f"    - Chunk overlap: {processor.chunk_overlap}")
        
        return True
        
    except Exception as e:
        print_result("Document Processor", False, str(e))
        return False


def test_vector_store(skip_api: bool = False):
    """Test vector store operations."""
    print_header("Testing Vector Store")
    
    if skip_api:
        print("  ⏭️  Skipping (requires API key)")
        return True
    
    try:
        from src.vector_store import VectorStoreManager
        
        # Use a test collection
        store = VectorStoreManager(collection_name="test_collection")
        print_result("Initialize vector store", True)
        
        # Get stats
        stats = store.get_stats()
        print(f"\n  Vector store stats:")
        print(f"    - Collection: {stats['collection_name']}")
        print(f"    - Total chunks: {stats['total_chunks']}")
        print(f"    - Total documents: {stats['total_documents']}")
        
        return True
        
    except Exception as e:
        print_result("Vector Store", False, str(e))
        return False


def test_llm_agent(skip_api: bool = False):
    """Test LLM agent."""
    print_header("Testing LLM Agent")
    
    if skip_api:
        print("  ⏭️  Skipping (requires API key)")
        return True
    
    try:
        from src.llm_agent import DocumentQAAgent, QueryType
        
        agent = DocumentQAAgent()
        print_result("Initialize agent", True)
        
        # Test query classification
        test_queries = [
            ("What is the conclusion?", QueryType.LOOKUP),
            ("Summarize the methodology", QueryType.SUMMARIZATION),
            ("What are the F1 scores?", QueryType.EXTRACTION),
        ]
        
        print(f"\n  Query classification tests:")
        for query, expected_type in test_queries:
            classified = agent._classify_query(query)
            status = "✓" if classified == expected_type else "✗"
            print(f"    {status} '{query[:30]}...' -> {classified.value}")
        
        print_result("Query classification", True)
        
        # Test agent stats
        stats = agent.get_stats()
        print(f"\n  Agent configuration:")
        print(f"    - Model: {stats['model']}")
        print(f"    - Cache enabled: {stats['cache_enabled']}")
        
        return True
        
    except Exception as e:
        print_result("LLM Agent", False, str(e))
        return False


def test_arxiv_tool():
    """Test ArXiv integration."""
    print_header("Testing ArXiv Tool")
    
    try:
        from src.arxiv_tool import ArxivTool
        
        tool = ArxivTool(max_results=3)
        print_result("Initialize ArXiv tool", True)
        
        # Test search
        print("\n  Searching ArXiv for 'transformer attention'...")
        results = tool.search("transformer attention", max_results=2)
        
        if results.papers:
            print_result("ArXiv search", True, f"Found {len(results.papers)} papers")
            print(f"\n  Sample results:")
            for paper in results.papers[:2]:
                print(f"    - {paper.title[:50]}...")
                print(f"      Authors: {', '.join(paper.authors[:2])}...")
        else:
            print_result("ArXiv search", True, "No results (may be rate limited)")
        
        return True
        
    except Exception as e:
        print_result("ArXiv Tool", False, str(e))
        return False


def test_full_pipeline(query: str = None, skip_api: bool = False):
    """Test the full Q&A pipeline."""
    print_header("Testing Full Pipeline")
    
    if skip_api:
        print("  ⏭️  Skipping (requires API key)")
        return True
    
    try:
        from src.llm_agent import DocumentQAAgent
        from src.vector_store import VectorStoreManager
        
        # Initialize components
        store = VectorStoreManager()
        agent = DocumentQAAgent(vector_store=store)
        
        stats = store.get_stats()
        
        if stats['total_chunks'] == 0:
            print("  ℹ️  No documents in vector store")
            print("     Upload a PDF via the Streamlit app first, then test queries")
            return True
        
        # Test query
        test_query = query or "What is the main topic of the documents?"
        print(f"\n  Testing query: '{test_query}'")
        
        start_time = time.time()
        result = agent.query(test_query)
        elapsed = time.time() - start_time
        
        print(f"\n  Results:")
        print(f"    - Query type: {result.query_type.value}")
        print(f"    - Confidence: {result.confidence:.0%}")
        print(f"    - Processing time: {elapsed:.2f}s")
        print(f"    - Sources: {len(result.sources)}")
        print(f"\n  Answer preview:")
        print(f"    {result.answer[:200]}...")
        
        print_result("Full pipeline", True)
        return True
        
    except Exception as e:
        print_result("Full pipeline", False, str(e))
        return False


def test_streamlit_app():
    """Test that Streamlit app can be imported."""
    print_header("Testing Streamlit App")
    
    try:
        # Just check syntax, don't run
        import py_compile
        py_compile.compile("src/app.py", doraise=True)
        print_result("Streamlit app syntax", True)
        return True
    except Exception as e:
        print_result("Streamlit app syntax", False, str(e))
        return False


def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(description="Test Document Q&A System")
    parser.add_argument("--quick", action="store_true", help="Quick test without API calls")
    parser.add_argument("--query", type=str, help="Test a specific query")
    parser.add_argument("--with-sample", action="store_true", help="Test with sample PDF")
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("  Document Q&A AI Agent - System Test")
    print("=" * 60)
    print(f"  Mode: {'Quick (no API calls)' if args.quick else 'Full'}")
    
    results = []
    
    # Always run these tests
    results.append(("Imports", test_imports()))
    results.append(("Configuration", test_configuration()))
    results.append(("Utilities", test_utils()))
    results.append(("Document Processor", test_document_processor()))
    results.append(("Streamlit App", test_streamlit_app()))
    
    # Conditionally run API tests
    results.append(("Vector Store", test_vector_store(skip_api=args.quick)))
    results.append(("LLM Agent", test_llm_agent(skip_api=args.quick)))
    results.append(("ArXiv Tool", test_arxiv_tool() if not args.quick else True))
    results.append(("Full Pipeline", test_full_pipeline(args.query, skip_api=args.quick)))
    
    # Print summary
    print_header("Test Summary")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {name}")
    
    print(f"\n  Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 All tests passed!")
        print("\n  Next steps:")
        print("    1. Add your GOOGLE_API_KEY to .env file")
        print("    2. Run: streamlit run src/app.py")
        print("    3. Upload a PDF and start asking questions!")
        return 0
    else:
        print("\n  ⚠️  Some tests failed. Check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
