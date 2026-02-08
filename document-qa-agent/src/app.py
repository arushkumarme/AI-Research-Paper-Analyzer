"""
Document Q&A AI Agent - Streamlit Web Application

A professional web interface for document question-answering using:
- Google Gemini API for LLM and embeddings
- ChromaDB for vector storage
- RAG-based retrieval and generation
- ArXiv integration for paper search

Run with: streamlit run src/app.py
"""

import logging
import time
from pathlib import Path

import streamlit as st

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("document_qa.app")

# Must be first Streamlit command
st.set_page_config(
    page_title="Document Q&A AI Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import application modules (after Streamlit config)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from config import get_settings, validate_environment, setup_logging
    from src.document_processor import DocumentProcessor
    from src.vector_store import VectorStoreManager
    from src.llm_agent import DocumentQAAgent, QueryType
    from src.arxiv_tool import ArxivTool, create_arxiv_agent, download_and_process_arxiv_paper
    MODULES_LOADED = True
except ImportError as e:
    MODULES_LOADED = False
    IMPORT_ERROR = str(e)


# ============================================
# Custom CSS Styling
# ============================================

CUSTOM_CSS = """
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Headers */
    h1 {
        color: #1E3A5F;
        border-bottom: 3px solid #4A90D9;
        padding-bottom: 0.5rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        padding-top: 1rem;
    }
    
    /* Upload area */
    .uploadedFile {
        border: 2px dashed #4A90D9;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Query result box */
    .result-box {
        background-color: #F0F7FF;
        border-left: 4px solid #4A90D9;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    /* Source citation */
    .source-citation {
        background-color: #E8F4EA;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        margin: 0.25rem 0;
        font-size: 0.9em;
    }
    
    /* Stats card */
    .stats-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    
    /* Example query button */
    .stButton > button {
        width: 100%;
        text-align: left;
        background-color: #F8F9FA;
        border: 1px solid #E9ECEF;
    }
    
    .stButton > button:hover {
        background-color: #E9ECEF;
        border-color: #4A90D9;
    }
    
    /* Processing indicator */
    .processing {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
</style>
"""


# ============================================
# Session State Initialization
# ============================================

def init_session_state():
    """Initialize Streamlit session state variables."""
    
    defaults = {
        "documents_processed": [],
        "total_chunks": 0,
        "agent": None,
        "vector_store": None,
        "processor": None,
        "arxiv_tool": None,
        "query_history": [],
        "processing": False,
        "initialized": False,
        "error_message": None,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def initialize_components():
    """Initialize document processor, vector store, and agent."""
    
    if st.session_state.initialized:
        return True
    
    try:
        with st.spinner("🔧 Initializing components..."):
            # Validate environment
            settings = get_settings()
            
            # Initialize components
            st.session_state.processor = DocumentProcessor(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
            
            st.session_state.vector_store = VectorStoreManager()
            
            st.session_state.agent = DocumentQAAgent(
                vector_store=st.session_state.vector_store,
            )
            
            st.session_state.arxiv_tool = ArxivTool()
            
            # Load existing documents from vector store
            stats = st.session_state.vector_store.get_stats()
            if stats["documents"]:
                st.session_state.documents_processed = [
                    doc["filename"] for doc in stats["documents"]
                ]
                st.session_state.total_chunks = stats["total_chunks"]
            
            st.session_state.initialized = True
            logger.info("Components initialized successfully")
            return True
            
    except Exception as e:
        st.session_state.error_message = str(e)
        logger.error(f"Initialization failed: {e}")
        return False


# ============================================
# UI Components
# ============================================

def render_header():
    """Render the main header and description."""
    
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.title("📚 Document Q&A AI Agent")
        st.markdown("""
        **Intelligent document question-answering powered by Google Gemini**
        
        Upload PDF documents and ask questions in natural language. The agent supports three query types:
        """)
    
    with col2:
        st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
    
    # Query type cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background-color:#E3F2FD;padding:1rem;border-radius:10px;height:120px;">
        <h4>🔍 Direct Lookup</h4>
        <p style="font-size:0.9em;">Find specific information in your documents</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color:#E8F5E9;padding:1rem;border-radius:10px;height:120px;">
        <h4>📝 Summarization</h4>
        <p style="font-size:0.9em;">Get concise summaries of sections or papers</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background-color:#FFF3E0;padding:1rem;border-radius:10px;height:120px;">
        <h4>📊 Metric Extraction</h4>
        <p style="font-size:0.9em;">Extract accuracy, F1-scores, and other metrics</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")


def render_sidebar():
    """Render the sidebar with upload and settings."""
    
    with st.sidebar:
        st.header("📁 Document Management")
        
        # File upload
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF documents to query",
        )
        
        # Process uploaded files
        if uploaded_files:
            process_uploads(uploaded_files)
        
        st.markdown("---")
        
        # Document statistics
        render_stats()
        
        st.markdown("---")
        
        # ArXiv Integration
        render_arxiv_section()
        
        st.markdown("---")
        
        # Actions
        st.subheader("⚙️ Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear All", use_container_width=True):
                clear_all_data()
        
        with col2:
            if st.button("🔄 Refresh", use_container_width=True):
                refresh_stats()
        
        # Settings expander
        with st.expander("⚙️ Settings"):
            st.number_input(
                "Results per query",
                min_value=1,
                max_value=10,
                value=5,
                key="num_results",
            )
            
            st.checkbox(
                "Show confidence scores",
                value=True,
                key="show_confidence",
            )
            
            st.checkbox(
                "Show source citations",
                value=True,
                key="show_sources",
            )


def render_stats():
    """Render document statistics."""
    
    st.subheader("📊 Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Documents",
            value=len(st.session_state.documents_processed),
        )
    
    with col2:
        st.metric(
            label="Chunks",
            value=st.session_state.total_chunks,
        )
    
    # List processed documents
    if st.session_state.documents_processed:
        with st.expander(f"📄 Uploaded Documents ({len(st.session_state.documents_processed)})"):
            for doc in st.session_state.documents_processed:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(f"• {doc}")
                with col2:
                    if st.button("🗑️", key=f"del_{doc}", help=f"Delete {doc}"):
                        delete_document(doc)


def render_arxiv_section():
    """Render ArXiv search section in sidebar."""
    
    st.subheader("🔬 ArXiv Search")
    
    arxiv_query = st.text_input(
        "Search ArXiv",
        placeholder="e.g., transformer attention",
        key="arxiv_query",
    )
    
    if st.button("🔍 Search Papers", use_container_width=True):
        if arxiv_query:
            search_arxiv(arxiv_query)
    
    # Show search results if any
    if "arxiv_results" in st.session_state and st.session_state.arxiv_results:
        with st.expander("📑 Search Results", expanded=True):
            for paper in st.session_state.arxiv_results[:5]:
                st.markdown(f"**{paper.title[:60]}...**")
                st.caption(f"Authors: {', '.join(paper.authors[:2])}...")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"[PDF]({paper.pdf_url})")
                with col2:
                    if st.button("📥 Add", key=f"add_{paper.arxiv_id}"):
                        download_arxiv_paper(paper.arxiv_id)
                
                st.markdown("---")


def render_query_interface():
    """Render the main query interface."""
    
    st.subheader("💬 Ask a Question")
    
    # Example queries
    with st.expander("📝 Example Queries", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🔍 Lookup**")
            if st.button("What is the conclusion?", key="ex1"):
                st.session_state.query_input = "What is the conclusion of the paper?"
            if st.button("What methodology was used?", key="ex2"):
                st.session_state.query_input = "What methodology was used in this research?"
        
        with col2:
            st.markdown("**📝 Summarization**")
            if st.button("Summarize the paper", key="ex3"):
                st.session_state.query_input = "Summarize the main findings of this paper"
            if st.button("Summarize methodology", key="ex4"):
                st.session_state.query_input = "Summarize the methodology section"
        
        with col3:
            st.markdown("**📊 Extraction**")
            if st.button("What are the accuracy scores?", key="ex5"):
                st.session_state.query_input = "What are the accuracy and F1 scores reported?"
            if st.button("List all metrics", key="ex6"):
                st.session_state.query_input = "Extract all performance metrics from the paper"
    
    # Query input
    query = st.text_area(
        "Your Question",
        value=st.session_state.get("query_input", ""),
        placeholder="Ask anything about your documents...",
        height=100,
        key="query_text",
    )
    
    # Document filter
    doc_options = ["All Documents"] + st.session_state.documents_processed
    selected_doc = st.selectbox(
        "Filter by document (optional)",
        doc_options,
        key="doc_filter",
    )
    
    # Submit button
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        submit_query = st.button(
            "🚀 Ask Question",
            type="primary",
            use_container_width=True,
        )
    
    with col2:
        use_arxiv = st.checkbox("Include ArXiv", key="use_arxiv")
    
    with col3:
        clear_query = st.button("Clear", use_container_width=True)
        if clear_query:
            st.session_state.query_input = ""
            st.rerun()
    
    # Process query
    if submit_query and query:
        doc_filter = None if selected_doc == "All Documents" else selected_doc
        process_query(query, doc_filter, use_arxiv)


def render_results():
    """Render query results."""
    
    if "current_result" not in st.session_state or not st.session_state.current_result:
        # Show placeholder
        st.info("👆 Upload documents and ask a question to get started!")
        return
    
    result = st.session_state.current_result
    
    st.subheader("📋 Answer")
    
    # Result metadata
    col1, col2, col3 = st.columns(3)
    
    with col1:
        query_type = result.get("query_type", "general")
        type_emoji = {
            "lookup": "🔍",
            "summarize": "📝",
            "extract": "📊",
            "compare": "⚖️",
            "general": "💡",
        }
        st.markdown(f"**Type:** {type_emoji.get(query_type, '💡')} {query_type.title()}")
    
    with col2:
        if st.session_state.get("show_confidence", True):
            confidence = result.get("confidence", 0)
            color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
            st.markdown(f"**Confidence:** :{color}[{confidence:.0%}]")
    
    with col3:
        processing_time = result.get("processing_time", 0)
        st.markdown(f"**Time:** {processing_time:.2f}s")
    
    # Answer content
    st.markdown("---")
    
    answer = result.get("answer", "No answer available.")
    st.markdown(
        f"""
        <div style="background-color:#F8F9FA;padding:1.5rem;border-radius:10px;border-left:4px solid #4A90D9;">
        {answer}
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Source citations
    if st.session_state.get("show_sources", True) and result.get("sources"):
        st.markdown("---")
        st.subheader("📚 Sources")
        
        for i, source in enumerate(result["sources"][:5], 1):
            with st.expander(f"Source {i}: {source.get('filename', 'Unknown')}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Page:** {source.get('page', 'N/A')}")
                    st.markdown(f"**Section:** {source.get('section', 'Unknown')}")
                
                with col2:
                    relevance = source.get("relevance_score", 0)
                    st.markdown(f"**Relevance:** {relevance:.2%}")
    
    # Query history
    if st.session_state.query_history:
        with st.expander("📜 Query History"):
            for i, hist in enumerate(reversed(st.session_state.query_history[-5:]), 1):
                st.markdown(f"**{i}.** {hist['query'][:80]}...")
                st.caption(f"Type: {hist['type']} | Time: {hist['time']:.2f}s")


# ============================================
# Processing Functions
# ============================================

def process_uploads(uploaded_files):
    """Process uploaded PDF files."""
    
    if not st.session_state.initialized:
        st.warning("⚠️ Please wait for initialization to complete.")
        return
    
    new_files = []
    for file in uploaded_files:
        if file.name not in st.session_state.documents_processed:
            new_files.append(file)
    
    if not new_files:
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, file in enumerate(new_files):
        try:
            status_text.text(f"Processing {file.name}...")
            progress_bar.progress((i) / len(new_files))
            
            # Read file bytes
            file_bytes = file.read()
            file.seek(0)  # Reset for potential re-read
            
            # Process PDF
            documents = st.session_state.processor.process_pdf_bytes(
                file_bytes,
                file.name,
            )
            
            # Add to vector store
            st.session_state.vector_store.add_documents(documents)
            
            # Update state
            st.session_state.documents_processed.append(file.name)
            st.session_state.total_chunks += len(documents)
            
            logger.info(f"Processed {file.name}: {len(documents)} chunks")
            
        except Exception as e:
            st.error(f"❌ Error processing {file.name}: {str(e)}")
            logger.error(f"Error processing {file.name}: {e}")
    
    progress_bar.progress(1.0)
    status_text.text("✅ Processing complete!")
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()
    
    st.success(f"✅ Successfully processed {len(new_files)} document(s)!")
    st.rerun()


def process_query(query: str, doc_filter: str = None, use_arxiv: bool = False):
    """Process a user query."""
    
    if not st.session_state.initialized:
        st.error("❌ System not initialized. Please check your configuration.")
        return
    
    if not st.session_state.documents_processed and not use_arxiv:
        st.warning("⚠️ Please upload at least one document first, or enable ArXiv search.")
        return
    
    try:
        with st.spinner("🤔 Thinking..."):
            start_time = time.time()
            
            # Use ArXiv-enabled agent if requested
            if use_arxiv:
                arxiv_agent = create_arxiv_agent()
                result = arxiv_agent.query_with_tools(query)
            else:
                result = st.session_state.agent.query(
                    query,
                    document_filter=doc_filter,
                    num_context=st.session_state.get("num_results", 5),
                )
            
            processing_time = time.time() - start_time
            
            # Store result
            st.session_state.current_result = {
                "answer": result.answer,
                "query_type": result.query_type.value,
                "sources": result.sources,
                "confidence": result.confidence,
                "processing_time": processing_time,
                "cached": result.cached,
            }
            
            # Add to history
            st.session_state.query_history.append({
                "query": query,
                "type": result.query_type.value,
                "time": processing_time,
            })
            
            logger.info(f"Query processed in {processing_time:.2f}s")
            
    except Exception as e:
        st.error(f"❌ Error processing query: {str(e)}")
        logger.error(f"Query error: {e}")


def search_arxiv(query: str):
    """Search ArXiv for papers."""
    
    if not st.session_state.arxiv_tool:
        st.error("❌ ArXiv tool not initialized.")
        return
    
    try:
        with st.spinner("🔍 Searching ArXiv..."):
            results = st.session_state.arxiv_tool.search(query, max_results=5)
            st.session_state.arxiv_results = results.papers
            
            if results.papers:
                st.success(f"✅ Found {len(results.papers)} papers!")
            else:
                st.info("No papers found for your query.")
                
    except Exception as e:
        st.error(f"❌ ArXiv search failed: {str(e)}")
        logger.error(f"ArXiv search error: {e}")


def download_arxiv_paper(arxiv_id: str):
    """Download and process an ArXiv paper."""
    
    try:
        with st.spinner(f"📥 Downloading {arxiv_id}..."):
            # Download PDF
            pdf_path = download_and_process_arxiv_paper(arxiv_id)
            
            if pdf_path and pdf_path.exists():
                # Process the PDF
                documents = st.session_state.processor.process_pdf(pdf_path)
                
                # Add to vector store
                st.session_state.vector_store.add_documents(documents)
                
                # Update state
                filename = pdf_path.name
                st.session_state.documents_processed.append(filename)
                st.session_state.total_chunks += len(documents)
                
                st.success(f"✅ Added {filename} ({len(documents)} chunks)")
                st.rerun()
            else:
                st.error("❌ Failed to download paper.")
                
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        logger.error(f"ArXiv download error: {e}")


def delete_document(filename: str):
    """Delete a document from the system."""
    
    try:
        # Remove from vector store
        chunks_deleted = st.session_state.vector_store.delete_document(filename)
        
        # Update state
        if filename in st.session_state.documents_processed:
            st.session_state.documents_processed.remove(filename)
            st.session_state.total_chunks -= chunks_deleted
        
        st.success(f"✅ Deleted {filename}")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error deleting document: {str(e)}")


def clear_all_data():
    """Clear all documents and reset the system."""
    
    try:
        # Clear vector store
        if st.session_state.vector_store:
            st.session_state.vector_store.clear_collection()
        
        # Clear agent cache
        if st.session_state.agent:
            st.session_state.agent.clear_cache()
        
        # Reset session state
        st.session_state.documents_processed = []
        st.session_state.total_chunks = 0
        st.session_state.current_result = None
        st.session_state.query_history = []
        st.session_state.arxiv_results = None
        
        st.success("✅ All data cleared!")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error clearing data: {str(e)}")


def refresh_stats():
    """Refresh statistics from vector store."""
    
    try:
        if st.session_state.vector_store:
            stats = st.session_state.vector_store.get_stats()
            st.session_state.documents_processed = [
                doc["filename"] for doc in stats["documents"]
            ]
            st.session_state.total_chunks = stats["total_chunks"]
            st.success("✅ Stats refreshed!")
            st.rerun()
    except Exception as e:
        st.error(f"❌ Error refreshing stats: {str(e)}")


# ============================================
# Error Display
# ============================================

def render_config_error():
    """Render configuration error page."""
    
    st.error("⚠️ Configuration Error")
    
    st.markdown("""
    ### Setup Required
    
    The application could not start due to a configuration issue.
    
    **Please follow these steps:**
    
    1. **Create a `.env` file** in the project root:
       ```
       cp .env.example .env
       ```
    
    2. **Add your Google Gemini API key**:
       ```
       GOOGLE_API_KEY=your_api_key_here
       ```
    
    3. **Get a free API key** from: [Google AI Studio](https://aistudio.google.com/apikey)
    
    4. **Restart the application**:
       ```
       streamlit run src/app.py
       ```
    """)
    
    if st.session_state.get("error_message"):
        with st.expander("🔍 Technical Details"):
            st.code(st.session_state.error_message)


def render_import_error():
    """Render import error page."""
    
    st.error("⚠️ Import Error")
    
    st.markdown(f"""
    ### Dependencies Missing
    
    Some required modules could not be imported.
    
    **Error:** `{IMPORT_ERROR}`
    
    **Please install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    
    Then restart the application.
    """)


# ============================================
# Main Application
# ============================================

def main():
    """Main application entry point."""
    
    # Check if modules loaded
    if not MODULES_LOADED:
        render_import_error()
        return
    
    # Initialize session state
    init_session_state()
    
    # Try to initialize components
    if not st.session_state.initialized:
        if not initialize_components():
            render_config_error()
            return
    
    # Render UI
    render_header()
    render_sidebar()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        render_query_interface()
    
    with col2:
        # Quick tips
        with st.expander("💡 Tips", expanded=True):
            st.markdown("""
            **Best practices:**
            - Upload relevant PDF documents first
            - Be specific in your questions
            - Use document filter for targeted queries
            - Check source citations for verification
            
            **Query tips:**
            - "What is..." for direct lookup
            - "Summarize..." for summaries
            - "Extract metrics..." for data extraction
            - "Compare..." for comparisons
            """)
    
    st.markdown("---")
    
    # Results section
    render_results()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align:center;color:#666;font-size:0.9em;">
        📚 Document Q&A AI Agent | Powered by Google Gemini | 
        <a href="https://github.com" target="_blank">GitHub</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
