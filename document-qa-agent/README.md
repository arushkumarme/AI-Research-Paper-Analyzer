# 📚 Document Q&A AI Agent

An intelligent document question-answering system powered by Google Gemini API. Upload PDF documents and ask questions in natural language with support for direct lookup, summarization, and metric extraction.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

- **🔍 Direct Lookup**: Find specific information in your documents
- **📝 Summarization**: Get concise summaries of sections or entire papers
- **📊 Metric Extraction**: Extract accuracy scores, F1-scores, and other metrics
- **🔬 ArXiv Integration**: Search and download papers directly from ArXiv
- **💾 Persistent Storage**: ChromaDB vector store with document persistence
- **⚡ Response Caching**: Cached responses for improved performance
- **🎯 Query Classification**: Automatic detection of query type for optimized responses

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Streamlit UI  │────▶│  Document Agent  │────▶│  Google Gemini  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌──────────────────┐
│  PDF Processor  │────▶│   ChromaDB       │
│   (PyMuPDF)     │     │  Vector Store    │
└─────────────────┘     └──────────────────┘
```

## 📋 Prerequisites

- Python 3.10 or higher
- Google Gemini API key (free tier available)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd document-qa-agent
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your Google API key
# Get your free API key from: https://aistudio.google.com/apikey
```

Your `.env` file should contain:

```env
GOOGLE_API_KEY=your_api_key_here
```

### 5. Run the Application

```bash
streamlit run src/app.py
```

The application will open in your browser at `http://localhost:8501`

## 📖 Usage Guide

### Uploading Documents

1. Use the sidebar to upload PDF files
2. Multiple files can be uploaded at once
3. Wait for processing to complete
4. View document statistics in the sidebar

### Asking Questions

**Direct Lookup:**
```
What is the conclusion of the paper?
What methodology was used in this research?
What does the paper say about neural networks?
```

**Summarization:**
```
Summarize the main findings of this paper
Give me an overview of the methodology section
Summarize the experimental results
```

**Metric Extraction:**
```
What are the accuracy and F1 scores reported?
Extract all performance metrics from the paper
What were the benchmark results?
```

**Comparison:**
```
Compare the approaches described in the uploaded papers
What are the differences in methodology between Paper A and Paper B?
```

### ArXiv Integration

1. Enter a search query in the ArXiv Search section
2. Click "Search Papers" to find relevant papers
3. Click "Add" to download and process a paper
4. Query the paper like any other document

## 🛠️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Google Gemini API key | Required |
| `GEMINI_MODEL` | Gemini model for generation | `gemini-1.5-flash` |
| `EMBEDDING_MODEL` | Model for embeddings | `models/embedding-001` |
| `CHROMA_PERSIST_DIR` | ChromaDB storage directory | `./chroma_db` |
| `CHUNK_SIZE` | Text chunk size | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |
| `LOG_LEVEL` | Logging level | `INFO` |

### Customizing Chunk Size

For longer documents, you may want to adjust chunking:

```env
CHUNK_SIZE=1500
CHUNK_OVERLAP=300
```

## 📁 Project Structure

```
document-qa-agent/
├── src/
│   ├── __init__.py           # Package initializer
│   ├── app.py                # Streamlit web application
│   ├── document_processor.py # PDF processing & chunking
│   ├── vector_store.py       # ChromaDB management
│   ├── llm_agent.py          # Gemini Q&A agent
│   ├── arxiv_tool.py         # ArXiv integration
│   └── utils.py              # Helper functions
├── config.py                 # Configuration management
├── requirements.txt          # Python dependencies
├── .env.example              # Environment template
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

## 🔧 API Reference

### DocumentProcessor

```python
from src.document_processor import DocumentProcessor

processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
documents = processor.process_pdf("paper.pdf")
```

### VectorStoreManager

```python
from src.vector_store import VectorStoreManager

store = VectorStoreManager()
store.add_documents(documents)
results = store.similarity_search("query", k=5)
```

### DocumentQAAgent

```python
from src.llm_agent import DocumentQAAgent

agent = DocumentQAAgent()
result = agent.query("What is the conclusion?")
print(result.answer)
print(result.sources)
```

### ArxivTool

```python
from src.arxiv_tool import ArxivTool

tool = ArxivTool()
results = tool.search("transformer attention", max_results=5)
paper = tool.get_paper("2301.00001")
```

## 🐛 Troubleshooting

### Common Issues

**"GOOGLE_API_KEY is required"**
- Ensure you've created a `.env` file with your API key
- Verify the key is valid at https://aistudio.google.com/apikey

**"ImportError: No module named..."**
- Run `pip install -r requirements.txt`
- Ensure your virtual environment is activated

**"Invalid PDF file"**
- Check that the PDF is not corrupted
- Ensure the file is not password-protected
- Try a different PDF to isolate the issue

**Slow Processing**
- Large PDFs take longer to process
- Consider reducing `CHUNK_SIZE` for faster processing
- Check your internet connection for API calls

### Logging

Enable debug logging for troubleshooting:

```env
LOG_LEVEL=DEBUG
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Google Gemini](https://ai.google.dev/) - LLM and embeddings
- [LangChain](https://langchain.com/) - LLM orchestration
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Streamlit](https://streamlit.io/) - Web framework
- [PyMuPDF](https://pymupdf.readthedocs.io/) - PDF processing
- [ArXiv](https://arxiv.org/) - Academic paper repository

---

Built with ❤️ for the Document Q&A AI Agent challenge
