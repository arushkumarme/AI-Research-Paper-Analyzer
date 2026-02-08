"""
Document Processor for PDF Ingestion

Handles PDF document processing including:
- Text extraction using PyMuPDF
- Structure preservation (sections, paragraphs, tables)
- Metadata extraction (title, authors, abstract)
- Intelligent text chunking for RAG

Usage:
    from src.document_processor import DocumentProcessor
    
    processor = DocumentProcessor()
    documents = processor.process_pdf("paper.pdf")
    
    # Access extracted content
    for doc in documents:
        print(doc.page_content)
        print(doc.metadata)
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Optional

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Configure module logger
logger = logging.getLogger("document_qa.processor")


@dataclass
class DocumentMetadata:
    """Structured metadata extracted from a document."""
    
    filename: str
    title: Optional[str] = None
    authors: Optional[str] = None
    abstract: Optional[str] = None
    total_pages: int = 0
    sections: list[str] = field(default_factory=list)
    file_hash: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert metadata to dictionary for storage."""
        return {
            "filename": self.filename,
            "title": self.title or "Unknown",
            "authors": self.authors or "Unknown",
            "abstract": self.abstract or "",
            "total_pages": self.total_pages,
            "sections": ", ".join(self.sections) if self.sections else "",
            "file_hash": self.file_hash or "",
        }


@dataclass
class ExtractedContent:
    """Container for extracted document content."""
    
    text: str
    metadata: DocumentMetadata
    pages: list[dict] = field(default_factory=list)  # Per-page content
    tables: list[dict] = field(default_factory=list)
    references: list[str] = field(default_factory=list)


class DocumentProcessor:
    """
    PDF document processor with structure-aware extraction and chunking.
    
    Attributes:
        chunk_size: Target size for text chunks (in characters)
        chunk_overlap: Overlap between consecutive chunks
        
    Example:
        processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
        docs = processor.process_pdf("research_paper.pdf")
        
        # Each doc has content and metadata
        for doc in docs:
            print(f"Chunk: {doc.page_content[:100]}...")
            print(f"Source: {doc.metadata['filename']}")
            print(f"Page: {doc.metadata['page_number']}")
    """
    
    # Common section headers in academic papers
    SECTION_PATTERNS = [
        r"^(?:abstract|introduction|background|related work|methodology|"
        r"methods|approach|experiments?|results?|discussion|conclusion|"
        r"references|acknowledgements?|appendix)",
        r"^\d+\.?\s+[A-Z]",  # Numbered sections like "1. Introduction"
        r"^[IVXLC]+\.?\s+[A-Z]",  # Roman numeral sections
    ]
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Target chunk size in characters (~tokens * 4)
            chunk_overlap: Character overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize text splitter with academic-friendly separators
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n\n",      # Major section breaks
                "\n\n",        # Paragraph breaks
                "\n",          # Line breaks
                ". ",          # Sentence breaks
                ", ",          # Clause breaks
                " ",           # Word breaks
                "",            # Character breaks (last resort)
            ],
            is_separator_regex=False,
        )
        
        logger.info(
            f"DocumentProcessor initialized (chunk_size={chunk_size}, "
            f"overlap={chunk_overlap})"
        )
    
    def process_pdf(
        self,
        source: str | Path | BinaryIO,
        filename: Optional[str] = None,
    ) -> list[Document]:
        """
        Process a PDF file and return chunked documents.
        
        Args:
            source: File path, Path object, or file-like object
            filename: Optional filename override (required for file-like objects)
            
        Returns:
            List of LangChain Document objects with content and metadata
            
        Raises:
            FileNotFoundError: If file path doesn't exist
            ValueError: If PDF cannot be processed
        """
        try:
            # Extract content from PDF
            extracted = self._extract_content(source, filename)
            
            # Create chunks with metadata
            documents = self._create_chunks(extracted)
            
            logger.info(
                f"Processed '{extracted.metadata.filename}': "
                f"{extracted.metadata.total_pages} pages, "
                f"{len(documents)} chunks"
            )
            
            return documents
            
        except fitz.FileDataError as e:
            logger.error(f"Invalid PDF file: {e}")
            raise ValueError(f"Invalid or corrupted PDF file: {e}") from e
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise
    
    def process_pdf_bytes(
        self,
        pdf_bytes: bytes,
        filename: str,
    ) -> list[Document]:
        """
        Process PDF from bytes (useful for Streamlit uploads).
        
        Args:
            pdf_bytes: Raw PDF file bytes
            filename: Name to use for the document
            
        Returns:
            List of chunked Document objects
        """
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            extracted = self._extract_from_fitz_doc(doc, filename, pdf_bytes)
            doc.close()
            
            documents = self._create_chunks(extracted)
            
            logger.info(
                f"Processed '{filename}' from bytes: "
                f"{extracted.metadata.total_pages} pages, "
                f"{len(documents)} chunks"
            )
            
            return documents
            
        except Exception as e:
            logger.error(f"Error processing PDF bytes: {e}")
            raise ValueError(f"Failed to process PDF: {e}") from e
    
    def _extract_content(
        self,
        source: str | Path | BinaryIO,
        filename: Optional[str] = None,
    ) -> ExtractedContent:
        """Extract content from PDF source."""
        
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"PDF file not found: {path}")
            
            filename = filename or path.name
            
            # Calculate file hash for deduplication
            file_hash = self._calculate_hash(path.read_bytes())
            
            doc = fitz.open(path)
            extracted = self._extract_from_fitz_doc(doc, filename)
            extracted.metadata.file_hash = file_hash
            doc.close()
            
        else:
            # File-like object
            if filename is None:
                raise ValueError("filename is required for file-like objects")
            
            content = source.read()
            if isinstance(content, str):
                content = content.encode()
            
            file_hash = self._calculate_hash(content)
            
            doc = fitz.open(stream=content, filetype="pdf")
            extracted = self._extract_from_fitz_doc(doc, filename)
            extracted.metadata.file_hash = file_hash
            doc.close()
        
        return extracted
    
    def _extract_from_fitz_doc(
        self,
        doc: fitz.Document,
        filename: str,
        raw_bytes: Optional[bytes] = None,
    ) -> ExtractedContent:
        """Extract content from an opened PyMuPDF document."""
        
        metadata = DocumentMetadata(
            filename=filename,
            total_pages=len(doc),
        )
        
        # Try to extract PDF metadata
        pdf_metadata = doc.metadata
        if pdf_metadata:
            metadata.title = pdf_metadata.get("title") or None
            metadata.authors = pdf_metadata.get("author") or None
        
        full_text = []
        pages = []
        
        for page_num, page in enumerate(doc, start=1):
            # Extract text with layout preservation
            page_text = page.get_text("text")
            
            # Extract text blocks for structure analysis
            blocks = page.get_text("blocks")
            
            page_data = {
                "page_number": page_num,
                "text": page_text,
                "blocks": len(blocks),
            }
            pages.append(page_data)
            full_text.append(page_text)
            
            # Try to extract title from first page
            if page_num == 1 and not metadata.title:
                metadata.title = self._extract_title(page_text, blocks)
            
            # Extract abstract from early pages
            if page_num <= 2 and not metadata.abstract:
                metadata.abstract = self._extract_abstract(page_text)
        
        # Join all pages
        combined_text = "\n\n".join(full_text)
        
        # Extract sections
        metadata.sections = self._extract_sections(combined_text)
        
        # Extract references
        references = self._extract_references(combined_text)
        
        # If title still not found, use filename
        if not metadata.title:
            metadata.title = Path(filename).stem.replace("_", " ").replace("-", " ")
        
        return ExtractedContent(
            text=combined_text,
            metadata=metadata,
            pages=pages,
            references=references,
        )
    
    def _extract_title(
        self,
        page_text: str,
        blocks: list,
    ) -> Optional[str]:
        """Extract document title from first page."""
        
        # Try to find title from text blocks (usually largest font on first page)
        # Blocks format: (x0, y0, x1, y1, "text", block_no, block_type)
        
        lines = page_text.strip().split("\n")
        
        # Title is often in the first few non-empty lines
        for line in lines[:10]:
            line = line.strip()
            # Skip empty lines and common headers
            if not line:
                continue
            if line.lower() in ["abstract", "introduction", "arxiv"]:
                continue
            if re.match(r"^[\d\.\-\/]+$", line):  # Skip dates/numbers
                continue
            if len(line) > 10 and len(line) < 200:
                # Likely a title
                return line
        
        return None
    
    def _extract_abstract(self, text: str) -> Optional[str]:
        """Extract abstract section from document."""
        
        # Common abstract patterns
        patterns = [
            r"(?i)abstract[:\s]*\n+(.*?)(?=\n\s*(?:introduction|keywords|1\.|I\.))",
            r"(?i)abstract[:\s]*(.*?)(?=\n\n)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                abstract = match.group(1).strip()
                # Clean up whitespace
                abstract = re.sub(r"\s+", " ", abstract)
                if len(abstract) > 50:  # Reasonable abstract length
                    return abstract[:1000]  # Truncate very long abstracts
        
        return None
    
    def _extract_sections(self, text: str) -> list[str]:
        """Extract section headers from document."""
        
        sections = []
        lines = text.split("\n")
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check against section patterns
            for pattern in self.SECTION_PATTERNS:
                if re.match(pattern, line, re.IGNORECASE):
                    # Clean up the section name
                    section = re.sub(r"^\d+\.?\s*", "", line)
                    section = re.sub(r"^[IVXLC]+\.?\s*", "", section)
                    if section and len(section) < 100:
                        sections.append(section)
                    break
        
        return sections
    
    def _extract_references(self, text: str) -> list[str]:
        """Extract references section from document."""
        
        references = []
        
        # Find references section
        ref_match = re.search(
            r"(?i)(?:references|bibliography)\s*\n(.*?)(?=\n\s*(?:appendix|$))",
            text,
            re.DOTALL,
        )
        
        if ref_match:
            ref_text = ref_match.group(1)
            # Split by reference numbers [1], [2], etc.
            refs = re.split(r"\[\d+\]", ref_text)
            for ref in refs:
                ref = ref.strip()
                if ref and len(ref) > 20:
                    references.append(ref[:500])  # Truncate long references
        
        return references[:50]  # Limit number of references
    
    def _create_chunks(self, extracted: ExtractedContent) -> list[Document]:
        """Create document chunks with metadata."""
        
        # Split the full text into chunks
        text_chunks = self.text_splitter.split_text(extracted.text)
        
        documents = []
        base_metadata = extracted.metadata.to_dict()
        
        for i, chunk in enumerate(text_chunks):
            # Determine which page this chunk is from (approximate)
            page_number = self._find_page_for_chunk(chunk, extracted.pages)
            
            # Detect section for this chunk
            section = self._detect_section(chunk, extracted.metadata.sections)
            
            # Create metadata for this chunk
            chunk_metadata = {
                **base_metadata,
                "chunk_index": i,
                "chunk_total": len(text_chunks),
                "page_number": page_number,
                "section": section or "Unknown",
                "chunk_size": len(chunk),
            }
            
            documents.append(Document(
                page_content=chunk,
                metadata=chunk_metadata,
            ))
        
        return documents
    
    def _find_page_for_chunk(
        self,
        chunk: str,
        pages: list[dict],
    ) -> int:
        """Find which page a chunk belongs to."""
        
        # Simple heuristic: find page with most overlap
        chunk_lower = chunk.lower()[:200]  # Use start of chunk
        
        best_page = 1
        best_overlap = 0
        
        for page in pages:
            page_text = page["text"].lower()
            # Count matching words
            chunk_words = set(chunk_lower.split())
            page_words = set(page_text.split())
            overlap = len(chunk_words & page_words)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_page = page["page_number"]
        
        return best_page
    
    def _detect_section(
        self,
        chunk: str,
        sections: list[str],
    ) -> Optional[str]:
        """Detect which section a chunk belongs to."""
        
        chunk_lower = chunk.lower()
        
        for section in sections:
            if section.lower() in chunk_lower:
                return section
        
        # Check for common section keywords
        section_keywords = {
            "abstract": "Abstract",
            "introduction": "Introduction",
            "background": "Background",
            "related work": "Related Work",
            "methodology": "Methodology",
            "method": "Methods",
            "experiment": "Experiments",
            "result": "Results",
            "discussion": "Discussion",
            "conclusion": "Conclusion",
            "reference": "References",
        }
        
        for keyword, section_name in section_keywords.items():
            if keyword in chunk_lower[:100]:  # Check beginning of chunk
                return section_name
        
        return None
    
    @staticmethod
    def _calculate_hash(content: bytes) -> str:
        """Calculate SHA-256 hash of content for deduplication."""
        return hashlib.sha256(content).hexdigest()[:16]
    
    def get_document_summary(self, extracted: ExtractedContent) -> str:
        """Generate a summary string of extracted document."""
        
        meta = extracted.metadata
        summary_parts = [
            f"Document: {meta.filename}",
            f"Title: {meta.title or 'Unknown'}",
            f"Authors: {meta.authors or 'Unknown'}",
            f"Pages: {meta.total_pages}",
        ]
        
        if meta.sections:
            summary_parts.append(f"Sections: {', '.join(meta.sections[:5])}")
        
        if meta.abstract:
            summary_parts.append(f"Abstract: {meta.abstract[:200]}...")
        
        return "\n".join(summary_parts)


# Convenience function for quick processing
def process_pdf_file(
    file_path: str | Path,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Document]:
    """
    Quick function to process a PDF file.
    
    Args:
        file_path: Path to PDF file
        chunk_size: Chunk size in characters
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of Document objects
        
    Example:
        docs = process_pdf_file("paper.pdf")
        for doc in docs:
            print(doc.page_content)
    """
    processor = DocumentProcessor(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return processor.process_pdf(file_path)


if __name__ == "__main__":
    # Example usage
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        print(f"Processing: {pdf_path}")
        
        processor = DocumentProcessor()
        docs = processor.process_pdf(pdf_path)
        
        print(f"\nExtracted {len(docs)} chunks")
        print("\nFirst chunk:")
        print("-" * 50)
        print(docs[0].page_content[:500])
        print("-" * 50)
        print(f"Metadata: {docs[0].metadata}")
    else:
        print("Usage: python document_processor.py <pdf_file>")
