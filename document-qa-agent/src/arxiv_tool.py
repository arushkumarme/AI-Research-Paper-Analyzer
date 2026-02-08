"""
ArXiv Integration Tool for Document Q&A Agent

Provides ArXiv paper search functionality with:
- Search by query, author, or category
- Paper metadata retrieval (title, authors, abstract, URL)
- PDF download capability
- Gemini function calling integration

Usage:
    from src.arxiv_tool import ArxivTool, create_arxiv_agent
    
    # Standalone usage
    tool = ArxivTool()
    results = tool.search("transformer attention mechanism", max_results=5)
    
    # With agent integration
    agent = create_arxiv_agent()
    response = agent.query_with_tools("Find recent papers on BERT fine-tuning")
"""

import logging
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

import arxiv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_settings
from src.llm_agent import AgentWithTools, QueryResult

# Configure module logger
logger = logging.getLogger("document_qa.arxiv")


@dataclass
class ArxivPaper:
    """Structured representation of an ArXiv paper."""
    
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: datetime
    updated: datetime
    categories: list[str]
    primary_category: str
    pdf_url: str
    entry_url: str
    comment: Optional[str] = None
    journal_ref: Optional[str] = None
    doi: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract[:500] + "..." if len(self.abstract) > 500 else self.abstract,
            "published": self.published.isoformat(),
            "updated": self.updated.isoformat(),
            "categories": self.categories,
            "primary_category": self.primary_category,
            "pdf_url": self.pdf_url,
            "entry_url": self.entry_url,
            "comment": self.comment,
            "journal_ref": self.journal_ref,
            "doi": self.doi,
        }
    
    def to_formatted_string(self) -> str:
        """Format paper for display."""
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += f" et al. ({len(self.authors)} authors)"
        
        return f"""**{self.title}**
- **ArXiv ID:** {self.arxiv_id}
- **Authors:** {authors_str}
- **Published:** {self.published.strftime('%Y-%m-%d')}
- **Category:** {self.primary_category}
- **PDF:** {self.pdf_url}
- **Abstract:** {self.abstract[:300]}{'...' if len(self.abstract) > 300 else ''}
"""


@dataclass
class SearchResult:
    """Container for ArXiv search results."""
    
    query: str
    papers: list[ArxivPaper] = field(default_factory=list)
    total_results: int = 0
    search_time: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "papers": [p.to_dict() for p in self.papers],
            "total_results": self.total_results,
            "search_time": self.search_time,
        }
    
    def to_formatted_string(self) -> str:
        """Format results for display."""
        if not self.papers:
            return f"No papers found for query: '{self.query}'"
        
        result = f"Found {self.total_results} papers for '{self.query}'\n"
        result += f"Showing top {len(self.papers)} results:\n\n"
        
        for i, paper in enumerate(self.papers, 1):
            result += f"### {i}. {paper.to_formatted_string()}\n"
        
        return result


class ArxivTool:
    """
    ArXiv search and retrieval tool.
    
    Provides search functionality for academic papers on ArXiv,
    with support for various query types and filtering options.
    
    Example:
        tool = ArxivTool()
        
        # Search by query
        results = tool.search("deep learning NLP", max_results=5)
        
        # Search by author
        results = tool.search_by_author("Yoshua Bengio", max_results=3)
        
        # Get specific paper
        paper = tool.get_paper("2301.00001")
        
        # Download PDF
        path = tool.download_pdf("2301.00001", "./papers/")
    """
    
    # ArXiv category mappings
    CATEGORIES = {
        "cs.AI": "Artificial Intelligence",
        "cs.CL": "Computation and Language (NLP)",
        "cs.CV": "Computer Vision",
        "cs.LG": "Machine Learning",
        "cs.NE": "Neural and Evolutionary Computing",
        "stat.ML": "Machine Learning (Statistics)",
    }
    
    def __init__(self, max_results: Optional[int] = None):
        """
        Initialize the ArXiv tool.
        
        Args:
            max_results: Default maximum results per search
        """
        settings = get_settings()
        self.default_max_results = max_results or settings.arxiv_max_results
        
        # Initialize ArXiv client
        self._client = arxiv.Client(
            page_size=100,
            delay_seconds=3.0,  # Respect rate limits
            num_retries=3,
        )
        
        logger.info(f"ArxivTool initialized (default max_results={self.default_max_results})")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,)),
        before_sleep=lambda retry_state: logger.warning(
            f"ArXiv search retry {retry_state.attempt_number}/3"
        ),
    )
    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        sort_by: str = "relevance",
        categories: Optional[list[str]] = None,
    ) -> SearchResult:
        """
        Search ArXiv for papers matching the query.
        
        Args:
            query: Search query string
            max_results: Maximum number of results
            sort_by: Sort order ('relevance', 'submitted', 'updated')
            categories: Optional list of ArXiv categories to filter by
            
        Returns:
            SearchResult containing matching papers
        """
        import time
        start_time = time.time()
        
        max_results = max_results or self.default_max_results
        
        # Build search query
        search_query = self._build_query(query, categories)
        
        # Configure sort
        sort_criteria = {
            "relevance": arxiv.SortCriterion.Relevance,
            "submitted": arxiv.SortCriterion.SubmittedDate,
            "updated": arxiv.SortCriterion.LastUpdatedDate,
        }
        sort = sort_criteria.get(sort_by, arxiv.SortCriterion.Relevance)
        
        # Execute search
        search = arxiv.Search(
            query=search_query,
            max_results=max_results,
            sort_by=sort,
            sort_order=arxiv.SortOrder.Descending,
        )
        
        papers = []
        try:
            for result in self._client.results(search):
                paper = self._convert_result(result)
                papers.append(paper)
        except Exception as e:
            logger.error(f"ArXiv search error: {e}")
            raise
        
        search_time = time.time() - start_time
        
        logger.info(f"ArXiv search for '{query}': found {len(papers)} papers in {search_time:.2f}s")
        
        return SearchResult(
            query=query,
            papers=papers,
            total_results=len(papers),
            search_time=search_time,
        )
    
    def search_by_author(
        self,
        author: str,
        max_results: Optional[int] = None,
    ) -> SearchResult:
        """
        Search for papers by a specific author.
        
        Args:
            author: Author name
            max_results: Maximum results
            
        Returns:
            SearchResult with author's papers
        """
        # ArXiv author search syntax
        query = f'au:"{author}"'
        
        max_results = max_results or self.default_max_results
        
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )
        
        papers = []
        for result in self._client.results(search):
            paper = self._convert_result(result)
            papers.append(paper)
        
        return SearchResult(
            query=f"Author: {author}",
            papers=papers,
            total_results=len(papers),
        )
    
    def search_by_id(self, arxiv_id: str) -> Optional[ArxivPaper]:
        """
        Get a specific paper by ArXiv ID.
        
        Args:
            arxiv_id: ArXiv paper ID (e.g., '2301.00001' or 'arxiv:2301.00001')
            
        Returns:
            ArxivPaper if found, None otherwise
        """
        # Clean up ID
        arxiv_id = arxiv_id.replace("arxiv:", "").strip()
        
        search = arxiv.Search(id_list=[arxiv_id])
        
        try:
            results = list(self._client.results(search))
            if results:
                return self._convert_result(results[0])
        except Exception as e:
            logger.error(f"Failed to fetch paper {arxiv_id}: {e}")
        
        return None
    
    def get_paper(self, arxiv_id: str) -> Optional[ArxivPaper]:
        """Alias for search_by_id."""
        return self.search_by_id(arxiv_id)
    
    def download_pdf(
        self,
        arxiv_id: str,
        output_dir: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Download PDF for a paper.
        
        Args:
            arxiv_id: ArXiv paper ID
            output_dir: Directory to save PDF (uses temp dir if not specified)
            
        Returns:
            Path to downloaded PDF, or None if failed
        """
        paper = self.search_by_id(arxiv_id)
        if not paper:
            logger.error(f"Paper not found: {arxiv_id}")
            return None
        
        # Prepare output path
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = Path(tempfile.gettempdir())
        
        # Clean filename
        safe_title = re.sub(r'[^\w\s-]', '', paper.title)[:50]
        filename = f"{paper.arxiv_id}_{safe_title}.pdf"
        filepath = output_path / filename
        
        try:
            logger.info(f"Downloading PDF: {paper.pdf_url}")
            urlretrieve(paper.pdf_url, str(filepath))
            logger.info(f"PDF saved to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"PDF download failed: {e}")
            return None
    
    def _build_query(
        self,
        query: str,
        categories: Optional[list[str]] = None,
    ) -> str:
        """Build ArXiv search query string."""
        
        # Base query - search in title and abstract
        search_query = f'all:"{query}"'
        
        # Add category filter if specified
        if categories:
            cat_query = " OR ".join(f"cat:{cat}" for cat in categories)
            search_query = f"({search_query}) AND ({cat_query})"
        
        return search_query
    
    def _convert_result(self, result: arxiv.Result) -> ArxivPaper:
        """Convert arxiv.Result to ArxivPaper."""
        
        return ArxivPaper(
            arxiv_id=result.entry_id.split("/")[-1],
            title=result.title,
            authors=[author.name for author in result.authors],
            abstract=result.summary,
            published=result.published,
            updated=result.updated,
            categories=result.categories,
            primary_category=result.primary_category,
            pdf_url=result.pdf_url,
            entry_url=result.entry_id,
            comment=result.comment,
            journal_ref=result.journal_ref,
            doi=result.doi,
        )


# ============================================
# Function calling integration for Gemini
# ============================================

def search_arxiv_papers(
    query: str,
    max_results: int = 5,
) -> str:
    """
    Search ArXiv for academic papers.
    
    This function is designed to be called by Gemini function calling.
    
    Args:
        query: Search query (keywords, topics, or paper titles)
        max_results: Maximum number of papers to return (1-10)
        
    Returns:
        Formatted string with search results
    """
    tool = ArxivTool()
    
    # Clamp max_results
    max_results = max(1, min(max_results, 10))
    
    try:
        results = tool.search(query, max_results=max_results)
        return results.to_formatted_string()
    except Exception as e:
        return f"Error searching ArXiv: {str(e)}"


def get_arxiv_paper_details(arxiv_id: str) -> str:
    """
    Get detailed information about a specific ArXiv paper.
    
    Args:
        arxiv_id: ArXiv paper ID (e.g., '2301.00001')
        
    Returns:
        Formatted paper details
    """
    tool = ArxivTool()
    
    try:
        paper = tool.get_paper(arxiv_id)
        if paper:
            return paper.to_formatted_string()
        return f"Paper not found: {arxiv_id}"
    except Exception as e:
        return f"Error fetching paper: {str(e)}"


def search_arxiv_by_author(author_name: str, max_results: int = 5) -> str:
    """
    Search for papers by a specific author.
    
    Args:
        author_name: Name of the author
        max_results: Maximum number of papers to return
        
    Returns:
        Formatted search results
    """
    tool = ArxivTool()
    
    try:
        results = tool.search_by_author(author_name, max_results=max_results)
        return results.to_formatted_string()
    except Exception as e:
        return f"Error searching for author: {str(e)}"


# Tool definitions for Gemini function calling
ARXIV_TOOL_DEFINITIONS = [
    {
        "name": "search_arxiv_papers",
        "description": "Search ArXiv for academic papers by keywords or topics. Use this to find research papers related to a query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query - keywords, topics, or paper titles to search for",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of papers to return (1-10, default 5)",
                },
            },
            "required": ["query"],
        },
        "function": search_arxiv_papers,
    },
    {
        "name": "get_arxiv_paper_details",
        "description": "Get detailed information about a specific ArXiv paper by its ID. Use this when you have an ArXiv ID and need full paper details.",
        "parameters": {
            "type": "object",
            "properties": {
                "arxiv_id": {
                    "type": "string",
                    "description": "ArXiv paper ID (e.g., '2301.00001' or '2301.00001v2')",
                },
            },
            "required": ["arxiv_id"],
        },
        "function": get_arxiv_paper_details,
    },
    {
        "name": "search_arxiv_by_author",
        "description": "Search for papers written by a specific author. Use this to find all papers by a researcher.",
        "parameters": {
            "type": "object",
            "properties": {
                "author_name": {
                    "type": "string",
                    "description": "Name of the author to search for",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of papers to return (1-10, default 5)",
                },
            },
            "required": ["author_name"],
        },
        "function": search_arxiv_by_author,
    },
]


def create_arxiv_agent() -> AgentWithTools:
    """
    Create a Document Q&A agent with ArXiv tools integrated.
    
    Returns:
        AgentWithTools configured with ArXiv function calling
        
    Example:
        agent = create_arxiv_agent()
        
        # Query that might use ArXiv
        result = agent.query_with_tools(
            "Find recent papers about transformer architectures"
        )
        print(result.answer)
    """
    agent = AgentWithTools()
    
    # Register all ArXiv tools
    for tool_def in ARXIV_TOOL_DEFINITIONS:
        agent.register_function_tool(
            func=tool_def["function"],
            name=tool_def["name"],
            description=tool_def["description"],
            parameters=tool_def["parameters"],
        )
    
    logger.info("Created agent with ArXiv tools")
    return agent


# ============================================
# Utility functions
# ============================================

def download_and_process_arxiv_paper(
    arxiv_id: str,
    output_dir: Optional[str] = None,
) -> Optional[Path]:
    """
    Download an ArXiv paper PDF for processing.
    
    Args:
        arxiv_id: ArXiv paper ID
        output_dir: Directory to save the PDF
        
    Returns:
        Path to downloaded PDF
        
    Example:
        from src.arxiv_tool import download_and_process_arxiv_paper
        from src.document_processor import DocumentProcessor
        
        pdf_path = download_and_process_arxiv_paper("2301.00001")
        if pdf_path:
            processor = DocumentProcessor()
            docs = processor.process_pdf(pdf_path)
    """
    tool = ArxivTool()
    return tool.download_pdf(arxiv_id, output_dir)


if __name__ == "__main__":
    # Test ArXiv tool
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    print("ArXiv Tool - Test")
    print("-" * 50)
    
    try:
        tool = ArxivTool()
        
        # Test search
        print("\nSearching for 'attention mechanism transformer'...")
        results = tool.search("attention mechanism transformer", max_results=3)
        
        print(f"\nFound {results.total_results} papers:")
        for i, paper in enumerate(results.papers, 1):
            print(f"\n{i}. {paper.title}")
            print(f"   Authors: {', '.join(paper.authors[:2])}...")
            print(f"   ArXiv ID: {paper.arxiv_id}")
            print(f"   Published: {paper.published.strftime('%Y-%m-%d')}")
        
        # Test function calling format
        print("\n" + "-" * 50)
        print("Testing function call format:")
        result = search_arxiv_papers("BERT fine-tuning", max_results=2)
        print(result)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
