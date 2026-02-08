"""
Utility Functions for Document Q&A AI Agent

Provides helper functions for:
- Text processing and cleaning
- Token counting and estimation
- File validation
- Formatting helpers
- Performance utilities

Usage:
    from src.utils import (
        clean_text,
        estimate_tokens,
        validate_pdf,
        format_sources,
    )
"""

import hashlib
import logging
import re
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

# Configure logger
logger = logging.getLogger("document_qa.utils")

# Type variable for generic functions
T = TypeVar("T")


# ============================================
# Text Processing
# ============================================

def clean_text(text: str) -> str:
    """
    Clean and normalize text content.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text with normalized whitespace
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)
    
    # Remove special characters that might cause issues
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    
    # Normalize quotes
    text = text.replace(""", '"').replace(""", '"')
    text = text.replace("'", "'").replace("'", "'")
    
    # Normalize dashes
    text = text.replace("–", "-").replace("—", "-")
    
    return text.strip()


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)].rsplit(" ", 1)[0] + suffix


def extract_sentences(text: str, max_sentences: int = 5) -> list[str]:
    """
    Extract sentences from text.
    
    Args:
        text: Input text
        max_sentences: Maximum sentences to extract
        
    Returns:
        List of sentences
    """
    # Simple sentence splitter
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences[:max_sentences] if s.strip()]


def remove_references_section(text: str) -> str:
    """
    Remove the references section from academic text.
    
    Args:
        text: Document text
        
    Returns:
        Text without references section
    """
    patterns = [
        r"(?i)\n\s*references\s*\n.*$",
        r"(?i)\n\s*bibliography\s*\n.*$",
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.DOTALL)
    
    return text


# ============================================
# Token Estimation
# ============================================

def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in text.
    
    Uses a simple heuristic: ~4 characters per token for English.
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    
    # Rough estimate: 1 token ≈ 4 characters for English
    char_count = len(text)
    return char_count // 4


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "gemini-1.5-flash",
) -> float:
    """
    Estimate API cost for token usage.
    
    Note: Gemini API pricing may change. These are approximate rates.
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: Model name
        
    Returns:
        Estimated cost in USD
    """
    # Gemini 1.5 Flash pricing (approximate, may vary)
    pricing = {
        "gemini-1.5-flash": {
            "input": 0.000075,   # per 1K tokens
            "output": 0.0003,    # per 1K tokens
        },
        "gemini-1.5-pro": {
            "input": 0.00125,
            "output": 0.005,
        },
    }
    
    rates = pricing.get(model, pricing["gemini-1.5-flash"])
    
    input_cost = (input_tokens / 1000) * rates["input"]
    output_cost = (output_tokens / 1000) * rates["output"]
    
    return round(input_cost + output_cost, 6)


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


# ============================================
# File Validation
# ============================================

def validate_pdf(file_path: Path | str) -> tuple[bool, str]:
    """
    Validate a PDF file.
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    path = Path(file_path)
    
    # Check existence
    if not path.exists():
        return False, f"File not found: {path}"
    
    # Check extension
    if path.suffix.lower() != ".pdf":
        return False, f"Invalid file type: {path.suffix}"
    
    # Check file size
    max_size_mb = 50
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_size_mb:
        return False, f"File too large: {size_mb:.1f}MB (max: {max_size_mb}MB)"
    
    # Check PDF header
    try:
        with open(path, "rb") as f:
            header = f.read(8)
            if not header.startswith(b"%PDF"):
                return False, "Invalid PDF file (missing PDF header)"
    except Exception as e:
        return False, f"Cannot read file: {e}"
    
    return True, ""


def validate_pdf_bytes(content: bytes, max_size_mb: int = 50) -> tuple[bool, str]:
    """
    Validate PDF content from bytes.
    
    Args:
        content: PDF file bytes
        max_size_mb: Maximum allowed size in MB
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check size
    size_mb = len(content) / (1024 * 1024)
    if size_mb > max_size_mb:
        return False, f"File too large: {size_mb:.1f}MB (max: {max_size_mb}MB)"
    
    # Check PDF header
    if not content.startswith(b"%PDF"):
        return False, "Invalid PDF file (missing PDF header)"
    
    return True, ""


def get_file_hash(content: bytes) -> str:
    """Generate SHA-256 hash of file content."""
    return hashlib.sha256(content).hexdigest()


# ============================================
# Formatting Helpers
# ============================================

def format_sources(sources: list[dict], max_sources: int = 5) -> str:
    """
    Format source citations for display.
    
    Args:
        sources: List of source dictionaries
        max_sources: Maximum sources to include
        
    Returns:
        Formatted source string
    """
    if not sources:
        return "No sources available."
    
    formatted = []
    for i, source in enumerate(sources[:max_sources], 1):
        filename = source.get("filename", "Unknown")
        page = source.get("page", "N/A")
        section = source.get("section", "Unknown")
        score = source.get("relevance_score", 0)
        
        formatted.append(
            f"[{i}] {filename} (Page {page}, {section}) - Relevance: {score:.0%}"
        )
    
    return "\n".join(formatted)


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable form.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "2.5s", "1m 30s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs}s"


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable form.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def format_number(num: int) -> str:
    """Format number with thousands separator."""
    return f"{num:,}"


# ============================================
# Performance Utilities
# ============================================

def timed(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to time function execution.
    
    Usage:
        @timed
        def my_function():
            ...
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.debug(f"{func.__name__} completed in {elapsed:.3f}s")
        return result
    return wrapper


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple = (Exception,),
) -> Callable:
    """
    Decorator for retrying with exponential backoff.
    
    Args:
        max_retries: Maximum retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exceptions: Exceptions to catch and retry
        
    Usage:
        @retry_with_backoff(max_retries=3)
        def api_call():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}), "
                            f"retrying in {delay:.1f}s: {e}"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"{func.__name__} failed after {max_retries + 1} attempts")
            
            raise last_exception
        return wrapper
    return decorator


class RateLimiter:
    """
    Simple rate limiter for API calls.
    
    Usage:
        limiter = RateLimiter(calls_per_minute=60)
        
        for item in items:
            limiter.wait()
            api_call(item)
    """
    
    def __init__(self, calls_per_minute: int = 60):
        self.min_interval = 60.0 / calls_per_minute
        self.last_call = 0.0
    
    def wait(self) -> None:
        """Wait if necessary to respect rate limit."""
        now = time.time()
        elapsed = now - self.last_call
        
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            time.sleep(sleep_time)
        
        self.last_call = time.time()


# ============================================
# Validation Helpers
# ============================================

def is_valid_query(query: str, min_length: int = 3) -> tuple[bool, str]:
    """
    Validate a user query.
    
    Args:
        query: User's query string
        min_length: Minimum query length
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not query:
        return False, "Query cannot be empty"
    
    query = query.strip()
    
    if len(query) < min_length:
        return False, f"Query too short (minimum {min_length} characters)"
    
    if len(query) > 2000:
        return False, "Query too long (maximum 2000 characters)"
    
    # Check for mostly special characters
    alpha_ratio = sum(c.isalnum() for c in query) / len(query)
    if alpha_ratio < 0.3:
        return False, "Query contains too many special characters"
    
    return True, ""


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename for safe storage.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove directory separators
    filename = filename.replace("/", "_").replace("\\", "_")
    
    # Remove or replace special characters
    filename = re.sub(r'[<>:"|?*]', "_", filename)
    
    # Limit length
    name, ext = filename.rsplit(".", 1) if "." in filename else (filename, "")
    if len(name) > 100:
        name = name[:100]
    
    return f"{name}.{ext}" if ext else name


# ============================================
# Data Extraction Helpers
# ============================================

def extract_metrics_from_text(text: str) -> list[dict]:
    """
    Extract numerical metrics from text.
    
    Args:
        text: Input text
        
    Returns:
        List of extracted metrics with context
    """
    metrics = []
    
    # Common metric patterns
    patterns = [
        # Percentage patterns
        (r"(\w+(?:\s+\w+)?)\s*[:=]\s*(\d+\.?\d*)\s*%", "percentage"),
        # Score patterns
        (r"(accuracy|precision|recall|f1|auc|mAP)\s*[:=]?\s*(\d+\.?\d*)", "score"),
        # With units
        (r"(\d+\.?\d*)\s*(ms|seconds?|minutes?|hours?|MB|GB|KB)", "measurement"),
    ]
    
    for pattern, metric_type in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            if metric_type == "percentage":
                name, value = match.groups()
                metrics.append({
                    "name": name.strip(),
                    "value": float(value),
                    "unit": "%",
                    "type": metric_type,
                })
            elif metric_type == "score":
                name, value = match.groups()
                metrics.append({
                    "name": name.strip(),
                    "value": float(value),
                    "unit": "" if float(value) <= 1 else "%",
                    "type": metric_type,
                })
            elif metric_type == "measurement":
                value, unit = match.groups()
                metrics.append({
                    "value": float(value),
                    "unit": unit,
                    "type": metric_type,
                })
    
    return metrics


def extract_citations(text: str) -> list[str]:
    """
    Extract citation references from text.
    
    Args:
        text: Input text
        
    Returns:
        List of citation strings
    """
    citations = []
    
    # Numbered citations [1], [2, 3], etc.
    numbered = re.findall(r"\[(\d+(?:,\s*\d+)*)\]", text)
    for nums in numbered:
        for num in nums.split(","):
            citations.append(f"[{num.strip()}]")
    
    # Author-year citations (Author, 2023)
    author_year = re.findall(r"\(([A-Z][a-z]+(?:\s+(?:et al\.|and|&)\s+[A-Z][a-z]+)?,\s*\d{4})\)", text)
    citations.extend([f"({c})" for c in author_year])
    
    return list(set(citations))


# ============================================
# Cache Key Generation
# ============================================

def generate_cache_key(*args: Any, **kwargs: Any) -> str:
    """
    Generate a cache key from arguments.
    
    Args:
        *args: Positional arguments to include in key
        **kwargs: Keyword arguments to include in key
        
    Returns:
        MD5 hash string suitable for use as cache key
    """
    key_parts = [str(arg) for arg in args]
    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
    key_string = "|".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()


def generate_document_key(filename: str, content_hash: str, chunk_index: int = 0) -> str:
    """
    Generate a unique key for a document chunk.
    
    Args:
        filename: Name of the document
        content_hash: Hash of content
        chunk_index: Index of the chunk
        
    Returns:
        Unique document key
    """
    return f"{filename}_{chunk_index}_{content_hash[:8]}"


def generate_query_key(query: str, context_hash: str) -> str:
    """
    Generate a cache key for a query with context.
    
    Args:
        query: The query string
        context_hash: Hash of the context documents
        
    Returns:
        Cache key for the query
    """
    normalized_query = query.lower().strip()
    combined = f"{normalized_query}:{context_hash}"
    return hashlib.md5(combined.encode()).hexdigest()


# ============================================
# Error Formatting
# ============================================

class AppError(Exception):
    """Base exception for application errors."""
    
    def __init__(self, message: str, code: str = "UNKNOWN", details: Optional[dict] = None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(message)
    
    def to_dict(self) -> dict:
        return {
            "error": self.code,
            "message": self.message,
            "details": self.details,
        }
    
    def to_user_message(self) -> str:
        """Return a user-friendly error message."""
        return self.message


class ConfigurationError(AppError):
    """Configuration-related errors."""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, "CONFIG_ERROR", details)


class DocumentProcessingError(AppError):
    """Document processing errors."""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, "DOCUMENT_ERROR", details)


class QueryError(AppError):
    """Query processing errors."""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, "QUERY_ERROR", details)


class APIError(AppError):
    """External API errors."""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, "API_ERROR", details)


def format_error_message(error: Exception, include_traceback: bool = False) -> str:
    """
    Format an exception into a user-friendly message.
    
    Args:
        error: The exception to format
        include_traceback: Whether to include traceback info
        
    Returns:
        Formatted error message
    """
    if isinstance(error, AppError):
        return error.to_user_message()
    
    # Map common exceptions to friendly messages
    error_messages = {
        FileNotFoundError: "The requested file could not be found.",
        PermissionError: "Permission denied. Please check file permissions.",
        ConnectionError: "Could not connect to the server. Please check your internet connection.",
        TimeoutError: "The request timed out. Please try again.",
        ValueError: f"Invalid value: {str(error)}",
    }
    
    for error_type, message in error_messages.items():
        if isinstance(error, error_type):
            return message
    
    # Default message
    base_message = f"An error occurred: {str(error)}"
    
    if include_traceback:
        import traceback
        tb = traceback.format_exc()
        return f"{base_message}\n\nDetails:\n{tb}"
    
    return base_message


def safe_execute(func: Callable[..., T], *args: Any, default: T = None, **kwargs: Any) -> T:
    """
    Execute a function safely, returning default on error.
    
    Args:
        func: Function to execute
        *args: Arguments to pass
        default: Default value on error
        **kwargs: Keyword arguments to pass
        
    Returns:
        Function result or default value
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.warning(f"Safe execute caught error in {func.__name__}: {e}")
        return default


# ============================================
# Logging Setup
# ============================================

def configure_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Configure application logging.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging
        format_string: Custom format string
        
    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    
    # Suppress noisy third-party loggers
    for noisy_logger in ["chromadb", "httpx", "urllib3", "google", "httpcore"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    
    return logging.getLogger("document_qa")


class LogContext:
    """
    Context manager for structured logging.
    
    Usage:
        with LogContext("Processing document", filename="test.pdf"):
            process_document()
    """
    
    def __init__(self, operation: str, **context):
        self.operation = operation
        self.context = context
        self.start_time = None
        self.logger = logging.getLogger("document_qa")
    
    def __enter__(self):
        self.start_time = time.time()
        context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
        self.logger.info(f"Starting: {self.operation} ({context_str})")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        if exc_type:
            self.logger.error(f"Failed: {self.operation} after {elapsed:.2f}s - {exc_val}")
        else:
            self.logger.info(f"Completed: {self.operation} in {elapsed:.2f}s")
        return False


# ============================================
# Performance Monitoring
# ============================================

class PerformanceMonitor:
    """
    Track performance metrics across operations.
    
    Usage:
        monitor = PerformanceMonitor()
        
        with monitor.track("embedding"):
            generate_embeddings()
        
        print(monitor.get_stats())
    """
    
    def __init__(self):
        self._timings: dict[str, list[float]] = {}
        self._counts: dict[str, int] = {}
    
    def track(self, operation: str):
        """Context manager to track operation timing."""
        return _TimingContext(self, operation)
    
    def record(self, operation: str, duration: float) -> None:
        """Record a timing for an operation."""
        if operation not in self._timings:
            self._timings[operation] = []
            self._counts[operation] = 0
        self._timings[operation].append(duration)
        self._counts[operation] += 1
    
    def get_stats(self) -> dict:
        """Get performance statistics."""
        stats = {}
        for op, timings in self._timings.items():
            if timings:
                stats[op] = {
                    "count": self._counts[op],
                    "total": sum(timings),
                    "avg": sum(timings) / len(timings),
                    "min": min(timings),
                    "max": max(timings),
                }
        return stats
    
    def reset(self) -> None:
        """Reset all metrics."""
        self._timings.clear()
        self._counts.clear()


class _TimingContext:
    """Internal context manager for PerformanceMonitor."""
    
    def __init__(self, monitor: PerformanceMonitor, operation: str):
        self.monitor = monitor
        self.operation = operation
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        duration = time.time() - self.start_time
        self.monitor.record(self.operation, duration)


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def timed_operation(operation_name: str):
    """
    Decorator to track function performance.
    
    Args:
        operation_name: Name for the operation in metrics
        
    Usage:
        @timed_operation("document_processing")
        def process_document(doc):
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            with performance_monitor.track(operation_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# ============================================
# Input Sanitization
# ============================================

def sanitize_input(text: str, max_length: int = 10000) -> str:
    """
    Sanitize user input for safety.
    
    Args:
        text: Input text to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized text
    """
    if not text:
        return ""
    
    # Truncate if too long
    text = text[:max_length]
    
    # Remove null bytes and control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()


def sanitize_for_display(text: str) -> str:
    """
    Sanitize text for safe display in UI.
    
    Args:
        text: Text to sanitize
        
    Returns:
        Display-safe text
    """
    if not text:
        return ""
    
    # Escape HTML-like characters
    text = text.replace("<", "&lt;").replace(">", "&gt;")
    
    return text


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    print("=" * 50)
    
    # Test text cleaning
    dirty_text = "  Hello   world\n\n\nwith   extra   spaces  "
    print(f"Clean text: '{clean_text(dirty_text)}'")
    
    # Test token estimation
    sample = "This is a sample text for token estimation."
    print(f"Estimated tokens: {estimate_tokens(sample)}")
    
    # Test metric extraction
    metrics_text = "The model achieved accuracy: 95.5% and F1 = 0.92"
    print(f"Extracted metrics: {extract_metrics_from_text(metrics_text)}")
    
    # Test cache key generation
    key = generate_cache_key("query", "context", model="gemini")
    print(f"Cache key: {key}")
    
    # Test performance monitor
    monitor = PerformanceMonitor()
    with monitor.track("test_operation"):
        time.sleep(0.1)
    print(f"Performance stats: {monitor.get_stats()}")
    
    # Test error formatting
    try:
        raise ValueError("Test error")
    except Exception as e:
        print(f"Formatted error: {format_error_message(e)}")
    
    print("\n✅ All utilities working!")
