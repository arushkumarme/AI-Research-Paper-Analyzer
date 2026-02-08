"""
Configuration Management for Document Q&A AI Agent

Handles environment variables, API configuration, and application settings
using Pydantic for validation and type safety.
"""

import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings with validation and environment variable loading.
    
    All settings can be overridden via environment variables or .env file.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # ========================
    # Google Gemini API
    # ========================
    google_api_key: str = Field(
        ...,
        description="Google Gemini API key from https://aistudio.google.com/apikey"
    )
    gemini_model: str = Field(
        default="gemini-1.5-flash",
        description="Gemini model for chat/generation"
    )
    embedding_model: str = Field(
        default="models/embedding-001",
        description="Gemini model for text embeddings"
    )
    
    # ========================
    # ChromaDB Configuration
    # ========================
    chroma_persist_dir: str = Field(
        default="./chroma_db",
        description="Directory for ChromaDB persistence"
    )
    chroma_collection_name: str = Field(
        default="documents",
        description="ChromaDB collection name"
    )
    
    # ========================
    # Document Processing
    # ========================
    chunk_size: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Text chunk size for splitting documents"
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        le=500,
        description="Overlap between text chunks"
    )
    max_upload_size_mb: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Maximum upload file size in MB"
    )
    
    # ========================
    # ArXiv Configuration
    # ========================
    arxiv_max_results: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum results from ArXiv search"
    )
    
    # ========================
    # Cache Configuration
    # ========================
    cache_enabled: bool = Field(
        default=True,
        description="Enable response caching"
    )
    cache_max_size: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Maximum cache entries"
    )
    embedding_cache_size: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Maximum embedding cache entries"
    )
    
    # ========================
    # Performance Settings
    # ========================
    max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum API retry attempts"
    )
    request_timeout: int = Field(
        default=60,
        ge=10,
        le=300,
        description="API request timeout in seconds"
    )
    batch_size: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Batch size for processing documents"
    )
    
    # ========================
    # Application Settings
    # ========================
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    log_file: Optional[str] = Field(
        default=None,
        description="Optional log file path"
    )
    debug_mode: bool = Field(
        default=False,
        description="Enable debug mode with verbose logging"
    )
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is a valid Python logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v_upper
    
    @field_validator("google_api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate API key is not empty or placeholder."""
        if not v or v == "your_gemini_api_key_here":
            raise ValueError(
                "GOOGLE_API_KEY is required. Get your key from: "
                "https://aistudio.google.com/apikey"
            )
        return v
    
    @property
    def chroma_persist_path(self) -> Path:
        """Return ChromaDB persist directory as Path object."""
        return Path(self.chroma_persist_dir)
    
    @property
    def max_upload_bytes(self) -> int:
        """Return max upload size in bytes."""
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings.
    
    Settings are loaded once and cached for performance.
    Use get_settings.cache_clear() to reload if needed.
    
    Returns:
        Settings: Validated application settings
        
    Raises:
        ValidationError: If required settings are missing or invalid
    """
    return Settings()


def setup_logging(level: Optional[str] = None) -> logging.Logger:
    """
    Configure application logging with consistent formatting.
    
    Args:
        level: Optional log level override. Uses settings if not provided.
        
    Returns:
        logging.Logger: Configured root logger
    """
    if level is None:
        try:
            level = get_settings().log_level
        except Exception:
            level = "INFO"
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    logger = logging.getLogger("document_qa")
    logger.info(f"Logging configured at {level} level")
    
    return logger


def validate_environment() -> bool:
    """
    Validate that all required environment variables are set.
    
    Returns:
        bool: True if environment is valid
        
    Raises:
        SystemExit: If critical configuration is missing
    """
    logger = logging.getLogger("document_qa.config")
    
    try:
        settings = get_settings()
        logger.info("Configuration validated successfully")
        logger.debug(f"Using Gemini model: {settings.gemini_model}")
        logger.debug(f"Using embedding model: {settings.embedding_model}")
        logger.debug(f"ChromaDB path: {settings.chroma_persist_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        print(f"\n❌ Configuration Error: {e}")
        print("\nPlease ensure you have:")
        print("  1. Copied .env.example to .env")
        print("  2. Set your GOOGLE_API_KEY in .env")
        print("  3. Get a free API key from: https://aistudio.google.com/apikey")
        sys.exit(1)


# Module-level initialization for convenience
if __name__ == "__main__":
    # Test configuration when run directly
    setup_logging("DEBUG")
    validate_environment()
    settings = get_settings()
    print(f"\n✅ Configuration loaded successfully!")
    print(f"   Model: {settings.gemini_model}")
    print(f"   Embeddings: {settings.embedding_model}")
