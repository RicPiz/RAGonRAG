"""
Configuration management for the RAG system.
"""

import os
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, skip loading
    pass


class EmbeddingConfig(BaseModel):
    """Configuration for embedding models."""
    model: str = Field(default="text-embedding-3-small", description="OpenAI embedding model")
    batch_size: int = Field(default=100, description="Batch size for embedding requests")
    max_retries: int = Field(default=3, description="Maximum retry attempts for API calls")
    timeout: int = Field(default=30, description="Request timeout in seconds")


class ChunkingConfig(BaseModel):
    """Configuration for text chunking."""
    max_chunk_size: int = Field(default=1500, description="Maximum chunk size in characters")
    chunk_overlap: int = Field(default=300, description="Overlap between chunks")
    use_semantic_chunking: bool = Field(default=True, description="Enable semantic chunking")
    min_chunk_size: int = Field(default=150, description="Minimum chunk size in characters")


class RetrievalConfig(BaseModel):
    """Configuration for retrieval systems."""
    tfidf_ngram_range: tuple = Field(default=(1, 2), description="N-gram range for TF-IDF")
    tfidf_max_features: Optional[int] = Field(default=None, description="Maximum features for TF-IDF")
    bm25_k1: float = Field(default=1.2, description="BM25 k1 parameter")
    bm25_b: float = Field(default=0.75, description="BM25 b parameter")
    hybrid_alpha: float = Field(default=0.5, description="Hybrid retrieval weight for embeddings")
    faiss_index_type: str = Field(default="IndexFlatIP", description="FAISS index type")


class GenerationConfig(BaseModel):
    """Configuration for text generation."""
    model: str = Field(default="gpt-4o", description="OpenAI generation model")
    temperature: float = Field(default=0.3, description="Generation temperature")
    top_p: float = Field(default=0.3, description="Top-p sampling parameter")
    max_tokens: int = Field(default=512, description="Maximum tokens in response")
    max_context_chunks: int = Field(default=5, description="Maximum chunks to include in context")


class CacheConfig(BaseModel):
    """Configuration for caching system."""
    embedding_cache_size: int = Field(default=10000, description="Max items in embedding cache")
    query_cache_size: int = Field(default=1000, description="Max items in query cache")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    enable_disk_cache: bool = Field(default=True, description="Enable disk-based caching")
    cache_dir: str = Field(default="cache", description="Directory for cache files")


class LoggingConfig(BaseModel):
    """Configuration for logging."""
    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    log_to_file: bool = Field(default=True, description="Enable file logging")
    log_file: str = Field(default="rag_system.log", description="Log file path")
    max_file_size: int = Field(default=10485760, description="Max log file size in bytes (10MB)")
    backup_count: int = Field(default=5, description="Number of backup log files")


class RAGConfig(BaseModel):
    """Main configuration class for the RAG system."""
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # Global settings
    data_dir: str = Field(default="data", description="Directory containing source documents")
    results_dir: str = Field(default="results", description="Directory for results")
    questions_dir: str = Field(default="questions", description="Directory for questions")
    answers_dir: str = Field(default="answers", description="Directory for answers")
    
    @field_validator("data_dir", "results_dir", "questions_dir", "answers_dir", mode="before")
    @classmethod
    def expand_paths(cls, v):
        """Expand relative paths to absolute paths for directory fields."""
        if isinstance(v, str):
            return str(Path(v).resolve())
        return v

    class Config:
        env_prefix = "RAG_"
        case_sensitive = False


def load_config(config_file: Optional[str] = None) -> RAGConfig:
    """Load configuration from file and environment variables.
    
    Args:
        config_file: Optional path to JSON configuration file.
        
    Returns:
        RAGConfig instance with loaded settings.
    """
    config_data: Dict[str, Any] = {}
    
    # Load from JSON file if provided
    if config_file and Path(config_file).exists():
        import json
        with open(config_file, 'r') as f:
            config_data = json.load(f)
    
    # Override with environment variables
    config = RAGConfig(**config_data)
    
    # Ensure cache directory exists
    Path(config.cache.cache_dir).mkdir(parents=True, exist_ok=True)
    
    return config


# Global configuration instance
_config: Optional[RAGConfig] = None


def get_config() -> RAGConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: RAGConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config


def get_openai_api_key() -> Optional[str]:
    """Get OpenAI API key from environment variables.

    Checks for OPENAI_API_KEY in environment variables (including .env file).

    Returns:
        API key if found, None otherwise
    """
    return os.getenv("OPENAI_API_KEY")


def validate_configuration(config: RAGConfig) -> None:
    """Validate the RAG configuration for consistency and correctness.

    Args:
        config: The RAGConfig instance to validate

    Raises:
        RAGError: If configuration is invalid
    """
    try:
        from .validation import RAGError, InputValidator
    except ImportError:
        from validation import RAGError, InputValidator

    errors = []

    # Validate embedding configuration
    try:
        InputValidator.validate_model_name(config.embedding.model)
    except Exception as e:
        errors.append(f"Invalid embedding model: {e}")

    if config.embedding.batch_size < 1 or config.embedding.batch_size > 1000:
        errors.append("Embedding batch size must be between 1 and 1000")

    if config.embedding.max_retries < 0 or config.embedding.max_retries > 10:
        errors.append("Embedding max retries must be between 0 and 10")

    if config.embedding.timeout < 1 or config.embedding.timeout > 300:
        errors.append("Embedding timeout must be between 1 and 300 seconds")

    # Validate chunking configuration
    if config.chunking.max_chunk_size < 100 or config.chunking.max_chunk_size > 10000:
        errors.append("Max chunk size must be between 100 and 10000 characters")

    if config.chunking.chunk_overlap < 0 or config.chunking.chunk_overlap >= config.chunking.max_chunk_size:
        errors.append("Chunk overlap must be >= 0 and < max_chunk_size")

    if config.chunking.min_chunk_size < 10 or config.chunking.min_chunk_size > config.chunking.max_chunk_size:
        errors.append("Min chunk size must be between 10 and max_chunk_size")

    # Validate retrieval configuration
    if config.retrieval.hybrid_alpha < 0.0 or config.retrieval.hybrid_alpha > 1.0:
        errors.append("Hybrid alpha must be between 0.0 and 1.0")

    if config.retrieval.bm25_k1 <= 0 or config.retrieval.bm25_k1 > 3:
        errors.append("BM25 k1 must be between 0 and 3")

    if config.retrieval.bm25_b < 0 or config.retrieval.bm25_b > 1:
        errors.append("BM25 b must be between 0 and 1")

    # Validate generation configuration
    try:
        InputValidator.validate_model_name(config.generation.model)
    except Exception as e:
        errors.append(f"Invalid generation model: {e}")

    if config.generation.temperature < 0 or config.generation.temperature > 2:
        errors.append("Generation temperature must be between 0 and 2")

    if config.generation.top_p < 0 or config.generation.top_p > 1:
        errors.append("Generation top_p must be between 0 and 1")

    if config.generation.max_tokens < 1 or config.generation.max_tokens > 32000:
        errors.append("Generation max_tokens must be between 1 and 32000")

    if config.generation.max_context_chunks < 1 or config.generation.max_context_chunks > 50:
        errors.append("Max context chunks must be between 1 and 50")

    # Validate cache configuration
    if config.cache.embedding_cache_size < 100 or config.cache.embedding_cache_size > 100000:
        errors.append("Embedding cache size must be between 100 and 100000")

    if config.cache.query_cache_size < 10 or config.cache.query_cache_size > 10000:
        errors.append("Query cache size must be between 10 and 10000")

    if config.cache.cache_ttl < 60 or config.cache.cache_ttl > 86400:
        errors.append("Cache TTL must be between 60 and 86400 seconds")

    # Validate logging configuration
    if config.logging.level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        errors.append("Logging level must be DEBUG, INFO, WARNING, ERROR, or CRITICAL")

    if config.logging.max_file_size < 1024 or config.logging.max_file_size > 104857600:  # 100MB
        errors.append("Max log file size must be between 1KB and 100MB")

    if config.logging.backup_count < 0 or config.logging.backup_count > 100:
        errors.append("Log backup count must be between 0 and 100")

    # Validate directory paths
    try:
        InputValidator.validate_directory_path(config.data_dir, must_exist=False)
    except Exception as e:
        errors.append(f"Invalid data directory: {e}")

    try:
        InputValidator.validate_directory_path(config.results_dir, must_exist=False)
    except Exception as e:
        errors.append(f"Invalid results directory: {e}")

    try:
        InputValidator.validate_directory_path(config.questions_dir, must_exist=False)
    except Exception as e:
        errors.append(f"Invalid questions directory: {e}")

    try:
        InputValidator.validate_directory_path(config.answers_dir, must_exist=False)
    except Exception as e:
        errors.append(f"Invalid answers directory: {e}")

    if errors:
        error_msg = f"Configuration validation failed ({len(errors)} errors):\n" + "\n".join(f"  - {error}" for error in errors)
        raise RAGError(error_msg, "VALIDATION_CONFIG_INVALID", "validation")


def validate_startup_configuration(require_api_key: bool = True) -> RAGConfig:
    """Validate configuration on system startup.

    This function should be called at system startup to ensure
    the configuration is valid before proceeding.

    Args:
        require_api_key: Whether to require API key (default True).
                        Set to False for UI initialization without API access.

    Returns:
        Validated RAGConfig instance

    Raises:
        RAGError: If configuration is invalid
    """
    config = get_config()

    # Validate base configuration (non-API parts)
    try:
        validate_configuration(config)
    except RAGError:
        # If validation fails but we're in UI context, let UI handle it
        if not require_api_key:
            raise
        # Otherwise re-raise
        raise

    # Additional startup validations only when API key is required
    if require_api_key:
        api_key = get_openai_api_key()
        if not api_key:
            raise RAGError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable or create .env file.",
                "VALIDATION_API_KEY_MISSING",
                "validation"
            )

    return config