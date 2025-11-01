"""
RAGonRAG - Enhanced Retrieval Augmented Generation System

A production-ready RAG (Retrieval-Augmented Generation) system featuring:

🚀 **Core Features:**
- FAISS-based vector search with intelligent index optimization
- Hybrid retrieval combining semantic and keyword search
- Comprehensive caching system (embeddings, queries, chunks)
- Async processing with proper resource management
- Real-time performance monitoring and profiling
- Structured logging with JSON support

🔧 **Technical Highlights:**
- Automatic FAISS index selection based on dataset size
- Multi-backend similarity computation (sklearn/scipy/numpy)
- Comprehensive error handling with standardized error codes
- Type-hinted codebase with full IDE support
- Production-ready async context management

📊 **Evaluation & Monitoring:**
- Ragas-based RAG quality metrics (answer relevancy, faithfulness, etc.)
- Real-time performance analytics and reporting
- Health check endpoints and system diagnostics
- Comprehensive logging with analysis tools

📚 **Quick Start:**
```python
import asyncio
from rag_system import create_rag_system

async def main():
    # Create and build RAG system
    system = await create_rag_system("data/", build_index=True)

    # Query the system
    result = await system.query("What is machine learning?")
    print(result["answer"])

asyncio.run(main())
```

For more examples and documentation, see the individual module docstrings.
"""

__version__ = "1.0.0"
__author__ = "RAGonRAG Team"
__description__ = "Enhanced Retrieval Augmented Generation System"

# Core system
try:
    from .rag_system import EnhancedRAGSystem, create_rag_system
except ImportError:
    try:
        from rag_system import EnhancedRAGSystem, create_rag_system
    except ImportError:
        class EnhancedRAGSystem: pass
        def create_rag_system(*args, **kwargs): return None

# Configuration
try:
    from .config import get_config, RAGConfig
except ImportError:
    try:
        from config import get_config, RAGConfig
    except ImportError:
        def get_config(): return None
        class RAGConfig: pass

# Main components
try:
    from .chunking import get_chunker, SmartChunker
    from .vector_db import get_vector_db_manager
    from .cache import get_embedding_cache, get_query_cache, get_chunk_cache
    from .async_utils import AsyncEmbeddingGenerator, AsyncHTTPClient, AsyncResourceManager
except ImportError:
    try:
        from chunking import get_chunker, SmartChunker
        from vector_db import get_vector_db_manager
        from cache import get_embedding_cache, get_query_cache, get_chunk_cache
        from async_utils import AsyncEmbeddingGenerator, AsyncHTTPClient, AsyncResourceManager
    except ImportError:
        def get_chunker(): return None
        class SmartChunker: pass
        def get_vector_db_manager(): return None
        def get_embedding_cache(): return None
        def get_query_cache(): return None
        def get_chunk_cache(): return None
        class AsyncEmbeddingGenerator: pass
        class AsyncHTTPClient: pass
        class AsyncResourceManager: pass

# Utilities
try:
    from .validation import InputValidator, ErrorHandler, RAGError
    from .metrics import get_metrics_collector, monitor_performance
    from .logger import get_logger, get_profiler, profile_function, create_performance_report
except ImportError:
    try:
        from validation import InputValidator, ErrorHandler, RAGError
        from metrics import get_metrics_collector, monitor_performance
        from logger import get_logger, get_profiler, profile_function, create_performance_report
    except ImportError:
        class InputValidator: pass
        class ErrorHandler: pass
        class RAGError(Exception): pass
        def get_metrics_collector(): return None
        def monitor_performance(): return lambda f: f
        def get_logger(name): return None
        def get_profiler(): return None
        def profile_function(): return lambda f: f
        def create_performance_report(): return None

# Health check utilities
try:
    from .health import health_check, system_status, quick_diagnostics
except ImportError:
    try:
        from health import health_check, system_status, quick_diagnostics
    except ImportError:
        def health_check(): return {"status": "error"}
        def system_status(): return {"status": "error"}
        def quick_diagnostics(): return {"issues": ["Imports failed"]}

# Re-export commonly used functions for convenience
__all__ = [
    # Core API
    "EnhancedRAGSystem", "create_rag_system",

    # Configuration
    "get_config", "RAGConfig",

    # Main Components
    "get_chunker", "SmartChunker",
    "get_vector_db_manager",
    "get_embedding_cache", "get_query_cache", "get_chunk_cache",
    "AsyncEmbeddingGenerator", "AsyncHTTPClient", "AsyncResourceManager",

    # Utilities
    "InputValidator", "ErrorHandler", "RAGError",
    "get_metrics_collector", "monitor_performance",
    "get_logger", "get_profiler", "profile_function", "create_performance_report",

    # Health & Monitoring
    "health_check", "system_status", "quick_diagnostics",

    # Metadata
    "__version__", "__author__", "__description__"
]

def quick_start_demo() -> str:
    """Return a quick start example for the system."""
    return """
# Quick Start Example
import asyncio
from rag_system import create_rag_system

async def demo():
    # Initialize system
    system = await create_rag_system("data/", build_index=True)

    # Ask questions
    result = await system.query("What are the key concepts in RAG?")
    print(result["answer"])

    # Check system health
    from rag_system.health import health_check
    status = health_check()
    print(f"System health: {status['status']}")

asyncio.run(demo())
"""

