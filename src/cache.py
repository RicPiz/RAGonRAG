"""
Comprehensive caching system for embeddings, queries, and results.
"""

import hashlib
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import diskcache as dc
import numpy as np

# Ensure src directory is in path for absolute imports
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from .config import get_config
    from .logger import get_logger
except ImportError:
    from config import get_config
    from logger import get_logger

logger = get_logger("cache")

# Cache schema version for compatibility checks
CACHE_SCHEMA_VERSION = "1.0"


class UnifiedCache:
    """Unified cache for different data types with memory and disk storage."""

    def __init__(self, cache_type: str, max_memory_size: int):
        """
        Initialize unified cache.

        Args:
            cache_type: Type of cache ('embeddings', 'queries', 'chunks')
            max_memory_size: Maximum items in memory cache
        """
        config = get_config().cache
        self.cache_type = cache_type
        self._memory_cache: Dict[str, Tuple[Any, float]] = {}  # {key: (data, timestamp)}
        self._max_memory_size = max_memory_size
        self._ttl = config.cache_ttl
        self._access_order: Dict[str, float] = {}  # Track access times for true LRU

        # Disk cache with size limits
        if config.enable_disk_cache:
            cache_dir = Path(config.cache_dir) / cache_type
            cache_dir.mkdir(parents=True, exist_ok=True)
            # Configure diskcache with eviction policy
            self._disk_cache = dc.Cache(
                str(cache_dir),
                size_limit=1024 ** 3,  # 1GB limit
                cull_limit=10  # Remove 10% when full
            )
        else:
            self._disk_cache = None

        logger.debug(f"Initialized {cache_type} cache", max_memory_size=max_memory_size)

    def _create_key(self, *args, **kwargs) -> str:
        """Create cache key from arguments with versioning."""
        import json

        # Build namespaced key with version
        if len(args) == 1 and not kwargs:
            # Simple key from single argument
            if isinstance(args[0], str):
                base_key = args[0]
            else:
                base_key = json.dumps(args[0], sort_keys=True, default=str)
        else:
            # Complex key from multiple arguments
            base_key = json.dumps({
                'args': args,
                'kwargs': sorted(kwargs.items())
            }, sort_keys=True, default=str)

        # Include version and cache type in the key
        versioned_key = f"v{CACHE_SCHEMA_VERSION}:{self.cache_type}:{base_key}"
        return hashlib.sha256(versioned_key.encode('utf-8')).hexdigest()

    def get(self, *args, **kwargs) -> Optional[Any]:
        """Get item from cache with true LRU tracking."""
        key = self._create_key(*args, **kwargs)

        # Try memory cache first
        if key in self._memory_cache:
            data, timestamp = self._memory_cache[key]
            if time.time() - timestamp < self._ttl:
                # Update access time for LRU
                self._access_order[key] = time.time()
                logger.debug(f"{self.cache_type} cache hit (memory)")
                return data
            else:
                # Expired
                del self._memory_cache[key]
                if key in self._access_order:
                    del self._access_order[key]

        # Try disk cache
        if self._disk_cache and key in self._disk_cache:
            cached_data = self._disk_cache[key]
            if time.time() - cached_data['timestamp'] < self._ttl:
                data = cached_data['data']
                # Store back in memory cache
                self._memory_cache[key] = (data, cached_data['timestamp'])
                self._access_order[key] = time.time()
                logger.debug(f"{self.cache_type} cache hit (disk)")
                return data
            else:
                # Expired
                del self._disk_cache[key]

        logger.debug(f"{self.cache_type} cache miss")
        return None
    
    def put(self, data: Any, *args, **kwargs) -> None:
        """Store item in cache with true LRU eviction."""
        key = self._create_key(*args, **kwargs)
        timestamp = time.time()

        # Memory cache with true LRU eviction based on access time
        if len(self._memory_cache) >= self._max_memory_size:
            # Remove least recently used entry
            if self._access_order:
                lru_key = min(self._access_order.keys(),
                             key=lambda k: self._access_order[k])
                del self._memory_cache[lru_key]
                del self._access_order[lru_key]
            else:
                # Fallback to oldest insertion if no access tracking
                oldest_key = min(self._memory_cache.keys(),
                               key=lambda k: self._memory_cache[k][1])
                del self._memory_cache[oldest_key]

        self._memory_cache[key] = (data, timestamp)
        self._access_order[key] = timestamp  # Initialize access time

        # Disk cache with version metadata
        if self._disk_cache:
            self._disk_cache[key] = {
                'data': data,
                'timestamp': timestamp,
                'version': CACHE_SCHEMA_VERSION
            }

        logger.debug(f"{self.cache_type} item cached")
    
    def clear(self) -> None:
        """Clear all caches."""
        self._memory_cache.clear()
        if self._disk_cache:
            self._disk_cache.clear()
        logger.info(f"{self.cache_type} cache cleared")


# Legacy classes for backward compatibility
class EmbeddingCache(UnifiedCache):
    """Cache for embedding vectors (backward compatibility)."""
    
    def __init__(self):
        config = get_config().cache
        super().__init__('embeddings', config.embedding_cache_size)

    def get(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache."""
        return super().get(text)

    def put(self, text: str, embedding: np.ndarray) -> None:
        """Store embedding in cache."""
        super().put(embedding, text)


class QueryCache(UnifiedCache):
    """Cache for query results (backward compatibility)."""

    def __init__(self):
        config = get_config().cache
        super().__init__('queries', config.query_cache_size)
    
    def get(self, query: str, retriever_type: str, top_k: int, **kwargs) -> Optional[Dict[str, Any]]:
        """Get cached query result."""
        return super().get(query, retriever_type, top_k, **kwargs)
    
    def put(self, query: str, retriever_type: str, top_k: int, result: Dict[str, Any], **kwargs) -> None:
        """Store query result in cache."""
        super().put(result, query, retriever_type, top_k, **kwargs)


class ChunkCache(UnifiedCache):
    """Cache for processed chunks (backward compatibility)."""
    
    def __init__(self):
        # Use unlimited size for chunk cache (rely on disk cache management)
        super().__init__('chunks', 10000)
    
    def get(self, doc_id: str, chunking_config: Dict[str, Any]) -> Optional[List[Any]]:
        """Get cached chunks for document."""
        return super().get(doc_id, chunking_config)
    
    def put(self, doc_id: str, chunking_config: Dict[str, Any], chunks: List[Any]) -> None:
        """Store chunks in cache."""
        super().put(chunks, doc_id, chunking_config)


# Global cache instances
_embedding_cache: Optional[EmbeddingCache] = None
_query_cache: Optional[QueryCache] = None
_chunk_cache: Optional[ChunkCache] = None


def get_embedding_cache() -> EmbeddingCache:
    """Get the global embedding cache instance."""
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = EmbeddingCache()
    return _embedding_cache


def get_query_cache() -> QueryCache:
    """Get the global query cache instance."""
    global _query_cache
    if _query_cache is None:
        _query_cache = QueryCache()
    return _query_cache


def get_chunk_cache() -> ChunkCache:
    """Get the global chunk cache instance."""
    global _chunk_cache
    if _chunk_cache is None:
        _chunk_cache = ChunkCache()
    return _chunk_cache


def clear_all_caches() -> None:
    """Clear all cache instances."""
    get_embedding_cache().clear()
    get_query_cache().clear()
    get_chunk_cache().clear()
    logger.info("All caches cleared")