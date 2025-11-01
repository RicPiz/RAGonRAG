"""
Tests for caching functionality.
"""

import pytest
import time
import numpy as np
from unittest.mock import patch, Mock

from src.cache import (
    EmbeddingCache, QueryCache, ChunkCache,
    get_embedding_cache, get_query_cache, get_chunk_cache,
    clear_all_caches
)


class TestEmbeddingCache:
    """Test embedding cache functionality."""
    
    def test_embedding_cache_basic(self, test_config):
        """Test basic embedding cache operations."""
        cache = EmbeddingCache()
        
        text = "test text"
        embedding = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        # Should be a miss initially
        assert cache.get(text) is None
        
        # Store embedding
        cache.put(text, embedding)
        
        # Should be a hit now
        retrieved = cache.get(text)
        assert retrieved is not None
        assert np.array_equal(retrieved, embedding)
    
    def test_embedding_cache_expiration(self, test_config):
        """Test cache expiration."""
        # Set very short TTL for testing
        test_config.cache.cache_ttl = 0.1
        
        cache = EmbeddingCache()
        text = "test text"
        embedding = np.array([0.1, 0.2, 0.3])
        
        cache.put(text, embedding)
        assert cache.get(text) is not None
        
        # Wait for expiration
        time.sleep(0.2)
        assert cache.get(text) is None
    
    def test_embedding_cache_memory_limit(self, test_config):
        """Test memory cache size limit."""
        # Set small cache size
        test_config.cache.embedding_cache_size = 2
        
        cache = EmbeddingCache()
        
        # Add items up to limit
        for i in range(3):
            text = f"text_{i}"
            embedding = np.array([float(i)] * 3)
            cache.put(text, embedding)
        
        # First item should be evicted
        assert cache.get("text_0") is None
        assert cache.get("text_1") is not None
        assert cache.get("text_2") is not None
    
    def test_embedding_cache_clear(self, test_config):
        """Test cache clearing."""
        cache = EmbeddingCache()
        
        text = "test text"
        embedding = np.array([0.1, 0.2, 0.3])
        cache.put(text, embedding)
        
        assert cache.get(text) is not None
        
        cache.clear()
        assert cache.get(text) is None
    
    @patch('src.cache.dc.Cache')
    def test_embedding_cache_disk_disabled(self, mock_disk_cache, test_config):
        """Test cache with disk caching disabled."""
        test_config.cache.enable_disk_cache = False
        
        cache = EmbeddingCache()
        
        # Should not create disk cache
        mock_disk_cache.assert_not_called()
        assert cache._disk_cache is None


class TestQueryCache:
    """Test query result cache functionality."""
    
    def test_query_cache_basic(self, test_config):
        """Test basic query cache operations."""
        cache = QueryCache()
        
        query = "What is machine learning?"
        retriever_type = "hybrid"
        top_k = 5
        result = {"answer": "ML is...", "hits": []}
        
        # Should be a miss initially
        assert cache.get(query, retriever_type, top_k) is None
        
        # Store result
        cache.put(query, retriever_type, top_k, result)
        
        # Should be a hit now
        retrieved = cache.get(query, retriever_type, top_k)
        assert retrieved == result
    
    def test_query_cache_different_params(self, test_config):
        """Test cache with different parameters."""
        cache = QueryCache()
        
        query = "What is machine learning?"
        result1 = {"answer": "Answer 1"}
        result2 = {"answer": "Answer 2"}
        
        # Store with different parameters
        cache.put(query, "hybrid", 5, result1)
        cache.put(query, "bm25", 5, result2)
        
        # Should retrieve correct results for each parameter set
        assert cache.get(query, "hybrid", 5) == result1
        assert cache.get(query, "bm25", 5) == result2
        assert cache.get(query, "tfidf", 5) is None
    
    def test_query_cache_with_kwargs(self, test_config):
        """Test cache with additional keyword arguments."""
        cache = QueryCache()
        
        query = "test query"
        result = {"answer": "test"}
        
        cache.put(query, "hybrid", 5, result, alpha=0.7, model="gpt-4")
        
        # Should match only with same kwargs
        assert cache.get(query, "hybrid", 5, alpha=0.7, model="gpt-4") == result
        assert cache.get(query, "hybrid", 5, alpha=0.5, model="gpt-4") is None
    
    def test_query_cache_expiration(self, test_config):
        """Test query cache expiration."""
        test_config.cache.cache_ttl = 0.1
        
        cache = QueryCache()
        query = "test query"
        result = {"answer": "test"}
        
        cache.put(query, "hybrid", 5, result)
        assert cache.get(query, "hybrid", 5) == result
        
        time.sleep(0.2)
        assert cache.get(query, "hybrid", 5) is None


class TestChunkCache:
    """Test chunk cache functionality."""
    
    def test_chunk_cache_basic(self, test_config):
        """Test basic chunk cache operations."""
        cache = ChunkCache()
        
        doc_id = "test.md"
        chunking_config = {"max_size": 1000, "overlap": 200}
        chunks = [Mock(), Mock()]
        
        # Should be a miss initially
        assert cache.get(doc_id, chunking_config) is None
        
        # Store chunks
        cache.put(doc_id, chunking_config, chunks)
        
        # Should be a hit now
        retrieved = cache.get(doc_id, chunking_config)
        assert retrieved == chunks
    
    def test_chunk_cache_different_config(self, test_config):
        """Test cache with different chunking configurations."""
        cache = ChunkCache()
        
        doc_id = "test.md"
        config1 = {"max_size": 1000}
        config2 = {"max_size": 500}
        chunks1 = [Mock()]
        chunks2 = [Mock(), Mock()]
        
        cache.put(doc_id, config1, chunks1)
        cache.put(doc_id, config2, chunks2)
        
        assert cache.get(doc_id, config1) == chunks1
        assert cache.get(doc_id, config2) == chunks2
    
    @patch('src.cache.dc.Cache')
    def test_chunk_cache_disk_disabled(self, mock_disk_cache, test_config):
        """Test chunk cache with disk caching disabled."""
        test_config.cache.enable_disk_cache = False
        
        cache = ChunkCache()
        
        # Should not create disk cache
        mock_disk_cache.assert_not_called()
        
        # Operations should still work (just no caching)
        doc_id = "test.md"
        config = {"max_size": 1000}
        chunks = [Mock()]
        
        cache.put(doc_id, config, chunks)  # Should not raise error
        assert cache.get(doc_id, config) is None  # Should return None


class TestCacheGlobalInstances:
    """Test global cache instances."""
    
    def test_get_embedding_cache_singleton(self, test_config):
        """Test embedding cache singleton behavior."""
        cache1 = get_embedding_cache()
        cache2 = get_embedding_cache()
        
        assert cache1 is cache2
    
    def test_get_query_cache_singleton(self, test_config):
        """Test query cache singleton behavior."""
        cache1 = get_query_cache()
        cache2 = get_query_cache()
        
        assert cache1 is cache2
    
    def test_get_chunk_cache_singleton(self, test_config):
        """Test chunk cache singleton behavior."""
        cache1 = get_chunk_cache()
        cache2 = get_chunk_cache()
        
        assert cache1 is cache2
    
    def test_clear_all_caches(self, test_config):
        """Test clearing all caches."""
        # Add some data to caches
        embedding_cache = get_embedding_cache()
        query_cache = get_query_cache()
        chunk_cache = get_chunk_cache()
        
        embedding_cache.put("text", np.array([1, 2, 3]))
        query_cache.put("query", "hybrid", 5, {"answer": "test"})
        chunk_cache.put("doc", {"config": "value"}, [Mock()])
        
        # Verify data exists
        assert embedding_cache.get("text") is not None
        assert query_cache.get("query", "hybrid", 5) is not None
        assert chunk_cache.get("doc", {"config": "value"}) is not None
        
        # Clear all caches
        clear_all_caches()
        
        # Verify data is cleared
        assert embedding_cache.get("text") is None
        assert query_cache.get("query", "hybrid", 5) is None
        assert chunk_cache.get("doc", {"config": "value"}) is None