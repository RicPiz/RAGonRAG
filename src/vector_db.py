"""
FAISS-based vector database for efficient similarity search and storage.
"""

import os
import pickle
import json
import hashlib
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import numpy as np
import faiss

# Import core dependencies
try:
    from .config import get_config
    from .logger import get_logger
    from .chunking import Chunk
except ImportError:
    try:
        from config import get_config
        from logger import get_logger
        from chunking import Chunk
    except ImportError:
        def get_config(): return None
        def get_logger(name): return None
        Chunk = object

logger = get_logger("vector_db")


class FAISSVectorDB:
    """FAISS-based vector database for storing and retrieving embeddings."""
    
    def __init__(self, dimension: int, index_type: str = "IndexFlatIP"):
        """Initialize FAISS vector database.
        
        Args:
            dimension: Dimension of the embedding vectors
            index_type: Type of FAISS index to use
        """
        self.dimension = dimension
        self.index_type = index_type
        self.index: Optional[faiss.Index] = None
        self.chunks: List[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.is_trained = False
        
        # Persistence paths
        config = get_config()
        self.db_dir = Path(config.cache.cache_dir) / "vector_db"
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.db_dir / f"faiss_index_{index_type}.bin"
        self.metadata_path = self.db_dir / f"metadata_{index_type}.json"  # Changed from .pkl to .json
        
        logger.info("Initialized FAISS vector database", 
                   dimension=dimension, index_type=index_type)
    
    def _create_index(self, num_vectors: int = 0) -> faiss.Index:
        """Create an optimized FAISS index based on dataset characteristics."""
        if self.index_type == "auto":
            index = self._select_optimal_index(num_vectors)
        else:
            index = self._create_specific_index(num_vectors)

        logger.info("Created FAISS index",
                   type=self.index_type,
                   num_vectors=num_vectors,
                   actual_type=type(index).__name__)
        return index

    def _select_optimal_index(self, num_vectors: int) -> faiss.Index:
        """Automatically select the best index type based on dataset size and characteristics."""
        if num_vectors <= 1000:
            # For small datasets, exact search is fastest
            logger.debug("Selected IndexFlatIP for small dataset")
            return faiss.IndexFlatIP(self.dimension)

        elif num_vectors <= 10000:
            # For medium datasets, IVF provides good balance
            nlist = min(100, max(4, num_vectors // 39))  # Rule of thumb: sqrt(n)/4
            quantizer = faiss.IndexFlatIP(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            logger.debug("Selected optimized IVF for medium dataset", nlist=nlist)
            return index

        else:
            # For large datasets, HNSW is more efficient
            index = faiss.IndexHNSWFlat(self.dimension, 32)
            # Optimize HNSW parameters based on dataset size
            index.hnsw.efConstruction = min(200, max(40, num_vectors // 1000))
            index.hnsw.efSearch = min(128, max(16, num_vectors // 5000))
            logger.debug("Selected optimized HNSW for large dataset",
                        ef_construction=index.hnsw.efConstruction,
                        ef_search=index.hnsw.efSearch)
            return index

    def _create_specific_index(self, num_vectors: int) -> faiss.Index:
        """Create the specifically requested index type with optimized parameters."""
        if self.index_type == "IndexFlatIP":
            return faiss.IndexFlatIP(self.dimension)

        elif self.index_type == "IndexFlatL2":
            return faiss.IndexFlatL2(self.dimension)

        elif self.index_type == "IndexIVFFlat":
            # Optimize IVF parameters based on dataset size
            if num_vectors <= 1000:
                nlist = 4
            elif num_vectors <= 10000:
                nlist = min(100, max(4, num_vectors // 39))
            else:
                nlist = min(1024, max(100, num_vectors // 39))

            quantizer = faiss.IndexFlatIP(self.dimension)  # Use IP for normalized embeddings
            index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)

            logger.debug("Created IVF index with optimized parameters",
                        nlist=nlist, num_vectors=num_vectors)
            return index

        elif self.index_type == "IndexHNSW":
            index = faiss.IndexHNSWFlat(self.dimension, 32)

            # Optimize HNSW parameters based on dataset size
            if num_vectors <= 10000:
                index.hnsw.efConstruction = 40
                index.hnsw.efSearch = 16
            elif num_vectors <= 100000:
                index.hnsw.efConstruction = 100
                index.hnsw.efSearch = 64
            else:
                index.hnsw.efConstruction = 200
                index.hnsw.efSearch = 128

            logger.debug("Created HNSW index with optimized parameters",
                        ef_construction=index.hnsw.efConstruction,
                        ef_search=index.hnsw.efSearch)
            return index

        else:
            raise ValueError(f"Unsupported index type: {self.index_type}. "
                           "Supported: auto, IndexFlatIP, IndexFlatL2, IndexIVFFlat, IndexHNSW")
    
    def build_index(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        """Build the FAISS index from chunks and their embeddings.
        
        Args:
            chunks: List of chunk objects
            embeddings: Numpy array of embeddings, shape (n_chunks, dimension)
        """
        if embeddings.shape[0] != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks")
        
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} doesn't match expected {self.dimension}")
        
        logger.info("Building FAISS index", num_chunks=len(chunks))
        
        # Store chunks and embeddings
        self.chunks = chunks
        self.embeddings = embeddings.astype(np.float32)
        
        # Normalize embeddings for cosine similarity (if using IndexFlatIP)
        if self.index_type == "IndexFlatIP":
            faiss.normalize_L2(self.embeddings)
        
        # Create and train index
        self.index = self._create_index(len(chunks))
        
        # Train index if needed
        if hasattr(self.index, 'train') and not self.index.is_trained:
            logger.debug("Training FAISS index")
            self.index.train(self.embeddings)
        
        # Add vectors to index
        self.index.add(self.embeddings)
        self.is_trained = True
        
        logger.info("FAISS index built successfully", 
                   total_vectors=self.index.ntotal)
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Chunk, float]]:
        """Search for similar chunks using the optimized query processing.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return

        Returns:
            List of (chunk, score) tuples sorted by similarity
        """
        if self.index is None or not self.is_trained:
            raise RuntimeError("Index not built. Call build_index() first.")

        if query_embedding.shape[0] != self.dimension:
            raise ValueError(f"Query embedding dimension {query_embedding.shape[0]} doesn't match expected {self.dimension}")

        # Prepare query
        query = query_embedding.reshape(1, -1).astype(np.float32)

        # Optimize search parameters based on index type
        k_search = self._optimize_search_k(k)

        # Normalize for cosine similarity (for IP-based indexes)
        if isinstance(self.index, faiss.IndexFlatIP) or \
           (hasattr(self.index, 'quantizer') and isinstance(self.index.quantizer, faiss.IndexFlatIP)):
            faiss.normalize_L2(query)

        # Perform search with optimized parameters
        scores, indices = self.index.search(query, k_search)

        # Convert to results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid result
                chunk = self.chunks[idx]
                similarity = self._convert_score_to_similarity(float(score))
                results.append((chunk, similarity))

        # Sort by similarity (descending) and limit to requested k
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:k]

        logger.debug("FAISS search completed",
                    query_dim=query_embedding.shape[0],
                    requested_k=k,
                    search_k=k_search,
                    results_count=len(results),
                    index_type=self.index_type)

        return results

    def _optimize_search_k(self, k: int) -> int:
        """Optimize the number of candidates to search based on index type."""
        if isinstance(self.index, faiss.IndexHNSWFlat):
            # HNSW benefits from searching more candidates
            return min(k * 3, len(self.chunks))
        elif hasattr(self.index, 'nlist') and self.index.nlist > 1:
            # IVF indexes: search more to account for quantization
            return min(k * 5, len(self.chunks))
        else:
            # Flat indexes: exact search
            return min(k, len(self.chunks))

    def _convert_score_to_similarity(self, score: float) -> float:
        """Convert FAISS score to normalized similarity score."""
        if isinstance(self.index, faiss.IndexFlatL2):
            # L2 distance: convert to similarity (0=identical, decreases with distance)
            return 1.0 / (1.0 + score)
        elif isinstance(self.index, faiss.IndexFlatIP):
            # Inner product: already a similarity measure (-1 to 1 for normalized vectors)
            return (score + 1.0) / 2.0  # Normalize to 0-1 range
        elif hasattr(self.index, 'quantizer'):
            # IVF indexes: assume IP quantizer for normalized vectors
            return (score + 1.0) / 2.0
        else:
            # Default: assume similarity score
            return max(0.0, min(1.0, score))
    
    def save(self, embedding_model: Optional[str] = None, corpus_files: Optional[List[str]] = None) -> None:
        """Persist the index and metadata to disk with JSON serialization.

        Args:
            embedding_model: Optional embedding model name for validation
            corpus_files: Optional list of corpus file names for validation
        """
        if self.index is None:
            logger.warning("No index to save")
            return

        logger.info("Saving FAISS index to disk")

        # Save FAISS index
        faiss.write_index(self.index, str(self.index_path))

        # Convert chunks to JSON-serializable format (no pickle!)
        chunks_data = [
            {
                'doc_id': c.doc_id,
                'title': c.title,
                'content': c.content,
                'source_path': c.source_path,
                'position': c.position,
                'start_char': c.start_char,
                'end_char': c.end_char,
                'semantic_similarity': c.semantic_similarity,
                'metadata': c.metadata if c.metadata else {}
            }
            for c in self.chunks
        ]

        # Calculate corpus hash for change detection
        corpus_hash = None
        if corpus_files:
            corpus_hash = hashlib.md5(''.join(sorted(corpus_files)).encode()).hexdigest()

        # Save metadata as JSON with version and validation info
        metadata = {
            'chunk_schema_version': '1.0',
            'chunks': chunks_data,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'is_trained': self.is_trained,
            'embeddings_shape': list(self.embeddings.shape) if self.embeddings is not None else None,
            'embedding_model': embedding_model,
            'corpus_hash': corpus_hash,
            'num_chunks': len(self.chunks)
        }

        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info("FAISS index saved successfully", format="JSON", num_chunks=len(self.chunks))
    
    def load(self) -> bool:
        """Load the index and metadata from disk with JSON deserialization.

        Returns:
            True if loaded successfully, False otherwise
        """
        if not self.index_path.exists() or not self.metadata_path.exists():
            logger.debug("No saved index found")
            return False

        try:
            logger.info("Loading FAISS index from disk")

            # Load FAISS index
            self.index = faiss.read_index(str(self.index_path))

            # Try loading JSON first, fall back to pickle for backward compatibility
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                # Reconstruct Chunk objects from JSON
                self.chunks = [
                    Chunk(
                        doc_id=c['doc_id'],
                        title=c['title'],
                        content=c['content'],
                        source_path=c['source_path'],
                        position=c['position'],
                        start_char=c['start_char'],
                        end_char=c['end_char'],
                        semantic_similarity=c.get('semantic_similarity', 0.0),
                        metadata=c.get('metadata', {})
                    )
                    for c in metadata['chunks']
                ]

                logger.info("Loaded index with JSON format (modern)")

            except (json.JSONDecodeError, KeyError) as json_err:
                # Fall back to pickle for old format
                logger.warning("JSON load failed, trying pickle format (legacy)", error=str(json_err))
                with open(self.metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                self.chunks = metadata['chunks']
                logger.warning("Loaded legacy pickle format. Consider rebuilding index for JSON format.")

            self.dimension = metadata['dimension']
            self.is_trained = metadata.get('is_trained', True)

            # Store validation metadata for later checks
            self.loaded_metadata = {
                'embedding_model': metadata.get('embedding_model'),
                'corpus_hash': metadata.get('corpus_hash'),
                'chunk_schema_version': metadata.get('chunk_schema_version', 'unknown')
            }

            # Note: embeddings are not loaded to prevent memory leaks
            # FAISS index contains all necessary information for search
            self.embeddings = None

            logger.info("FAISS index loaded successfully",
                       total_vectors=self.index.ntotal,
                       format="JSON" if 'chunk_schema_version' in metadata else "pickle")
            return True

        except Exception as e:
            logger.error("Failed to load FAISS index", exception=e)
            return False
    
    def clear(self) -> None:
        """Clear the index and remove persisted files."""
        self.index = None
        self.chunks = []
        self.embeddings = None
        self.is_trained = False
        
        # Remove persisted files
        if self.index_path.exists():
            self.index_path.unlink()
        if self.metadata_path.exists():
            self.metadata_path.unlink()
        
        logger.info("FAISS index cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the vector database."""
        stats = {
            'dimension': self.dimension,
            'index_type': self.index_type,
            'is_trained': self.is_trained,
            'num_chunks': len(self.chunks),
            'total_vectors': self.index.ntotal if self.index else 0,
            'index_memory_mb': self._estimate_memory_usage(),
            'index_size_mb': self.index_path.stat().st_size / 1024 / 1024 if self.index_path.exists() else 0,
            'performance_metrics': self._get_performance_metrics()
        }
        return stats

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage of the index in MB."""
        if self.index is None:
            return 0.0

        # Rough estimation based on index type and size
        base_memory = self.index.ntotal * self.dimension * 4  # 4 bytes per float32

        if hasattr(self.index, 'nlist'):
            # IVF indexes have additional overhead
            base_memory *= 1.2  # 20% overhead for IVF structures
        elif isinstance(self.index, faiss.IndexHNSWFlat):
            # HNSW has significant memory overhead
            base_memory *= 2.5  # HNSW graph structures

        return base_memory / (1024 * 1024)  # Convert to MB

    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance recommendations and metrics."""
        metrics = {}

        if self.index is None:
            return metrics

        num_vectors = len(self.chunks)

        # Performance recommendations based on dataset size
        if num_vectors <= 1000:
            recommended_index = "IndexFlatIP"
            metrics['recommendation'] = "Current setup is optimal for small datasets"
        elif num_vectors <= 10000:
            recommended_index = "IndexIVFFlat"
            if self.index_type != "IndexIVFFlat" and self.index_type != "auto":
                metrics['recommendation'] = "Consider IndexIVFFlat for better performance on medium datasets"
        else:
            recommended_index = "IndexHNSW"
            if self.index_type not in ["IndexHNSW", "auto"]:
                metrics['recommendation'] = "Consider IndexHNSW for better performance on large datasets"

        metrics.update({
            'recommended_index': recommended_index,
            'current_index': self.index_type,
            'dataset_size': num_vectors,
            'estimated_query_time': self._estimate_query_time()
        })

        return metrics

    def _estimate_query_time(self) -> str:
        """Estimate typical query time based on index type and size."""
        num_vectors = len(self.chunks)

        if isinstance(self.index, faiss.IndexFlatIP):
            if num_vectors <= 1000:
                return "< 1ms"
            elif num_vectors <= 10000:
                return "1-5ms"
            else:
                return "5-20ms"
        elif isinstance(self.index, faiss.IndexHNSWFlat):
            if num_vectors <= 10000:
                return "2-10ms"
            elif num_vectors <= 100000:
                return "5-25ms"
            else:
                return "10-50ms"
        elif hasattr(self.index, 'nlist'):
            return "3-15ms"
        else:
            return "Unknown"

    def optimize_for_query_pattern(self, query_pattern: str = "single") -> None:
        """Optimize index parameters for specific query patterns.

        Args:
            query_pattern: 'single', 'batch', or 'mixed'
        """
        if self.index is None:
            return

        if isinstance(self.index, faiss.IndexHNSWFlat):
            if query_pattern == "single":
                # Optimize for single queries
                self.index.hnsw.efSearch = min(32, self.index.hnsw.efSearch)
            elif query_pattern == "batch":
                # Optimize for batch queries
                self.index.hnsw.efSearch = min(128, self.index.hnsw.efSearch)
            logger.info("Optimized HNSW for query pattern", pattern=query_pattern, ef_search=self.index.hnsw.efSearch)
        elif hasattr(self.index, 'nprobe'):
            # IVF index optimization
            if query_pattern == "single":
                self.index.nprobe = min(4, self.index.nlist // 4)
            elif query_pattern == "batch":
                self.index.nprobe = min(16, self.index.nlist // 4)
            logger.info("Optimized IVF for query pattern", pattern=query_pattern, nprobe=self.index.nprobe)


class VectorDBManager:
    """Manager class for vector databases with different configurations."""
    
    def __init__(self):
        self._databases: Dict[str, FAISSVectorDB] = {}
        self.config = get_config()
    
    def get_db(self, db_name: str, dimension: int, index_type: Optional[str] = None) -> FAISSVectorDB:
        """Get or create a vector database instance.
        
        Args:
            db_name: Name identifier for the database
            dimension: Embedding dimension
            index_type: FAISS index type (defaults to config)
            
        Returns:
            FAISSVectorDB instance
        """
        if index_type is None:
            index_type = self.config.retrieval.faiss_index_type
        
        db_key = f"{db_name}_{dimension}_{index_type}"
        
        if db_key not in self._databases:
            self._databases[db_key] = FAISSVectorDB(dimension, index_type)
            
            # Try to load existing index
            if not self._databases[db_key].load():
                logger.debug("No existing index found, will create new one", db_name=db_name)
        
        return self._databases[db_key]
    
    def clear_all(self) -> None:
        """Clear all vector databases."""
        for db in self._databases.values():
            db.clear()
        self._databases.clear()
        logger.info("All vector databases cleared")
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all vector databases."""
        return {name: db.get_stats() for name, db in self._databases.items()}


# Global manager instance
_vector_db_manager: Optional[VectorDBManager] = None


def get_vector_db_manager() -> VectorDBManager:
    """Get the global vector database manager."""
    global _vector_db_manager
    if _vector_db_manager is None:
        _vector_db_manager = VectorDBManager()
    return _vector_db_manager