"""
Robust cosine similarity utilities with multiple backend support.
"""

import numpy as np
from typing import Union, Tuple, Optional, List

try:
    from .logger import get_logger
except ImportError:
    try:
        from logger import get_logger
    except ImportError:
        def get_logger(name): return None

logger = get_logger("similarity_utils")

class CosineSimilarityCalculator:
    """Robust cosine similarity calculator with multiple backends."""
    
    def __init__(self):
        self._backend = self._initialize_backend()
        logger.info(f"Cosine similarity calculator initialized with backend: {self._backend}")
    
    def _initialize_backend(self) -> str:
        """Initialize the best available backend for cosine similarity."""
        
        # Try sklearn first (most reliable)
        try:
            from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
            self._sklearn_cosine = sklearn_cosine
            return "sklearn"
        except ImportError:
            logger.warning("sklearn not available for cosine similarity")
        
        # Try scipy as fallback
        try:
            from scipy.spatial.distance import cosine as scipy_cosine
            self._scipy_cosine = lambda x, y: 1 - scipy_cosine(x.flatten(), y.flatten())
            return "scipy"
        except ImportError:
            logger.warning("scipy not available for cosine similarity")
        
        # Use numpy as last resort
        logger.warning("Using numpy implementation for cosine similarity")
        return "numpy"
    
    def _numpy_cosine_similarity(self, x: np.ndarray, y: np.ndarray) -> float:
        """Pure numpy implementation of cosine similarity."""
        x_flat = x.flatten()
        y_flat = y.flatten()
        
        dot_product = np.dot(x_flat, y_flat)
        norm_x = np.linalg.norm(x_flat)
        norm_y = np.linalg.norm(y_flat)
        
        if norm_x == 0 or norm_y == 0:
            return 0.0
        
        return dot_product / (norm_x * norm_y)
    
    def compute_similarity(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            x: First vector (can be 1D or 2D)
            y: Second vector (can be 1D or 2D)
            
        Returns:
            Cosine similarity as float
        """
        # Ensure arrays are numpy arrays
        x = np.asarray(x)
        y = np.asarray(y)
        
        if self._backend == "sklearn":
            # sklearn expects 2D arrays
            if x.ndim == 1:
                x = x.reshape(1, -1)
            if y.ndim == 1:
                y = y.reshape(1, -1)
            return float(self._sklearn_cosine(x, y)[0][0])
        
        elif self._backend == "scipy":
            # scipy works with 1D arrays
            if x.ndim > 1:
                x = x.flatten()
            if y.ndim > 1:
                y = y.flatten()
            return float(self._scipy_cosine(x, y))
        
        else:  # numpy backend
            return float(self._numpy_cosine_similarity(x, y))
    
    def compute_pairwise_similarities(self, matrix: np.ndarray) -> np.ndarray:
        """
        Compute pairwise cosine similarities for a matrix of vectors.
        
        Args:
            matrix: 2D array where each row is a vector
            
        Returns:
            Similarity matrix
        """
        if self._backend == "sklearn":
            return self._sklearn_cosine(matrix)
        
        # For other backends, compute pairwise manually
        n = matrix.shape[0]
        similarities = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                sim = self.compute_similarity(matrix[i], matrix[j])
                similarities[i, j] = sim
                similarities[j, i] = sim
        
        return similarities
    
    def compute_query_similarities(self, query_vector: Union[np.ndarray, object], matrix: Union[np.ndarray, object]) -> np.ndarray:
        """
        Compute similarities between a query vector and a matrix of vectors.
        
        Args:
            query_vector: Query vector (1D, 2D array, or sparse matrix)
            matrix: Matrix of vectors to compare against (dense or sparse)
            
        Returns:
            Array of similarities
        """
        if self._backend == "sklearn":
            # sklearn can handle sparse matrices directly
            # Ensure proper shapes for sklearn
            if hasattr(query_vector, 'ndim') and query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)
            elif hasattr(query_vector, 'shape') and len(query_vector.shape) == 1:
                query_vector = query_vector.reshape(1, -1)
            return self._sklearn_cosine(query_vector, matrix)[0]
        
        # For other backends, convert sparse matrices to dense and compute similarities one by one
        if hasattr(matrix, 'toarray'):  # Sparse matrix
            matrix = matrix.toarray()
        if hasattr(query_vector, 'toarray'):  # Sparse query
            query_vector = query_vector.toarray()
            
        similarities = []
        for i in range(matrix.shape[0]):
            sim = self.compute_similarity(query_vector, matrix[i])
            similarities.append(sim)
        
        return np.array(similarities)


# Global instance
_cosine_calculator: CosineSimilarityCalculator = None


def get_cosine_calculator() -> CosineSimilarityCalculator:
    """Get the global cosine similarity calculator instance."""
    global _cosine_calculator
    if _cosine_calculator is None:
        _cosine_calculator = CosineSimilarityCalculator()
    return _cosine_calculator


def cosine_similarity(x: Union[np.ndarray, object], y: Union[np.ndarray, object]) -> Union[float, np.ndarray]:
    """
    Compute cosine similarity between vectors or matrices.
    
    This is a drop-in replacement for sklearn's cosine_similarity that works
    with multiple backends and is more robust.
    
    Args:
        x: First vector/matrix (dense or sparse)
        y: Second vector/matrix (dense or sparse)
        
    Returns:
        Cosine similarity (float for vectors, array for matrices)
    """
    calculator = get_cosine_calculator()
    
    # Don't convert to asarray if they're sparse matrices - let sklearn handle them
    if not (hasattr(x, 'toarray') or hasattr(y, 'toarray')):
        x = np.asarray(x)
        y = np.asarray(y)
    
    # Get shapes - handle both dense and sparse matrices
    x_shape = x.shape
    y_shape = y.shape
    x_ndim = len(x_shape)
    y_ndim = len(y_shape)
    
    # Handle different input shapes to mimic sklearn behavior
    if x_ndim == 1 and y_ndim == 1:
        # Two vectors - return single similarity
        return calculator.compute_similarity(x, y)
    
    elif x_ndim == 2 and y_ndim == 2:
        # Matrix vs matrix operations
        if x_shape[0] == 1 and y_shape[0] == 1:
            # Single vectors in matrix form - return scalar
            return calculator.compute_similarity(x[0], y[0])
        elif x_shape[0] == 1:
            # Single query vs multiple documents - this is the TF-IDF case
            return calculator.compute_query_similarities(x, y)
        else:
            # Multiple queries - compute all pairwise
            similarities = []
            for i in range(x_shape[0]):
                sims = calculator.compute_query_similarities(x[i], y)
                similarities.append(sims)
            return np.array(similarities)
    
    elif x_ndim == 1 and y_ndim == 2:
        # Single vector vs matrix
        return calculator.compute_query_similarities(x, y)
    
    elif x_ndim == 2 and y_ndim == 1:
        # Matrix vs single vector
        return calculator.compute_query_similarities(y, x)
    
    else:
        # Fallback for sparse matrices or unusual cases
        if calculator._backend == "sklearn":
            # Let sklearn handle it directly
            return calculator._sklearn_cosine(x, y)[0] if x_shape[0] == 1 else calculator._sklearn_cosine(x, y)
        else:
            # Convert to dense and try again
            if hasattr(x, 'toarray'):
                x = x.toarray()
            if hasattr(y, 'toarray'):
                y = y.toarray()
            return calculator.compute_similarity(x.flatten(), y.flatten())


def compute_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarities for a set of vectors.
    
    Args:
        vectors: 2D array where each row is a vector
        
    Returns:
        Symmetric similarity matrix
    """
    calculator = get_cosine_calculator()
    return calculator.compute_pairwise_similarities(vectors)