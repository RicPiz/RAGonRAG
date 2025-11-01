"""
Enhanced RAG system with FAISS vector database, async processing, caching, and comprehensive error handling.

This module provides a complete rewrite of the RAG system with:
- FAISS-based vector storage for efficient similarity search
- Comprehensive caching for embeddings and query results  
- Async processing for concurrent operations
- Robust error handling and input validation
- Performance monitoring and metrics
- Configurable chunking strategies including semantic chunking
- Structured logging throughout

Key classes:
- EnhancedRAGSystem: Main RAG pipeline with all improvements
- AsyncEmbeddingService: Async embedding generation with caching
- FAISSRetriever: FAISS-based retrieval with multiple index types
- HybridRetriever: Combines multiple retrieval strategies
"""

import os
import asyncio
import time
import sys
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import numpy as np

# Ensure src directory is in path for absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import core dependencies
try:
    # Try relative imports first (for module execution)
    from .config import get_config, get_openai_api_key
    from .logger import get_logger
    from .validation import validate_rag_query_params, InputValidator, RAGError, ErrorHandler
except ImportError:
    # Fall back to absolute imports (for direct execution)
    try:
        from config import get_config, get_openai_api_key
        from logger import get_logger
        from validation import validate_rag_query_params, InputValidator, RAGError, ErrorHandler
    except ImportError:
        def get_config(): return None
        def get_openai_api_key(): return None
        def get_logger(name): return None
        def validate_rag_query_params(*args): return args[0] if args else {}
        class InputValidator: pass
        class RAGError(Exception): pass
        class ErrorHandler: pass
try:
    from .cache import get_embedding_cache, get_query_cache
    from .chunking import get_chunker, Chunk
    from .vector_db import get_vector_db_manager
    from .async_utils import AsyncEmbeddingGenerator, AsyncTaskExecutor, get_async_executor
    from .metrics import monitor_performance, get_metrics_collector, record_metric
except ImportError:
    try:
        from cache import get_embedding_cache, get_query_cache
        from chunking import get_chunker, Chunk
        from vector_db import get_vector_db_manager
        from async_utils import AsyncEmbeddingGenerator, AsyncTaskExecutor, get_async_executor
        from metrics import monitor_performance, get_metrics_collector, record_metric
    except ImportError:
        def get_embedding_cache(): return None
        def get_query_cache(): return None
        def get_chunker(): return None
        Chunk = object
        def get_vector_db_manager(): return None
        class AsyncEmbeddingGenerator: pass
        class AsyncTaskExecutor: pass
        def get_async_executor(): return None
        def monitor_performance(): return lambda f: f
        def get_metrics_collector(): return None
        def record_metric(*args, **kwargs): pass
try:
    from .similarity_utils import cosine_similarity
except ImportError:
    try:
        from similarity_utils import cosine_similarity
    except ImportError:
        def cosine_similarity(x, y): return 0.0

logger = get_logger("rag_system_v2")


class AsyncEmbeddingService:
    """Async embedding service with caching and error handling."""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.api_key = InputValidator.validate_api_key(api_key, "OpenAI")
        self.model = InputValidator.validate_model_name(model)
        self.embedding_generator = AsyncEmbeddingGenerator(self.api_key, self.model)
        self.cache = get_embedding_cache()
        self.config = get_config()
        
        logger.info("Embedding service initialized", model=self.model)
    
    async def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for texts with caching and error handling.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            ValidationError: If texts are invalid
            RuntimeError: If embedding generation fails
        """
        if not texts:
            return []
        
        # Validate inputs
        for i, text in enumerate(texts):
            try:
                InputValidator.validate_query(text)
            except RAGError as e:
                logger.warning("Invalid text for embedding", index=i, error=str(e))
                raise RAGError(f"Invalid text at index {i}: {e.message}", "VALIDATION_INVALID_TYPE", "validation")
        
        with monitor_performance("embedding_generation", 
                                num_texts=len(texts), 
                                model=self.model):
            
            # Check cache for each text
            cached_embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(texts):
                cached = self.cache.get(text)
                if cached is not None:
                    cached_embeddings.append((i, cached))
                    record_metric("cache_hit", 0, metadata={"type": "embedding"})
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
                    record_metric("cache_miss", 0, metadata={"type": "embedding"})
            
            # Generate embeddings for uncached texts
            if uncached_texts:
                try:
                    new_embeddings_list = await self.embedding_generator.generate_embeddings(uncached_texts)
                    
                    # Convert to numpy arrays and cache
                    for text, embedding_list in zip(uncached_texts, new_embeddings_list):
                        embedding_array = np.array(embedding_list, dtype=np.float32)
                        self.cache.put(text, embedding_array)
                    
                    # Combine cached and new embeddings
                    all_embeddings = [None] * len(texts)
                    
                    # Place cached embeddings
                    for i, embedding in cached_embeddings:
                        all_embeddings[i] = embedding
                    
                    # Place new embeddings
                    for i, embedding_list in zip(uncached_indices, new_embeddings_list):
                        all_embeddings[i] = np.array(embedding_list, dtype=np.float32)
                    
                    return all_embeddings
                    
                except Exception as e:
                    error_info = ErrorHandler.handle_api_error(e, "embedding_generation", 
                                                             {"num_texts": len(uncached_texts)})
                    raise RuntimeError(f"Embedding generation failed: {error_info['message']}") from e
            
            else:
                # All embeddings were cached
                return [embedding for _, embedding in sorted(cached_embeddings)]
    
    async def get_single_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        embeddings = await self.get_embeddings([text])
        return embeddings[0]


class FAISSRetriever:
    """FAISS-based retriever with vector database storage."""
    
    def __init__(self, chunks: List[Chunk], embeddings: np.ndarray, index_type: str = "IndexFlatIP"):
        self.chunks = chunks
        self.config = get_config()
        self.db_manager = get_vector_db_manager()
        
        # Get embedding dimension
        if embeddings.size == 0:
            raise ValueError("No embeddings provided")
        
        dimension = embeddings.shape[1]
        
        # Initialize vector database
        self.vector_db = self.db_manager.get_db("faiss_retriever", dimension, index_type)
        
        # Build index if not already built or if data changed
        if not self.vector_db.is_trained or len(self.vector_db.chunks) != len(chunks):
            with monitor_performance("faiss_index_build", 
                                   num_chunks=len(chunks), 
                                   dimension=dimension,
                                   index_type=index_type):
                self.vector_db.build_index(chunks, embeddings)
                self.vector_db.save()
        
        logger.info("FAISS retriever initialized", 
                   num_chunks=len(chunks), 
                   dimension=dimension,
                   index_type=index_type)
    
    @classmethod
    def from_existing(cls, vector_db):
        """Create FAISSRetriever from existing vector database."""
        # Create instance without calling __init__
        instance = cls.__new__(cls)
        instance.chunks = vector_db.chunks
        instance.config = get_config()
        instance.db_manager = get_vector_db_manager()
        instance.vector_db = vector_db
        
        logger.info("FAISS retriever created from existing database", 
                   num_chunks=len(instance.chunks))
        return instance
    
    def retrieve(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """Retrieve similar chunks using FAISS.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of (chunk, similarity_score) tuples
        """
        with monitor_performance("faiss_retrieval", top_k=top_k):
            return self.vector_db.search(query_embedding, top_k)


class HybridRetriever:
    """Enhanced hybrid retriever combining multiple strategies."""
    
    def __init__(self, 
                 faiss_retriever: FAISSRetriever,
                 embedding_service: AsyncEmbeddingService,
                 alpha: float = 0.5):
        self.faiss_retriever = faiss_retriever
        self.embedding_service = embedding_service
        self.alpha = InputValidator.validate_float_range(alpha, "alpha", 0.0, 1.0)
        self.config = get_config()
        
        # Initialize TF-IDF for keyword retrieval
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=self.config.retrieval.tfidf_ngram_range,
            max_features=self.config.retrieval.tfidf_max_features,
            stop_words='english',
            sublinear_tf=True
        )
        
        # Build TF-IDF index
        corpus = [chunk.as_text() for chunk in self.faiss_retriever.chunks]
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
        
        logger.info("Hybrid retriever initialized", 
                   alpha=alpha,
                   num_chunks=len(self.faiss_retriever.chunks))
    
    async def retrieve(self, query: str, top_k: int = 5, alpha: Optional[float] = None) -> List[Tuple[Chunk, float]]:
        """Retrieve using hybrid approach.
        
        Args:
            query: Search query
            top_k: Number of results to return
            alpha: Override alpha for this query (0.0=keyword, 1.0=semantic)
            
        Returns:
            List of (chunk, similarity_score) tuples
        """
        with monitor_performance("hybrid_retrieval", query_length=len(query), top_k=top_k):
            
            # Get query embedding for semantic search
            query_embedding = await self.embedding_service.get_single_embedding(query)
            
            # Semantic retrieval
            semantic_results = self.faiss_retriever.retrieve(query_embedding, top_k * 2)
            
            # Keyword retrieval using TF-IDF
            query_vector = self.tfidf_vectorizer.transform([query])
            tfidf_similarities = cosine_similarity(query_vector, self.tfidf_matrix)

            # Ensure tfidf_similarities is a 1D dense numpy array
            tfidf_similarities = np.asarray(tfidf_similarities).ravel()

            # Get top TF-IDF results using argsort correctly
            tfidf_indices = np.argsort(tfidf_similarities)[::-1][:top_k * 2]
            keyword_results = [
                (self.faiss_retriever.chunks[i], float(tfidf_similarities[i]))
                for i in tfidf_indices
            ]
            
            # Combine results using weighted scoring with optional alpha override
            return self._combine_results(semantic_results, keyword_results, top_k, alpha)
    
    def _combine_results(self,
                        semantic_results: List[Tuple[Chunk, float]],
                        keyword_results: List[Tuple[Chunk, float]],
                        top_k: int,
                        alpha: Optional[float] = None) -> List[Tuple[Chunk, float]]:
        """Combine semantic and keyword retrieval results."""

        # Use provided alpha or fallback to instance alpha
        effective_alpha = alpha if alpha is not None else self.alpha

        # Create stable chunk key using doc_id and position
        def chunk_key(chunk: Chunk) -> tuple:
            return (chunk.doc_id, chunk.position)

        # Create score maps using stable identifiers
        semantic_scores = {chunk_key(chunk): score for chunk, score in semantic_results}
        keyword_scores = {chunk_key(chunk): score for chunk, score in keyword_results}
        
        # Normalize scores
        if semantic_scores:
            sem_values = list(semantic_scores.values())
            sem_min, sem_max = min(sem_values), max(sem_values)
            if sem_max > sem_min:
                semantic_scores = {
                    chunk_id: (score - sem_min) / (sem_max - sem_min)
                    for chunk_id, score in semantic_scores.items()
                }
        
        if keyword_scores:
            kw_values = list(keyword_scores.values())
            kw_min, kw_max = min(kw_values), max(kw_values)
            if kw_max > kw_min:
                keyword_scores = {
                    chunk_id: (score - kw_min) / (kw_max - kw_min)
                    for chunk_id, score in keyword_scores.items()
                }
        
        # Combine scores
        all_chunk_keys = set(semantic_scores.keys()) | set(keyword_scores.keys())
        combined_results = []

        # Create a map from chunk_key to chunk object for efficient lookup
        chunk_map = {}
        for chunk, _ in semantic_results + keyword_results:
            key = chunk_key(chunk)
            if key not in chunk_map:
                chunk_map[key] = chunk

        for key in all_chunk_keys:
            sem_score = semantic_scores.get(key, 0.0)
            kw_score = keyword_scores.get(key, 0.0)
            combined_score = effective_alpha * sem_score + (1 - effective_alpha) * kw_score

            # Find the chunk object using the map
            chunk = chunk_map.get(key)

            if chunk:
                combined_results.append((chunk, combined_score))
        
        # Sort by combined score and return top_k
        combined_results.sort(key=lambda x: x[1], reverse=True)
        return combined_results[:top_k]


class EnhancedRAGSystem:
    """Enhanced RAG system with all improvements."""
    
    def __init__(self, 
                 data_dir: str,
                 api_key: Optional[str] = None,
                 config_file: Optional[str] = None):
        
        # Load configuration
        if config_file:
            from .config import load_config
            config = load_config(config_file)
        else:
            config = get_config()
        
        # Validate API key
        if not api_key:
            api_key = get_openai_api_key()
        
        if not api_key:
            raise RAGError("OpenAI API key is required", "VALIDATION_API_KEY_INVALID", "validation")
        
        self.api_key = InputValidator.validate_api_key(api_key, "OpenAI")
        self.data_dir = InputValidator.validate_directory_path(data_dir)
        self.config = config
        self.query_cache = get_query_cache()
        self.metrics_collector = get_metrics_collector()
        
        # Initialize services
        self.embedding_service = AsyncEmbeddingService(self.api_key, config.embedding.model)
        self.chunker = get_chunker("auto")
        
        # Initialize components (will be set during build)
        self.chunks: List[Chunk] = []
        self.retriever: Optional[HybridRetriever] = None
        self.generation_model = config.generation.model
        
        logger.info("Enhanced RAG system initialized", 
                   data_dir=str(self.data_dir),
                   embedding_model=config.embedding.model,
                   generation_model=config.generation.model)
    
    async def build_index(self) -> None:
        """Build the retrieval index from documents in data directory."""
        
        with monitor_performance("index_building", data_dir=str(self.data_dir)):
            
            # First, try to load existing index
            try:
                from .vector_db import FAISSVectorDB
            except ImportError:
                try:
                    from vector_db import FAISSVectorDB  # type: ignore
                except ImportError:
                    import sys
                    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
                    from src.vector_db import FAISSVectorDB  # type: ignore

            # Try to load existing index (without specifying dimension)
            # The dimension will be read from metadata
            temp_vector_db = FAISSVectorDB(
                dimension=1,  # Placeholder, will be set from loaded metadata
                index_type=self.config.retrieval.faiss_index_type
            )

            if temp_vector_db.load():
                logger.info("Loaded existing FAISS index from disk", dimension=temp_vector_db.dimension)

                # Validate loaded index against current configuration
                needs_rebuild = False
                rebuild_reason = None

                if hasattr(temp_vector_db, 'loaded_metadata'):
                    loaded_meta = temp_vector_db.loaded_metadata

                    # Check if embedding model has changed
                    if loaded_meta.get('embedding_model') != self.config.embedding.model:
                        needs_rebuild = True
                        rebuild_reason = f"Embedding model mismatch: loaded={loaded_meta.get('embedding_model')}, current={self.config.embedding.model}"

                    # Check if corpus has changed (compare hash)
                    current_corpus_files = sorted([str(f.name) for f in self.data_dir.glob("*.md")])
                    if current_corpus_files:
                        import hashlib
                        current_corpus_hash = hashlib.md5(''.join(current_corpus_files).encode()).hexdigest()
                        loaded_corpus_hash = loaded_meta.get('corpus_hash')

                        if loaded_corpus_hash and loaded_corpus_hash != current_corpus_hash:
                            needs_rebuild = True
                            rebuild_reason = "Corpus files have changed since index was built"

                if needs_rebuild:
                    logger.warning("Index validation failed, rebuilding index", reason=rebuild_reason)
                    # Fall through to rebuild logic below
                else:
                    # Index is valid, use it
                    self.chunks = temp_vector_db.chunks

                    # Initialize FAISS retriever with loaded data
                    faiss_retriever = FAISSRetriever.from_existing(temp_vector_db)

                    # Initialize hybrid retriever
                    self.retriever = HybridRetriever(
                        faiss_retriever,
                        self.embedding_service,
                        self.config.retrieval.hybrid_alpha
                    )

                    logger.info("RAG system initialized from existing index",
                               num_chunks=len(self.chunks))
                    return

            # If no existing index, build from scratch
            logger.info("No existing index found, building from documents")

            # Load and process documents
            documents = self._load_documents()
            logger.info("Loaded documents", num_documents=len(documents))

            # Process documents into chunks
            all_chunks = []
            for doc_name, doc_content in documents.items():
                doc_path = self.data_dir / doc_name
                chunks = self.chunker.split(doc_content,
                                          doc_id=doc_name,
                                          source_path=str(doc_path))
                all_chunks.extend(chunks)

            self.chunks = all_chunks
            logger.info("Created chunks", num_chunks=len(self.chunks))

            # Generate embeddings for all chunks
            chunk_texts = [chunk.as_text() for chunk in self.chunks]
            embeddings = await self.embedding_service.get_embeddings(chunk_texts)
            embeddings_array = np.array(embeddings)

            # Derive dimension from actual embeddings
            actual_dimension = embeddings_array.shape[1]
            logger.info("Detected embedding dimension", dimension=actual_dimension)

            # Initialize FAISS retriever with actual dimension
            faiss_retriever = FAISSRetriever(
                self.chunks,
                embeddings_array,
                self.config.retrieval.faiss_index_type
            )

            # Save the index for future use with metadata for validation
            corpus_files = [str(f.name) for f in self.data_dir.glob("*.md")]
            faiss_retriever.vector_db.save(
                embedding_model=self.config.embedding.model,
                corpus_files=corpus_files
            )
            logger.info("FAISS index saved to disk for future use",
                       embedding_model=self.config.embedding.model,
                       num_corpus_files=len(corpus_files))

            # Initialize hybrid retriever
            self.retriever = HybridRetriever(
                faiss_retriever,
                self.embedding_service,
                self.config.retrieval.hybrid_alpha
            )

            logger.info("RAG index built and saved successfully")
    
    def _load_documents(self) -> Dict[str, str]:
        """Load markdown documents from data directory."""
        documents = {}
        
        for file_path in self.data_dir.glob("*.md"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    documents[file_path.name] = f.read()
                logger.debug("Loaded document", file=file_path.name, size=len(documents[file_path.name]))
            except Exception as e:
                error_info = ErrorHandler.handle_file_error(e, str(file_path), "read")
                logger.error("Failed to load document", file=file_path.name, error=error_info)
        
        if not documents:
            raise RuntimeError(f"No markdown documents found in {self.data_dir}")
        
        return documents
    
    async def query(self,
                   query: str,
                   top_k: int = 5,
                   use_cache: bool = True,
                   hybrid_alpha: Optional[float] = None) -> Dict[str, Any]:
        """Query the RAG system with intelligent retrieval and generation.

        This method performs hybrid retrieval (semantic + keyword search) followed by
        LLM generation to provide comprehensive answers based on your document collection.

        Args:
            query: The question or query string to process. Should be 3-5000 characters.
                  The system will validate and sanitize the input automatically.
            top_k: Number of most relevant document chunks to retrieve and use for
                  answer generation. Default is 5. Higher values may improve answer
                  quality but increase processing time.
            use_cache: Whether to check for cached results before processing.
                      Default is True. Disable for fresh results or testing.
            hybrid_alpha: Controls the balance between semantic and keyword search.
                         0.0 = pure keyword search, 1.0 = pure semantic search.
                         Default uses system configuration (typically 0.5 for balanced).

        Returns:
            Dictionary containing:
            - "query": The processed query string
            - "answer": The generated answer from the LLM
            - "chunks": List of retrieved document chunks with metadata
            - "metadata": Performance metrics and technical details

        Raises:
            RAGError: If query validation fails or parameters are invalid
            RuntimeError: If the RAG system hasn't been built or is in an invalid state

        Examples:
            Basic query:
            ```python
            result = await system.query("What is machine learning?")
            print(result["answer"])
            ```

            Advanced query with custom parameters:
            ```python
            result = await system.query(
                query="Explain neural networks",
                top_k=10,
                use_cache=False,
                hybrid_alpha=0.3  # More keyword-focused
            )
            ```

            Accessing retrieved context:
            ```python
            result = await system.query("How does RAG work?")
            for chunk in result["chunks"]:
                print(f"Source: {chunk['doc_id']}")
                print(f"Content: {chunk['content'][:100]}...")
            ```

            Performance monitoring:
            ```python
            result = await system.query("Complex question here")
            meta = result["metadata"]
            print(f"Retrieval time: {meta['retrieval_time_ms']:.1f}ms")
            print(f"Generation time: {meta['generation_time_ms']:.1f}ms")
            ```

        Note:
            - Results are automatically cached to improve performance on repeated queries
            - The system uses intelligent chunking and embedding for optimal retrieval
            - Performance metrics are included in the response for monitoring
            - Context chunks are limited to 500 characters in the response for readability
        """
        
        # Validate inputs
        params = validate_rag_query_params(query, top_k, "hybrid")
        query = params["query"]
        top_k = params["top_k"]
        
        # Validate hybrid_alpha if provided
        if hybrid_alpha is not None:
            hybrid_alpha = InputValidator.validate_float_range(hybrid_alpha, "hybrid_alpha", 0.0, 1.0)
        else:
            hybrid_alpha = self.config.retrieval.hybrid_alpha
        
        if not self.retriever:
            raise RuntimeError("RAG system not built. Call build_index() first.")
        
        # Check query cache
        cache_key_params = {
            "embedding_model": self.config.embedding.model,
            "generation_model": self.config.generation.model,
            "hybrid_alpha": hybrid_alpha
        }
        
        if use_cache:
            cached_result = self.query_cache.get(query, "hybrid", top_k, **cache_key_params)
            if cached_result:
                logger.debug("Query cache hit", query=query[:50])
                record_metric("cache_hit", 0, metadata={"type": "query"})
                return cached_result
            else:
                record_metric("cache_miss", 0, metadata={"type": "query"})
        
        with monitor_performance("rag_query", query_length=len(query), top_k=top_k):
            
            # Retrieve relevant chunks
            start_time = time.perf_counter()
            retrieved_chunks = await self.retriever.retrieve(query, top_k, alpha=hybrid_alpha)
            retrieval_time = (time.perf_counter() - start_time) * 1000
            
            # Generate answer
            start_time = time.perf_counter()
            answer, generation_metadata = await self._generate_answer(query, retrieved_chunks)
            generation_time = (time.perf_counter() - start_time) * 1000
            
            # Prepare result
            result = {
                "query": query,
                "answer": answer,
                "chunks": [
                    {
                        "doc_id": chunk.doc_id,
                        "title": chunk.title,
                        "content": chunk.content[:500] + "..." if len(chunk.content) > 500 else chunk.content,
                        "similarity_score": float(score),
                        "position": chunk.position,
                        "source_path": chunk.source_path
                    }
                    for chunk, score in retrieved_chunks
                ],
                "metadata": {
                    "retrieval_time_ms": retrieval_time,
                    "generation_time_ms": generation_time,
                    "total_time_ms": retrieval_time + generation_time,
                    "num_chunks_retrieved": len(retrieved_chunks),
                    "embedding_model": self.config.embedding.model,
                    "generation_model": self.config.generation.model,
                    **generation_metadata
                }
            }
            
            # Cache the result
            if use_cache:
                self.query_cache.put(query, "hybrid", top_k, result, **cache_key_params)
            
            return result
    
    async def _generate_answer(self, 
                             query: str, 
                             chunks: List[Tuple[Chunk, float]]) -> Tuple[str, Dict[str, Any]]:
        """Generate answer using LLM.
        
        Args:
            query: User query
            chunks: Retrieved chunks with scores
            
        Returns:
            Tuple of (answer, metadata)
        """
        
        # Prepare context
        context_parts = []
        for i, (chunk, score) in enumerate(chunks[:self.config.generation.max_context_chunks]):
            context_parts.append(
                f"[Document {i+1}: {chunk.doc_id} - {chunk.title}]\n{chunk.content}"
            )
        
        context = "\n\n".join(context_parts)
        
        # Prepare messages
        system_prompt = (
            "You are a helpful assistant that answers questions based on provided context. "
            "Use only the information from the context to answer questions. "
            "If the answer cannot be found in the context, say so clearly. "
            "Be concise and accurate in your responses."
        )
        
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        
        # Generate using OpenAI API with timeout and retries
        try:
            from openai import AsyncOpenAI
            import asyncio

            client = AsyncOpenAI(
                api_key=self.api_key,
                timeout=self.config.embedding.timeout,  # Use config timeout
                max_retries=2  # Basic retry on 5xx errors
            )

            # Wrap the API call with timeout
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=self.config.generation.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.config.generation.temperature,
                    top_p=self.config.generation.top_p,
                    max_tokens=self.config.generation.max_tokens
                ),
                timeout=self.config.embedding.timeout + 10  # Add buffer to config timeout
            )

            # Handle empty content safely
            answer = response.choices[0].message.content
            if answer is None:
                logger.warning("LLM returned empty content", finish_reason=response.choices[0].finish_reason)
                answer = "I couldn't generate an answer. Please try rephrasing your question."

            metadata = {
                "model_used": self.config.generation.model,
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
                "finish_reason": response.choices[0].finish_reason
            }

            return answer.strip(), metadata

        except asyncio.TimeoutError:
            logger.error("Answer generation timed out", timeout=self.config.embedding.timeout)
            raise RuntimeError(f"Answer generation timed out after {self.config.embedding.timeout}s")
        except Exception as e:
            # Mask API keys in error context
            error_context = {"query_prefix": query[:50], "context_length": len(context)}
            error_info = ErrorHandler.handle_api_error(e, "answer_generation", error_context)
            raise RuntimeError(f"Answer generation failed: {error_info['message']}") from e
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        stats = {
            "num_chunks": len(self.chunks),
            "data_directory": str(self.data_dir),
            "embedding_model": self.config.embedding.model,
            "generation_model": self.config.generation.model,
            "index_built": self.retriever is not None,
        }
        
        # Add vector database stats if available
        if self.retriever and hasattr(self.retriever, 'faiss_retriever'):
            vector_stats = self.retriever.faiss_retriever.vector_db.get_stats()
            stats.update({f"vector_db_{k}": v for k, v in vector_stats.items()})
        
        # Add metrics stats
        system_metrics = self.metrics_collector.get_system_metrics()
        stats.update({f"metrics_{k}": v for k, v in system_metrics.items()})
        
        return stats
    
    async def close(self) -> None:
        """Clean up resources."""
        logger.info("Closing RAG system")


# Factory function for easy initialization
async def create_rag_system(data_dir: str,
                           api_key: Optional[str] = None,
                           config_file: Optional[str] = None,
                           build_index: bool = True) -> EnhancedRAGSystem:
    """Create and configure a RAG system with optimal settings.

    This factory function provides the main entry point for creating RAG systems.
    It handles configuration loading, API key validation, and optional index building.

    Args:
        data_dir: Path to directory containing markdown documents.
                 Should contain .md files that will be indexed for retrieval.
        api_key: OpenAI API key. If None, will attempt to read from
                OPENAI_API_KEY environment variable.
        config_file: Path to JSON configuration file. If None, uses default config.
        build_index: Whether to immediately build the FAISS index.
                    Set to False if you want to defer index building.

    Returns:
        EnhancedRAGSystem: Configured RAG system ready for queries.

    Raises:
        RAGError: If configuration is invalid or API key is missing.
        RuntimeError: If data directory is empty or inaccessible.

    Examples:
        Basic usage:
        ```python
        system = await create_rag_system("data/")
        result = await system.query("What is machine learning?")
        ```

        Advanced configuration:
        ```python
        system = await create_rag_system(
            data_dir="my_docs/",
            api_key="sk-your-key-here",
            config_file="custom_config.json",
            build_index=True
        )
        ```

        Deferred index building:
        ```python
        system = await create_rag_system("data/", build_index=False)
        # ... do other setup ...
        await system.build_index()
        ```

    Note:
        The system automatically selects optimal FAISS index types based on
        your dataset size. For small datasets (<1K docs), it uses IndexFlatIP.
        For larger datasets, it uses IndexIVFFlat or IndexHNSW for better performance.
    """
    # Validate configuration on startup
    try:
        from .config import validate_startup_configuration
    except ImportError:
        from config import validate_startup_configuration
    config = validate_startup_configuration()

    system = EnhancedRAGSystem(data_dir, api_key, config_file)
    
    if build_index:
        await system.build_index()
    
    return system