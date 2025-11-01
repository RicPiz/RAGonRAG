<div align="center">
  <img src="RAGonRAG_Logo.png" alt="RAGonRAG Logo" width="400"/>
</div>

# RAGonRAG - Production-Ready RAG System

A Retrieval-Augmented Generation (RAG) system with advanced caching, hybrid retrieval, and comprehensive monitoring capabilities. The knowledge base is derived from transcripts of lectures from the DeepLearning.AI course on RAG.

## Features

- **Hybrid Retrieval System**: Combines semantic search (FAISS) with keyword-based retrieval (TF-IDF + BM25) for optimal results
- **Advanced Caching**: Multi-level caching (memory + disk) with versioning and true LRU eviction
- **Async Processing**: Fully asynchronous pipeline for maximum throughput
- **Smart Chunking**: Multiple strategies including semantic chunking for context-aware document splitting
- **Performance Monitoring**: Real-time metrics tracking with alerting system
- **Web Interface**: User-friendly Streamlit interface with performance dashboard
- **Evaluation Framework**: Built-in evaluation system with support for custom metrics

## Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

```bash
# Clone the repository
git clone https://github.com/ricpiz/RAGonRAG.git
cd RAGonRAG

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Run Web Interface

```bash
streamlit run src/app.py
```

Access the interface at [http://localhost:8501](http://localhost:8501)


## Architecture

### Core Components

- **RAG System** ([src/rag_system.py](src/rag_system.py)): Main orchestration layer
- **Vector Database** ([src/vector_db.py](src/vector_db.py)): FAISS-based vector storage with JSON persistence
- **Chunking** ([src/chunking.py](src/chunking.py)): Smart document splitting with semantic awareness
- **Cache** ([src/cache.py](src/cache.py)): Multi-level caching with versioning
- **Validation** ([src/validation.py](src/validation.py)): Input sanitization and error handling

### Hybrid Retrieval

The system combines two retrieval approaches:

1. **Semantic Search**: Uses OpenAI embeddings + FAISS for meaning-based retrieval
2. **Keyword Search**: TF-IDF + BM25 for exact term matching

Results are combined using a configurable alpha parameter (default: 0.5).

### Caching Strategy

Three-level caching system:

- **Embedding Cache**: Stores computed embeddings to avoid redundant API calls
- **Query Cache**: Caches complete query results for instant responses
- **Chunk Cache**: Caches processed document chunks

All caches support both memory (LRU) and disk persistence with automatic invalidation.

## Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=your_api_key_here

# Optional - see config.py for all options
RAG_EMBEDDING_MODEL=text-embedding-3-small
RAG_GENERATION_MODEL=gpt-4o
RAG_CACHE_ENABLE_DISK_CACHE=true
RAG_LOGGING_LEVEL=INFO
```

### Configuration File

Create `config.json` to customize behavior:

```json
{
  "embedding": {
    "model": "text-embedding-3-small",
    "batch_size": 100,
    "max_retries": 3
  },
  "generation": {
    "model": "gpt-4o",
    "temperature": 0.3,
    "max_tokens": 512
  },
  "chunking": {
    "max_chunk_size": 1500,
    "chunk_overlap": 300,
    "use_semantic_chunking": true
  },
  "retrieval": {
    "hybrid_alpha": 0.5,
    "faiss_index_type": "IndexFlatIP"
  },
  "cache": {
    "enable_disk_cache": true,
    "cache_ttl": 3600
  }
}
```

## Performance

Typical performance metrics (on standard hardware):

- **Query latency**: 500-1500ms (first query), <100ms (cached)
- **Indexing throughput**: ~1000 chunks/minute
- **Cache hit rate**: >80% for repeated queries
- **Memory usage**: ~200MB base + 1KB per chunk

## Security Features

- **API Key Masking**: Sensitive data is masked in all logs
- **Input Validation**: All user inputs are sanitized
- **Path Traversal Protection**: File paths are validated
- **Rate Limiting**: Built-in retry logic with exponential backoff
- **Error Hiding**: Production mode hides stack traces (set `DEBUG=false`)

## Advanced Features

### Index Validation

The system automatically validates loaded indices against:
- Embedding model changes
- Corpus file changes (via MD5 hash)
- Schema version compatibility

If validation fails, the index is automatically rebuilt.

## License

MIT License - see LICENSE file for details.