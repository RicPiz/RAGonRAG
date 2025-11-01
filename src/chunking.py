"""
Advanced chunking strategies including semantic chunking and overlap management.
"""

import re
import nltk
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import numpy as np

# Import core dependencies
try:
    from .config import get_config
    from .logger import get_logger
    from .cache import get_chunk_cache
    from .similarity_utils import cosine_similarity
except ImportError:
    from config import get_config
    from logger import get_logger
    from cache import get_chunk_cache
    from similarity_utils import cosine_similarity

logger = get_logger("chunking")

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except Exception as e:
        # Degrade gracefully if download fails (offline mode)
        logger.warning(f"Failed to download NLTK punkt tokenizer: {e}. Will use regex fallback.")

# Fallback regex-based sentence splitter
def _simple_sentence_split(text: str) -> List[str]:
    """Simple regex-based sentence splitter as fallback."""
    import re
    # Split on common sentence endings
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


@dataclass
class Chunk:
    """Enhanced chunk representation with metadata and semantic information."""
    doc_id: str
    title: str
    content: str
    source_path: str
    position: int
    start_char: int
    end_char: int
    semantic_similarity: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def as_text(self) -> str:
        """Return the text used for indexing: title + content."""
        return f"{self.title}\n{self.content}".strip()

    def get_length(self) -> int:
        """Get the character length of the chunk."""
        return len(self.as_text())


class BaseChunker:
    """Base class for all chunking strategies."""
    
    def __init__(self):
        self.config = get_config().chunking
        self.cache = get_chunk_cache()
    
    def split(self, text: str, *, doc_id: str, source_path: str) -> List[Chunk]:
        """Split text into chunks. To be implemented by subclasses."""
        raise NotImplementedError
    
    def _create_cache_key(self) -> Dict[str, Any]:
        """Create configuration key for caching."""
        return {
            'chunker_type': self.__class__.__name__,
            'max_chunk_size': self.config.max_chunk_size,
            'chunk_overlap': self.config.chunk_overlap,
            'min_chunk_size': self.config.min_chunk_size
        }


class SmartChunker(BaseChunker):
    """Unified chunker that combines title-based and semantic chunking strategies."""

    def __init__(self, use_semantic: bool = True, similarity_threshold: float = 0.6):
        super().__init__()
        self.use_semantic = use_semantic
        self.similarity_threshold = similarity_threshold
        self.heading_re = re.compile(r"^(#+)\s+(.*)")

        # Initialize semantic model only if needed
        if self.use_semantic:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Initialized smart chunker with semantic chunking", threshold=similarity_threshold)
        else:
            logger.info("Initialized smart chunker with title-based chunking")

    def split(self, text: str, *, doc_id: str, source_path: str) -> List[Chunk]:
        """Split text using hybrid title + semantic approach."""
        cache_key = self._create_cache_key()
        cache_key['use_semantic'] = self.use_semantic
        cache_key['similarity_threshold'] = self.similarity_threshold

        # Try cache first
        cached_chunks = self.cache.get(doc_id, cache_key)
        if cached_chunks:
            logger.debug("Using cached chunks", doc_id=doc_id)
            return cached_chunks

        logger.debug("Smart chunking document", doc_id=doc_id, semantic=self.use_semantic)

        if self.use_semantic:
            return self._hybrid_chunking(text, doc_id, source_path, cache_key)
        else:
            return self._title_chunking(text, doc_id, source_path, cache_key)

    def _title_chunking(self, text: str, doc_id: str, source_path: str, cache_key: dict) -> List[Chunk]:
        """Title-based chunking with overlap."""
        lines = text.splitlines()
        chunks: List[Chunk] = []
        current_title = doc_id
        buffer: List[str] = []
        position = 0
        start_char = 0

        for line_idx, line in enumerate(lines):
            m = self.heading_re.match(line)
            if m:
                # Flush previous chunk
                if buffer:
                    chunk_content = "\n".join(buffer).strip()
                    if len(chunk_content) >= self.config.min_chunk_size:
                        chunks.append(
                            Chunk(
                                doc_id=doc_id,
                                title=current_title.strip(),
                                content=chunk_content,
                                source_path=source_path,
                                position=position,
                                start_char=start_char,
                                end_char=start_char + len(chunk_content),
                                metadata={'line_start': start_char, 'line_end': line_idx}
                            )
                        )
                        position += 1

                    # Handle overlap
                    if self.config.chunk_overlap > 0 and buffer:
                        overlap_text = "\n".join(buffer[-3:])  # Last 3 lines for overlap
                        if len(overlap_text) < self.config.chunk_overlap:
                            start_char = max(0, start_char + len(chunk_content) - len(overlap_text))
                        else:
                            start_char = start_char + len(chunk_content) - self.config.chunk_overlap
                    else:
                        start_char = start_char + len(chunk_content) + 1

                    buffer = []

                # Start new chunk
                current_title = m.group(2)
            else:
                buffer.append(line)

        # Flush remaining content
        if buffer or not chunks:
            chunk_content = "\n".join(buffer).strip()
            if len(chunk_content) >= self.config.min_chunk_size:
                chunks.append(
                    Chunk(
                        doc_id=doc_id,
                        title=current_title.strip(),
                        content=chunk_content,
                        source_path=source_path,
                        position=position,
                        start_char=start_char,
                        end_char=start_char + len(chunk_content),
                        metadata={'line_start': start_char, 'line_end': len(lines)}
                    )
                )

        # Post-process chunks to ensure size limits
        chunks = self._split_oversized_chunks(chunks, doc_id, source_path)

        # Cache the results
        self.cache.put(doc_id, cache_key, chunks)

        logger.info("Title chunking completed", doc_id=doc_id, num_chunks=len(chunks))
        return chunks

    def _hybrid_chunking(self, text: str, doc_id: str, source_path: str, cache_key: dict) -> List[Chunk]:
        """Hybrid title + semantic chunking."""
        # First, do title-based chunking to get logical sections
        title_chunks = self._get_title_chunks(text, doc_id, source_path)

        # Apply semantic chunking to large chunks
        final_chunks = []
        for chunk in title_chunks:
            if chunk.get_length() > self.config.max_chunk_size:
                # Use semantic chunking for this chunk
                semantic_chunks = self._semantic_split(chunk.content, doc_id, source_path, chunk.position)
                # Preserve original title
                for sc in semantic_chunks:
                    sc.title = chunk.title
                    sc.position = len(final_chunks)
                    final_chunks.append(sc)
            else:
                final_chunks.append(chunk)

        # Cache the results
        self.cache.put(doc_id, cache_key, final_chunks)

        logger.info("Hybrid chunking completed", doc_id=doc_id, num_chunks=len(final_chunks))
        return final_chunks

    def _get_title_chunks(self, text: str, doc_id: str, source_path: str) -> List[Chunk]:
        """Get title-based chunks without caching."""
        lines = text.splitlines()
        chunks: List[Chunk] = []
        current_title = doc_id
        buffer: List[str] = []
        position = 0
        start_char = 0

        for line_idx, line in enumerate(lines):
            m = self.heading_re.match(line)
            if m:
                # Flush previous chunk
                if buffer:
                    chunk_content = "\n".join(buffer).strip()
                    if len(chunk_content) >= self.config.min_chunk_size:
                        chunks.append(
                            Chunk(
                                doc_id=doc_id,
                                title=current_title.strip(),
                                content=chunk_content,
                                source_path=source_path,
                                position=position,
                                start_char=start_char,
                                end_char=start_char + len(chunk_content),
                                metadata={'line_start': start_char, 'line_end': line_idx}
                            )
                        )
                        position += 1
                    buffer = []
                # Start new chunk
                current_title = m.group(2)
                start_char = line_idx
            else:
                buffer.append(line)

        # Flush remaining content
        if buffer or not chunks:
            chunk_content = "\n".join(buffer).strip()
            if len(chunk_content) >= self.config.min_chunk_size:
                chunks.append(
                    Chunk(
                        doc_id=doc_id,
                        title=current_title.strip(),
                        content=chunk_content,
                        source_path=source_path,
                        position=position,
                        start_char=start_char,
                        end_char=start_char + len(chunk_content),
                        metadata={'line_start': start_char, 'line_end': len(lines)}
                    )
                )

        return chunks

    def _semantic_split(self, text: str, doc_id: str, source_path: str, base_position: int) -> List[Chunk]:
        """Split text using semantic similarity."""
        try:
            sentences = nltk.sent_tokenize(text)
        except LookupError:
            # Fallback to simple regex split if NLTK data unavailable
            logger.warning("NLTK tokenizer unavailable, using regex fallback")
            sentences = _simple_sentence_split(text)
        if len(sentences) <= 1:
            return [Chunk(
                doc_id=doc_id,
                title=f"{doc_id}_chunk_{base_position}",
                content=text,
                source_path=source_path,
                position=base_position,
                start_char=0,
                end_char=len(text)
            )]

        # Get sentence embeddings
        embeddings = self.model.encode(sentences)

        # Group semantically similar sentences
        chunks = []
        current_chunk_sentences = [sentences[0]]
        current_chunk_start = 0

        for i in range(1, len(sentences)):
            # Calculate similarity with current chunk's last sentence
            prev_embedding = embeddings[i-1].reshape(1, -1)
            curr_embedding = embeddings[i].reshape(1, -1)
            similarity = cosine_similarity(prev_embedding, curr_embedding)

            # Check chunk size constraints
            current_chunk_text = " ".join(current_chunk_sentences)
            would_exceed_size = len(current_chunk_text) + len(sentences[i]) > self.config.max_chunk_size

            if similarity >= self.similarity_threshold and not would_exceed_size:
                current_chunk_sentences.append(sentences[i])
            else:
                # Create chunk from current sentences
                chunk_content = " ".join(current_chunk_sentences)
                if len(chunk_content) >= self.config.min_chunk_size:
                    avg_similarity = self._calculate_average_similarity(
                        embeddings[current_chunk_start:i]
                    )
                    chunks.append(
                        Chunk(
                            doc_id=doc_id,
                            title=f"{doc_id}_chunk_{base_position + len(chunks)}",
                            content=chunk_content,
                            source_path=source_path,
                            position=base_position + len(chunks),
                            start_char=0,
                            end_char=len(chunk_content),
                            semantic_similarity=avg_similarity,
                            metadata={'sentence_range': (current_chunk_start, i-1)}
                        )
                    )

                # Start new chunk
                current_chunk_sentences = [sentences[i]]
                current_chunk_start = i

        # Handle last chunk
        if current_chunk_sentences:
            chunk_content = " ".join(current_chunk_sentences)
            if len(chunk_content) >= self.config.min_chunk_size:
                avg_similarity = self._calculate_average_similarity(
                    embeddings[current_chunk_start:]
                )
                chunks.append(
                    Chunk(
                        doc_id=doc_id,
                        title=f"{doc_id}_chunk_{base_position + len(chunks)}",
                        content=chunk_content,
                        source_path=source_path,
                        position=base_position + len(chunks),
                        start_char=0,
                        end_char=len(chunk_content),
                        semantic_similarity=avg_similarity,
                        metadata={'sentence_range': (current_chunk_start, len(sentences)-1)}
                    )
                )

        return chunks

    def _calculate_average_similarity(self, embeddings: np.ndarray) -> float:
        """Calculate average pairwise similarity within a chunk."""
        if len(embeddings) <= 1:
            return 1.0

        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = cosine_similarity(
                    embeddings[i],
                    embeddings[j]
                )
                similarities.append(sim)

        return float(np.mean(similarities)) if similarities else 1.0

    def _split_oversized_chunks(self, chunks: List[Chunk], doc_id: str, source_path: str) -> List[Chunk]:
        """Split chunks that exceed maximum size."""
        result_chunks = []

        for chunk in chunks:
            if chunk.get_length() <= self.config.max_chunk_size:
                result_chunks.append(chunk)
            else:
                # Split by sentences
                sentences = nltk.sent_tokenize(chunk.content)
                sub_chunks = []
                current_text = []
                current_length = len(chunk.title) + 1  # +1 for newline

                for sentence in sentences:
                    if current_length + len(sentence) <= self.config.max_chunk_size:
                        current_text.append(sentence)
                        current_length += len(sentence)
                    else:
                        if current_text:
                            sub_content = " ".join(current_text)
                            sub_chunks.append(
                                Chunk(
                                    doc_id=doc_id,
                                    title=chunk.title,
                                    content=sub_content,
                                    source_path=source_path,
                                    position=len(result_chunks) + len(sub_chunks),
                                    start_char=chunk.start_char,
                                    end_char=chunk.start_char + len(sub_content),
                                    metadata=chunk.metadata.copy()
                                )
                            )
                        current_text = [sentence]
                        current_length = len(chunk.title) + 1 + len(sentence)

                if current_text:
                    sub_content = " ".join(current_text)
                    sub_chunks.append(
                        Chunk(
                            doc_id=doc_id,
                            title=chunk.title,
                            content=sub_content,
                            source_path=source_path,
                            position=len(result_chunks) + len(sub_chunks),
                            start_char=chunk.start_char,
                            end_char=chunk.start_char + len(sub_content),
                            metadata=chunk.metadata.copy()
                        )
                    )

                result_chunks.extend(sub_chunks)

        return result_chunks


# Legacy class names for backward compatibility
TitleChunker = SmartChunker
SemanticChunker = SmartChunker
HybridChunker = SmartChunker


def get_chunker(chunker_type: str = "auto") -> BaseChunker:
    """Factory function to get appropriate chunker."""
    config = get_config().chunking

    if chunker_type == "auto":
        # Use smart chunker with configuration-based behavior
        return SmartChunker(use_semantic=config.use_semantic_chunking)
    elif chunker_type == "title":
        return SmartChunker(use_semantic=False)
    elif chunker_type == "semantic":
        return SmartChunker(use_semantic=True)
    elif chunker_type == "hybrid":
        return SmartChunker(use_semantic=True)
    else:
        raise ValueError(f"Unknown chunker type: {chunker_type}. Use 'auto', 'title', 'semantic', or 'hybrid'.")