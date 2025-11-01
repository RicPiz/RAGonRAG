"""
Tests for text chunking functionality.
"""

import pytest
from unittest.mock import Mock, patch

from src.chunking import TitleChunker, SemanticChunker, HybridChunker, get_chunker, Chunk


class TestTitleChunker:
    """Test title-based chunking."""
    
    def test_title_chunker_basic(self, test_config):
        """Test basic title chunking."""
        chunker = TitleChunker()
        text = """# First Section

This is the first section content.
It has multiple lines.

## Subsection

This is a subsection.

# Second Section

This is the second section.
"""
        
        chunks = chunker.split(text, doc_id="test.md", source_path="/test/test.md")
        
        assert len(chunks) == 3
        assert chunks[0].title == "First Section"
        assert "first section content" in chunks[0].content.lower()
        assert chunks[1].title == "Subsection"
        assert chunks[2].title == "Second Section"
    
    def test_title_chunker_no_headers(self, test_config):
        """Test chunking text without headers."""
        chunker = TitleChunker()
        text = "This is just plain text without any headers."
        
        chunks = chunker.split(text, doc_id="test.md", source_path="/test/test.md")
        
        assert len(chunks) == 1
        assert chunks[0].title == "test.md"
        assert chunks[0].content == text
    
    def test_title_chunker_empty_content(self, test_config):
        """Test chunking with empty content sections."""
        chunker = TitleChunker()
        text = """# First Section

# Second Section

Some content here.
"""
        
        chunks = chunker.split(text, doc_id="test.md", source_path="/test/test.md")
        
        # Should only create chunks with sufficient content
        assert len(chunks) == 1
        assert chunks[0].title == "Second Section"
    
    def test_title_chunker_oversized_chunks(self, test_config):
        """Test splitting oversized chunks."""
        # Set a small max chunk size for testing
        test_config.chunking.max_chunk_size = 100
        
        chunker = TitleChunker()
        long_content = "This is a very long sentence. " * 20  # Should exceed 100 chars
        text = f"# Long Section\n\n{long_content}"
        
        chunks = chunker.split(text, doc_id="test.md", source_path="/test/test.md")
        
        # Should be split into multiple chunks
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.as_text()) <= test_config.chunking.max_chunk_size
    
    def test_chunk_metadata(self, test_config):
        """Test chunk metadata creation."""
        chunker = TitleChunker()
        text = "# Test Section\n\nTest content."
        
        chunks = chunker.split(text, doc_id="test.md", source_path="/test/test.md")
        
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.doc_id == "test.md"
        assert chunk.source_path == "/test/test.md"
        assert chunk.position == 0
        assert chunk.start_char >= 0
        assert chunk.end_char > chunk.start_char
        assert isinstance(chunk.metadata, dict)


class TestSemanticChunker:
    """Test semantic chunking."""
    
    @patch('src.chunking.SentenceTransformer')
    def test_semantic_chunker_basic(self, mock_transformer, test_config):
        """Test basic semantic chunking."""
        # Mock the sentence transformer
        mock_model = Mock()
        mock_transformer.return_value = mock_model
        
        # Mock embeddings - similar sentences should have higher similarity
        mock_embeddings = [
            [0.1, 0.2, 0.3],  # First sentence
            [0.15, 0.25, 0.35],  # Similar to first (high similarity)
            [0.8, 0.9, 0.7],  # Very different (low similarity)
        ]
        mock_model.encode.return_value = mock_embeddings
        
        chunker = SemanticChunker(similarity_threshold=0.7)
        text = "First sentence about cats. Second sentence also about cats. This sentence is about dogs."
        
        chunks = chunker.split(text, doc_id="test.md", source_path="/test/test.md")
        
        # Should create chunks based on semantic similarity
        assert len(chunks) >= 1
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert all(chunk.semantic_similarity >= 0 for chunk in chunks)
    
    @patch('src.chunking.SentenceTransformer')
    def test_semantic_chunker_single_sentence(self, mock_transformer, test_config):
        """Test semantic chunking with single sentence."""
        mock_model = Mock()
        mock_transformer.return_value = mock_model
        
        chunker = SemanticChunker()
        text = "Single sentence."
        
        chunks = chunker.split(text, doc_id="test.md", source_path="/test/test.md")
        
        assert len(chunks) == 1
        assert chunks[0].content == text


class TestHybridChunker:
    """Test hybrid chunking strategy."""
    
    @patch('src.chunking.SentenceTransformer')
    def test_hybrid_chunker(self, mock_transformer, test_config):
        """Test hybrid chunking approach."""
        # Set small chunk size to force semantic chunking on oversized chunks
        test_config.chunking.max_chunk_size = 50
        
        mock_model = Mock()
        mock_transformer.return_value = mock_model
        mock_model.encode.return_value = [[0.1, 0.2], [0.8, 0.9]]
        
        chunker = HybridChunker()
        text = """# Small Section

Short content.

# Large Section

This is a very long section with lots of content that should exceed the maximum chunk size and trigger semantic chunking.
"""
        
        chunks = chunker.split(text, doc_id="test.md", source_path="/test/test.md")
        
        # Should have at least 2 chunks (one small, one or more from large section)
        assert len(chunks) >= 2
        
        # First chunk should be from title chunking
        assert chunks[0].title == "Small Section"
        
        # Large section should be split semantically
        large_section_chunks = [c for c in chunks if c.title == "Large Section"]
        assert len(large_section_chunks) >= 1


class TestChunkerFactory:
    """Test chunker factory function."""
    
    def test_get_chunker_auto(self, test_config):
        """Test automatic chunker selection."""
        test_config.chunking.use_semantic_chunking = True
        chunker = get_chunker("auto")
        assert isinstance(chunker, HybridChunker)
        
        test_config.chunking.use_semantic_chunking = False
        chunker = get_chunker("auto")
        assert isinstance(chunker, TitleChunker)
    
    def test_get_chunker_explicit_types(self, test_config):
        """Test explicit chunker type selection."""
        assert isinstance(get_chunker("title"), TitleChunker)
        assert isinstance(get_chunker("hybrid"), HybridChunker)
    
    def test_get_chunker_invalid_type(self, test_config):
        """Test invalid chunker type."""
        with pytest.raises(ValueError, match="Unknown chunker type"):
            get_chunker("invalid_type")


class TestChunkClass:
    """Test Chunk data class."""
    
    def test_chunk_as_text(self):
        """Test chunk text representation."""
        chunk = Chunk(
            doc_id="test.md",
            title="Test Title",
            content="Test content here.",
            source_path="/test/test.md",
            position=0,
            start_char=0,
            end_char=20
        )
        
        expected = "Test Title\nTest content here."
        assert chunk.as_text() == expected
    
    def test_chunk_get_length(self):
        """Test chunk length calculation."""
        chunk = Chunk(
            doc_id="test.md",
            title="Test",
            content="Content",
            source_path="/test/test.md",
            position=0,
            start_char=0,
            end_char=10
        )
        
        # Length should be title + newline + content
        expected_length = len("Test\nContent")
        assert chunk.get_length() == expected_length
    
    def test_chunk_metadata_initialization(self):
        """Test chunk metadata initialization."""
        # Without metadata
        chunk1 = Chunk(
            doc_id="test.md",
            title="Test",
            content="Content",
            source_path="/test/test.md",
            position=0,
            start_char=0,
            end_char=10
        )
        assert chunk1.metadata == {}
        
        # With metadata
        metadata = {"custom_field": "value"}
        chunk2 = Chunk(
            doc_id="test.md",
            title="Test",
            content="Content",
            source_path="/test/test.md",
            position=0,
            start_char=0,
            end_char=10,
            metadata=metadata
        )
        assert chunk2.metadata == metadata