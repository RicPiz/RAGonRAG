"""
Pytest configuration and fixtures.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any, List
import asyncio

# Add src to path for testing
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.config import RAGConfig, set_config
from src.chunking import Chunk


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path)


@pytest.fixture
def test_config(temp_dir: Path) -> RAGConfig:
    """Create a test configuration."""
    config = RAGConfig()
    config.data_dir = str(temp_dir / "data")
    config.results_dir = str(temp_dir / "results")
    config.cache.cache_dir = str(temp_dir / "cache")
    config.cache.enable_disk_cache = True
    config.logging.log_to_file = False  # Don't create log files during tests
    
    # Create directories
    Path(config.data_dir).mkdir(parents=True, exist_ok=True)
    Path(config.results_dir).mkdir(parents=True, exist_ok=True)
    Path(config.cache.cache_dir).mkdir(parents=True, exist_ok=True)
    
    set_config(config)
    return config


@pytest.fixture
def sample_markdown_content() -> str:
    """Sample markdown content for testing."""
    return """# Introduction to RAG

Retrieval Augmented Generation (RAG) is a technique that combines retrieval and generation.

## Key Components

RAG systems typically consist of three main components:

1. Document store
2. Retrieval system  
3. Generation model

### Document Store

The document store contains the knowledge base that the system will search through.

### Retrieval System

The retrieval system finds relevant documents based on the query.

## Benefits

RAG provides several benefits:

- Up-to-date information
- Reduced hallucinations
- Cite sources
"""


@pytest.fixture
def sample_chunks() -> List[Chunk]:
    """Sample chunks for testing."""
    return [
        Chunk(
            doc_id="test_doc.md",
            title="Introduction to RAG",
            content="Retrieval Augmented Generation (RAG) is a technique that combines retrieval and generation.",
            source_path="/test/test_doc.md",
            position=0,
            start_char=0,
            end_char=100
        ),
        Chunk(
            doc_id="test_doc.md",
            title="Key Components",
            content="RAG systems typically consist of three main components: Document store, Retrieval system, Generation model.",
            source_path="/test/test_doc.md",
            position=1,
            start_char=100,
            end_char=200
        ),
        Chunk(
            doc_id="test_doc.md",
            title="Benefits",
            content="RAG provides several benefits: Up-to-date information, Reduced hallucinations, Cite sources.",
            source_path="/test/test_doc.md",
            position=2,
            start_char=200,
            end_char=300
        )
    ]


@pytest.fixture
def mock_openai_response() -> Dict[str, Any]:
    """Mock OpenAI API response."""
    return {
        "data": [
            {"embedding": [0.1, 0.2, 0.3, 0.4, 0.5]},
            {"embedding": [0.5, 0.4, 0.3, 0.2, 0.1]},
            {"embedding": [0.3, 0.3, 0.3, 0.3, 0.3]}
        ]
    }


@pytest.fixture
def mock_chat_response() -> Dict[str, Any]:
    """Mock OpenAI chat completion response."""
    return {
        "choices": [
            {
                "message": {
                    "content": "Based on the provided context, RAG is a technique that combines retrieval and generation."
                }
            }
        ]
    }


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def create_test_files(temp_dir: Path, sample_markdown_content: str):
    """Create test markdown files."""
    data_dir = temp_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Create test markdown files
    files = {
        "Module_1.md": sample_markdown_content,
        "Module_2.md": sample_markdown_content.replace("RAG", "Vector Search"),
        "Module_3.md": "# Simple Document\n\nThis is a simple test document."
    }
    
    for filename, content in files.items():
        (data_dir / filename).write_text(content, encoding='utf-8')
    
    return data_dir


@pytest.fixture
def create_test_qa_files(temp_dir: Path):
    """Create test Q&A files."""
    questions_dir = temp_dir / "questions"
    answers_dir = temp_dir / "answers"
    
    questions_dir.mkdir(exist_ok=True)
    answers_dir.mkdir(exist_ok=True)
    
    # Sample questions
    questions_content = """1. What is RAG?
2. What are the key components of RAG?
3. What are the benefits of using RAG?"""
    
    # Sample answers
    answers_content = """1. RAG is Retrieval Augmented Generation, a technique that combines retrieval and generation.
2. The key components are document store, retrieval system, and generation model.
3. Benefits include up-to-date information, reduced hallucinations, and ability to cite sources."""
    
    (questions_dir / "Module_1_Questions.md").write_text(questions_content)
    (answers_dir / "Module_1_Answers.md").write_text(answers_content)
    
    return questions_dir, answers_dir