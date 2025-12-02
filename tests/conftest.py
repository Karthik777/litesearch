"""Pytest fixtures for litesearch tests."""
import pytest

@pytest.fixture
def mem_db():
    """In-memory database without semantic search (avoids usearch dependency)."""
    # Import here to avoid import side-effects during collection
    from litesearch.core import setup_db
    return setup_db(':memory:', sem_search=False)

@pytest.fixture
def sample_content():
    """Sample content for testing."""
    return [
        {"content": "Machine learning is a subset of AI", "metadata": "{}"},
        {"content": "Python is a programming language", "metadata": "{}"},
        {"content": "SQLite is an embedded database", "metadata": "{}"},
    ]
