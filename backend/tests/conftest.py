import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add the backend directory to Python path for importing modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import Course, Lesson, CourseChunk
from vector_store import SearchResults
from config import Config


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    config = Mock(spec=Config)
    config.ANTHROPIC_API_KEY = "test-api-key"
    config.ANTHROPIC_MODEL = "claude-3-sonnet-20240229"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.MAX_RESULTS = 5
    config.MAX_HISTORY = 2
    config.CHROMA_PATH = "./test_chroma_db"
    return config


@pytest.fixture
def sample_lesson():
    """Sample lesson for testing"""
    return Lesson(
        lesson_number=1,
        title="Introduction to Computer Use",
        lesson_link="https://example.com/lesson1"
    )


@pytest.fixture
def sample_course(sample_lesson):
    """Sample course for testing"""
    return Course(
        title="Building Towards Computer Use with Anthropic",
        course_link="https://example.com/course",
        instructor="Colt Steele",
        lessons=[sample_lesson]
    )


@pytest.fixture
def sample_course_chunk():
    """Sample course chunk for testing"""
    return CourseChunk(
        content="This is sample course content about computer use with large language models.",
        course_title="Building Towards Computer Use with Anthropic",
        lesson_number=1,
        chunk_index=0
    )


@pytest.fixture
def sample_search_results():
    """Sample search results for testing"""
    return SearchResults(
        documents=["This is sample course content about computer use."],
        metadata=[{
            "course_title": "Building Towards Computer Use with Anthropic",
            "lesson_number": 1,
            "chunk_index": 0
        }],
        distances=[0.1]
    )


@pytest.fixture
def empty_search_results():
    """Empty search results for testing"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[]
    )


@pytest.fixture
def error_search_results():
    """Error search results for testing"""
    return SearchResults.empty("Database connection error")


@pytest.fixture
def mock_chroma_client():
    """Mock ChromaDB client"""
    mock_client = Mock()
    mock_collection = Mock()
    
    # Configure mock collection behavior
    mock_collection.query.return_value = {
        'documents': [["This is sample course content"]],
        'metadatas': [[{"course_title": "Test Course", "lesson_number": 1}]],
        'distances': [[0.1]]
    }
    mock_collection.add.return_value = None
    mock_collection.get.return_value = {
        'ids': ["test_course"],
        'metadatas': [{"title": "Test Course", "instructor": "Test Instructor"}]
    }
    
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_client.delete_collection.return_value = None
    
    return mock_client


@pytest.fixture
def mock_sentence_transformer():
    """Mock sentence transformer for embeddings"""
    with patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction') as mock:
        mock_instance = Mock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic API client"""
    mock_client = Mock()
    
    # Mock response structure for non-tool use
    mock_response = Mock()
    mock_response.stop_reason = "end_turn"
    mock_response.content = [Mock(text="This is a test response")]
    mock_client.messages.create.return_value = mock_response
    
    return mock_client


@pytest.fixture
def mock_anthropic_tool_response():
    """Mock Anthropic API response with tool use"""
    mock_response = Mock()
    mock_response.stop_reason = "tool_use"
    
    # Mock tool use content block
    mock_tool_block = Mock()
    mock_tool_block.type = "tool_use"
    mock_tool_block.name = "search_course_content"
    mock_tool_block.id = "tool_123"
    mock_tool_block.input = {"query": "test query"}
    
    mock_response.content = [mock_tool_block]
    return mock_response


@pytest.fixture
def mock_anthropic_final_response():
    """Mock final Anthropic API response after tool use"""
    mock_response = Mock()
    mock_response.stop_reason = "end_turn"
    mock_response.content = [Mock(text="Based on the search results, here is the answer.")]
    return mock_response


@pytest.fixture
def temp_chroma_path():
    """Temporary directory for ChromaDB testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture(autouse=True)
def mock_sentence_transformers():
    """Auto-applied mock for sentence transformers to avoid downloading models in tests"""
    with patch('sentence_transformers.SentenceTransformer'):
        yield


# Test data fixtures
@pytest.fixture
def multiple_courses():
    """Multiple sample courses for testing"""
    course1 = Course(
        title="Course 1: Basic Concepts",
        course_link="https://example.com/course1",
        instructor="Instructor 1",
        lessons=[
            Lesson(lesson_number=1, title="Introduction", lesson_link="https://example.com/c1l1"),
            Lesson(lesson_number=2, title="Fundamentals", lesson_link="https://example.com/c1l2")
        ]
    )
    
    course2 = Course(
        title="Course 2: Advanced Topics",
        course_link="https://example.com/course2",
        instructor="Instructor 2",
        lessons=[
            Lesson(lesson_number=1, title="Advanced Introduction", lesson_link="https://example.com/c2l1")
        ]
    )
    
    return [course1, course2]


@pytest.fixture
def multiple_course_chunks():
    """Multiple course chunks for testing"""
    return [
        CourseChunk(
            content="Basic concepts introduction content",
            course_title="Course 1: Basic Concepts",
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="Fundamentals chapter content",
            course_title="Course 1: Basic Concepts",
            lesson_number=2,
            chunk_index=1
        ),
        CourseChunk(
            content="Advanced topics introduction",
            course_title="Course 2: Advanced Topics",
            lesson_number=1,
            chunk_index=0
        )
    ]


@pytest.fixture
def mock_tool_manager():
    """Mock tool manager for testing"""
    mock_manager = Mock()
    mock_manager.get_tool_definitions.return_value = [
        {
            "name": "search_course_content",
            "description": "Search course content",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    ]
    mock_manager.execute_tool.return_value = "Mock search results"
    mock_manager.get_last_sources.return_value = []
    mock_manager.reset_sources.return_value = None
    return mock_manager


# Session fixtures
@pytest.fixture
def mock_session_manager():
    """Mock session manager for testing"""
    mock_manager = Mock()
    mock_manager.create_session.return_value = "test_session_123"
    mock_manager.get_conversation_history.return_value = "Previous conversation context"
    mock_manager.add_exchange.return_value = None
    return mock_manager


# Error simulation fixtures
@pytest.fixture
def chroma_connection_error():
    """Fixture to simulate ChromaDB connection errors"""
    def _raise_error(*args, **kwargs):
        raise ConnectionError("Failed to connect to ChromaDB")
    return _raise_error


@pytest.fixture
def anthropic_api_error():
    """Fixture to simulate Anthropic API errors"""
    def _raise_error(*args, **kwargs):
        raise Exception("Anthropic API rate limit exceeded")
    return _raise_error