"""
Unit tests for VectorStore class.

Tests cover:
- Initialization and configuration
- Search functionality and error handling
- Course name resolution
- Filter building logic
- Data addition and retrieval
- Error scenarios that could cause "query failed"
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from pathlib import Path

from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk
from tests.test_data import TestDataGenerator, MockResponses, ErrorScenarios


class TestVectorStore:
    """Test cases for VectorStore class"""
    
    def test_initialization_with_valid_config(self, mock_config, mock_chroma_client, mock_sentence_transformer):
        """Test VectorStore initialization with valid configuration"""
        with patch('chromadb.PersistentClient', return_value=mock_chroma_client):
            store = VectorStore(
                chroma_path=mock_config.CHROMA_PATH,
                embedding_model=mock_config.EMBEDDING_MODEL,
                max_results=mock_config.MAX_RESULTS
            )
            
            assert store.max_results == mock_config.MAX_RESULTS
            assert store.client == mock_chroma_client
            mock_chroma_client.get_or_create_collection.assert_called()
    
    def test_initialization_with_invalid_path(self):
        """Test VectorStore initialization with invalid ChromaDB path"""
        with patch('chromadb.PersistentClient') as mock_client:
            mock_client.side_effect = Exception("Invalid path")
            
            with pytest.raises(Exception):
                VectorStore(
                    chroma_path="/invalid/path",
                    embedding_model="all-MiniLM-L6-v2",
                    max_results=5
                )
    
    def test_search_with_valid_query(self, mock_config, mock_chroma_client, mock_sentence_transformer):
        """Test search functionality with valid query"""
        # Setup mock response
        mock_chroma_client.get_or_create_collection.return_value.query.return_value = MockResponses.successful_search_response()
        
        with patch('chromadb.PersistentClient', return_value=mock_chroma_client):
            store = VectorStore(
                chroma_path=mock_config.CHROMA_PATH,
                embedding_model=mock_config.EMBEDDING_MODEL,
                max_results=mock_config.MAX_RESULTS
            )
            
            results = store.search("How to use Anthropic API?")
            
            assert isinstance(results, SearchResults)
            assert not results.is_empty()
            assert len(results.documents) == 2
            assert results.error is None
    
    def test_search_with_course_name_filter(self, mock_config, mock_chroma_client, mock_sentence_transformer):
        """Test search with course name filtering"""
        # Setup course catalog response for course name resolution
        catalog_collection = Mock()
        catalog_collection.query.return_value = MockResponses.course_catalog_response()
        
        # Setup content collection response
        content_collection = Mock()
        content_collection.query.return_value = MockResponses.successful_search_response()
        
        # Configure mock to return different collections
        def mock_get_collection(name):
            if name == "course_catalog":
                return catalog_collection
            elif name == "course_content":
                return content_collection
            return Mock()
        
        mock_chroma_client.get_or_create_collection.side_effect = mock_get_collection
        
        with patch('chromadb.PersistentClient', return_value=mock_chroma_client):
            store = VectorStore(
                chroma_path=mock_config.CHROMA_PATH,
                embedding_model=mock_config.EMBEDDING_MODEL,
                max_results=mock_config.MAX_RESULTS
            )
            
            results = store.search(
                query="API usage",
                course_name="Computer Use"
            )
            
            assert isinstance(results, SearchResults)
            # Verify course name resolution was attempted
            catalog_collection.query.assert_called_once()
            # Verify content search was performed with filter
            content_collection.query.assert_called_once()
    
    def test_search_with_nonexistent_course(self, mock_config, mock_chroma_client, mock_sentence_transformer):
        """Test search with non-existent course name"""
        # Setup empty course catalog response
        catalog_collection = Mock()
        catalog_collection.query.return_value = MockResponses.empty_search_response()
        
        mock_chroma_client.get_or_create_collection.return_value = catalog_collection
        
        with patch('chromadb.PersistentClient', return_value=mock_chroma_client):
            store = VectorStore(
                chroma_path=mock_config.CHROMA_PATH,
                embedding_model=mock_config.EMBEDDING_MODEL,
                max_results=mock_config.MAX_RESULTS
            )
            
            results = store.search(
                query="API usage",
                course_name="Nonexistent Course"
            )
            
            assert isinstance(results, SearchResults)
            assert results.error is not None
            assert "No course found matching" in results.error
    
    def test_search_with_lesson_number_filter(self, mock_config, mock_chroma_client, mock_sentence_transformer):
        """Test search with lesson number filtering"""
        mock_chroma_client.get_or_create_collection.return_value.query.return_value = MockResponses.successful_search_response()
        
        with patch('chromadb.PersistentClient', return_value=mock_chroma_client):
            store = VectorStore(
                chroma_path=mock_config.CHROMA_PATH,
                embedding_model=mock_config.EMBEDDING_MODEL,
                max_results=mock_config.MAX_RESULTS
            )
            
            results = store.search(
                query="API usage",
                lesson_number=1
            )
            
            assert isinstance(results, SearchResults)
            # Verify that query was called with lesson number filter
            store.course_content.query.assert_called_with(
                query_texts=["API usage"],
                n_results=5,
                where={"lesson_number": 1}
            )
    
    def test_search_with_combined_filters(self, mock_config, mock_chroma_client, mock_sentence_transformer):
        """Test search with both course name and lesson number filters"""
        # Setup course resolution
        catalog_collection = Mock()
        catalog_collection.query.return_value = MockResponses.course_catalog_response()
        
        content_collection = Mock()
        content_collection.query.return_value = MockResponses.successful_search_response()
        
        def mock_get_collection(name):
            if name == "course_catalog":
                return catalog_collection
            elif name == "course_content":
                return content_collection
            return Mock()
        
        mock_chroma_client.get_or_create_collection.side_effect = mock_get_collection
        
        with patch('chromadb.PersistentClient', return_value=mock_chroma_client):
            store = VectorStore(
                chroma_path=mock_config.CHROMA_PATH,
                embedding_model=mock_config.EMBEDDING_MODEL,
                max_results=mock_config.MAX_RESULTS
            )
            
            results = store.search(
                query="API usage",
                course_name="Computer Use",
                lesson_number=1
            )
            
            assert isinstance(results, SearchResults)
            # Verify combined filter was used
            content_collection.query.assert_called_with(
                query_texts=["API usage"],
                n_results=5,
                where={
                    "$and": [
                        {"course_title": "Building Towards Computer Use with Anthropic"},
                        {"lesson_number": 1}
                    ]
                }
            )
    
    def test_search_with_empty_results(self, mock_config, mock_chroma_client, mock_sentence_transformer):
        """Test search that returns no results"""
        mock_chroma_client.get_or_create_collection.return_value.query.return_value = MockResponses.empty_search_response()
        
        with patch('chromadb.PersistentClient', return_value=mock_chroma_client):
            store = VectorStore(
                chroma_path=mock_config.CHROMA_PATH,
                embedding_model=mock_config.EMBEDDING_MODEL,
                max_results=mock_config.MAX_RESULTS
            )
            
            results = store.search("nonexistent query")
            
            assert isinstance(results, SearchResults)
            assert results.is_empty()
            assert results.error is None
    
    def test_search_with_chroma_error(self, mock_config, mock_chroma_client, mock_sentence_transformer):
        """Test search when ChromaDB throws an error"""
        mock_chroma_client.get_or_create_collection.return_value.query.side_effect = ConnectionError("Database connection failed")
        
        with patch('chromadb.PersistentClient', return_value=mock_chroma_client):
            store = VectorStore(
                chroma_path=mock_config.CHROMA_PATH,
                embedding_model=mock_config.EMBEDDING_MODEL,
                max_results=mock_config.MAX_RESULTS
            )
            
            results = store.search("test query")
            
            assert isinstance(results, SearchResults)
            assert results.error is not None
            assert "Search error:" in results.error
            assert "Database connection failed" in results.error
    
    def test_build_filter_no_parameters(self, mock_config, mock_chroma_client, mock_sentence_transformer):
        """Test filter building with no parameters"""
        with patch('chromadb.PersistentClient', return_value=mock_chroma_client):
            store = VectorStore(
                chroma_path=mock_config.CHROMA_PATH,
                embedding_model=mock_config.EMBEDDING_MODEL,
                max_results=mock_config.MAX_RESULTS
            )
            
            filter_dict = store._build_filter(None, None)
            assert filter_dict is None
    
    def test_build_filter_course_only(self, mock_config, mock_chroma_client, mock_sentence_transformer):
        """Test filter building with course title only"""
        with patch('chromadb.PersistentClient', return_value=mock_chroma_client):
            store = VectorStore(
                chroma_path=mock_config.CHROMA_PATH,
                embedding_model=mock_config.EMBEDDING_MODEL,
                max_results=mock_config.MAX_RESULTS
            )
            
            filter_dict = store._build_filter("Test Course", None)
            assert filter_dict == {"course_title": "Test Course"}
    
    def test_build_filter_lesson_only(self, mock_config, mock_chroma_client, mock_sentence_transformer):
        """Test filter building with lesson number only"""
        with patch('chromadb.PersistentClient', return_value=mock_chroma_client):
            store = VectorStore(
                chroma_path=mock_config.CHROMA_PATH,
                embedding_model=mock_config.EMBEDDING_MODEL,
                max_results=mock_config.MAX_RESULTS
            )
            
            filter_dict = store._build_filter(None, 1)
            assert filter_dict == {"lesson_number": 1}
    
    def test_build_filter_both_parameters(self, mock_config, mock_chroma_client, mock_sentence_transformer):
        """Test filter building with both course and lesson"""
        with patch('chromadb.PersistentClient', return_value=mock_chroma_client):
            store = VectorStore(
                chroma_path=mock_config.CHROMA_PATH,
                embedding_model=mock_config.EMBEDDING_MODEL,
                max_results=mock_config.MAX_RESULTS
            )
            
            filter_dict = store._build_filter("Test Course", 1)
            expected = {
                "$and": [
                    {"course_title": "Test Course"},
                    {"lesson_number": 1}
                ]
            }
            assert filter_dict == expected
    
    def test_resolve_course_name_success(self, mock_config, mock_chroma_client, mock_sentence_transformer):
        """Test successful course name resolution"""
        mock_chroma_client.get_or_create_collection.return_value.query.return_value = MockResponses.course_catalog_response()
        
        with patch('chromadb.PersistentClient', return_value=mock_chroma_client):
            store = VectorStore(
                chroma_path=mock_config.CHROMA_PATH,
                embedding_model=mock_config.EMBEDDING_MODEL,
                max_results=mock_config.MAX_RESULTS
            )
            
            resolved_title = store._resolve_course_name("Computer Use")
            assert resolved_title == "Building Towards Computer Use with Anthropic"
    
    def test_resolve_course_name_failure(self, mock_config, mock_chroma_client, mock_sentence_transformer):
        """Test course name resolution failure"""
        mock_chroma_client.get_or_create_collection.return_value.query.return_value = MockResponses.empty_search_response()
        
        with patch('chromadb.PersistentClient', return_value=mock_chroma_client):
            store = VectorStore(
                chroma_path=mock_config.CHROMA_PATH,
                embedding_model=mock_config.EMBEDDING_MODEL,
                max_results=mock_config.MAX_RESULTS
            )
            
            resolved_title = store._resolve_course_name("Nonexistent Course")
            assert resolved_title is None
    
    def test_resolve_course_name_with_error(self, mock_config, mock_chroma_client, mock_sentence_transformer):
        """Test course name resolution when ChromaDB throws error"""
        mock_chroma_client.get_or_create_collection.return_value.query.side_effect = Exception("Database error")
        
        with patch('chromadb.PersistentClient', return_value=mock_chroma_client):
            store = VectorStore(
                chroma_path=mock_config.CHROMA_PATH,
                embedding_model=mock_config.EMBEDDING_MODEL,
                max_results=mock_config.MAX_RESULTS
            )
            
            resolved_title = store._resolve_course_name("Computer Use")
            assert resolved_title is None
    
    def test_add_course_metadata(self, mock_config, mock_chroma_client, mock_sentence_transformer, sample_course):
        """Test adding course metadata to vector store"""
        with patch('chromadb.PersistentClient', return_value=mock_chroma_client):
            store = VectorStore(
                chroma_path=mock_config.CHROMA_PATH,
                embedding_model=mock_config.EMBEDDING_MODEL,
                max_results=mock_config.MAX_RESULTS
            )
            
            store.add_course_metadata(sample_course)
            
            # Verify the course catalog collection add method was called
            store.course_catalog.add.assert_called_once()
            
            # Check the call arguments
            call_args = store.course_catalog.add.call_args
            assert call_args[1]['documents'] == [sample_course.title]
            assert call_args[1]['ids'] == [sample_course.title]
            assert 'metadatas' in call_args[1]
    
    def test_add_course_content(self, mock_config, mock_chroma_client, mock_sentence_transformer, multiple_course_chunks):
        """Test adding course content chunks to vector store"""
        with patch('chromadb.PersistentClient', return_value=mock_chroma_client):
            store = VectorStore(
                chroma_path=mock_config.CHROMA_PATH,
                embedding_model=mock_config.EMBEDDING_MODEL,
                max_results=mock_config.MAX_RESULTS
            )
            
            store.add_course_content(multiple_course_chunks)
            
            # Verify the course content collection add method was called
            store.course_content.add.assert_called_once()
            
            # Check the call arguments
            call_args = store.course_content.add.call_args
            assert len(call_args[1]['documents']) == len(multiple_course_chunks)
            assert len(call_args[1]['metadatas']) == len(multiple_course_chunks)
            assert len(call_args[1]['ids']) == len(multiple_course_chunks)
    
    def test_add_empty_course_content(self, mock_config, mock_chroma_client, mock_sentence_transformer):
        """Test adding empty course content list"""
        with patch('chromadb.PersistentClient', return_value=mock_chroma_client):
            store = VectorStore(
                chroma_path=mock_config.CHROMA_PATH,
                embedding_model=mock_config.EMBEDDING_MODEL,
                max_results=mock_config.MAX_RESULTS
            )
            
            store.add_course_content([])
            
            # Verify the course content collection add method was NOT called
            store.course_content.add.assert_not_called()
    
    def test_get_existing_course_titles(self, mock_config, mock_chroma_client, mock_sentence_transformer):
        """Test retrieving existing course titles"""
        mock_chroma_client.get_or_create_collection.return_value.get.return_value = {
            'ids': ['Course 1', 'Course 2', 'Course 3']
        }
        
        with patch('chromadb.PersistentClient', return_value=mock_chroma_client):
            store = VectorStore(
                chroma_path=mock_config.CHROMA_PATH,
                embedding_model=mock_config.EMBEDDING_MODEL,
                max_results=mock_config.MAX_RESULTS
            )
            
            titles = store.get_existing_course_titles()
            assert titles == ['Course 1', 'Course 2', 'Course 3']
    
    def test_get_existing_course_titles_with_error(self, mock_config, mock_chroma_client, mock_sentence_transformer):
        """Test retrieving course titles when error occurs"""
        mock_chroma_client.get_or_create_collection.return_value.get.side_effect = Exception("Database error")
        
        with patch('chromadb.PersistentClient', return_value=mock_chroma_client):
            store = VectorStore(
                chroma_path=mock_config.CHROMA_PATH,
                embedding_model=mock_config.EMBEDDING_MODEL,
                max_results=mock_config.MAX_RESULTS
            )
            
            titles = store.get_existing_course_titles()
            assert titles == []
    
    def test_get_course_count(self, mock_config, mock_chroma_client, mock_sentence_transformer):
        """Test getting course count"""
        mock_chroma_client.get_or_create_collection.return_value.get.return_value = {
            'ids': ['Course 1', 'Course 2']
        }
        
        with patch('chromadb.PersistentClient', return_value=mock_chroma_client):
            store = VectorStore(
                chroma_path=mock_config.CHROMA_PATH,
                embedding_model=mock_config.EMBEDDING_MODEL,
                max_results=mock_config.MAX_RESULTS
            )
            
            count = store.get_course_count()
            assert count == 2
    
    def test_clear_all_data(self, mock_config, mock_chroma_client, mock_sentence_transformer):
        """Test clearing all data from collections"""
        with patch('chromadb.PersistentClient', return_value=mock_chroma_client):
            store = VectorStore(
                chroma_path=mock_config.CHROMA_PATH,
                embedding_model=mock_config.EMBEDDING_MODEL,
                max_results=mock_config.MAX_RESULTS
            )
            
            store.clear_all_data()
            
            # Verify collections were deleted
            assert mock_chroma_client.delete_collection.call_count == 2
            delete_calls = [call[0][0] for call in mock_chroma_client.delete_collection.call_args_list]
            assert "course_catalog" in delete_calls
            assert "course_content" in delete_calls


class TestSearchResults:
    """Test cases for SearchResults class"""
    
    def test_from_chroma_with_valid_data(self):
        """Test creating SearchResults from ChromaDB response"""
        chroma_response = MockResponses.successful_search_response()
        results = SearchResults.from_chroma(chroma_response)
        
        assert len(results.documents) == 2
        assert len(results.metadata) == 2
        assert len(results.distances) == 2
        assert results.error is None
    
    def test_from_chroma_with_empty_data(self):
        """Test creating SearchResults from empty ChromaDB response"""
        chroma_response = MockResponses.empty_search_response()
        results = SearchResults.from_chroma(chroma_response)
        
        assert len(results.documents) == 0
        assert len(results.metadata) == 0
        assert len(results.distances) == 0
        assert results.error is None
    
    def test_empty_with_error_message(self):
        """Test creating empty SearchResults with error message"""
        error_msg = "Database connection failed"
        results = SearchResults.empty(error_msg)
        
        assert results.is_empty()
        assert results.error == error_msg
    
    def test_is_empty_true(self):
        """Test is_empty returns True for empty results"""
        results = SearchResults(documents=[], metadata=[], distances=[])
        assert results.is_empty()
    
    def test_is_empty_false(self):
        """Test is_empty returns False for non-empty results"""
        results = SearchResults(
            documents=["test doc"],
            metadata=[{"test": "meta"}],
            distances=[0.1]
        )
        assert not results.is_empty()


# Integration-style tests that test multiple components together
class TestVectorStoreIntegration:
    """Integration tests for VectorStore with real-like scenarios"""
    
    def test_complete_search_workflow(self, mock_config, mock_chroma_client, mock_sentence_transformer):
        """Test complete search workflow from query to results"""
        # Setup realistic mock responses
        catalog_collection = Mock()
        catalog_collection.query.return_value = MockResponses.course_catalog_response()
        
        content_collection = Mock()
        content_collection.query.return_value = MockResponses.successful_search_response()
        
        def mock_get_collection(name):
            if name == "course_catalog":
                return catalog_collection
            elif name == "course_content":
                return content_collection
            return Mock()
        
        mock_chroma_client.get_or_create_collection.side_effect = mock_get_collection
        
        with patch('chromadb.PersistentClient', return_value=mock_chroma_client):
            store = VectorStore(
                chroma_path=mock_config.CHROMA_PATH,
                embedding_model=mock_config.EMBEDDING_MODEL,
                max_results=mock_config.MAX_RESULTS
            )
            
            # Test the complete workflow
            results = store.search(
                query="How to use the Anthropic API?",
                course_name="Computer Use",
                lesson_number=1
            )
            
            # Verify the workflow completed successfully
            assert isinstance(results, SearchResults)
            assert not results.is_empty()
            assert results.error is None
            
            # Verify both course resolution and content search occurred
            catalog_collection.query.assert_called_once()
            content_collection.query.assert_called_once()
    
    def test_error_recovery_scenarios(self, mock_config, mock_chroma_client, mock_sentence_transformer):
        """Test various error scenarios and recovery"""
        # Test with different types of errors
        error_scenarios = [
            ConnectionError("Database connection failed"),
            ValueError("Invalid query parameters"),
            RuntimeError("ChromaDB service unavailable")
        ]
        
        for error in error_scenarios:
            mock_chroma_client.reset_mock()
            mock_chroma_client.get_or_create_collection.return_value.query.side_effect = error
            
            with patch('chromadb.PersistentClient', return_value=mock_chroma_client):
                store = VectorStore(
                    chroma_path=mock_config.CHROMA_PATH,
                    embedding_model=mock_config.EMBEDDING_MODEL,
                    max_results=mock_config.MAX_RESULTS
                )
                
                results = store.search("test query")
                
                # Verify error is properly handled
                assert isinstance(results, SearchResults)
                assert results.error is not None
                assert "Search error:" in results.error
                assert str(error) in results.error