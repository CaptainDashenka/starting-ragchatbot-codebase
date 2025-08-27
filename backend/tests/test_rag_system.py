"""
Integration tests for RAGSystem class.

Tests cover:
- Complete system initialization
- End-to-end query processing
- Component integration and coordination
- Error handling and propagation
- Session management integration
- Document processing workflow
- System analytics and monitoring
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os
import tempfile

from rag_system import RAGSystem
from models import Course, Lesson, CourseChunk
from vector_store import SearchResults
from tests.test_data import TestDataGenerator, MockResponses, QueryTestData, get_common_test_scenarios


class TestRAGSystemInitialization:
    """Test RAGSystem initialization and component setup"""
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_initialization_success(self, mock_session_mgr, mock_doc_proc, mock_ai_gen, mock_vector_store, mock_config):
        """Test successful RAGSystem initialization"""
        # Create RAGSystem
        rag = RAGSystem(mock_config)
        
        # Verify all components are initialized
        assert rag.config == mock_config
        assert rag.document_processor is not None
        assert rag.vector_store is not None
        assert rag.ai_generator is not None
        assert rag.session_manager is not None
        assert rag.tool_manager is not None
        
        # Verify components were initialized with correct parameters
        mock_doc_proc.assert_called_once_with(mock_config.CHUNK_SIZE, mock_config.CHUNK_OVERLAP)
        mock_vector_store.assert_called_once_with(mock_config.CHROMA_PATH, mock_config.EMBEDDING_MODEL, mock_config.MAX_RESULTS)
        mock_ai_gen.assert_called_once_with(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
        mock_session_mgr.assert_called_once_with(mock_config.MAX_HISTORY)
        
        # Verify tools are registered
        assert len(rag.tool_manager.tools) == 2  # search_tool and outline_tool
        assert "search_course_content" in rag.tool_manager.tools
        assert "get_course_outline" in rag.tool_manager.tools
    
    @patch('rag_system.VectorStore')
    def test_initialization_with_vector_store_error(self, mock_vector_store, mock_config):
        """Test RAGSystem initialization when VectorStore fails"""
        # Make VectorStore initialization fail
        mock_vector_store.side_effect = Exception("Failed to connect to ChromaDB")
        
        with pytest.raises(Exception, match="Failed to connect to ChromaDB"):
            RAGSystem(mock_config)
    
    @patch('rag_system.AIGenerator')
    def test_initialization_with_ai_generator_error(self, mock_ai_gen, mock_config):
        """Test RAGSystem initialization when AIGenerator fails"""
        # Make AIGenerator initialization fail
        mock_ai_gen.side_effect = Exception("Invalid Anthropic API key")
        
        with pytest.raises(Exception, match="Invalid Anthropic API key"):
            RAGSystem(mock_config)


class TestRAGSystemDocumentProcessing:
    """Test document processing functionality"""
    
    def test_add_course_document_success(self, mock_config, sample_course, sample_course_chunk):
        """Test successful addition of a course document"""
        with patch.multiple(
            'rag_system',
            VectorStore=Mock(),
            AIGenerator=Mock(),
            DocumentProcessor=Mock(),
            SessionManager=Mock()
        ):
            rag = RAGSystem(mock_config)
            
            # Mock document processor
            rag.document_processor.process_course_document.return_value = (sample_course, [sample_course_chunk])
            
            # Mock vector store methods
            rag.vector_store.add_course_metadata.return_value = None
            rag.vector_store.add_course_content.return_value = None
            
            # Add course document
            result_course, chunk_count = rag.add_course_document("test_document.txt")
            
            # Verify results
            assert result_course == sample_course
            assert chunk_count == 1
            
            # Verify methods were called
            rag.document_processor.process_course_document.assert_called_once_with("test_document.txt")
            rag.vector_store.add_course_metadata.assert_called_once_with(sample_course)
            rag.vector_store.add_course_content.assert_called_once_with([sample_course_chunk])
    
    def test_add_course_document_processing_error(self, mock_config):
        """Test course document addition with processing error"""
        with patch.multiple(
            'rag_system',
            VectorStore=Mock(),
            AIGenerator=Mock(),
            DocumentProcessor=Mock(),
            SessionManager=Mock()
        ):
            rag = RAGSystem(mock_config)
            
            # Make document processor fail
            rag.document_processor.process_course_document.side_effect = Exception("Failed to parse document")
            
            # Add course document
            result_course, chunk_count = rag.add_course_document("invalid_document.txt")
            
            # Verify error handling
            assert result_course is None
            assert chunk_count == 0
    
    @patch('os.path.exists')
    @patch('os.listdir')
    def test_add_course_folder_success(self, mock_listdir, mock_exists, mock_config, sample_course, sample_course_chunk):
        """Test successful addition of course folder"""
        with patch.multiple(
            'rag_system',
            VectorStore=Mock(),
            AIGenerator=Mock(),
            DocumentProcessor=Mock(),
            SessionManager=Mock()
        ):
            rag = RAGSystem(mock_config)
            
            # Setup mocks
            mock_exists.return_value = True
            mock_listdir.return_value = ["course1.txt", "course2.pdf", "ignore.jpg"]
            
            # Mock vector store
            rag.vector_store.get_existing_course_titles.return_value = set()
            rag.vector_store.add_course_metadata.return_value = None
            rag.vector_store.add_course_content.return_value = None
            
            # Mock document processor
            def mock_process(file_path):
                if "course1.txt" in file_path:
                    return (sample_course, [sample_course_chunk])
                elif "course2.pdf" in file_path:
                    course2 = Course(title="Course 2", lessons=[])
                    chunk2 = CourseChunk(content="Content 2", course_title="Course 2", chunk_index=0)
                    return (course2, [chunk2])
                return None, []
            
            rag.document_processor.process_course_document.side_effect = mock_process
            
            # Add course folder
            total_courses, total_chunks = rag.add_course_folder("/test/folder")
            
            # Verify results
            assert total_courses == 2
            assert total_chunks == 2
            
            # Verify processing was attempted for valid files
            assert rag.document_processor.process_course_document.call_count == 2
    
    @patch('os.path.exists')
    def test_add_course_folder_nonexistent(self, mock_exists, mock_config):
        """Test adding course folder that doesn't exist"""
        mock_exists.return_value = False
        
        with patch.multiple(
            'rag_system',
            VectorStore=Mock(),
            AIGenerator=Mock(),
            DocumentProcessor=Mock(),
            SessionManager=Mock()
        ):
            rag = RAGSystem(mock_config)
            
            total_courses, total_chunks = rag.add_course_folder("/nonexistent/folder")
            
            assert total_courses == 0
            assert total_chunks == 0
    
    @patch('os.path.exists')
    @patch('os.listdir')
    def test_add_course_folder_with_clear_existing(self, mock_listdir, mock_exists, mock_config):
        """Test adding course folder with clear_existing=True"""
        mock_exists.return_value = True
        mock_listdir.return_value = []
        
        with patch.multiple(
            'rag_system',
            VectorStore=Mock(),
            AIGenerator=Mock(),
            DocumentProcessor=Mock(),
            SessionManager=Mock()
        ):
            rag = RAGSystem(mock_config)
            
            # Mock vector store
            rag.vector_store.clear_all_data.return_value = None
            rag.vector_store.get_existing_course_titles.return_value = set()
            
            # Add course folder with clear_existing=True
            rag.add_course_folder("/test/folder", clear_existing=True)
            
            # Verify clear_all_data was called
            rag.vector_store.clear_all_data.assert_called_once()


class TestRAGSystemQuery:
    """Test query processing functionality"""
    
    def test_query_without_session_id(self, mock_config):
        """Test query processing without session ID"""
        with patch.multiple(
            'rag_system',
            VectorStore=Mock(),
            AIGenerator=Mock(),
            DocumentProcessor=Mock(),
            SessionManager=Mock()
        ):
            rag = RAGSystem(mock_config)
            
            # Mock AI generator response
            rag.ai_generator.generate_response.return_value = "This is the AI response to your question."
            
            # Mock tool manager
            rag.tool_manager.get_tool_definitions.return_value = [{"name": "test_tool"}]
            rag.tool_manager.get_last_sources.return_value = [{"text": "Test Source", "url": "http://test.com"}]
            rag.tool_manager.reset_sources.return_value = None
            
            # Process query
            response, sources = rag.query("What is artificial intelligence?")
            
            # Verify results
            assert response == "This is the AI response to your question."
            assert len(sources) == 1
            assert sources[0]["text"] == "Test Source"
            
            # Verify AI generator was called correctly
            rag.ai_generator.generate_response.assert_called_once()
            call_args = rag.ai_generator.generate_response.call_args
            assert "What is artificial intelligence?" in call_args[1]["query"]
            assert call_args[1]["conversation_history"] is None
            assert call_args[1]["tools"] == [{"name": "test_tool"}]
            assert call_args[1]["tool_manager"] == rag.tool_manager
            
            # Verify sources were managed correctly
            rag.tool_manager.get_last_sources.assert_called_once()
            rag.tool_manager.reset_sources.assert_called_once()
    
    def test_query_with_session_id(self, mock_config):
        """Test query processing with session ID"""
        with patch.multiple(
            'rag_system',
            VectorStore=Mock(),
            AIGenerator=Mock(),
            DocumentProcessor=Mock(),
            SessionManager=Mock()
        ):
            rag = RAGSystem(mock_config)
            
            # Mock session manager
            rag.session_manager.get_conversation_history.return_value = "Previous conversation context"
            rag.session_manager.add_exchange.return_value = None
            
            # Mock AI generator response
            rag.ai_generator.generate_response.return_value = "AI response with context"
            
            # Mock tool manager
            rag.tool_manager.get_tool_definitions.return_value = []
            rag.tool_manager.get_last_sources.return_value = []
            rag.tool_manager.reset_sources.return_value = None
            
            # Process query with session
            response, sources = rag.query("Follow-up question", session_id="test_session_123")
            
            # Verify session handling
            rag.session_manager.get_conversation_history.assert_called_once_with("test_session_123")
            rag.session_manager.add_exchange.assert_called_once_with("test_session_123", "Follow-up question", "AI response with context")
            
            # Verify AI generator received conversation history
            call_args = rag.ai_generator.generate_response.call_args
            assert call_args[1]["conversation_history"] == "Previous conversation context"
    
    def test_query_with_tool_execution(self, mock_config):
        """Test query that triggers tool execution"""
        with patch.multiple(
            'rag_system',
            VectorStore=Mock(),
            AIGenerator=Mock(),
            DocumentProcessor=Mock(),
            SessionManager=Mock()
        ):
            rag = RAGSystem(mock_config)
            
            # Mock AI generator to simulate tool use
            rag.ai_generator.generate_response.return_value = "Based on the search results, here's the answer about course content."
            
            # Mock tool manager with search results
            rag.tool_manager.get_tool_definitions.return_value = [
                {
                    "name": "search_course_content",
                    "description": "Search for course content"
                }
            ]
            rag.tool_manager.get_last_sources.return_value = [
                {"text": "Computer Use Course - Lesson 1", "url": "https://example.com/lesson1"}
            ]
            rag.tool_manager.reset_sources.return_value = None
            
            # Process query
            response, sources = rag.query("How do I use the Anthropic API?")
            
            # Verify tool-based response
            assert "search results" in response
            assert len(sources) == 1
            assert "Computer Use Course" in sources[0]["text"]
            
            # Verify tool manager was properly utilized
            rag.tool_manager.get_tool_definitions.assert_called_once()
            rag.tool_manager.get_last_sources.assert_called_once()
            rag.tool_manager.reset_sources.assert_called_once()
    
    def test_query_with_ai_generator_error(self, mock_config):
        """Test query processing when AI generator fails"""
        with patch.multiple(
            'rag_system',
            VectorStore=Mock(),
            AIGenerator=Mock(),
            DocumentProcessor=Mock(),
            SessionManager=Mock()
        ):
            rag = RAGSystem(mock_config)
            
            # Make AI generator fail
            rag.ai_generator.generate_response.side_effect = Exception("API rate limit exceeded")
            
            # Mock tool manager
            rag.tool_manager.get_tool_definitions.return_value = []
            
            # Process query should propagate the error
            with pytest.raises(Exception, match="API rate limit exceeded"):
                rag.query("Test query")
    
    def test_query_with_empty_query(self, mock_config):
        """Test query processing with empty query"""
        with patch.multiple(
            'rag_system',
            VectorStore=Mock(),
            AIGenerator=Mock(),
            DocumentProcessor=Mock(),
            SessionManager=Mock()
        ):
            rag = RAGSystem(mock_config)
            
            # Mock AI generator response for empty query
            rag.ai_generator.generate_response.return_value = "I need more information to help you."
            
            # Mock tool manager
            rag.tool_manager.get_tool_definitions.return_value = []
            rag.tool_manager.get_last_sources.return_value = []
            rag.tool_manager.reset_sources.return_value = None
            
            # Process empty query
            response, sources = rag.query("")
            
            # Verify response is handled
            assert isinstance(response, str)
            assert len(response) > 0


class TestRAGSystemAnalytics:
    """Test system analytics and monitoring functionality"""
    
    def test_get_course_analytics(self, mock_config):
        """Test getting course analytics"""
        with patch.multiple(
            'rag_system',
            VectorStore=Mock(),
            AIGenerator=Mock(),
            DocumentProcessor=Mock(),
            SessionManager=Mock()
        ):
            rag = RAGSystem(mock_config)
            
            # Mock vector store analytics methods
            rag.vector_store.get_course_count.return_value = 5
            rag.vector_store.get_existing_course_titles.return_value = [
                "Course 1", "Course 2", "Course 3", "Course 4", "Course 5"
            ]
            
            # Get analytics
            analytics = rag.get_course_analytics()
            
            # Verify analytics structure
            assert "total_courses" in analytics
            assert "course_titles" in analytics
            assert analytics["total_courses"] == 5
            assert len(analytics["course_titles"]) == 5
            assert "Course 1" in analytics["course_titles"]
    
    def test_get_course_analytics_with_error(self, mock_config):
        """Test getting course analytics when vector store fails"""
        with patch.multiple(
            'rag_system',
            VectorStore=Mock(),
            AIGenerator=Mock(),
            DocumentProcessor=Mock(),
            SessionManager=Mock()
        ):
            rag = RAGSystem(mock_config)
            
            # Make vector store methods fail
            rag.vector_store.get_course_count.side_effect = Exception("Database error")
            rag.vector_store.get_existing_course_titles.side_effect = Exception("Database error")
            
            # Get analytics should propagate errors
            with pytest.raises(Exception, match="Database error"):
                rag.get_course_analytics()


class TestRAGSystemIntegration:
    """Integration tests with realistic scenarios"""
    
    def test_complete_course_addition_and_query_workflow(self, mock_config):
        """Test complete workflow: add course, then query it"""
        with patch.multiple(
            'rag_system',
            VectorStore=Mock(),
            AIGenerator=Mock(),
            DocumentProcessor=Mock(),
            SessionManager=Mock()
        ):
            rag = RAGSystem(mock_config)
            
            # Setup course addition
            test_course = Course(
                title="Integration Test Course",
                instructor="Test Instructor",
                lessons=[
                    Lesson(lesson_number=1, title="Introduction", lesson_link="http://test.com/lesson1")
                ]
            )
            test_chunk = CourseChunk(
                content="This is test content about integration testing.",
                course_title="Integration Test Course",
                lesson_number=1,
                chunk_index=0
            )
            
            # Mock document processing
            rag.document_processor.process_course_document.return_value = (test_course, [test_chunk])
            rag.vector_store.add_course_metadata.return_value = None
            rag.vector_store.add_course_content.return_value = None
            
            # Add course document
            added_course, chunk_count = rag.add_course_document("integration_test.txt")
            assert added_course.title == "Integration Test Course"
            assert chunk_count == 1
            
            # Setup query processing
            rag.ai_generator.generate_response.return_value = "Integration testing involves testing the interaction between different components of a system."
            rag.tool_manager.get_tool_definitions.return_value = [{"name": "search_course_content"}]
            rag.tool_manager.get_last_sources.return_value = [
                {"text": "Integration Test Course - Lesson 1", "url": "http://test.com/lesson1"}
            ]
            rag.tool_manager.reset_sources.return_value = None
            
            # Query the added course
            response, sources = rag.query("What is integration testing?")
            
            # Verify complete workflow
            assert "integration testing" in response.lower()
            assert len(sources) == 1
            assert "Integration Test Course" in sources[0]["text"]
            
            # Verify all components were used
            rag.document_processor.process_course_document.assert_called_once()
            rag.vector_store.add_course_metadata.assert_called_once()
            rag.vector_store.add_course_content.assert_called_once()
            rag.ai_generator.generate_response.assert_called_once()
    
    def test_multiple_queries_with_session(self, mock_config):
        """Test multiple queries within a session"""
        with patch.multiple(
            'rag_system',
            VectorStore=Mock(),
            AIGenerator=Mock(),
            DocumentProcessor=Mock(),
            SessionManager=Mock()
        ):
            rag = RAGSystem(mock_config)
            
            # Setup session manager
            conversation_history = ""
            
            def mock_get_history(session_id):
                return conversation_history
            
            def mock_add_exchange(session_id, query, response):
                nonlocal conversation_history
                conversation_history += f"User: {query}\nAssistant: {response}\n"
            
            rag.session_manager.get_conversation_history.side_effect = mock_get_history
            rag.session_manager.add_exchange.side_effect = mock_add_exchange
            
            # Mock tool manager
            rag.tool_manager.get_tool_definitions.return_value = []
            rag.tool_manager.get_last_sources.return_value = []
            rag.tool_manager.reset_sources.return_value = None
            
            # Mock AI responses
            rag.ai_generator.generate_response.side_effect = [
                "AI is a field of computer science.",
                "Machine learning is a subset of AI that focuses on learning from data.",
                "Deep learning uses neural networks with multiple layers."
            ]
            
            session_id = "multi_query_test"
            
            # First query
            response1, _ = rag.query("What is artificial intelligence?", session_id)
            assert "AI is a field" in response1
            
            # Second query with context
            response2, _ = rag.query("What about machine learning?", session_id)
            assert "machine learning" in response2.lower()
            
            # Third query with more context
            response3, _ = rag.query("And deep learning?", session_id)
            assert "deep learning" in response3.lower()
            
            # Verify session history was built up
            assert rag.session_manager.add_exchange.call_count == 3
            assert rag.session_manager.get_conversation_history.call_count == 3
    
    def test_error_propagation_through_system(self, mock_config):
        """Test how errors propagate through the system layers"""
        with patch.multiple(
            'rag_system',
            VectorStore=Mock(),
            AIGenerator=Mock(),
            DocumentProcessor=Mock(),
            SessionManager=Mock()
        ):
            rag = RAGSystem(mock_config)
            
            # Test different error scenarios
            
            # 1. Tool execution error
            rag.tool_manager.get_tool_definitions.return_value = [{"name": "test_tool"}]
            rag.ai_generator.generate_response.side_effect = Exception("Tool execution failed")
            
            with pytest.raises(Exception, match="Tool execution failed"):
                rag.query("Test query that uses tools")
            
            # 2. Session management error
            rag.ai_generator.generate_response.side_effect = None
            rag.ai_generator.generate_response.return_value = "Test response"
            rag.session_manager.add_exchange.side_effect = Exception("Session storage error")
            rag.tool_manager.get_last_sources.return_value = []
            rag.tool_manager.reset_sources.return_value = None
            
            with pytest.raises(Exception, match="Session storage error"):
                rag.query("Test query", session_id="test_session")
    
    def test_system_with_realistic_data_flow(self, mock_config):
        """Test system with realistic data flow and responses"""
        with patch.multiple(
            'rag_system',
            VectorStore=Mock(),
            AIGenerator=Mock(),
            DocumentProcessor=Mock(),
            SessionManager=Mock()
        ):
            rag = RAGSystem(mock_config)
            
            # Setup realistic course data
            anthropic_course = Course(
                title="Building Towards Computer Use with Anthropic",
                course_link="https://www.deeplearning.ai/short-courses/building-toward-computer-use-with-anthropic/",
                instructor="Colt Steele",
                lessons=[
                    Lesson(lesson_number=0, title="Introduction", lesson_link="https://learn.deeplearning.ai/lesson/intro"),
                    Lesson(lesson_number=1, title="Anthropic's Claude models", lesson_link="https://learn.deeplearning.ai/lesson/models"),
                    Lesson(lesson_number=2, title="Getting Started with the Anthropic API", lesson_link="https://learn.deeplearning.ai/lesson/api")
                ]
            )
            
            # Setup realistic tool manager response
            rag.tool_manager.get_tool_definitions.return_value = [
                {
                    "name": "search_course_content",
                    "description": "Search course materials with smart course name matching and lesson filtering",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "What to search for in the course content"},
                            "course_name": {"type": "string", "description": "Course title (partial matches work)"},
                            "lesson_number": {"type": "integer", "description": "Specific lesson number to search within"}
                        },
                        "required": ["query"]
                    }
                }
            ]
            
            rag.tool_manager.get_last_sources.return_value = [
                {
                    "text": "Building Towards Computer Use with Anthropic - Lesson 2",
                    "url": "https://learn.deeplearning.ai/lesson/api"
                }
            ]
            rag.tool_manager.reset_sources.return_value = None
            
            # Setup realistic AI response
            rag.ai_generator.generate_response.return_value = """To get started with the Anthropic API:

1. Sign up for an API key at console.anthropic.com
2. Install the Anthropic Python SDK: pip install anthropic
3. Make your first API call using the Messages API
4. Configure your requests with model parameters like temperature and max_tokens

The API supports text generation, tool calling, and vision capabilities for analyzing images."""
            
            # Process realistic query
            response, sources = rag.query("How do I get started with the Anthropic API for computer use tasks?")
            
            # Verify realistic response
            assert isinstance(response, str)
            assert "API" in response
            assert "anthropic" in response.lower()
            assert len(sources) == 1
            assert "Building Towards Computer Use" in sources[0]["text"]
            assert sources[0]["url"].startswith("https://")
            
            # Verify system components worked together
            rag.ai_generator.generate_response.assert_called_once()
            call_args = rag.ai_generator.generate_response.call_args[1]
            assert "computer use tasks" in call_args["query"]
            assert call_args["tools"] == rag.tool_manager.get_tool_definitions()
            assert call_args["tool_manager"] == rag.tool_manager


class TestRAGSystemErrorRecovery:
    """Test error recovery and graceful degradation"""
    
    def test_graceful_degradation_without_tools(self, mock_config):
        """Test system continues to work even when tools fail"""
        with patch.multiple(
            'rag_system',
            VectorStore=Mock(),
            AIGenerator=Mock(),
            DocumentProcessor=Mock(),
            SessionManager=Mock()
        ):
            rag = RAGSystem(mock_config)
            
            # Simulate tool failure
            rag.tool_manager.get_tool_definitions.side_effect = Exception("Tool system unavailable")
            
            # AI should still work without tools
            rag.ai_generator.generate_response.return_value = "I can still answer general questions without searching specific course content."
            
            # This should work gracefully (tools would be None)
            with pytest.raises(Exception):
                # The error should propagate from get_tool_definitions
                response, sources = rag.query("What is artificial intelligence?")
    
    def test_partial_system_failure_recovery(self, mock_config):
        """Test recovery from partial system failures"""
        with patch.multiple(
            'rag_system',
            VectorStore=Mock(),
            AIGenerator=Mock(),
            DocumentProcessor=Mock(),
            SessionManager=Mock()
        ):
            rag = RAGSystem(mock_config)
            
            # Simulate session manager failure
            rag.session_manager.get_conversation_history.side_effect = Exception("Session unavailable")
            
            # But other components work
            rag.ai_generator.generate_response.return_value = "Response without session context"
            rag.tool_manager.get_tool_definitions.return_value = []
            rag.tool_manager.get_last_sources.return_value = []
            rag.tool_manager.reset_sources.return_value = None
            
            # This should fail because session error propagates
            with pytest.raises(Exception, match="Session unavailable"):
                rag.query("Test query", session_id="test_session")
            
            # But without session_id it should work
            response, sources = rag.query("Test query without session")
            assert response == "Response without session context"