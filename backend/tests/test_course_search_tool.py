"""
Unit tests for CourseSearchTool and ToolManager classes.

Tests cover:
- Tool definition structure and validation
- Tool execution with various parameters
- Result formatting and source tracking
- Error handling and propagation
- Integration with VectorStore
- ToolManager functionality
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json

from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager, Tool
from vector_store import SearchResults
from models import Course, Lesson
from tests.test_data import TestDataGenerator, MockResponses, QueryTestData


class TestCourseSearchTool:
    """Test cases for CourseSearchTool class"""
    
    def test_tool_definition_structure(self):
        """Test that tool definition has correct structure for Anthropic tool calling"""
        mock_vector_store = Mock()
        tool = CourseSearchTool(mock_vector_store)
        
        definition = tool.get_tool_definition()
        
        # Verify required fields
        assert "name" in definition
        assert "description" in definition
        assert "input_schema" in definition
        
        # Verify tool name
        assert definition["name"] == "search_course_content"
        
        # Verify schema structure
        schema = definition["input_schema"]
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema
        
        # Verify required parameters
        assert "query" in schema["required"]
        
        # Verify parameter definitions
        properties = schema["properties"]
        assert "query" in properties
        assert "course_name" in properties
        assert "lesson_number" in properties
        
        # Verify parameter types
        assert properties["query"]["type"] == "string"
        assert properties["course_name"]["type"] == "string"
        assert properties["lesson_number"]["type"] == "integer"
    
    def test_execute_with_query_only(self, sample_search_results):
        """Test tool execution with query parameter only"""
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        
        tool = CourseSearchTool(mock_vector_store)
        
        result = tool.execute(query="How to use Anthropic API?")
        
        # Verify vector store was called correctly
        mock_vector_store.search.assert_called_once_with(
            query="How to use Anthropic API?",
            course_name=None,
            lesson_number=None
        )
        
        # Verify result is properly formatted string
        assert isinstance(result, str)
        assert "Building Towards Computer Use with Anthropic" in result
        assert "Lesson 1" in result
    
    def test_execute_with_course_filter(self, sample_search_results):
        """Test tool execution with course name filter"""
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        
        tool = CourseSearchTool(mock_vector_store)
        
        result = tool.execute(
            query="API usage examples",
            course_name="Computer Use"
        )
        
        # Verify vector store was called with course filter
        mock_vector_store.search.assert_called_once_with(
            query="API usage examples",
            course_name="Computer Use",
            lesson_number=None
        )
        
        assert isinstance(result, str)
    
    def test_execute_with_lesson_filter(self, sample_search_results):
        """Test tool execution with lesson number filter"""
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        
        tool = CourseSearchTool(mock_vector_store)
        
        result = tool.execute(
            query="API usage examples",
            lesson_number=1
        )
        
        # Verify vector store was called with lesson filter
        mock_vector_store.search.assert_called_once_with(
            query="API usage examples",
            course_name=None,
            lesson_number=1
        )
        
        assert isinstance(result, str)
    
    def test_execute_with_all_parameters(self, sample_search_results):
        """Test tool execution with all parameters"""
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        
        tool = CourseSearchTool(mock_vector_store)
        
        result = tool.execute(
            query="API usage examples",
            course_name="Computer Use",
            lesson_number=1
        )
        
        # Verify vector store was called with all parameters
        mock_vector_store.search.assert_called_once_with(
            query="API usage examples",
            course_name="Computer Use",
            lesson_number=1
        )
        
        assert isinstance(result, str)
    
    def test_execute_with_empty_results(self, empty_search_results):
        """Test tool execution when search returns no results"""
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = empty_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        
        result = tool.execute(query="nonexistent topic")
        
        # Verify appropriate message is returned
        assert isinstance(result, str)
        assert "No relevant content found" in result
    
    def test_execute_with_empty_results_and_filters(self, empty_search_results):
        """Test tool execution with filters when search returns no results"""
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = empty_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        
        result = tool.execute(
            query="nonexistent topic",
            course_name="Test Course",
            lesson_number=1
        )
        
        # Verify filter information is included in message
        assert isinstance(result, str)
        assert "No relevant content found" in result
        assert "Test Course" in result
        assert "lesson 1" in result
    
    def test_execute_with_search_error(self, error_search_results):
        """Test tool execution when vector store returns an error"""
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = error_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        
        result = tool.execute(query="test query")
        
        # Verify error is properly returned
        assert isinstance(result, str)
        assert error_search_results.error in result
    
    def test_format_results_with_lesson_links(self):
        """Test result formatting includes lesson links when available"""
        mock_vector_store = Mock()
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        
        tool = CourseSearchTool(mock_vector_store)
        
        # Create sample search results
        results = SearchResults(
            documents=["Sample course content about API usage"],
            metadata=[{
                "course_title": "API Course",
                "lesson_number": 1,
                "chunk_index": 0
            }],
            distances=[0.1]
        )
        
        formatted = tool._format_results(results)
        
        # Verify formatting
        assert isinstance(formatted, str)
        assert "[API Course - Lesson 1]" in formatted
        assert "Sample course content about API usage" in formatted
        
        # Verify sources were tracked
        assert len(tool.last_sources) == 1
        source = tool.last_sources[0]
        assert source["text"] == "API Course - Lesson 1"
        assert source["url"] == "https://example.com/lesson1"
    
    def test_format_results_without_lesson_links(self):
        """Test result formatting when lesson links are not available"""
        mock_vector_store = Mock()
        mock_vector_store.get_lesson_link.return_value = None
        
        tool = CourseSearchTool(mock_vector_store)
        
        # Create sample search results without lesson number
        results = SearchResults(
            documents=["Sample course content"],
            metadata=[{
                "course_title": "Test Course",
                "lesson_number": None,
                "chunk_index": 0
            }],
            distances=[0.1]
        )
        
        formatted = tool._format_results(results)
        
        # Verify formatting
        assert isinstance(formatted, str)
        assert "[Test Course]" in formatted
        assert "Sample course content" in formatted
        
        # Verify sources were tracked
        assert len(tool.last_sources) == 1
        source = tool.last_sources[0]
        assert source["text"] == "Test Course"
        assert source["url"] is None
    
    def test_format_results_source_validation(self):
        """Test that source validation handles invalid source text"""
        mock_vector_store = Mock()
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        
        tool = CourseSearchTool(mock_vector_store)
        
        # Create search results with problematic metadata
        results = SearchResults(
            documents=["Sample content"],
            metadata=[{
                "course_title": "",  # Empty course title
                "lesson_number": 1,
                "chunk_index": 0
            }],
            distances=[0.1]
        )
        
        formatted = tool._format_results(results)
        
        # Verify source validation worked
        assert len(tool.last_sources) == 1
        source = tool.last_sources[0]
        # Should have been corrected to a valid format
        assert source["text"].strip()  # Should not be empty after validation
    
    def test_multiple_search_results(self):
        """Test formatting multiple search results"""
        mock_vector_store = Mock()
        mock_vector_store.get_lesson_link.side_effect = [
            "https://example.com/lesson1",
            "https://example.com/lesson2"
        ]
        
        tool = CourseSearchTool(mock_vector_store)
        
        # Create multiple search results
        results = SearchResults(
            documents=[
                "First piece of content",
                "Second piece of content"
            ],
            metadata=[
                {
                    "course_title": "Course A",
                    "lesson_number": 1,
                    "chunk_index": 0
                },
                {
                    "course_title": "Course B", 
                    "lesson_number": 2,
                    "chunk_index": 1
                }
            ],
            distances=[0.1, 0.2]
        )
        
        formatted = tool._format_results(results)
        
        # Verify both results are formatted
        assert "[Course A - Lesson 1]" in formatted
        assert "[Course B - Lesson 2]" in formatted
        assert "First piece of content" in formatted
        assert "Second piece of content" in formatted
        
        # Verify both sources are tracked
        assert len(tool.last_sources) == 2
        assert tool.last_sources[0]["text"] == "Course A - Lesson 1"
        assert tool.last_sources[1]["text"] == "Course B - Lesson 2"


class TestCourseOutlineTool:
    """Test cases for CourseOutlineTool class"""
    
    def test_tool_definition_structure(self):
        """Test that outline tool definition has correct structure"""
        mock_vector_store = Mock()
        tool = CourseOutlineTool(mock_vector_store)
        
        definition = tool.get_tool_definition()
        
        # Verify required fields
        assert definition["name"] == "get_course_outline"
        assert "description" in definition
        assert "input_schema" in definition
        
        # Verify schema structure
        schema = definition["input_schema"]
        assert "course_title" in schema["required"]
        assert "course_title" in schema["properties"]
        assert schema["properties"]["course_title"]["type"] == "string"
    
    def test_execute_with_valid_course(self):
        """Test outline tool execution with valid course"""
        mock_outline = {
            "course_title": "Test Course",
            "course_link": "https://example.com/course",
            "instructor": "Test Instructor",
            "lessons": [
                {
                    "lesson_number": 1,
                    "lesson_title": "Introduction",
                    "lesson_link": "https://example.com/lesson1"
                }
            ]
        }
        
        mock_vector_store = Mock()
        mock_vector_store.get_course_outline.return_value = mock_outline
        
        tool = CourseOutlineTool(mock_vector_store)
        
        result = tool.execute(course_title="Test Course")
        
        # Verify vector store was called
        mock_vector_store.get_course_outline.assert_called_once_with("Test Course")
        
        # Verify result formatting
        assert isinstance(result, str)
        assert "**Test Course**" in result
        assert "Test Instructor" in result
        assert "Lesson 1: Introduction" in result
        assert "https://example.com/course" in result
    
    def test_execute_with_nonexistent_course(self):
        """Test outline tool execution with non-existent course"""
        mock_vector_store = Mock()
        mock_vector_store.get_course_outline.return_value = None
        
        tool = CourseOutlineTool(mock_vector_store)
        
        result = tool.execute(course_title="Nonexistent Course")
        
        # Verify appropriate error message
        assert isinstance(result, str)
        assert "No course found matching 'Nonexistent Course'" in result
    
    def test_format_outline_complete(self):
        """Test outline formatting with complete data"""
        mock_vector_store = Mock()
        tool = CourseOutlineTool(mock_vector_store)
        
        outline = {
            "course_title": "Complete Course",
            "course_link": "https://example.com/course",
            "instructor": "Dr. Test",
            "lessons": [
                {
                    "lesson_number": 1,
                    "lesson_title": "Lesson One",
                    "lesson_link": "https://example.com/lesson1"
                },
                {
                    "lesson_number": 2,
                    "lesson_title": "Lesson Two",
                    "lesson_link": "https://example.com/lesson2"
                }
            ]
        }
        
        formatted = tool._format_outline(outline)
        
        # Verify all elements are included
        assert "**Complete Course**" in formatted
        assert "Course Link: https://example.com/course" in formatted
        assert "Instructor: Dr. Test" in formatted
        assert "**Lessons:**" in formatted
        assert "Lesson 1: Lesson One (https://example.com/lesson1)" in formatted
        assert "Lesson 2: Lesson Two (https://example.com/lesson2)" in formatted
    
    def test_format_outline_minimal(self):
        """Test outline formatting with minimal data"""
        mock_vector_store = Mock()
        tool = CourseOutlineTool(mock_vector_store)
        
        outline = {
            "course_title": "Minimal Course",
            "lessons": []
        }
        
        formatted = tool._format_outline(outline)
        
        # Verify minimal formatting
        assert "**Minimal Course**" in formatted
        assert "No lesson information available" in formatted
        assert "Course Link:" not in formatted
        assert "Instructor:" not in formatted


class TestToolManager:
    """Test cases for ToolManager class"""
    
    def test_tool_registration(self):
        """Test registering tools with ToolManager"""
        manager = ToolManager()
        mock_vector_store = Mock()
        
        # Create a tool
        search_tool = CourseSearchTool(mock_vector_store)
        
        # Register the tool
        manager.register_tool(search_tool)
        
        # Verify tool is registered
        assert "search_course_content" in manager.tools
        assert manager.tools["search_course_content"] == search_tool
    
    def test_tool_registration_without_name(self):
        """Test that tool registration fails without tool name"""
        manager = ToolManager()
        
        # Create a mock tool without name in definition
        mock_tool = Mock(spec=Tool)
        mock_tool.get_tool_definition.return_value = {"description": "Test tool"}
        
        with pytest.raises(ValueError, match="Tool must have a 'name' in its definition"):
            manager.register_tool(mock_tool)
    
    def test_get_tool_definitions(self):
        """Test getting all tool definitions"""
        manager = ToolManager()
        mock_vector_store = Mock()
        
        # Register multiple tools
        search_tool = CourseSearchTool(mock_vector_store)
        outline_tool = CourseOutlineTool(mock_vector_store)
        
        manager.register_tool(search_tool)
        manager.register_tool(outline_tool)
        
        definitions = manager.get_tool_definitions()
        
        # Verify definitions are returned
        assert len(definitions) == 2
        tool_names = [defn["name"] for defn in definitions]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names
    
    def test_execute_tool_success(self):
        """Test successful tool execution"""
        manager = ToolManager()
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = SearchResults(
            documents=["Test result"],
            metadata=[{"course_title": "Test"}],
            distances=[0.1]
        )
        mock_vector_store.get_lesson_link.return_value = None
        
        # Register tool
        search_tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(search_tool)
        
        # Execute tool
        result = manager.execute_tool(
            "search_course_content",
            query="test query"
        )
        
        assert isinstance(result, str)
        assert "Test result" in result
    
    def test_execute_nonexistent_tool(self):
        """Test executing non-existent tool"""
        manager = ToolManager()
        
        result = manager.execute_tool("nonexistent_tool", query="test")
        
        assert "Tool 'nonexistent_tool' not found" in result
    
    def test_get_last_sources(self):
        """Test retrieving sources from last search"""
        manager = ToolManager()
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1]
        )
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        
        # Register tool and execute search
        search_tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(search_tool)
        manager.execute_tool("search_course_content", query="test")
        
        # Get sources
        sources = manager.get_last_sources()
        
        assert len(sources) == 1
        assert sources[0]["text"] == "Test Course - Lesson 1"
        assert sources[0]["url"] == "https://example.com/lesson1"
    
    def test_reset_sources(self):
        """Test resetting sources from all tools"""
        manager = ToolManager()
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1]
        )
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        
        # Register tool and execute search
        search_tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(search_tool)
        manager.execute_tool("search_course_content", query="test")
        
        # Verify sources exist
        assert len(manager.get_last_sources()) == 1
        
        # Reset sources
        manager.reset_sources()
        
        # Verify sources are cleared
        assert len(manager.get_last_sources()) == 0
    
    def test_get_last_sources_multiple_tools(self):
        """Test getting sources when multiple tools are registered"""
        manager = ToolManager()
        mock_vector_store = Mock()
        
        # Create tools
        search_tool1 = CourseSearchTool(mock_vector_store)
        search_tool2 = CourseSearchTool(mock_vector_store)
        
        # Mock only one tool having sources
        search_tool1.last_sources = [{"text": "Source 1", "url": "url1"}]
        search_tool2.last_sources = []
        
        # Register tools
        manager.tools["tool1"] = search_tool1
        manager.tools["tool2"] = search_tool2
        
        # Get sources
        sources = manager.get_last_sources()
        
        # Should return sources from the tool that has them
        assert len(sources) == 1
        assert sources[0]["text"] == "Source 1"


# Error handling tests
class TestToolErrorHandling:
    """Test error handling in tool execution"""
    
    def test_search_tool_with_vector_store_error(self):
        """Test search tool when vector store throws error"""
        mock_vector_store = Mock()
        mock_vector_store.search.side_effect = Exception("Database connection failed")
        
        tool = CourseSearchTool(mock_vector_store)
        
        # This should not raise an exception but return error in SearchResults
        result = tool.execute(query="test query")
        
        # The error should be handled gracefully
        assert isinstance(result, str)
        # The exact error handling depends on how VectorStore.search handles exceptions
    
    def test_outline_tool_with_vector_store_error(self):
        """Test outline tool when vector store throws error"""
        mock_vector_store = Mock()
        mock_vector_store.get_course_outline.side_effect = Exception("Database error")
        
        tool = CourseOutlineTool(mock_vector_store)
        
        # This should not raise an exception
        with pytest.raises(Exception):  # We expect this to propagate up
            tool.execute(course_title="test course")
    
    def test_tool_manager_with_tool_execution_error(self):
        """Test tool manager when tool execution throws error"""
        manager = ToolManager()
        
        # Create a mock tool that raises an error
        mock_tool = Mock(spec=Tool)
        mock_tool.get_tool_definition.return_value = {"name": "error_tool"}
        mock_tool.execute.side_effect = Exception("Tool execution failed")
        
        manager.register_tool(mock_tool)
        
        # Execute tool - error should be handled
        with pytest.raises(Exception):
            manager.execute_tool("error_tool", test_param="test")


# Integration tests combining multiple components
class TestToolIntegration:
    """Integration tests for tools working together"""
    
    def test_search_tool_with_real_search_results(self):
        """Test search tool with realistic search results"""
        # Create realistic mock responses
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = SearchResults(
            documents=[
                "The Anthropic API allows you to integrate Claude into your applications.",
                "Computer use with Claude enables automated task execution on computers."
            ],
            metadata=[
                {
                    "course_title": "Building Towards Computer Use with Anthropic",
                    "lesson_number": 2,
                    "chunk_index": 5
                },
                {
                    "course_title": "Building Towards Computer Use with Anthropic",
                    "lesson_number": 3,
                    "chunk_index": 12
                }
            ],
            distances=[0.15, 0.23]
        )
        mock_vector_store.get_lesson_link.side_effect = [
            "https://learn.deeplearning.ai/lesson2",
            "https://learn.deeplearning.ai/lesson3"
        ]
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(
            query="How to use Anthropic API for computer use?",
            course_name="Computer Use"
        )
        
        # Verify comprehensive result
        assert isinstance(result, str)
        assert "Building Towards Computer Use with Anthropic" in result
        assert "Anthropic API" in result
        assert "computer use" in result
        assert "Lesson 2" in result
        assert "Lesson 3" in result
        
        # Verify sources are properly tracked
        assert len(tool.last_sources) == 2
        assert all(source["url"] is not None for source in tool.last_sources)
    
    def test_tool_manager_complete_workflow(self):
        """Test complete tool manager workflow with multiple tools"""
        manager = ToolManager()
        mock_vector_store = Mock()
        
        # Setup search tool
        mock_vector_store.search.return_value = SearchResults(
            documents=["Sample search content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1]
        )
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        
        # Setup outline tool
        mock_vector_store.get_course_outline.return_value = {
            "course_title": "Test Course",
            "instructor": "Test Instructor",
            "lessons": [{"lesson_number": 1, "lesson_title": "Test Lesson"}]
        }
        
        # Register both tools
        search_tool = CourseSearchTool(mock_vector_store)
        outline_tool = CourseOutlineTool(mock_vector_store)
        manager.register_tool(search_tool)
        manager.register_tool(outline_tool)
        
        # Test search execution
        search_result = manager.execute_tool(
            "search_course_content",
            query="test content"
        )
        assert "Sample search content" in search_result
        
        # Test outline execution
        outline_result = manager.execute_tool(
            "get_course_outline",
            course_title="Test Course"
        )
        assert "Test Course" in outline_result
        assert "Test Instructor" in outline_result
        
        # Test tool definitions
        definitions = manager.get_tool_definitions()
        assert len(definitions) == 2
        
        # Test source tracking
        sources = manager.get_last_sources()
        assert len(sources) == 1  # Only search tool tracks sources