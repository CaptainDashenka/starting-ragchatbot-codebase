"""
Unit tests for AIGenerator class.

Tests cover:
- Anthropic API integration
- Tool calling functionality
- Tool execution handling
- Response generation
- Error handling and recovery
- Conversation history handling
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import anthropic

from ai_generator import AIGenerator
from tests.test_data import MockResponses, QueryTestData


class TestAIGenerator:
    """Test cases for AIGenerator class"""
    
    def test_initialization(self, mock_config):
        """Test AIGenerator initialization with valid configuration"""
        generator = AIGenerator(
            api_key=mock_config.ANTHROPIC_API_KEY,
            model=mock_config.ANTHROPIC_MODEL
        )
        
        assert generator.model == mock_config.ANTHROPIC_MODEL
        assert generator.base_params["model"] == mock_config.ANTHROPIC_MODEL
        assert generator.base_params["temperature"] == 0
        assert generator.base_params["max_tokens"] == 800
    
    def test_system_prompt_content(self):
        """Test that system prompt has expected content"""
        # Verify key elements of the system prompt
        assert "AI assistant specialized in course materials" in AIGenerator.SYSTEM_PROMPT
        assert "search tool" in AIGenerator.SYSTEM_PROMPT
        assert "course content questions" in AIGenerator.SYSTEM_PROMPT
        assert "Brief, Concise and focused" in AIGenerator.SYSTEM_PROMPT
    
    @patch('anthropic.Anthropic')
    def test_generate_response_without_tools(self, mock_anthropic_class, mock_config):
        """Test response generation without tool usage"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Setup mock response
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text=MockResponses.anthropic_text_response())]
        mock_client.messages.create.return_value = mock_response
        
        generator = AIGenerator(
            api_key=mock_config.ANTHROPIC_API_KEY,
            model=mock_config.ANTHROPIC_MODEL
        )
        
        result = generator.generate_response("What is artificial intelligence?")
        
        # Verify API was called correctly
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args[1]
        
        assert call_args["model"] == mock_config.ANTHROPIC_MODEL
        assert call_args["temperature"] == 0
        assert call_args["max_tokens"] == 800
        assert len(call_args["messages"]) == 1
        assert call_args["messages"][0]["role"] == "user"
        assert "tools" not in call_args
        
        # Verify result
        assert isinstance(result, str)
        assert "Anthropic API" in result
    
    @patch('anthropic.Anthropic')
    def test_generate_response_with_conversation_history(self, mock_anthropic_class, mock_config):
        """Test response generation with conversation history"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Setup mock response
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Response with context")]
        mock_client.messages.create.return_value = mock_response
        
        generator = AIGenerator(
            api_key=mock_config.ANTHROPIC_API_KEY,
            model=mock_config.ANTHROPIC_MODEL
        )
        
        conversation_history = "User: Previous question\nAssistant: Previous answer"
        
        result = generator.generate_response(
            "Follow-up question",
            conversation_history=conversation_history
        )
        
        # Verify system prompt includes history
        call_args = mock_client.messages.create.call_args[1]
        assert conversation_history in call_args["system"]
        
        assert result == "Response with context"
    
    @patch('anthropic.Anthropic')
    def test_generate_response_with_tools(self, mock_anthropic_class, mock_config, mock_tool_manager):
        """Test response generation with tool availability"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Setup mock response without tool use
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Direct response")]
        mock_client.messages.create.return_value = mock_response
        
        generator = AIGenerator(
            api_key=mock_config.ANTHROPIC_API_KEY,
            model=mock_config.ANTHROPIC_MODEL
        )
        
        tools = mock_tool_manager.get_tool_definitions()
        
        result = generator.generate_response(
            "General question",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        # Verify tools were provided in API call
        call_args = mock_client.messages.create.call_args[1]
        assert "tools" in call_args
        assert call_args["tools"] == tools
        assert call_args["tool_choice"]["type"] == "auto"
        
        assert result == "Direct response"
    
    @patch('anthropic.Anthropic')
    def test_generate_response_with_tool_use(self, mock_anthropic_class, mock_config, mock_tool_manager):
        """Test response generation when AI decides to use tools"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # First response: tool use
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "test query"}
        mock_tool_response.content = [mock_tool_block]
        
        # Second response: final answer
        mock_final_response = Mock()
        mock_final_response.stop_reason = "end_turn"
        mock_final_response.content = [Mock(text="Based on search results, here's the answer")]
        
        # Configure mock to return different responses on consecutive calls
        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        
        generator = AIGenerator(
            api_key=mock_config.ANTHROPIC_API_KEY,
            model=mock_config.ANTHROPIC_MODEL
        )
        
        tools = mock_tool_manager.get_tool_definitions()
        
        result = generator.generate_response(
            "How to use the Anthropic API?",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="test query"
        )
        
        # Verify API was called twice (initial + final)
        assert mock_client.messages.create.call_count == 2
        
        # Verify final result
        assert result == "Based on search results, here's the answer"
    
    @patch('anthropic.Anthropic')
    def test_handle_tool_execution_single_tool(self, mock_anthropic_class, mock_config, mock_tool_manager):
        """Test tool execution handling with single tool call"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Setup final response
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Final answer with tool results")]
        mock_client.messages.create.return_value = mock_final_response
        
        generator = AIGenerator(
            api_key=mock_config.ANTHROPIC_API_KEY,
            model=mock_config.ANTHROPIC_MODEL
        )
        
        # Create mock initial response with tool use
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_456"
        mock_tool_block.input = {"query": "API documentation", "course_name": "Computer Use"}
        mock_initial_response.content = [mock_tool_block]
        
        # Base parameters
        base_params = {
            "model": mock_config.ANTHROPIC_MODEL,
            "messages": [{"role": "user", "content": "Test query"}],
            "system": "System prompt"
        }
        
        # Mock tool execution result
        mock_tool_manager.execute_tool.return_value = "Tool search results"
        
        result = generator._handle_tool_execution(
            mock_initial_response,
            base_params,
            mock_tool_manager
        )
        
        # Verify tool was executed with correct parameters
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="API documentation",
            course_name="Computer Use"
        )
        
        # Verify final API call structure
        call_args = mock_client.messages.create.call_args[1]
        assert len(call_args["messages"]) == 3  # original + assistant + tool results
        
        # Check assistant message
        assistant_msg = call_args["messages"][1]
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["content"] == mock_initial_response.content
        
        # Check tool results message
        tool_msg = call_args["messages"][2]
        assert tool_msg["role"] == "user"
        assert len(tool_msg["content"]) == 1
        assert tool_msg["content"][0]["type"] == "tool_result"
        assert tool_msg["content"][0]["tool_use_id"] == "tool_456"
        assert tool_msg["content"][0]["content"] == "Tool search results"
        
        assert result == "Final answer with tool results"
    
    @patch('anthropic.Anthropic')
    def test_handle_tool_execution_multiple_tools(self, mock_anthropic_class, mock_config, mock_tool_manager):
        """Test tool execution handling with multiple tool calls"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Setup final response
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Final answer using multiple tools")]
        mock_client.messages.create.return_value = mock_final_response
        
        generator = AIGenerator(
            api_key=mock_config.ANTHROPIC_API_KEY,
            model=mock_config.ANTHROPIC_MODEL
        )
        
        # Create mock initial response with multiple tool uses
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        
        mock_tool_block1 = Mock()
        mock_tool_block1.type = "tool_use"
        mock_tool_block1.name = "search_course_content"
        mock_tool_block1.id = "tool_1"
        mock_tool_block1.input = {"query": "API basics"}
        
        mock_tool_block2 = Mock()
        mock_tool_block2.type = "tool_use"
        mock_tool_block2.name = "get_course_outline"
        mock_tool_block2.id = "tool_2"
        mock_tool_block2.input = {"course_title": "Computer Use"}
        
        mock_initial_response.content = [mock_tool_block1, mock_tool_block2]
        
        # Base parameters
        base_params = {
            "model": mock_config.ANTHROPIC_MODEL,
            "messages": [{"role": "user", "content": "Complex query"}],
            "system": "System prompt"
        }
        
        # Mock tool execution results
        mock_tool_manager.execute_tool.side_effect = [
            "Search results content",
            "Course outline content"
        ]
        
        result = generator._handle_tool_execution(
            mock_initial_response,
            base_params,
            mock_tool_manager
        )
        
        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        
        # Check tool calls
        call_1 = mock_tool_manager.execute_tool.call_args_list[0]
        assert call_1[0] == ("search_course_content",)
        assert call_1[1] == {"query": "API basics"}
        
        call_2 = mock_tool_manager.execute_tool.call_args_list[1]
        assert call_2[0] == ("get_course_outline",)
        assert call_2[1] == {"course_title": "Computer Use"}
        
        # Verify tool results message structure
        call_args = mock_client.messages.create.call_args[1]
        tool_msg = call_args["messages"][2]
        assert len(tool_msg["content"]) == 2
        
        # Check both tool results
        assert tool_msg["content"][0]["tool_use_id"] == "tool_1"
        assert tool_msg["content"][0]["content"] == "Search results content"
        assert tool_msg["content"][1]["tool_use_id"] == "tool_2"
        assert tool_msg["content"][1]["content"] == "Course outline content"
        
        assert result == "Final answer using multiple tools"
    
    @patch('anthropic.Anthropic')
    def test_handle_tool_execution_mixed_content(self, mock_anthropic_class, mock_config, mock_tool_manager):
        """Test tool execution handling with mixed content (text + tool use)"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Setup final response
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Final answer")]
        mock_client.messages.create.return_value = mock_final_response
        
        generator = AIGenerator(
            api_key=mock_config.ANTHROPIC_API_KEY,
            model=mock_config.ANTHROPIC_MODEL
        )
        
        # Create mock initial response with mixed content
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        
        mock_text_block = Mock()
        mock_text_block.type = "text"
        mock_text_block.text = "Let me search for that information."
        
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_789"
        mock_tool_block.input = {"query": "mixed content test"}
        
        mock_initial_response.content = [mock_text_block, mock_tool_block]
        
        # Base parameters
        base_params = {
            "model": mock_config.ANTHROPIC_MODEL,
            "messages": [{"role": "user", "content": "Test query"}],
            "system": "System prompt"
        }
        
        mock_tool_manager.execute_tool.return_value = "Mixed content search results"
        
        result = generator._handle_tool_execution(
            mock_initial_response,
            base_params,
            mock_tool_manager
        )
        
        # Verify only tool blocks were executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="mixed content test"
        )
        
        # Verify tool results message contains only tool results
        call_args = mock_client.messages.create.call_args[1]
        tool_msg = call_args["messages"][2]
        assert len(tool_msg["content"]) == 1
        assert tool_msg["content"][0]["type"] == "tool_result"
        
        assert result == "Final answer"
    
    @patch('anthropic.Anthropic')
    def test_anthropic_api_error_handling(self, mock_anthropic_class, mock_config):
        """Test handling of Anthropic API errors"""
        # Setup mock client that raises an error
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("API rate limit exceeded")
        
        generator = AIGenerator(
            api_key=mock_config.ANTHROPIC_API_KEY,
            model=mock_config.ANTHROPIC_MODEL
        )
        
        # Verify error is propagated
        with pytest.raises(Exception, match="API rate limit exceeded"):
            generator.generate_response("Test query")
    
    @patch('anthropic.Anthropic')
    def test_tool_execution_error_handling(self, mock_anthropic_class, mock_config, mock_tool_manager):
        """Test handling of tool execution errors"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Setup initial tool use response
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_error"
        mock_tool_block.input = {"query": "test"}
        mock_tool_response.content = [mock_tool_block]
        
        # Setup final response
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Handled tool error")]
        
        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        
        # Make tool execution fail
        mock_tool_manager.execute_tool.side_effect = Exception("Tool execution failed")
        
        generator = AIGenerator(
            api_key=mock_config.ANTHROPIC_API_KEY,
            model=mock_config.ANTHROPIC_MODEL
        )
        
        # Should propagate the tool execution error
        with pytest.raises(Exception, match="Tool execution failed"):
            generator.generate_response(
                "Test query",
                tools=mock_tool_manager.get_tool_definitions(),
                tool_manager=mock_tool_manager
            )
    
    def test_empty_query_handling(self, mock_config, mock_anthropic_client):
        """Test handling of empty or whitespace queries"""
        with patch('anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator(
                api_key=mock_config.ANTHROPIC_API_KEY,
                model=mock_config.ANTHROPIC_MODEL
            )
            
            # Test with empty string
            result = generator.generate_response("")
            assert isinstance(result, str)
            
            # Test with whitespace
            result = generator.generate_response("   ")
            assert isinstance(result, str)
    
    def test_very_long_query_handling(self, mock_config, mock_anthropic_client):
        """Test handling of very long queries"""
        with patch('anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator(
                api_key=mock_config.ANTHROPIC_API_KEY,
                model=mock_config.ANTHROPIC_MODEL
            )
            
            # Create a very long query
            long_query = "How do I use the API? " * 1000
            
            result = generator.generate_response(long_query)
            assert isinstance(result, str)
            
            # Verify the query was passed to the API
            mock_anthropic_client.messages.create.assert_called_once()
            call_args = mock_anthropic_client.messages.create.call_args[1]
            assert long_query in call_args["messages"][0]["content"]


class TestAIGeneratorConfiguration:
    """Test AIGenerator configuration and parameter handling"""
    
    def test_base_params_configuration(self, mock_config):
        """Test that base parameters are configured correctly"""
        generator = AIGenerator(
            api_key=mock_config.ANTHROPIC_API_KEY,
            model=mock_config.ANTHROPIC_MODEL
        )
        
        expected_params = {
            "model": mock_config.ANTHROPIC_MODEL,
            "temperature": 0,
            "max_tokens": 800
        }
        
        assert generator.base_params == expected_params
    
    def test_system_prompt_immutability(self):
        """Test that system prompt is static and immutable"""
        generator1 = AIGenerator("key1", "model1")
        generator2 = AIGenerator("key2", "model2")
        
        # System prompt should be the same for all instances
        assert generator1.SYSTEM_PROMPT == generator2.SYSTEM_PROMPT
        assert generator1.SYSTEM_PROMPT is generator2.SYSTEM_PROMPT
    
    @patch('anthropic.Anthropic')
    def test_api_parameter_passing(self, mock_anthropic_class, mock_config):
        """Test that API parameters are passed correctly"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Setup mock response
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Test response")]
        mock_client.messages.create.return_value = mock_response
        
        generator = AIGenerator(
            api_key=mock_config.ANTHROPIC_API_KEY,
            model=mock_config.ANTHROPIC_MODEL
        )
        
        generator.generate_response(
            "Test query",
            conversation_history="Test history"
        )
        
        # Verify API call parameters
        call_args = mock_client.messages.create.call_args[1]
        
        assert call_args["model"] == mock_config.ANTHROPIC_MODEL
        assert call_args["temperature"] == 0
        assert call_args["max_tokens"] == 800
        assert "Test history" in call_args["system"]
        assert len(call_args["messages"]) == 1
        assert call_args["messages"][0]["role"] == "user"
        assert call_args["messages"][0]["content"] == "Test query"


# Integration tests
class TestAIGeneratorIntegration:
    """Integration tests for AIGenerator with realistic scenarios"""
    
    @patch('anthropic.Anthropic')
    def test_complete_tool_calling_workflow(self, mock_anthropic_class, mock_config, mock_tool_manager):
        """Test complete workflow from query to tool execution to final response"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Setup tool use response
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "integration_test"
        mock_tool_block.input = {
            "query": "How to use Claude for computer use?",
            "course_name": "Computer Use",
            "lesson_number": 2
        }
        mock_tool_response.content = [mock_tool_block]
        
        # Setup final response
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Claude can be used for computer use by utilizing its vision capabilities and tool calling features to interact with applications and interfaces.")]
        
        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        
        # Setup tool manager response
        mock_tool_manager.execute_tool.return_value = "[Computer Use Course - Lesson 2]\nClaude's computer use capability allows it to view screens, understand interfaces, and take actions like clicking buttons and typing text."
        
        generator = AIGenerator(
            api_key=mock_config.ANTHROPIC_API_KEY,
            model=mock_config.ANTHROPIC_MODEL
        )
        
        result = generator.generate_response(
            "How can I use Claude for computer automation tasks?",
            conversation_history="User: Hello\nAssistant: Hi! How can I help you today?",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )
        
        # Verify complete workflow
        assert mock_client.messages.create.call_count == 2
        mock_tool_manager.execute_tool.assert_called_once()
        
        # Verify tool execution parameters
        tool_call = mock_tool_manager.execute_tool.call_args
        assert tool_call[0] == ("search_course_content",)
        assert tool_call[1]["query"] == "How to use Claude for computer use?"
        assert tool_call[1]["course_name"] == "Computer Use"
        assert tool_call[1]["lesson_number"] == 2
        
        # Verify final result
        assert isinstance(result, str)
        assert "computer use" in result.lower()
        assert "claude" in result.lower()
    
    @patch('anthropic.Anthropic')
    def test_no_tool_use_scenario(self, mock_anthropic_class, mock_config, mock_tool_manager):
        """Test scenario where AI decides not to use tools"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Setup direct response (no tool use)
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Artificial Intelligence is a field of computer science that focuses on creating systems capable of performing tasks that typically require human intelligence.")]
        mock_client.messages.create.return_value = mock_response
        
        generator = AIGenerator(
            api_key=mock_config.ANTHROPIC_API_KEY,
            model=mock_config.ANTHROPIC_MODEL
        )
        
        result = generator.generate_response(
            "What is artificial intelligence?",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )
        
        # Verify no tool execution occurred
        mock_tool_manager.execute_tool.assert_not_called()
        
        # Verify single API call
        assert mock_client.messages.create.call_count == 1
        
        # Verify tools were provided but not used
        call_args = mock_client.messages.create.call_args[1]
        assert "tools" in call_args
        
        # Verify result
        assert isinstance(result, str)
        assert "Artificial Intelligence" in result
    
    def test_realistic_error_scenarios(self, mock_config):
        """Test realistic error scenarios that might cause 'query failed'"""
        # Test different error conditions that might occur in production
        
        error_scenarios = [
            # API key issues
            ("Invalid API key", lambda: AIGenerator("invalid_key", mock_config.ANTHROPIC_MODEL)),
            
            # Model issues  
            ("Invalid model", lambda: AIGenerator(mock_config.ANTHROPIC_API_KEY, "nonexistent-model")),
        ]
        
        for error_desc, generator_factory in error_scenarios:
            # These would typically fail during API calls, not initialization
            generator = generator_factory()
            assert isinstance(generator, AIGenerator)
            # The actual errors would occur during API calls in real scenarios