"""Tests for react_agent module."""

import pytest
import math
from unittest.mock import AsyncMock, Mock, patch
import asyncio
import time
import logging


# Configure pytest to use anyio for async tests - only asyncio backend
pytestmark = pytest.mark.anyio


class TestReactAgentMCP:
    """Test cases for ReactAgentMCP class."""

    # Set specific backend to avoid trio dependency issues
    @pytest.fixture(scope="module")
    def anyio_backend(self):
        return "asyncio"

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Setup all necessary mocks before each test."""
        # Mock all external dependencies
        self.mock_modules = {
            "langgraph": Mock(),
            "langgraph.prebuilt": Mock(),
            "langchain_core": Mock(),
            "langchain_core.language_models": Mock(),
            "langchain_core.language_models.chat_models": Mock(),
            "langchain_core.tools": Mock(),
            "langchain_mcp_adapters": Mock(),
            "langchain_mcp_adapters.tools": Mock(),
            "mcp": Mock(),
            "mcp.client": Mock(),
            "mcp.client.stdio": Mock(),
        }

        # Create specific mock functions
        self.mock_create_react_agent = Mock()
        self.mock_load_ai_search_tools = AsyncMock()
        self.mock_cleanup_ai_search_client = AsyncMock()

        self.mock_modules["langgraph.prebuilt"].create_react_agent = self.mock_create_react_agent
        self.mock_modules["langchain_mcp_adapters.tools"].load_mcp_tools = AsyncMock()

        # Apply module mocks
        self.module_patcher = patch.dict("sys.modules", self.mock_modules)
        self.module_patcher.start()

        # Mock the ai_search_tool functions
        self.ai_search_patcher = patch.dict(
            "sys.modules",
            {
                "generic_rag.mcp.ai_search_tool": Mock(
                    load_ai_search_tools=self.mock_load_ai_search_tools,
                    cleanup_ai_search_client=self.mock_cleanup_ai_search_client,
                )
            },
        )
        self.ai_search_patcher.start()

        yield

        # Cleanup
        self.module_patcher.stop()
        self.ai_search_patcher.stop()

    def get_react_agent_class(self):
        """Import and return the ReactAgentMCP class after mocks are in place."""
        from generic_rag.graphs.react_agent import ReactAgentMCP, SYSTEM_PROMPT

        return ReactAgentMCP, SYSTEM_PROMPT

    @pytest.fixture
    def mock_chat_model(self):
        """Fixture to provide a mock chat model."""
        mock_model = Mock()
        mock_model.__class__.__name__ = "MockChatModel"
        return mock_model

    @pytest.fixture
    def mock_tools(self):
        """Fixture to provide mock tools."""
        return [
            Mock(name="search_tool", spec=["name", "description"]),
            Mock(name="another_tool", spec=["name", "description"]),
        ]

    @pytest.fixture
    def mock_agent(self):
        """Fixture to provide a mock LangGraph agent."""
        mock_agent = AsyncMock()
        mock_agent.ainvoke = AsyncMock()
        return mock_agent

    @pytest.fixture
    def mock_ai_message(self):
        """Fixture to provide a mock AI message."""
        mock_message = Mock()
        mock_message.__class__.__name__ = "AIMessage"
        mock_message.content = "This is a test response from the AI agent."
        return mock_message

    @pytest.fixture
    def mock_tool_message(self):
        """Fixture to provide a mock tool message."""
        mock_message = Mock()
        mock_message.__class__.__name__ = "ToolMessage"
        mock_message.content = "Tool execution result"
        return mock_message

    def test_init_with_default_system_prompt(self, mock_chat_model):
        """Test ReactAgentMCP initialization with default system prompt."""
        ReactAgentMCP, SYSTEM_PROMPT = self.get_react_agent_class()

        agent = ReactAgentMCP(agent_model=mock_chat_model, system_prompt=None)

        assert agent.agent_model == mock_chat_model
        assert agent.system_prompt == SYSTEM_PROMPT
        assert agent.tools is None
        assert agent.agent is None
        assert agent.instance_id > 0

    def test_init_with_custom_system_prompt(self, mock_chat_model):
        """Test ReactAgentMCP initialization with custom system prompt."""
        ReactAgentMCP, _ = self.get_react_agent_class()

        custom_prompt = "Custom system prompt for testing"
        agent = ReactAgentMCP(agent_model=mock_chat_model, system_prompt=custom_prompt)

        assert agent.agent_model == mock_chat_model
        assert agent.system_prompt == custom_prompt
        assert agent.tools is None
        assert agent.agent is None

    def test_init_with_tools(self, mock_chat_model, mock_tools):
        """Test ReactAgentMCP initialization with provided tools."""
        ReactAgentMCP, _ = self.get_react_agent_class()

        agent = ReactAgentMCP(agent_model=mock_chat_model, system_prompt="Test prompt", tools=mock_tools)

        assert agent.tools == mock_tools

    def test_instance_counter(self, mock_chat_model):
        """Test that instance counter is correctly incremented."""
        ReactAgentMCP, _ = self.get_react_agent_class()

        initial_count = ReactAgentMCP._instance_count

        agent1 = ReactAgentMCP(agent_model=mock_chat_model, system_prompt="Test")
        agent2 = ReactAgentMCP(agent_model=mock_chat_model, system_prompt="Test")

        assert agent1.instance_id == initial_count + 1
        assert agent2.instance_id == initial_count + 2

    async def test_create_agent_with_provided_tools(self, mock_chat_model, mock_tools, mock_agent):
        """Test create_agent when tools are provided."""
        ReactAgentMCP, _ = self.get_react_agent_class()

        self.mock_create_react_agent.return_value = mock_agent

        agent = ReactAgentMCP(agent_model=mock_chat_model, system_prompt="Test prompt", tools=mock_tools)

        result = await agent.create_agent()

        assert result == agent
        assert agent.agent == mock_agent
        self.mock_create_react_agent.assert_called_once_with(
            model=mock_chat_model, tools=mock_tools, prompt="Test prompt"
        )

    async def test_create_agent_loads_tools_when_none_provided(self, mock_chat_model, mock_tools, mock_agent):
        """Test create_agent when no tools are provided - should load them."""
        ReactAgentMCP, _ = self.get_react_agent_class()

        self.mock_create_react_agent.return_value = mock_agent
        self.mock_load_ai_search_tools.return_value = mock_tools

        agent = ReactAgentMCP(agent_model=mock_chat_model, system_prompt="Test prompt")

        result = await agent.create_agent()

        assert result == agent
        assert agent.agent == mock_agent
        assert agent.tools == mock_tools
        self.mock_load_ai_search_tools.assert_called_once()
        self.mock_create_react_agent.assert_called_once_with(
            model=mock_chat_model, tools=mock_tools, prompt="Test prompt"
        )

    async def test_create_agent_handles_empty_tools(self, mock_chat_model, mock_agent):
        """Test create_agent when load_ai_search_tools returns None or empty list."""
        ReactAgentMCP, _ = self.get_react_agent_class()

        self.mock_create_react_agent.return_value = mock_agent
        self.mock_load_ai_search_tools.return_value = None

        agent = ReactAgentMCP(agent_model=mock_chat_model, system_prompt="Test prompt")

        result = await agent.create_agent()

        assert result == agent
        assert agent.agent == mock_agent
        assert agent.tools is None
        self.mock_load_ai_search_tools.assert_called_once()
        self.mock_create_react_agent.assert_called_once()

    async def test_stream_with_ai_message(self, mock_chat_model, mock_tools, mock_agent, mock_ai_message):
        """Test stream method with AIMessage response."""
        ReactAgentMCP, _ = self.get_react_agent_class()

        # Setup agent response
        mock_agent.ainvoke.return_value = {"messages": [mock_ai_message]}

        agent = ReactAgentMCP(agent_model=mock_chat_model, system_prompt="Test prompt", tools=mock_tools)
        agent.agent = mock_agent

        # Collect streamed chunks
        chunks = []
        async for chunk in agent.stream("Test message"):
            chunks.append(chunk)

        # Verify the content was streamed
        full_content = "".join(chunks)
        assert full_content == mock_ai_message.content
        assert len(chunks) >= 1  # Should have at least one chunk

        # Verify agent was called correctly
        mock_agent.ainvoke.assert_called_once_with(
            {"messages": [{"role": "user", "content": "Test message"}]}, config=None
        )

    async def test_stream_with_mixed_messages(
        self, mock_chat_model, mock_tools, mock_agent, mock_ai_message, mock_tool_message
    ):
        """Test stream method with mixed message types - should only stream AIMessage."""
        ReactAgentMCP, _ = self.get_react_agent_class()

        # Setup agent response with mixed message types
        mock_agent.ainvoke.return_value = {"messages": [mock_tool_message, mock_ai_message]}

        agent = ReactAgentMCP(agent_model=mock_chat_model, system_prompt="Test prompt", tools=mock_tools)
        agent.agent = mock_agent

        # Collect streamed chunks
        chunks = []
        async for chunk in agent.stream("Test message"):
            chunks.append(chunk)

        # Verify only AIMessage content was streamed
        full_content = "".join(chunks)
        assert full_content == mock_ai_message.content
        assert mock_tool_message.content not in full_content

    async def test_stream_with_config(self, mock_chat_model, mock_tools, mock_agent, mock_ai_message):
        """Test stream method with custom config."""
        ReactAgentMCP, _ = self.get_react_agent_class()

        mock_agent.ainvoke.return_value = {"messages": [mock_ai_message]}

        agent = ReactAgentMCP(agent_model=mock_chat_model, system_prompt="Test prompt", tools=mock_tools)
        agent.agent = mock_agent

        config = {"temperature": 0.5, "max_tokens": 100}

        chunks = []
        async for chunk in agent.stream("Test message", config=config):
            chunks.append(chunk)

        # Verify agent was called with config
        mock_agent.ainvoke.assert_called_once_with(
            {"messages": [{"role": "user", "content": "Test message"}]}, config=config
        )

    async def test_stream_with_empty_ai_message(self, mock_chat_model, mock_tools, mock_agent):
        """Test stream method with AIMessage that has no content."""
        ReactAgentMCP, _ = self.get_react_agent_class()

        mock_empty_ai_message = Mock()
        mock_empty_ai_message.__class__.__name__ = "AIMessage"
        mock_empty_ai_message.content = ""

        mock_agent.ainvoke.return_value = {"messages": [mock_empty_ai_message]}

        agent = ReactAgentMCP(agent_model=mock_chat_model, system_prompt="Test prompt", tools=mock_tools)
        agent.agent = mock_agent

        chunks = []
        async for chunk in agent.stream("Test message"):
            chunks.append(chunk)

        # Should yield nothing for empty content
        assert chunks == []

    async def test_stream_with_no_messages(self, mock_chat_model, mock_tools, mock_agent):
        """Test stream method when agent returns no messages."""
        ReactAgentMCP, _ = self.get_react_agent_class()

        mock_agent.ainvoke.return_value = {"messages": []}

        agent = ReactAgentMCP(agent_model=mock_chat_model, system_prompt="Test prompt", tools=mock_tools)
        agent.agent = mock_agent

        chunks = []
        async for chunk in agent.stream("Test message"):
            chunks.append(chunk)

        assert chunks == []

    async def test_stream_exception_handling(self, mock_chat_model, mock_tools, mock_agent):
        """Test stream method exception handling."""
        ReactAgentMCP, _ = self.get_react_agent_class()

        mock_agent.ainvoke.side_effect = Exception("Test exception")

        agent = ReactAgentMCP(agent_model=mock_chat_model, system_prompt="Test prompt", tools=mock_tools)
        agent.agent = mock_agent

        with pytest.raises(Exception, match="Test exception"):
            async for chunk in agent.stream("Test message"):
                pass

    async def test_cleanup(self, mock_chat_model):
        """Test cleanup method."""
        ReactAgentMCP, _ = self.get_react_agent_class()

        agent = ReactAgentMCP(agent_model=mock_chat_model, system_prompt="Test")

        await agent.cleanup()

        self.mock_cleanup_ai_search_client.assert_called_once()

    def test_system_prompt_contains_service_info(self):
        """Test that SYSTEM_PROMPT contains expected service information."""
        _, SYSTEM_PROMPT = self.get_react_agent_class()

        assert "Azure AI Search" in SYSTEM_PROMPT
        assert "search_index_query" in SYSTEM_PROMPT
        assert "service" in SYSTEM_PROMPT
        assert "index" in SYSTEM_PROMPT
        assert "Dutch" in SYSTEM_PROMPT
        assert "English" in SYSTEM_PROMPT

    @pytest.mark.parametrize(
        "content_length,expected_chunks",
        [
            (25, 1),  # Short content
            (50, 1),  # Exactly chunk size
            (75, 2),  # One and a half chunks
            (150, 3),  # Multiple chunks
            (200, 4),  # Many chunks
            (1, 1),  # Single character
            (0, 0),  # Empty content (edge case)
        ],
    )
    async def test_stream_chunking_various_sizes(
        self, content_length, expected_chunks, mock_chat_model, mock_tools, mock_agent
    ):
        """Test streaming behavior with various content lengths."""
        ReactAgentMCP, _ = self.get_react_agent_class()

        # Create content of specified length
        content = "A" * content_length
        mock_ai_message = Mock()
        mock_ai_message.__class__.__name__ = "AIMessage"
        mock_ai_message.content = content

        mock_agent.ainvoke.return_value = {"messages": [mock_ai_message]}

        agent = ReactAgentMCP(agent_model=mock_chat_model, system_prompt="Test prompt", tools=mock_tools)
        agent.agent = mock_agent

        chunks = []
        async for chunk in agent.stream("Test message"):
            chunks.append(chunk)

        # Verify chunking behavior
        if content_length == 0:
            assert chunks == []
        else:
            chunk_size = 50  # From the implementation
            expected_chunk_count = math.ceil(content_length / chunk_size)
            assert len(chunks) == expected_chunk_count
            assert all(len(chunk) <= chunk_size for chunk in chunks)
            assert "".join(chunks) == content

            # Verify last chunk might be shorter
            if content_length > chunk_size:
                expected_last_chunk_size = content_length % chunk_size
                if expected_last_chunk_size > 0:
                    assert len(chunks[-1]) == expected_last_chunk_size

    async def test_stream_chunking_behavior(self, mock_chat_model, mock_tools, mock_agent):
        """Test that stream properly chunks content - legacy test for backward compatibility."""
        ReactAgentMCP, _ = self.get_react_agent_class()

        # Create a longer message to test chunking
        long_content = "A" * 150  # 150 characters
        mock_ai_message = Mock()
        mock_ai_message.__class__.__name__ = "AIMessage"
        mock_ai_message.content = long_content

        mock_agent.ainvoke.return_value = {"messages": [mock_ai_message]}

        agent = ReactAgentMCP(agent_model=mock_chat_model, system_prompt="Test prompt", tools=mock_tools)
        agent.agent = mock_agent

        chunks = []
        async for chunk in agent.stream("Test message"):
            chunks.append(chunk)

        # Verify chunking using math.ceil for robustness
        chunk_size = 50
        expected_chunks = math.ceil(len(long_content) / chunk_size)
        assert len(chunks) == expected_chunks
        assert all(len(chunk) <= chunk_size for chunk in chunks)
        assert "".join(chunks) == long_content

    async def test_stream_handles_invalid_result_format(self, mock_chat_model, mock_tools, mock_agent):
        """Test stream method with invalid result format."""
        ReactAgentMCP, _ = self.get_react_agent_class()

        # Return invalid format (no 'messages' key)
        mock_agent.ainvoke.return_value = {"invalid": "format"}

        agent = ReactAgentMCP(agent_model=mock_chat_model, system_prompt="Test prompt", tools=mock_tools)
        agent.agent = mock_agent

        chunks = []
        async for chunk in agent.stream("Test message"):
            chunks.append(chunk)

        # Should handle gracefully and return no chunks
        assert chunks == []

    # Integration Tests
    async def test_full_agent_lifecycle(self, mock_chat_model, mock_tools, mock_agent, mock_ai_message):
        """Test complete lifecycle: init -> create_agent -> stream -> cleanup."""
        ReactAgentMCP, _ = self.get_react_agent_class()

        # Setup mocks
        self.mock_create_react_agent.return_value = mock_agent
        self.mock_load_ai_search_tools.return_value = mock_tools
        mock_agent.ainvoke.return_value = {"messages": [mock_ai_message]}

        # 1. Initialize agent
        agent = ReactAgentMCP(agent_model=mock_chat_model, system_prompt="Test prompt")
        assert agent.agent is None
        assert agent.tools is None

        # 2. Create agent
        result = await agent.create_agent()
        assert result == agent
        assert agent.agent == mock_agent
        assert agent.tools == mock_tools

        # 3. Stream messages
        chunks = []
        async for chunk in agent.stream("Test query"):
            chunks.append(chunk)
        assert "".join(chunks) == mock_ai_message.content

        # 4. Cleanup
        await agent.cleanup()
        self.mock_cleanup_ai_search_client.assert_called_once()

    async def test_agent_reuse_after_cleanup(self, mock_chat_model, mock_tools, mock_agent, mock_ai_message):
        """Test that agent can be recreated and used after cleanup."""
        ReactAgentMCP, _ = self.get_react_agent_class()

        # Setup mocks
        self.mock_create_react_agent.return_value = mock_agent
        self.mock_load_ai_search_tools.return_value = mock_tools
        mock_agent.ainvoke.return_value = {"messages": [mock_ai_message]}

        agent = ReactAgentMCP(agent_model=mock_chat_model, system_prompt="Test prompt")

        # First lifecycle
        await agent.create_agent()
        await agent.cleanup()

        # Reset mocks for second use
        self.mock_create_react_agent.reset_mock()
        self.mock_load_ai_search_tools.reset_mock()
        self.mock_cleanup_ai_search_client.reset_mock()

        # Second lifecycle - should work fine
        await agent.create_agent()
        chunks = []
        async for chunk in agent.stream("Second query"):
            chunks.append(chunk)
        assert "".join(chunks) == mock_ai_message.content

        await agent.cleanup()
        self.mock_cleanup_ai_search_client.assert_called_once()

    async def test_concurrent_streaming_calls(self, mock_chat_model, mock_tools, mock_agent):
        """Test multiple concurrent stream calls on the same agent instance."""
        ReactAgentMCP, _ = self.get_react_agent_class()

        # Create different responses for concurrent calls
        response1 = Mock()
        response1.__class__.__name__ = "AIMessage"
        response1.content = "Response 1"

        response2 = Mock()
        response2.__class__.__name__ = "AIMessage"
        response2.content = "Response 2"

        # Mock agent to return different responses
        mock_agent.ainvoke.side_effect = [{"messages": [response1]}, {"messages": [response2]}]

        agent = ReactAgentMCP(agent_model=mock_chat_model, system_prompt="Test prompt", tools=mock_tools)
        agent.agent = mock_agent

        async def stream_and_collect(message):
            chunks = []
            async for chunk in agent.stream(message):
                chunks.append(chunk)
            return "".join(chunks)

        # Run concurrent streams
        results = await asyncio.gather(stream_and_collect("Query 1"), stream_and_collect("Query 2"))

        # Both should complete successfully
        assert len(results) == 2
        assert "Response 1" in results
        assert "Response 2" in results

    # Error Scenarios and Edge Cases
    async def test_stream_before_agent_creation(self, mock_chat_model, mock_tools):
        """Test streaming fails gracefully when agent is not created yet."""
        ReactAgentMCP, _ = self.get_react_agent_class()

        agent = ReactAgentMCP(agent_model=mock_chat_model, system_prompt="Test prompt", tools=mock_tools)
        # Don't call create_agent()

        with pytest.raises(AttributeError):
            async for chunk in agent.stream("Test message"):
                pass

    async def test_create_agent_idempotency(self, mock_chat_model, mock_tools, mock_agent):
        """Test that calling create_agent multiple times doesn't cause issues."""
        ReactAgentMCP, _ = self.get_react_agent_class()

        self.mock_create_react_agent.return_value = mock_agent
        self.mock_load_ai_search_tools.return_value = mock_tools

        agent = ReactAgentMCP(agent_model=mock_chat_model, system_prompt="Test prompt")

        # Call create_agent multiple times
        result1 = await agent.create_agent()
        result2 = await agent.create_agent()
        result3 = await agent.create_agent()

        # All should return the same agent instance
        assert result1 == agent
        assert result2 == agent
        assert result3 == agent

        # Should have been called multiple times (not cached)
        assert self.mock_create_react_agent.call_count == 3

    async def test_cleanup_multiple_calls(self, mock_chat_model):
        """Test that cleanup can be called multiple times safely."""
        ReactAgentMCP, _ = self.get_react_agent_class()

        agent = ReactAgentMCP(agent_model=mock_chat_model, system_prompt="Test")

        # Multiple cleanup calls should not raise errors
        await agent.cleanup()
        await agent.cleanup()
        await agent.cleanup()

        # Should have been called multiple times
        assert self.mock_cleanup_ai_search_client.call_count == 3

    @pytest.mark.parametrize(
        "message_type,should_stream",
        [
            ("AIMessage", True),
            ("ToolMessage", False),
            ("HumanMessage", False),
            ("SystemMessage", False),
            ("FunctionMessage", False),
        ],
    )
    async def test_stream_message_type_filtering(
        self, message_type, should_stream, mock_chat_model, mock_tools, mock_agent
    ):
        """Test that only AIMessage types are streamed."""
        ReactAgentMCP, _ = self.get_react_agent_class()

        # Create mock message of specified type
        mock_message = Mock()
        mock_message.__class__.__name__ = message_type
        mock_message.content = f"Content from {message_type}"

        mock_agent.ainvoke.return_value = {"messages": [mock_message]}

        agent = ReactAgentMCP(agent_model=mock_chat_model, system_prompt="Test prompt", tools=mock_tools)
        agent.agent = mock_agent

        chunks = []
        async for chunk in agent.stream("Test message"):
            chunks.append(chunk)

        if should_stream:
            assert len(chunks) > 0
            assert "".join(chunks) == mock_message.content
        else:
            assert chunks == []

    @pytest.mark.parametrize(
        "config", [None, {}, {"temperature": 0.5}, {"temperature": 0.8, "max_tokens": 100}, {"custom_param": "value"}]
    )
    async def test_stream_with_various_configs(self, config, mock_chat_model, mock_tools, mock_agent, mock_ai_message):
        """Test streaming with various configuration options."""
        ReactAgentMCP, _ = self.get_react_agent_class()

        mock_agent.ainvoke.return_value = {"messages": [mock_ai_message]}

        agent = ReactAgentMCP(agent_model=mock_chat_model, system_prompt="Test prompt", tools=mock_tools)
        agent.agent = mock_agent

        chunks = []
        async for chunk in agent.stream("Test message", config=config):
            chunks.append(chunk)

        # Verify agent was called with the config
        mock_agent.ainvoke.assert_called_once_with(
            {"messages": [{"role": "user", "content": "Test message"}]}, config=config
        )

        # Content should be streamed correctly regardless of config
        assert "".join(chunks) == mock_ai_message.content

    # Logging Tests
    async def test_logging_during_streaming(self, caplog, mock_chat_model, mock_tools, mock_agent, mock_ai_message):
        """Test that appropriate log messages are generated during streaming."""
        ReactAgentMCP, _ = self.get_react_agent_class()
        mock_agent.ainvoke.return_value = {"messages": [mock_ai_message]}

        agent = ReactAgentMCP(agent_model=mock_chat_model, system_prompt="Test prompt", tools=mock_tools)
        agent.agent = mock_agent

        with caplog.at_level(logging.DEBUG):
            chunks = []
            async for chunk in agent.stream("Test message"):
                chunks.append(chunk)

        # Check for expected log messages
        log_messages = [record.message for record in caplog.records]
        assert any("Using stream for message: Test message" in msg for msg in log_messages)
        assert any("stream result:" in msg for msg in log_messages)
        assert any("Streaming AIMessage content:" in msg for msg in log_messages)

    async def test_logging_during_agent_creation(self, caplog, mock_chat_model, mock_tools, mock_agent):
        """Test logging during agent creation process."""
        ReactAgentMCP, _ = self.get_react_agent_class()
        self.mock_create_react_agent.return_value = mock_agent
        self.mock_load_ai_search_tools.return_value = mock_tools

        agent = ReactAgentMCP(agent_model=mock_chat_model, system_prompt="Test prompt")

        with caplog.at_level(logging.DEBUG):
            await agent.create_agent()

        log_messages = [record.message for record in caplog.records]
        assert any("Loading AI search tools..." in msg for msg in log_messages)
        assert any("Creating react agent with model:" in msg for msg in log_messages)
        assert any("Agent created:" in msg for msg in log_messages)

    # Performance-related Tests
    async def test_streaming_large_content_performance(self, mock_chat_model, mock_tools, mock_agent):
        """Test streaming performance with very large content."""
        ReactAgentMCP, _ = self.get_react_agent_class()

        # Create very large content (10KB)
        large_content = "A" * 10000
        mock_ai_message = Mock()
        mock_ai_message.__class__.__name__ = "AIMessage"
        mock_ai_message.content = large_content

        mock_agent.ainvoke.return_value = {"messages": [mock_ai_message]}

        agent = ReactAgentMCP(agent_model=mock_chat_model, system_prompt="Test prompt", tools=mock_tools)
        agent.agent = mock_agent
        start_time = time.time()

        chunks = []
        async for chunk in agent.stream("Test message"):
            chunks.append(chunk)

        end_time = time.time()

        # Verify content integrity
        assert "".join(chunks) == large_content

        # Performance check - should complete within reasonable time (5 seconds)
        assert (end_time - start_time) < 5.0

        # Verify proper chunking of large content
        chunk_size = 50
        expected_chunks = math.ceil(len(large_content) / chunk_size)
        assert len(chunks) == expected_chunks
