import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import logging

# Import the module under test
from generic_rag.mcp.ai_search_tool import (
    initialize_ai_search_client,
    cleanup_ai_search_client,
    load_ai_search_tools,
    SERVER_PARAMS,
)

# Configure pytest to use anyio for async tests - only asyncio backend
pytestmark = pytest.mark.anyio


# Set specific backend to avoid trio dependency issues
@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"


@pytest.fixture
def mock_tools():
    """Fixture providing mock tools for testing"""
    return [MagicMock(name="tool1", description="Test tool 1"), MagicMock(name="tool2", description="Test tool 2")]


@pytest.fixture
def mock_stdio_client():
    """Fixture providing a mocked stdio_client"""
    with patch("generic_rag.mcp.ai_search_tool.stdio_client") as mock:
        mock_stdio_context = AsyncMock()
        mock_read, mock_write = MagicMock(), MagicMock()
        mock_stdio_context.__aenter__.return_value = (mock_read, mock_write)
        mock.return_value = mock_stdio_context
        yield mock


@pytest.fixture
def mock_client_session():
    """Fixture providing a mocked ClientSession"""
    with patch("generic_rag.mcp.ai_search_tool.ClientSession") as mock:
        mock_session = AsyncMock()
        mock.return_value = mock_session
        yield mock


@pytest.fixture
def mock_load_mcp_tools():
    """Fixture providing a mocked load_mcp_tools function"""
    with patch("generic_rag.mcp.ai_search_tool.load_mcp_tools") as mock:
        yield mock


@pytest.fixture
def ai_search_module():
    """Fixture providing the ai_search_tool module for testing"""
    import generic_rag.mcp.ai_search_tool as module

    return module


@pytest.fixture
def full_mcp_setup(mock_stdio_client, mock_client_session, mock_load_mcp_tools, mock_tools):
    """Fixture providing a complete MCP setup with all necessary mocks"""
    # Configure load_mcp_tools to return mock_tools
    mock_load_mcp_tools.return_value = mock_tools

    return {
        "stdio_client": mock_stdio_client,
        "client_session": mock_client_session,
        "load_mcp_tools": mock_load_mcp_tools,
        "tools": mock_tools,
    }


class TestAISearchTool:
    """Test suite for AI search tool functionality"""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self, ai_search_module):
        """Setup and teardown for each test"""
        # Reset global state before each test
        ai_search_module._tools = None
        ai_search_module._client_session = None
        ai_search_module._stdio_client_context = None
        yield
        # Cleanup after each test
        ai_search_module._tools = None
        ai_search_module._client_session = None
        ai_search_module._stdio_client_context = None

    async def test_server_params_configuration(self):
        """Test that SERVER_PARAMS is configured correctly"""
        assert SERVER_PARAMS.command == "npx"
        assert SERVER_PARAMS.args == ["-y", "@azure/mcp@latest", "server", "start", "--namespace", "search"]
        assert SERVER_PARAMS.env is None

    async def test_initialize_ai_search_client_success(self, full_mcp_setup):
        """Test successful initialization of AI search client"""
        mocks = full_mcp_setup

        # Call the function
        result = await initialize_ai_search_client()

        # Assertions
        assert result == mocks["tools"]
        mocks["stdio_client"].assert_called_once_with(SERVER_PARAMS)
        mocks["stdio_client"].return_value.__aenter__.assert_called_once()
        mocks["client_session"].assert_called_once()
        mocks["client_session"].return_value.__aenter__.assert_called_once()
        mocks["client_session"].return_value.initialize.assert_called_once()
        mocks["load_mcp_tools"].assert_called_once_with(mocks["client_session"].return_value)

    async def test_initialize_ai_search_client_already_initialized(self, mock_stdio_client):
        """Test that initialization returns existing tools if already initialized"""
        # Set up existing tools
        existing_tools = [MagicMock(name="existing_tool")]

        with patch("generic_rag.mcp.ai_search_tool._tools", existing_tools):
            result = await initialize_ai_search_client()

            # Should return existing tools without calling stdio_client
            assert result == existing_tools
            mock_stdio_client.assert_not_called()

    async def test_initialize_ai_search_client_failure(self, mock_stdio_client):
        """Test initialization failure and cleanup"""
        with patch("generic_rag.mcp.ai_search_tool.cleanup_ai_search_client") as mock_cleanup:
            # Make stdio_client raise an exception
            mock_stdio_client.side_effect = Exception("Connection failed")

            # Call should raise the exception
            with pytest.raises(Exception, match="Connection failed"):
                await initialize_ai_search_client()

            # Cleanup should be called
            mock_cleanup.assert_called_once()

    async def test_cleanup_ai_search_client_full_cleanup(self, ai_search_module):
        """Test cleanup when both session and stdio context exist"""
        with (
            patch("generic_rag.mcp.ai_search_tool._client_session") as mock_session,
            patch("generic_rag.mcp.ai_search_tool._stdio_client_context") as mock_stdio_context,
        ):
            # Setup mocks
            mock_session_obj = AsyncMock()
            mock_session.__bool__ = lambda: True
            mock_session_obj.__aexit__ = AsyncMock()

            mock_stdio_obj = AsyncMock()
            mock_stdio_context.__bool__ = lambda: True
            mock_stdio_obj.__aexit__ = AsyncMock()

            # Patch the module globals directly
            ai_search_module._client_session = mock_session_obj
            ai_search_module._stdio_client_context = mock_stdio_obj
            ai_search_module._tools = ["some_tools"]

            await cleanup_ai_search_client()

            # Verify cleanup calls
            mock_session_obj.__aexit__.assert_called_once_with(None, None, None)
            mock_stdio_obj.__aexit__.assert_called_once_with(None, None, None)

            # Verify global state reset
            assert ai_search_module._client_session is None
            assert ai_search_module._stdio_client_context is None
            assert ai_search_module._tools is None

    async def test_cleanup_ai_search_client_with_exceptions(self, ai_search_module):
        """Test cleanup handles exceptions gracefully"""
        # Create mock objects that raise exceptions
        mock_session = AsyncMock()
        mock_session.__aexit__.side_effect = Exception("Session cleanup error")

        mock_stdio_context = AsyncMock()
        mock_stdio_context.__aexit__.side_effect = Exception("Stdio cleanup error")

        ai_search_module._client_session = mock_session
        ai_search_module._stdio_client_context = mock_stdio_context
        ai_search_module._tools = ["some_tools"]

        # Should not raise exceptions
        await cleanup_ai_search_client()

        # Verify global state is still reset despite exceptions
        assert ai_search_module._client_session is None
        assert ai_search_module._stdio_client_context is None
        assert ai_search_module._tools is None

    async def test_cleanup_ai_search_client_partial_state(self, ai_search_module):
        """Test cleanup when only some components exist"""
        # Only set session, no stdio context
        mock_session = AsyncMock()
        ai_search_module._client_session = mock_session
        ai_search_module._stdio_client_context = None
        ai_search_module._tools = ["some_tools"]

        await cleanup_ai_search_client()

        # Verify session cleanup was called
        mock_session.__aexit__.assert_called_once_with(None, None, None)

        # Verify state reset
        assert ai_search_module._client_session is None
        assert ai_search_module._stdio_client_context is None
        assert ai_search_module._tools is None

    async def test_load_ai_search_tools_delegates_to_initialize(self):
        """Test that load_ai_search_tools delegates to initialize_ai_search_client"""
        mock_tools = [MagicMock(name="test_tool")]

        with patch(
            "generic_rag.mcp.ai_search_tool.initialize_ai_search_client", return_value=mock_tools
        ) as mock_initialize:
            result = await load_ai_search_tools()

            assert result == mock_tools
            mock_initialize.assert_called_once()

    async def test_concurrent_initialization(self, mock_stdio_client, mock_client_session, mock_load_mcp_tools):
        """Test that concurrent calls to initialize don't cause issues"""
        call_count = 0

        async def mock_initialize_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Simulate some async work
            await asyncio.sleep(0.01)
            tools = [MagicMock(name=f"tool_{call_count}")]
            return tools

        mock_load_mcp_tools.side_effect = mock_initialize_side_effect

        # Make multiple concurrent calls
        tasks = [initialize_ai_search_client() for _ in range(3)]
        results = await asyncio.gather(*tasks)

        # Should only initialize once due to the lock
        assert call_count == 1
        # All results should be the same
        assert all(result == results[0] for result in results)

    async def test_logging_during_initialization(self, caplog, full_mcp_setup):
        """Test that appropriate logging occurs during initialization"""
        with caplog.at_level(logging.DEBUG):
            await initialize_ai_search_client()

            # Check for expected log messages
            log_messages = [record.message for record in caplog.records]
            assert any("Starting MCP client initialization" in msg for msg in log_messages)
            assert any("MCP client initialized successfully" in msg for msg in log_messages)
            assert any("Available tools:" in msg for msg in log_messages)

    async def test_error_logging_during_initialization(self, caplog, mock_stdio_client):
        """Test that errors are logged during initialization failure"""
        with caplog.at_level(logging.ERROR):
            with patch("generic_rag.mcp.ai_search_tool.cleanup_ai_search_client"):
                mock_stdio_client.side_effect = Exception("Test error")

                with pytest.raises(Exception):
                    await initialize_ai_search_client()

                # Check for error log
                assert any(
                    "Failed to initialize AI search client: Test error" in record.message
                    for record in caplog.records
                    if record.levelno == logging.ERROR
                )
