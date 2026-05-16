from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
import logging
import asyncio
from typing import Optional, List

SERVER_PARAMS = StdioServerParameters(
    command="npx", args=["-y", "@azure/mcp@latest", "server", "start", "--namespace", "search"], env=None
)

# Cached tools and session for singleton pattern
_tools: Optional[List] = None
_client_session: Optional[ClientSession] = None
_stdio_client_context = None
_init_lock = asyncio.Lock()


async def initialize_ai_search_client():
    """
    Initialize and keep the MCP client connection alive (singleton pattern)
    """
    global _client_session, _stdio_client_context, _tools

    # Simple double-check pattern with lock
    if _tools is not None:
        logging.debug("Returning existing AI search tools")
        return _tools

    async with _init_lock:
        # Double-check inside lock
        if _tools is not None:
            logging.debug("Tools were initialized by another thread")
            return _tools

        logging.debug("Starting MCP client initialization...")

        try:
            # Create and store context managers for persistent connection
            _stdio_client_context = stdio_client(SERVER_PARAMS)
            read, write = await _stdio_client_context.__aenter__()

            # Create and initialize session
            _client_session = ClientSession(read, write)
            await _client_session.__aenter__()
            await _client_session.initialize()

            # Load tools with the persistent session
            _tools = await load_mcp_tools(_client_session)

            logging.debug("Available tools:")
            for tool in _tools:
                logging.debug(f"- {tool.name}: {tool.description}")

            logging.debug(f"MCP client initialized successfully with {len(_tools)} tools")
            return _tools

        except Exception as e:
            logging.error(f"Failed to initialize AI search client: {e}")
            await cleanup_ai_search_client()
            raise


async def cleanup_ai_search_client():
    """
    Clean up the MCP client connection
    """
    global _client_session, _stdio_client_context, _tools

    logging.debug("Cleaning up MCP client...")

    # Properly close session and stdio client
    if _client_session:
        try:
            await _client_session.__aexit__(None, None, None)
        except Exception as e:
            logging.debug(f"Error closing client session: {e}")

    if _stdio_client_context:
        try:
            await _stdio_client_context.__aexit__(None, None, None)
        except Exception as e:
            logging.debug(f"Error closing stdio client: {e}")

    # Reset all global state
    _client_session = None
    _stdio_client_context = None
    _tools = None

    logging.debug("MCP client cleanup completed")


async def load_ai_search_tools() -> List:
    """
    Returns the AI search tools, initializing the connection if needed.
    """
    return await initialize_ai_search_client()
