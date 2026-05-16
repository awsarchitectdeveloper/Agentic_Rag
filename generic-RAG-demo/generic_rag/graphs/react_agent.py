from langgraph.prebuilt import create_react_agent
from generic_rag.parsers.config import AppSettings, load_settings
import logging
from generic_rag.mcp.ai_search_tool import load_ai_search_tools
from pathlib import Path
import sys
from langchain_core.language_models.chat_models import BaseChatModel
from typing import Any, AsyncGenerator


logger = logging.getLogger("sogeti-rag")
logger.setLevel(logging.DEBUG)
# Suppress Azure HTTP logging
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_FILE_PATH = PROJECT_ROOT / "config.yaml"
try:
    settings: AppSettings = load_settings(CONFIG_FILE_PATH)
except (FileNotFoundError, Exception):
    logger.error(f"Failed to load configuration from {CONFIG_FILE_PATH}. Exiting.")
    sys.exit(1)

service_name = settings.azure_ai_search.endpoint.split("/")[2].split(".")[0]

SYSTEM_PROMPT = f"""You are an assistant for question-answering tasks. If the question is in Dutch, answer in Dutch. If the question is in English, answer in English.
                You have access to Azure AI Search tools. To search for documents, use the search_index_query tool with these parameters:
                - service: "{service_name}"
                - index: "{settings.azure_ai_search.index_name}"
                - query: your search query text

                If you don't know the answer after searching, say that you don't know."""


class ReactAgentMCP:
    _instance_count = 0

    def __init__(self, agent_model: BaseChatModel, system_prompt: str, tools: list = None):
        ReactAgentMCP._instance_count += 1
        self.instance_id = ReactAgentMCP._instance_count
        logger.debug(f"Creating ReactAgentMCP instance #{self.instance_id}")

        self.agent_model = agent_model
        self.system_prompt = SYSTEM_PROMPT if system_prompt is None else system_prompt
        self.tools = tools
        self.agent = None

    async def create_agent(self):
        # Load tools if not provided
        if self.tools is None:
            logger.debug(f"Instance #{self.instance_id}: Loading AI search tools...")
            self.tools = await load_ai_search_tools()
            logger.debug(f"Instance #{self.instance_id}: Loaded {len(self.tools) if self.tools else 0} tools")

        # create langgraph react agent
        logger.debug(f"Instance #{self.instance_id}: Creating react agent with model: {type(self.agent_model)}")
        self.agent = create_react_agent(model=self.agent_model, tools=self.tools, prompt=self.system_prompt)
        logger.debug(f"Instance #{self.instance_id}: Agent created: {type(self.agent)}")
        return self

    async def stream(self, message: str, config: dict | None = None) -> AsyncGenerator[Any, Any]:
        """Simplified streaming approach - process and then stream result"""
        logger.debug(f"Using stream for message: {message}")

        try:
            # Get the full result first
            result = await self.agent.ainvoke({"messages": [{"role": "user", "content": message}]}, config=config)

            logger.debug(f"stream result: {type(result)} - {result}")

            # Extract and stream the final response (only AIMessage with content)
            if isinstance(result, dict) and "messages" in result:
                for msg in result["messages"]:
                    # Only stream AIMessage objects with content
                    if hasattr(msg, "content") and msg.content and msg.__class__.__name__ == "AIMessage":
                        logger.debug(f"Streaming AIMessage content: {msg.content[:100]}...")
                        # Stream the content in chunks for better UX
                        content = msg.content
                        chunk_size = 50  # Stream in 50-character chunks
                        for i in range(0, len(content), chunk_size):
                            chunk = content[i : i + chunk_size]
                            yield chunk
                    else:
                        logger.debug(f"Skipping non-AIMessage: {msg.__class__.__name__}")
        except Exception as e:
            logger.error(f"Error in stream: {e}")
            raise

    async def cleanup(self):
        """Clean up resources"""
        from generic_rag.mcp.ai_search_tool import cleanup_ai_search_client

        await cleanup_ai_search_client()
