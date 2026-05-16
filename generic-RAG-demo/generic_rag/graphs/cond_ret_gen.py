import ast
import logging
import re
from pathlib import Path
from typing import Any, AsyncGenerator

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import InjectedStore, ToolNode, tools_condition
from typing_extensions import Annotated

logger = logging.getLogger("sogeti-rag")


class CondRetGenLangGraph:
    def __init__(
        self, vector_store: Chroma, chat_model: BaseChatModel, embedding_model: Embeddings, system_prompt: str
    ):
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.system_prompt = system_prompt

        memory = MemorySaver()
        tools = ToolNode([self._retrieve])

        graph_builder = StateGraph(MessagesState)
        graph_builder.add_node(self._query_or_respond)
        graph_builder.add_node(tools)
        graph_builder.add_node(self._generate)

        graph_builder.set_entry_point("_query_or_respond")
        graph_builder.add_conditional_edges("_query_or_respond", tools_condition, {END: END, "tools": "tools"})
        graph_builder.add_edge("tools", "_generate")
        graph_builder.add_edge("_generate", END)

        self.graph = graph_builder.compile(checkpointer=memory, store=vector_store)

        self.file_path_pattern = r"'file_path'\s*:\s*'((?:[^'\\]|\\.)*)'"
        self.source_pattern = r"'source'\s*:\s*'((?:[^'\\]|\\.)*)'"
        self.page_pattern = r"'page'\s*:\s*(\d+)"
        self.pattern = r"Source:\s*(\{.*?\})"

        self.last_retrieved_docs = {}
        self.last_retrieved_sources = set()

    async def stream(self, message: str, config: RunnableConfig | None = None) -> AsyncGenerator[Any, Any]:
        async for llm_response, metadata in self.graph.astream(
            {"messages": [{"role": "user", "content": message}]}, stream_mode="messages", config=config
        ):
            if llm_response.content and metadata["langgraph_node"] == "_generate":
                yield llm_response.content
            elif llm_response.name == "_retrieve":
                dictionary_strings = re.findall(
                    self.pattern, llm_response.content, re.DOTALL
                )  # Use re.DOTALL if dicts might span newlines
                for dict_str in dictionary_strings:
                    parsed_dict = ast.literal_eval(dict_str)
                    if "filetype" in parsed_dict and parsed_dict["filetype"] == "web":
                        self.last_retrieved_sources.add(parsed_dict["source"])
                    elif Path(parsed_dict["source"]).suffix == ".pdf":
                        if parsed_dict["source"] in self.last_retrieved_docs:
                            self.last_retrieved_docs[parsed_dict["source"]].add(parsed_dict["page"])
                        else:
                            self.last_retrieved_docs[parsed_dict["source"]] = {parsed_dict["page"]}

    @tool(response_format="content_and_artifact")
    def _retrieve(
        query: str, full_user_content: str, vector_store: Annotated[Any, InjectedStore()]
    ) -> tuple[str, list[Document]]:
        """
        Retrieve information related to a query and user content.
        """
        # This method is used as a tool in the graph.
        # It's doc-string is used for the pydantic model, please consider doc-string text carefully.
        # Furthermore, it can not and should not have the `self` parameter.
        # If you want to pass on state, please refer to:
        # https://python.langchain.com/docs/concepts/tools/#special-type-annotations
        logger.debug(f"query: {query}")
        logger.debug(f"user content: {full_user_content}")

        retrieved_docs = []
        retrieved_docs = vector_store.similarity_search(query, k=4)
        retrieved_docs = vector_store.similarity_search(full_user_content, k=4)
        serialized = "\n\n".join((f"Source: {doc.metadata}\nContent: {doc.page_content}") for doc in retrieved_docs)

        return serialized, retrieved_docs

    def _query_or_respond(self, state: MessagesState) -> dict[str, BaseMessage]:
        """Generate tool call for retrieval or respond."""
        # Reset last retrieved docs
        self.last_retrieved_docs = {}
        self.last_retrieved_sources = set()

        llm_with_tools = self.chat_model.bind_tools([self._retrieve])
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def _generate(self, state: MessagesState) -> dict[str, BaseMessage]:
        """Generate answer."""
        # get generated ToolMessages
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        # format into prompt
        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        system_message_content = self.system_prompt + f"\n\n{docs_content}"
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system") or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(system_message_content)] + conversation_messages

        # run
        response = self.chat_model.invoke(prompt)
        return {"messages": [response]}
