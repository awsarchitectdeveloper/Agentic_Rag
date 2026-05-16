import logging
from pathlib import Path
from typing import Any, AsyncGenerator

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.retrievers import BaseRetriever
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import List, TypedDict

logger = logging.getLogger("sogeti-rag")


class State(TypedDict):
    question: str
    context: List[Document]
    answer: BaseMessage


class RetGenLangGraph:
    def __init__(
        self,
        vector_store: Chroma,
        chat_model: BaseChatModel,
        embedding_model: Embeddings,
        system_prompt: str,
        compression_model: BaseRetriever | None = None,
    ):
        self.vector_store = vector_store
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.system_prompt = system_prompt
        self.compression_model = compression_model

        memory = MemorySaver()
        graph_builder = StateGraph(State).add_sequence([self._retrieve, self._generate])
        graph_builder.add_edge(START, "_retrieve")
        graph_builder.add_edge("_retrieve", "_generate")
        graph_builder.add_edge("_generate", END)

        self.graph = graph_builder.compile(memory)
        self.last_retrieved_docs = []

    async def stream(self, message: str, config: RunnableConfig | None = None) -> AsyncGenerator[Any, Any]:
        async for response, _ in self.graph.astream({"question": message}, stream_mode="messages", config=config):
            yield response.content

    def _retrieve(self, state: State) -> dict[str, list]:
        logger.debug(f"querying VS for: {state['question']}")
        if self.compression_model:
            self.last_retrieved_docs = self.compression_model.invoke(state["question"])
        else:
            self.last_retrieved_docs = self.vector_store.similarity_search(state["question"])
        return {"context": self.last_retrieved_docs}

    def _generate(self, state: State) -> dict[str, list]:
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        system_message_content = self.system_prompt + f"\n\n{docs_content}"

        prompt = [SystemMessage(system_message_content)] + [state["question"]]

        response = self.chat_model.invoke(prompt)
        return {"answer": [response]}

    def get_last_pdf_sources(self) -> dict[str, list[int]]:
        """
        Method that retrieves the PDF sources used during the last invoke.
        """
        pdf_sources = {}

        if not self.last_retrieved_docs:
            return pdf_sources

        for doc in self.last_retrieved_docs:
            if "source" in doc.metadata and Path(doc.metadata["source"]).suffix.lower() == ".pdf":
                source = doc.metadata["source"]
            else:
                continue

            if source not in pdf_sources:
                pdf_sources[source] = set()

            # The page numbers are in the `page_numer` and `page` fields.
            if "page_number" in doc.metadata:
                pdf_sources[source].add(doc.metadata["page_number"])

            if "page" in doc.metadata:
                pdf_sources[source].add(doc.metadata["page"])

            if len(pdf_sources[source]) == 0:
                logger.warning(f"PDF source {source} has no page number. Please check the metadata of the document.")

        return pdf_sources

    def get_last_web_sources(self) -> set:
        """
        Method that retrieves the web sources used during the last invoke.
        """
        web_sources = set()

        if not self.last_retrieved_docs:
            return web_sources

        for doc in self.last_retrieved_docs:
            if "filetype" in doc.metadata and doc.metadata["filetype"] == "web":
                web_sources.add(doc.metadata["source"])

        return web_sources
