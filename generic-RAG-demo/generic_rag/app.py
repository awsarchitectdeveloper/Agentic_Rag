import logging
import sys
import asyncio
from pathlib import Path
import fitz
import chainlit as cl

from generic_rag.backend.models import get_chat_model, get_compression_model, get_embedding_model, get_agent_model
from generic_rag.backend.vectordb import get_vector_store
from generic_rag.graphs.cond_ret_gen import CondRetGenLangGraph
from generic_rag.graphs.ret_gen import RetGenLangGraph
from generic_rag.parsers.config import AppSettings, load_settings
from generic_rag.backend.summarization import summarize_text

from langchain_core.messages import SystemMessage, HumanMessage
from generic_rag.backend.realtime import RealtimeSTT, TTS_model
from generic_rag.graphs.react_agent import ReactAgentMCP


logger = logging.getLogger("sogeti-rag")
logger.setLevel(logging.DEBUG)

# Suppress Azure HTTP logging
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE_PATH = PROJECT_ROOT / "config.yaml"

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "If the question is in Dutch, answer in Dutch. If the question is in English, answer in English. "
    "You MUST ONLY use the provided context to answer the question. "
    "Do NOT use any other knowledge outside the provided context. "
    "If the answer is not in the context, respond with exactly: 'I don't know.' "
    "Do NOT make up information. "
    "Do NOT reference any document unless it is explicitly mentioned in the provided context."
)

try:
    settings: AppSettings = load_settings(CONFIG_FILE_PATH)
except (FileNotFoundError, Exception):
    logger.error(f"Failed to load configuration from {CONFIG_FILE_PATH}. Exiting.")
    sys.exit(1)

embedding_function = get_embedding_model(settings)

chat_function = get_chat_model(settings)

vector_store = get_vector_store(settings=settings, embedding_function=embedding_function, collection_name="generic_rag")

# Initialize graph variable
graph = None
react_agent = None


async def initialize_graph_once():
    """Initialize the graph exactly once"""
    global graph, react_agent

    if graph is not None:
        logger.debug("Graph already initialized, skipping...")
        return graph

    logger.debug(f"Initializing graph with selection: {settings.graph_selection}")

    if settings.graph_selection == "conditional_graph":
        graph = CondRetGenLangGraph(
            vector_store=vector_store,
            chat_model=chat_function,
            embedding_model=embedding_function,
            system_prompt=system_prompt,
        )
    elif settings.graph_selection == "react_agent":
        if not settings.vector_db.type == "ai_search":
            logger.error("AI Search is not configured. Please set type under vector_db to 'ai_search' in your config.")
            sys.exit(1)
        agent_model = get_agent_model(settings)
        react_agent = ReactAgentMCP(agent_model=agent_model, system_prompt=None)
        graph = await react_agent.create_agent()
    elif settings.graph_selection == "ret_gen":
        graph = RetGenLangGraph(
            vector_store=vector_store,
            chat_model=chat_function,
            embedding_model=embedding_function,
            system_prompt=system_prompt,
            compression_model=(
                get_compression_model("BAAI/bge-reranker-base", vector_store) if settings.use_reranker else None
            ),
        )
    else:
        logger.error(
            f"Invalid graph selection: {settings.graph_selection}. Please choose 'ret_gen', 'conditional_graph', or 'react_agent'."
        )
        sys.exit(1)

    logger.debug("Graph initialization completed")
    return graph


# Initialize the graph at module level
_initialization_task = None


def get_or_create_initialization_task():
    global _initialization_task
    if _initialization_task is None:
        _initialization_task = asyncio.create_task(initialize_graph_once())
    return _initialization_task


@cl.on_chat_start
async def on_chat_start():
    # Ensure graph is initialized before chat starts
    await get_or_create_initialization_task()


@cl.on_chat_end
async def on_chat_end():
    # Cleanup resources if using react agent
    if isinstance(graph, ReactAgentMCP):
        await graph.cleanup()


@cl.on_message
async def on_message(message: cl.Message):
    config = {"configurable": {"thread_id": cl.user_session.get("id")}}

    if settings.use_summarization:
        # PDF Upload
        pdfs = [file for file in message.elements if "pdf" in file.mime]
        if pdfs:
            try:
                full_text = await process_pdf(pdfs)
            except Exception as e:
                await cl.Message(content=f"Failed to read PDF: {str(e)}").send()
                return

            # Summarization only works with PDF upload
            if await detect_summarization_intent(message.content, chat_function):
                summary = summarize_text(full_text, chat_function)
                await cl.Message(content=f"**Summary of document contents:**\n{summary}").send()
                return

    chainlit_response = cl.Message(content="")

    async for response in graph.stream(message.content, config=config):
        await chainlit_response.stream_token(response)

    if isinstance(graph, RetGenLangGraph):
        await add_sources(chainlit_response, graph.get_last_pdf_sources(), graph.get_last_web_sources())
    if isinstance(graph, CondRetGenLangGraph):
        await add_sources(chainlit_response, graph.last_retrieved_docs, graph.last_retrieved_sources)

    await chainlit_response.send()


async def add_sources(chainlit_response: cl.Message, pdf_sources: dict, web_sources: set | list) -> None:
    if len(pdf_sources) > 0:
        await chainlit_response.stream_token("\n\nThe following PDF source were consulted:\n")
        for source, page_numbers in pdf_sources.items():
            filename = Path(source).name
            await chainlit_response.stream_token(f"- {filename} on page(s): {sorted(page_numbers)}\n")
            chainlit_response.elements.append(
                cl.Pdf(name=filename, display="side", path=source, page=sorted(page_numbers)[0])
            )

    if len(web_sources) > 0:
        await chainlit_response.stream_token("\n\nThe following web sources were consulted:\n")
        for source in web_sources:
            await chainlit_response.stream_token(f"- {source}\n")


async def detect_summarization_intent(query: str, client) -> bool:
    messages = [
        SystemMessage(content="You are a system that detects user intent."),
        HumanMessage(
            content=f"Is the following query asking for a summary of a document or content? Answer only 'yes' or 'no'.\n\nQuery: {query}"
        ),
    ]
    response = client.generate([messages])
    answer = response.generations[0][0].text.strip().lower()
    return "yes" in answer


async def process_pdf(pdfs):
    doc = fitz.open(pdfs[0].path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return full_text


@cl.set_starters
async def set_starters():
    if settings.chainlit_starters is None:
        return

    starters = []
    for starter in settings.chainlit_starters:
        try:
            starters.append(cl.Starter(label=starter["label"], message=starter["message"]))
        except KeyError:
            logger.warning(
                "CHAINLIT_STARTERS environment is not a list with dictionaries containing 'label' and 'message' keys."
            )

    return starters


if settings.voice_chat:

    async def realtime_api(settings):
        realtime = RealtimeSTT(settings)
        tts_function = TTS_model(settings)
        cl.user_session.set("realtime", realtime)
        cl.user_session.set("tts_function", tts_function)
        cl.user_session.set("track_id", cl.user_session.get("id"))
        cl.user_session.set("is_speaking", False)

        # Ensure the lock is initialized once per session
        if not cl.user_session.get("input_lock"):
            cl.user_session.set("input_lock", asyncio.Lock())

        async def audio_transcription(event):
            # Make sure that new user input is send before the assistants response to the previous
            lock = cl.user_session.get("input_lock")

            if lock.locked() or cl.user_session.get("is_speaking"):
                await cl.Message(content="⏳ Still processing your previous input...").send()
                return

            async with lock:
                user_content = event["transcript"]
                await cl.Message(content=user_content, type="user_message", author="You").send()

                config = {"configurable": {"thread_id": cl.user_session.get("id")}}
                chainlit_response = cl.Message(content="")

                async for response in graph.stream(user_content, config=config):
                    await chainlit_response.stream_token(response)

                if isinstance(graph, RetGenLangGraph):
                    await add_sources(chainlit_response, graph.get_last_pdf_sources(), graph.get_last_web_sources())
                if isinstance(graph, CondRetGenLangGraph):
                    await add_sources(chainlit_response, graph.last_retrieved_docs, graph.last_retrieved_sources)

                await chainlit_response.send()

                cl.user_session.set("is_speaking", True)
                full_audio = await tts_function.create_audio(chainlit_response.content)

                await cl.context.emitter.send_audio_chunk(
                    cl.OutputAudioChunk(mimeType="pcm", data=full_audio, track=cl.user_session.get("track_id"))
                )
                # determine audio duration by dividing by sample rate * bytes per sample
                await asyncio.sleep(len(full_audio) / 48000)  # sleep during that time
                cl.user_session.set("is_speaking", False)

        realtime.on("server.conversation.item.input_audio_transcription.completed", audio_transcription)

    @cl.on_chat_start
    async def on_chat_start():
        await get_or_create_initialization_task()
        await realtime_api(settings.azure)

    @cl.on_audio_start
    async def on_audio_start():
        try:
            realtime: RealtimeSTT = cl.user_session.get("realtime")
            tts_function: TTS_model = cl.user_session.get("tts_function")
            if not realtime.is_connected():
                await realtime.connect()
            logger.info("Connected to realtime client")
            if not tts_function.is_connected():
                await tts_function.connect()
            logger.info("Loaded TTS function")
            return True
        except Exception as e:
            await cl.ErrorMessage(content=f"Failed to connect to realtime client or load tts function: {e}").send()
            return False

    @cl.on_audio_chunk
    async def on_audio_chunk(chunk: cl.InputAudioChunk):
        realtime: RealtimeSTT = cl.user_session.get("realtime")
        if realtime:
            if realtime.is_connected():
                await realtime.append_audio_chunk(chunk.data)
            else:
                logger.info("RealtimeAPI is not connected")

    @cl.on_audio_end
    async def on_audio_end():
        realtime: RealtimeSTT = cl.user_session.get("realtime")
        if realtime and realtime.is_connected():
            await realtime.disconnect()
            logger.info("Disconnected from realtime API")
        lock = cl.user_session.get("input_lock")
        if lock.locked():
            lock.release()
