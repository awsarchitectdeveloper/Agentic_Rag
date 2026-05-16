import logging
import os
import sys
import asyncio
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from langchain_chroma import Chroma

from generic_rag.backend.models import get_chat_model, get_compression_model, get_embedding_model
from generic_rag.graphs.cond_ret_gen import CondRetGenLangGraph
from generic_rag.graphs.ret_gen import RetGenLangGraph
from generic_rag.parsers.config import AppSettings, load_settings

from evaluation.llm_judge import evaluate_answer
from evaluation.logger import log_evaluation_to_blob, read_questions_csv_from_blob, write_results_to_blob

load_dotenv()

logger = logging.getLogger("sogeti-rag")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE_PATH = PROJECT_ROOT / "config.yaml"

print(f"Resolved PROJECT_ROOT: {PROJECT_ROOT}")
print(f"Looking for config at: {CONFIG_FILE_PATH}")
print(f"Azure account URL is: {os.environ['AZURE_STORAGE_ACCOUNT_URL']}")

try:
    settings: AppSettings = load_settings(CONFIG_FILE_PATH)
except (FileNotFoundError, Exception) as e:
    logger.error(f"Failed to load configuration from {CONFIG_FILE_PATH}: {e}")
    sys.exit(1)

embedding_function = get_embedding_model(settings)
chat_function = get_chat_model(settings)

vector_store = Chroma(
    collection_name="generic_rag",
    embedding_function=embedding_function,
    persist_directory=str(settings.chroma_db.location),
)

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "If the question is in Dutch, answer in Dutch. If the question is in English, answer in English. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know."
)

if settings.use_conditional_graph:
    graph = CondRetGenLangGraph(
        vector_store=vector_store,
        chat_model=chat_function,
        embedding_model=embedding_function,
        system_prompt=system_prompt,
    )
else:
    graph = RetGenLangGraph(
        vector_store=vector_store,
        chat_model=chat_function,
        embedding_model=embedding_function,
        system_prompt=system_prompt,
        compression_model=(
            get_compression_model("BAAI/bge-reranker-base", vector_store) if settings.use_reranker else None
        ),
    )


async def run_blob_batch_evaluation():
    try:
        df = read_questions_csv_from_blob()
    except Exception as e:
        logger.error(f"Failed to read questions CSV from Blob: {e}")
        return

    if df.empty:
        logger.warning("Questions CSV is empty. Nothing to evaluate.")
        return

    run_id = datetime.utcnow().strftime("run_%Y%m%dT%H%M%SZ")
    logger.info(f"[Batch:{run_id}] Starting batch evaluation for {len(df)} questions")

    results = []

    for _, row in df.iterrows():
        question = str(row.get("question", "")).strip()
        if not question:
            continue

        try:
            tokens = []
            async for token in graph.stream(question, config={"configurable": {"thread_id": "batch"}}):
                tokens.append(token)
            answer = "".join(tokens)

            context = ""
            try:
                if hasattr(graph, "get_last_context"):
                    chunks = graph.get_last_context()
                    context = "\n\n".join(getattr(c, "page_content", c) for c in chunks)
                elif hasattr(graph, "last_retrieved_docs"):
                    chunks = getattr(graph, "last_retrieved_docs", [])
                    context = "\n\n".join(getattr(c, "page_content", str(c)) for c in chunks)
            except Exception:
                context = ""

            judge = await evaluate_answer(question, answer, chat_model=chat_function, context=context)

            results.append(
                {
                    "question": question,
                    "generated_answer": answer,
                    "llm_score": judge["score"],
                    "llm_explanation": judge["explanation"],
                    "criteria": judge.get("criteria", {}),
                    "raw_judge": judge.get("raw", ""),
                }
            )

            try:
                log_evaluation_to_blob(
                    question=question, answer=answer, score=str(judge["score"]), explanation=judge["explanation"]
                )
            except Exception as log_e:
                logger.warning(f"[Batch:{run_id}] Append log failed: {log_e}")

        except Exception as e:
            logger.error(f"[Batch:{run_id}] Error for question '{question}': {e}")
            results.append(
                {
                    "question": question,
                    "generated_answer": "",
                    "llm_score": "",
                    "llm_explanation": f"ERROR: {e}",
                    "criteria": "",
                    "raw_judge": "",
                }
            )

    out_df = pd.DataFrame(
        results, columns=["question", "generated_answer", "llm_score", "llm_explanation", "criteria", "raw_judge"]
    )

    try:
        blob_path = write_results_to_blob(out_df, run_id)
        logger.info(f"[Batch:{run_id}] Completed. Results saved to {blob_path}")
    except Exception as e:
        logger.error(f"[Batch:{run_id}] Failed to write results CSV to Blob: {e}")


if __name__ == "__main__":
    if settings.evaluation_enabled:
        asyncio.run(run_blob_batch_evaluation())
    else:
        logger.info("Evaluation is disabled in config. Nothing to run.")
