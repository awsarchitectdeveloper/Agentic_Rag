import io
import os
import csv
import pandas as pd
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.core.exceptions import ResourceNotFoundError

load_dotenv()
ACCOUNT_URL = os.getenv("AZURE_STORAGE_ACCOUNT_URL")
print(ACCOUNT_URL)

INPUT_CONTAINER_NAME = os.getenv("INPUT_CONTAINER_NAME", "aitrial")
INPUT_BLOB_NAME = os.getenv("INPUT_BLOB_NAME", "questions.csv")

OUTPUT_CONTAINER_NAME = os.getenv("OUTPUT_CONTAINER_NAME", "aitrial")
OUTPUT_BLOB_PREFIX = os.getenv("OUTPUT_BLOB_PREFIX", "batch_eval")

EVAL_TO_BLOB = os.getenv("EVAL_TO_BLOB", "false").lower() == "true"
EVAL_LOG_CONTAINER_NAME = os.getenv("EVAL_LOG_CONTAINER_NAME", OUTPUT_CONTAINER_NAME)
EVAL_LOG_BLOB_NAME = os.getenv("EVAL_LOG_BLOB_NAME", "evaluations_log.csv")


def _get_blob_service() -> BlobServiceClient:
    if not ACCOUNT_URL:
        raise RuntimeError("AZURE_STORAGE_ACCOUNT_URL is not set.")
    return BlobServiceClient(account_url=ACCOUNT_URL, credential=DefaultAzureCredential())


def _ensure_container(container_name: str):
    bsc = _get_blob_service()
    cc = bsc.get_container_client(container_name)
    try:
        cc.get_container_properties()
    except ResourceNotFoundError:
        cc.create_container()
    return cc


def read_questions_csv_from_blob() -> pd.DataFrame:
    """
    Reads a CSV with at least a 'question' column from Azure Blob.
    """
    cc = _ensure_container(INPUT_CONTAINER_NAME)
    bc = cc.get_blob_client(INPUT_BLOB_NAME)
    data = bc.download_blob().readall()
    df = pd.read_csv(io.BytesIO(data))
    if "question" not in df.columns:
        raise ValueError("CSV must contain a 'question' column.")
    return df


def write_results_to_blob(df: pd.DataFrame, run_id: str) -> str:
    """
    Writes the batch results CSV to Azure Blob: <OUTPUT_BLOB_PREFIX>_<run_id>.csv
    Returns the 'container/blobname' path string.
    """
    cc = _ensure_container(OUTPUT_CONTAINER_NAME)
    out_blob_name = f"{OUTPUT_BLOB_PREFIX}_{run_id}.csv"
    bc = cc.get_blob_client(out_blob_name)

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    bc.upload_blob(
        buf.getvalue().encode("utf-8"), overwrite=True, content_settings=ContentSettings(content_type="text/csv")
    )
    return f"{OUTPUT_CONTAINER_NAME}/{out_blob_name}"


def _ensure_append_blob(container_name: str, blob_name: str):
    cc = _ensure_container(container_name)
    bc = cc.get_blob_client(blob_name)
    try:
        bc.get_blob_properties()
    except ResourceNotFoundError:
        bc.create_append_blob()
    return bc


def _sanitize_for_csv(cell: str) -> str:
    if cell and isinstance(cell, str) and cell[0] in ("=", "+", "-", "@"):
        return "'" + cell
    return cell


def log_evaluation_to_blob(
    question: str,
    answer: str,
    score: str,
    explanation: str,
    criteria: Optional[Dict[str, Any]] = None,
    raw: Optional[str] = None,
    run_id: Optional[str] = None,
) -> None:
    if not EVAL_TO_BLOB:
        return

    bc = _ensure_append_blob(EVAL_LOG_CONTAINER_NAME, EVAL_LOG_BLOB_NAME)

    try:
        size = bc.get_blob_properties().size
    except ResourceNotFoundError:
        size = 0

    buf = io.StringIO()
    writer = csv.writer(buf)

    if size == 0:
        writer.writerow(
            ["run_id", "question", "generated_answer", "llm_score", "llm_explanation", "criteria_json", "raw_judge"]
        )

    row = [
        _sanitize_for_csv(run_id or ""),
        _sanitize_for_csv(question),
        _sanitize_for_csv(answer),
        _sanitize_for_csv(str(score)),
        _sanitize_for_csv(explanation),
        _sanitize_for_csv(str(criteria) if criteria else ""),
        _sanitize_for_csv(raw or ""),
    ]
    writer.writerow(row)

    data = buf.getvalue().encode("utf-8")
    bc.append_block(data, length=len(data), content_settings=ContentSettings(content_type="text/csv"))
