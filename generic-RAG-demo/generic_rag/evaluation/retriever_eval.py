import mlflow
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from generic_rag.parsers.config import load_settings
from generic_rag.backend.models import get_embedding_model
from generic_rag.backend.vectordb import get_vector_store
from generic_rag.evaluation.utils import load_questions
from generic_rag.evaluation.schemas import load_or_extract_config


load_dotenv()
CONFIG_FILE_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


# Load questions once (path will be resolved inside evaluate_retriever_only)
def _normalize_filename(filename: str) -> str:
    """Normalize filename for robust comparison (lowercase, strip whitespace)."""
    if not filename:
        return ""
    return str(filename).lower().strip()


def _normalize_gt(eval_df: pd.DataFrame) -> pd.DataFrame:
    eval_df["GT_source"] = eval_df["GT_source"].apply(lambda x: x if isinstance(x, list) else [])
    # Normalize GT source filenames for consistent comparison
    eval_df["GT_source"] = eval_df["GT_source"].apply(lambda sources: [_normalize_filename(src) for src in sources])
    return eval_df


def _extract_doc_identifier(doc) -> str:
    """Return a robust identifier for a retrieved document using metadata fallbacks."""
    md = getattr(doc, "metadata", {}) or {}
    # Prefer full source path, filename, then filename field, then id, then truncated content
    source = md.get("source") or md.get("filename") or md.get("file") or md.get("id")
    if source:
        # Normalize filename for consistent comparison with GT sources
        return _normalize_filename(str(Path(source).name))
    # Fallback to content-based fingerprint
    content = getattr(doc, "page_content", "") or getattr(doc, "content", "")
    return _normalize_filename(content[:120])


def evaluate_retriever_only(config_path: Path = None, fetch_k: int = None, retriever=None) -> pd.DataFrame:
    """Evaluate only the retriever component."""
    config = load_or_extract_config(config_path=config_path, default_path=CONFIG_FILE_PATH)
    eval_cfg = config["evaluation"]

    # Build dynamic metrics from config
    retriever_metrics_cfg = eval_cfg["metrics"]["retriever"]
    extra_metrics = []

    # Add precision@k metrics
    for k in retriever_metrics_cfg.get("precision_at_k", []):
        extra_metrics.append(mlflow.metrics.precision_at_k(k))

    # Add recall@k metrics
    for k in retriever_metrics_cfg.get("recall_at_k", []):
        extra_metrics.append(mlflow.metrics.recall_at_k(k))

    # Add ndcg@k metrics
    for k in retriever_metrics_cfg.get("ndcg_at_k", []):
        extra_metrics.append(mlflow.metrics.ndcg_at_k(k))

    # MLflow evaluation
    mlflow.set_tracking_uri(eval_cfg["mlflow_tracking_uri"])
    # Use provided retriever if supplied; otherwise build one from settings
    if retriever is None:
        actual_config_path = config_path or CONFIG_FILE_PATH
        settings = load_settings(actual_config_path)
        embed = get_embedding_model(settings)
        vectorstore = get_vector_store(settings, embedding_function=embed)
        if fetch_k is None:
            fetch_k = eval_cfg.get("fetch_k")
        retriever = vectorstore.as_retriever(search_kwargs={"k": fetch_k})

    # Load and prepare question dataframe
    questions_path = Path(eval_cfg["questions_file"])
    eval_df = load_questions(questions_path)
    eval_df = _normalize_gt(eval_df)
    has_GT_source_df = eval_df[eval_df["GT_source"].apply(lambda x: len(x) > 0)]

    # Define model function used by mlflow.evaluate
    def retriever_model_function(question_df: pd.DataFrame) -> pd.Series:
        def retrieve_docs(question: str) -> list[str]:
            docs = retriever.invoke(question)
            return [_extract_doc_identifier(d) for d in docs]

        return question_df["question"].apply(retrieve_docs)

    results_dir = Path(eval_cfg["results_dir"])

    with mlflow.start_run(run_name="retriever_eval"):
        result = mlflow.evaluate(
            model=retriever_model_function,
            data=has_GT_source_df,
            model_type="retriever",
            targets="GT_source",
            evaluators="default",
            extra_metrics=extra_metrics,
        )

        eval_results_table = result.tables["eval_results_table"]
        results_dir.mkdir(parents=True, exist_ok=True)
        eval_results_table.to_csv(results_dir / "retriever_eval.csv", index=False)
        mlflow.log_artifact(str(results_dir / "retriever_eval.csv"))

    print("✓ Retriever evaluation completed.")
    return has_GT_source_df


def main():
    """Main entry point for retriever evaluation."""
    result_df = evaluate_retriever_only()
    print("\nRetriever Evaluation Summary:")
    print(f"Evaluated {len(result_df)} queries with ground truth sources.")


if __name__ == "__main__":
    main()
