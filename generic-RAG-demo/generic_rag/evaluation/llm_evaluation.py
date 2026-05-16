"""Full LLM evaluation script - evaluates both retriever and generation."""

import copy
import pandas as pd
import mlflow
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from generic_rag.parsers.config import load_settings
from generic_rag.backend.models import get_chat_model, get_embedding_model
from generic_rag.backend.vectordb import get_vector_store
from generic_rag.evaluation.utils import load_questions
from generic_rag.evaluation.qa_generator import build_qa_chain, generate_answers
from generic_rag.evaluation.schemas import create_judge_schema, load_or_extract_config
from generic_rag.evaluation.judge import llm_judge, build_judge_prompt

load_dotenv()


def run_full_evaluation(config_path: Path = None) -> pd.DataFrame:
    """Run complete evaluation (retriever + generation + judging)."""
    default_config_path = Path(__file__).resolve().parents[2] / "config.yaml"
    raw_config = load_or_extract_config(config_path=config_path, default_path=default_config_path)

    # Load configuration
    actual_config_path = config_path or default_config_path
    settings = load_settings(actual_config_path)
    eval_cfg = raw_config["evaluation"]
    questions_path = Path(eval_cfg["questions_file"])
    results_path = Path(eval_cfg.get("results_file", "generic_rag/evaluation/results/llm_eval_results.csv"))
    fetch_k = eval_cfg["fetch_k"]

    # Initialize models
    chat_model = get_chat_model(settings)
    embed = get_embedding_model(settings)
    vectorstore = get_vector_store(settings, embedding_function=embed)
    retriever = vectorstore.as_retriever(search_kwargs={"k": fetch_k})

    # Load questions
    eval_df = load_questions(questions_path)
    print(f"Loaded {len(eval_df)} evaluation questions")

    # Generate answers
    qa_chain = build_qa_chain(chat_model, retriever, config=copy.deepcopy(raw_config))
    eval_df = generate_answers(eval_df, qa_chain)
    print("Generated answers for all questions")

    # Judge answers with dynamic schema
    DynamicJudgeSchema = create_judge_schema(config=copy.deepcopy(raw_config))
    parser = JsonOutputParser(pydantic_object=DynamicJudgeSchema)
    judge_prompt = build_judge_prompt(config=copy.deepcopy(raw_config))
    print("Running LLM judge evaluation...")
    metrics = eval_cfg["metrics"]["llm_judge"]
    scores = eval_df.apply(
        lambda row: llm_judge(row, chat_model, retriever, parser, judge_prompt, metrics=metrics), axis=1
    )
    scores_df = pd.DataFrame(scores.tolist())
    eval_df = pd.concat([eval_df, scores_df], axis=1)

    # Log results with configurable metrics
    mlflow.set_tracking_uri(eval_cfg["mlflow_tracking_uri"])
    with mlflow.start_run(run_name="llm_judge_eval"):
        # Get configured metrics instead of hardcoded ones
        configured_metrics = eval_cfg["metrics"]["llm_judge"]
        available_metrics = [col for col in configured_metrics if col in eval_df.columns]

        for metric in available_metrics:
            avg_score = eval_df[metric].mean()
            mlflow.log_metric(metric, avg_score)
            print(f"Average {metric}: {avg_score:.2f}")

        results_path.parent.mkdir(parents=True, exist_ok=True)
        eval_df.to_csv(results_path, index=False)
        mlflow.log_artifact(str(results_path))

    print(f"✓ LLM-based evaluation completed. Results saved to {results_path}")
    return eval_df


def main():
    """Main entry point for full LLM evaluation."""
    result_df = run_full_evaluation()
    print("\n--- Summary Metrics ---")

    score_columns = [
        col for col in result_df.columns if col in ["correctness", "relevance", "conciseness", "faithfulness"]
    ]
    if score_columns:
        for col in score_columns:
            print(f"{col}: {result_df[col].mean():.2f}")
    else:
        print("No score metrics found in results")


if __name__ == "__main__":
    main()
