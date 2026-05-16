#!/usr/bin/env python3
"""
Main entry point for running RAG evaluation.

This script provides a single entry point to run different types of evaluation:
- retriever_only: Evaluate only document retrieval
- full_eval: Evaluate retrieval + generation + judging

Usage:
    python run_evaluation.py --mode retriever_only
    python run_evaluation.py --mode full_eval
    python run_evaluation.py --mode full_eval --fetch_k 5
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from generic_rag.evaluation.retriever_eval import evaluate_retriever_only
from generic_rag.evaluation.llm_evaluation import run_full_evaluation


def main():
    """Main entry point with configurable evaluation modes."""
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument(
        "--mode",
        choices=["retriever_only", "full_eval"],
        default="full_eval",
        help="Evaluation mode: retriever_only or full_eval",
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to config file (default: config.yaml)")
    parser.add_argument("--fetch_k", type=int, default=None, help="Number of documents to retrieve (overrides config)")

    args = parser.parse_args()

    print(f" Running RAG evaluation in mode: {args.mode}")
    print("-" * 50)

    if args.mode == "retriever_only":
        print(" Evaluating retriever component only...")
        result_df = evaluate_retriever_only(config_path=args.config, fetch_k=args.fetch_k)
        print(f"Retriever evaluation completed with {len(result_df)} queries")

    elif args.mode == "full_eval":
        print(" Running full evaluation (retriever + generation + judging)...")
        result_df = run_full_evaluation(config_path=args.config)
        print(f"Full evaluation completed with {len(result_df)} questions")

    print("\n Evaluation finished successfully!")


if __name__ == "__main__":
    main()
