from pathlib import Path
import json
import pandas as pd


def load_questions(path: Path) -> pd.DataFrame:
    """Load evaluation questions from structured files (JSON, CSV, Excel)."""
    # Allowed file types for questions
    allowed_extensions = [".json", ".csv", ".xlsx"]

    # Validate extension
    if path.suffix not in allowed_extensions:
        raise ValueError(f"Invalid file type: {path.suffix}. Supported for questions: {', '.join(allowed_extensions)}")

    # Load based on type
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)

    elif path.suffix == ".csv":
        return pd.read_csv(path)

    elif path.suffix == ".xlsx":
        return pd.read_excel(path)


def format_docs(docs):
    """Format retrieved documents into a single string for evaluation."""
    if not docs:
        return "No documents retrieved"

    formatted = []
    for i, doc in enumerate(docs, 1):
        content = doc.page_content if hasattr(doc, "page_content") else str(doc)
        # Truncate very long documents
        if len(content) > 500:
            content = content[:500] + "..."
        formatted.append(f"Document {i}:\n{content}")

    return "\n\n".join(formatted)
