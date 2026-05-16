import json
import re
from typing import Dict, Any

JSON_INSTRUCTIONS = """\
You are an impartial evaluator for RAG answers.
Evaluate on a scale of 1 (worst) to 10 (best) based on:
- Relevance: Does the answer address the question asked?
- Correctness: Is it factually correct with respect to the given context?
- Completeness: Does it cover the key points needed to satisfy the query?

Penalties:
- If the answer contradicts the provided context, reduce heavily.
- If important context evidence is missing from the answer, reduce accordingly.
- If the model says "I don't know" appropriately when context is insufficient, do NOT penalize.

Return ONLY this JSON object (no extra text):
{
  "score": <integer 1-10>,
  "explanation": "<brief reasoned explanation>",
  "criteria": {
    "relevance": <integer 1-10>,
    "correctness": <integer 1-10>,
    "completeness": <integer 1-10>
  }
}
"""


def build_judge_prompt(question: str, answer: str, context: str = "") -> str:
    return f"""
{JSON_INSTRUCTIONS}

Question:
{question}

Answer:
{answer}

Context (optional):
{context}
"""


async def evaluate_answer(question: str, answer: str, chat_model, context: str = "") -> Dict[str, Any]:
    prompt = build_judge_prompt(question, answer, context)
    resp = await chat_model.ainvoke(prompt)
    raw = getattr(resp, "content", resp)

    try:
        parsed = json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", str(raw), flags=re.DOTALL)
        if not match:
            raise ValueError(f"Judge returned non-JSON: {raw!r}")
        parsed = json.loads(match.group(0))

    score = int(parsed.get("score"))
    explanation = str(parsed.get("explanation", "")).strip()
    criteria = parsed.get("criteria", {}) or {}
    criteria = {
        "relevance": int(criteria.get("relevance", score)),
        "correctness": int(criteria.get("correctness", score)),
        "completeness": int(criteria.get("completeness", score)),
    }

    return {"score": score, "explanation": explanation, "criteria": criteria, "raw": raw}
