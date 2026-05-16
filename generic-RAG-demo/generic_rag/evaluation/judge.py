import re
import json
import logging
from langchain_core.prompts import ChatPromptTemplate
from generic_rag.evaluation.utils import format_docs

logger = logging.getLogger(__name__)


def build_judge_prompt(config: dict):
    """Build judge prompt with a required preloaded `config` dict.

    This enforces that callers provide the config to avoid file I/O during
    evaluation and in CI environments.
    """
    judge_system_prompt = config["evaluation"]["prompts"]["judge_system"]

    return ChatPromptTemplate.from_messages(
        [
            ("system", judge_system_prompt),
            ("human", "Question: {question}\nContext: {context}\nAnswer: {answer}\nFormat: {format_instructions}"),
        ]
    )


def llm_judge(row, chat_model, retriever, parser, judge_prompt, metrics: list):
    """Run the judge LLM on a single row and return parsed results or None metrics.

    Args:
        row: Dictionary with 'question' and 'generated_answer' keys
        chat_model: LLM model for judging
        retriever: Document retriever for context
        parser: Parser for LLM output
        judge_prompt: Prompt template for judging
        metrics: List of metric names to return (required)

    Returns:
        Dictionary with metric scores or None values for failed parsing
    """
    context = format_docs(retriever.invoke(row["question"]))
    judge_input = judge_prompt.format(
        question=row["question"],
        context=context,
        answer=row["generated_answer"],
        format_instructions=parser.get_format_instructions(),
    )
    judgment = chat_model.invoke(judge_input)
    # Support objects with .content or raw string
    judgment_text = getattr(judgment, "content", str(judgment))
    try:
        return parser.parse(judgment_text)
    except Exception as e:
        logger.debug(f"Primary parsing failed: {e}. Attempting JSON block extraction...")
        match = re.search(r"\{.*\}", judgment_text, re.DOTALL)
        if match:
            try:
                logger.debug(f"Found JSON block, attempting to parse: {match.group()[:100]}...")
                return json.loads(match.group())
            except Exception as json_e:
                logger.warning(f"JSON block parsing also failed: {json_e}")
        else:
            logger.warning("No JSON block found in judgment text for fallback parsing")

        # Fallback: return None for all configured metrics
        logger.info(f"Returning None values for metrics: {metrics}")
        return {metric: None for metric in metrics}
