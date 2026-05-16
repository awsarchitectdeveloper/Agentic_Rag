"""Test/results module for LLM evaluation.

This module imports judge functionality from the canonical judge module.
Results and test-related code should live here, not production judge logic.
"""

from unittest.mock import Mock

# Import canonical judge functions
from generic_rag.evaluation.judge import llm_judge, build_judge_prompt

# Re-export for any existing imports
__all__ = ["llm_judge", "build_judge_prompt"]


# ============================================================================
# Tests for build_judge_prompt
# ============================================================================


def test_build_judge_prompt_with_config():
    """Test build_judge_prompt with a valid config dictionary."""
    config = {"evaluation": {"prompts": {"judge_system": "You are a helpful judge."}}}

    prompt = build_judge_prompt(config=config)

    # Check that the prompt has 2 messages (system and human)
    assert len(prompt.messages) == 2
    assert "You are a helpful judge." in str(prompt.messages[0])
    assert "Question" in str(prompt.messages[1])


def test_build_judge_prompt_with_none_config():
    """Test build_judge_prompt when a config is provided explicitly."""
    config = {"evaluation": {"prompts": {"judge_system": "System prompt from config."}}}

    prompt = build_judge_prompt(config=config)

    assert prompt is not None
    assert len(prompt.messages) == 2


# ============================================================================
# Tests for llm_judge
# ============================================================================


def test_llm_judge_successful_parsing():
    """Test llm_judge with successful parsing."""
    # Mock row data
    row = {"question": "What is Python?", "generated_answer": "Python is a programming language."}

    # Mock chat model
    mock_chat_model = Mock()
    mock_chat_model.invoke.return_value = Mock(content='{"accuracy": 0.9, "relevance": 0.8}')

    # Mock retriever
    mock_retriever = Mock()
    mock_retriever.invoke.return_value = ["Python is a high-level programming language."]

    # Mock parser
    mock_parser = Mock()
    mock_parser.get_format_instructions.return_value = "Return JSON format"
    mock_parser.parse.return_value = {"accuracy": 0.9, "relevance": 0.8}

    # Mock judge prompt
    mock_judge_prompt = Mock()
    mock_judge_prompt.format.return_value = "Formatted prompt"

    result = llm_judge(
        row, mock_chat_model, mock_retriever, mock_parser, mock_judge_prompt, metrics=["accuracy", "relevance"]
    )

    assert result == {"accuracy": 0.9, "relevance": 0.8}
    mock_parser.parse.assert_called_once()


def test_llm_judge_parsing_fails():
    """Test llm_judge when parsing fails and fallback is used."""
    row = {"question": "What is AI?", "generated_answer": "AI is artificial intelligence."}

    mock_chat_model = Mock()
    mock_chat_model.invoke.return_value = Mock(content="Invalid response")

    mock_retriever = Mock()
    mock_retriever.invoke.return_value = ["AI context"]

    mock_parser = Mock()
    mock_parser.get_format_instructions.return_value = "JSON format"
    mock_parser.parse.side_effect = Exception("Parsing failed")

    mock_judge_prompt = Mock()
    mock_judge_prompt.format.return_value = "Formatted prompt"

    result = llm_judge(
        row, mock_chat_model, mock_retriever, mock_parser, mock_judge_prompt, metrics=["accuracy", "relevance"]
    )

    assert result == {"accuracy": None, "relevance": None}
