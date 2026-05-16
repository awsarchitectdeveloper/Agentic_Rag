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
    """Provide a minimal config when calling build_judge_prompt at import-time."""
    config = {"evaluation": {"prompts": {"judge_system": "System prompt"}}}
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


def test_llm_judge_extracts_json_from_messy_response():
    """Test llm_judge can extract JSON from LLM responses with extra text."""
    row = {"question": "What is machine learning?", "generated_answer": "Machine learning is a subset of AI."}

    # Mock LLM response with extra text before/after JSON
    messy_response = """
    Here's my evaluation of the answer:

    Looking at the question and response, I need to assess accuracy and relevance.

    {"accuracy": 0.8, "relevance": 0.9}

    The answer is quite good overall, though it could be more detailed.
    """

    mock_chat_model = Mock()
    mock_chat_model.invoke.return_value = Mock(content=messy_response)

    mock_retriever = Mock()
    mock_retriever.invoke.return_value = ["ML context"]

    mock_parser = Mock()
    mock_parser.get_format_instructions.return_value = "Return JSON"
    # Simulate parser failing on the messy response (triggers JSON extraction fallback)
    mock_parser.parse.side_effect = Exception("Parser failed on messy response")

    mock_judge_prompt = Mock()
    mock_judge_prompt.format.return_value = "Formatted prompt"

    result = llm_judge(
        row, mock_chat_model, mock_retriever, mock_parser, mock_judge_prompt, metrics=["accuracy", "relevance"]
    )

    # Should extract the JSON block and return the scores
    assert result == {"accuracy": 0.8, "relevance": 0.9}


def test_llm_judge_handles_empty_context():
    """Test that llm_judge handles empty retrieval context gracefully."""
    row = {"question": "What is AI?", "generated_answer": "AI is artificial intelligence."}

    mock_chat_model = Mock()
    mock_chat_model.invoke.return_value = Mock(content='{"accuracy": 0.5, "relevance": 0.6}')

    mock_retriever = Mock()
    # Simulate empty retrieval results
    mock_retriever.invoke.return_value = []

    mock_parser = Mock()
    mock_parser.get_format_instructions.return_value = "Return JSON"
    mock_parser.parse.return_value = {"accuracy": 0.5, "relevance": 0.6}

    mock_judge_prompt = Mock()
    mock_judge_prompt.format.return_value = "Formatted prompt with empty context"

    result = llm_judge(
        row, mock_chat_model, mock_retriever, mock_parser, mock_judge_prompt, metrics=["accuracy", "relevance"]
    )

    # Should handle empty context and still return valid scores
    assert result == {"accuracy": 0.5, "relevance": 0.6}
    # Verify that format was called (prompt template should handle empty context)
    mock_judge_prompt.format.assert_called_once()


def test_llm_judge_handles_none_context():
    """Test that llm_judge handles None retrieval context gracefully."""
    row = {"question": "What is machine learning?", "generated_answer": "ML is a subset of AI."}

    mock_chat_model = Mock()
    mock_chat_model.invoke.return_value = Mock(content='{"accuracy": 0.4, "relevance": 0.5}')

    mock_retriever = Mock()
    # Simulate retriever returning None
    mock_retriever.invoke.return_value = None

    mock_parser = Mock()
    mock_parser.get_format_instructions.return_value = "Return JSON"
    mock_parser.parse.return_value = {"accuracy": 0.4, "relevance": 0.5}

    mock_judge_prompt = Mock()
    mock_judge_prompt.format.return_value = "Formatted prompt with no context"

    result = llm_judge(
        row, mock_chat_model, mock_retriever, mock_parser, mock_judge_prompt, metrics=["accuracy", "relevance"]
    )

    # Should handle None context and still return valid scores
    assert result == {"accuracy": 0.4, "relevance": 0.5}
    # Verify that format was called (prompt template should handle None context)
    mock_judge_prompt.format.assert_called_once()
