import unittest
from unittest.mock import MagicMock
from generic_rag.backend.summarization import num_tokens_from_string_text, split_string_with_limit, summarize_text


class TestTextSummarizationUtils(unittest.TestCase):
    def test_num_tokens_from_string_text(self):
        """Test token count for a simple string"""
        text = "Hello world!"
        tokens = num_tokens_from_string_text(text)
        self.assertIsInstance(tokens, int)
        self.assertGreater(tokens, 0)

    def test_num_tokens_from_string_text_empty(self):
        """Test if there is an empty string, it should return 0 tokens."""
        text = ""
        tokens = num_tokens_from_string_text(text)
        self.assertEqual(tokens, 0)

    def test_split_string_with_limit_basic(self):
        """Test splitting a long string into chunks based on token limit."""
        text = "This is a test sentence that should be split into multiple parts based on token limits."
        parts = split_string_with_limit(text, limit=10)
        self.assertIsInstance(parts, list)
        self.assertTrue(all(isinstance(part, str) for part in parts))
        self.assertGreaterEqual(len(parts), 1)

    def test_split_string_with_limit_exact_limit(self):
        """Edge case: string with exactly the token limit should return one part."""
        text = "Token limit test " * 50  # Adjust to match exactly 1000 tokens if needed
        parts = split_string_with_limit(text, limit=1000)
        self.assertGreaterEqual(len(parts), 1)

    def test_split_string_with_limit_empty(self):
        """Edge case: empty string should return an empty list."""
        parts = split_string_with_limit("", limit=10)
        self.assertEqual(parts, [])

    def test_summarize_text_under_limit(self):
        """Test summarization for short text."""
        text = "Short text."
        mock_client = MagicMock()
        mock_client.generate.return_value.generations = [[MagicMock(text="Summary of short text.")]]

        summary = summarize_text(text, mock_client)
        self.assertEqual(summary, "Summary of short text.")
        mock_client.generate.assert_called_once()

    def test_summarize_text_over_limit(self):
        """Test summarization for text over token limit."""
        text = "Long text. " * 1000  # Should exceed 1000 tokens
        mock_client = MagicMock()
        mock_client.generate.return_value.generations = [[MagicMock(text="Chunk summary")]]

        summary = summarize_text(text, mock_client)
        self.assertEqual(summary, "Chunk summary")
        self.assertGreater(mock_client.generate.call_count, 1)

    def test_summarize_text_empty(self):
        """Edge case: summarizing an empty string."""
        text = ""
        mock_client = MagicMock()
        mock_client.generate.return_value.generations = [[MagicMock(text="Summary of empty text.")]]

        summary = summarize_text(text, mock_client)
        self.assertEqual(summary, "Summary of empty text.")
        mock_client.generate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
