from typing import List
import tiktoken
from langchain_core.messages import SystemMessage, HumanMessage


def num_tokens_from_string_text(string_text: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string_text))
    return num_tokens


def split_string_with_limit(text: str, limit: int) -> List[str]:
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    parts = []
    current_part = []

    for token in tokens:
        current_part.append(token)
        if len(current_part) >= limit:
            parts.append(current_part)
            current_part = []
    if current_part:
        parts.append(current_part)

    text_parts = [encoding.decode(part) for part in parts]

    return text_parts


def summarize_text(text: str, client):
    """
    Public method to summarize text based on token length and chunking.
    """
    count_tokens = num_tokens_from_string_text(text)
    print(count_tokens)
    if count_tokens > 1000:
        output_chunks = []
        chunked_text = split_string_with_limit(text, 1000)
        for chunk in chunked_text:
            summary = _summarize_text_chunk(chunk, client)
            output_chunks.append(summary)
        summary = _summarize_text_chunk("\n".join(output_chunks), client)
    else:
        summary = _summarize_text_chunk(text, client)

    return summary


def _summarize_text_chunk(text: str, client):
    """
    Private method to summarize text using the Azure OpenAI service.
    """
    messages = [SystemMessage(content="Summarize the following text."), HumanMessage(content=text)]
    response = client.generate([messages])
    return response.generations[0][0].text
