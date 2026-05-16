from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

from langchain.document_loaders import TextLoader
from langchain_openai import AzureChatOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import textwrap
from generic_rag.parsers.config import AppSettings, ChatBackend


def load_document(doc):
    loader = TextLoader(doc)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    return docs


def summarize_mapreduce(settings: AppSettings):
    if settings.chat_backend == ChatBackend.azure:
        llm = AzureChatOpenAI(
            openai_api_base=settings.azure.llm_endpoint,
            openai_api_version=settings.azure.llm_api_version,
            deployment_name=settings.azure.llm_deployment_name,
            credential=DefaultAzureCredential(),
            openai_api_key=get_bearer_token_provider(settings.azure.bearer_token_scope),
            openai_api_type="azure_ad",
            max_tokens=1800,
        )
        text = load_document()

    chain = load_summarize_chain(llm=llm, chain_type="map_reduce")
    output_summary = chain.run(text)
    wrapped_text = textwrap.fill(output_summary, width=100)
    print(wrapped_text)
