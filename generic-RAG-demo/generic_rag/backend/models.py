import logging
import os

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFacePipeline

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, ChatOpenAI, OpenAIEmbeddings
from langchain_together import ChatTogether, TogetherEmbeddings
from generic_rag.parsers.config import AppSettings, ChatBackend, EmbeddingBackend
from azure.identity import DefaultAzureCredential, get_bearer_token_provider


logger = logging.getLogger(__name__)


def get_chat_model(settings: AppSettings) -> BaseChatModel:
    """
    Initializes and returns a chat model based on the backend type and configuration.

    Args:
        settings: The loaded AppSettings object containing configurations.

    Returns:
        An instance of BaseChatModel.

    Raises:
        ValueError: If the backend type is unknown or required configuration is missing.
    """
    logger.info(f"Initializing chat model for backend: {settings.chat_backend.value}")

    if settings.chat_backend == ChatBackend.azure:
        if not settings.azure:
            raise ValueError("Azure chat backend selected, but 'azure' configuration section is missing in config.")
        if (
            not settings.azure.llm_endpoint
            or not settings.azure.llm_deployment_name
            or not settings.azure.llm_api_version
        ):
            raise ValueError(
                "Azure configuration requires 'llm_endpoint', 'llm_deployment_name', and 'llm_api_version'."
            )
        if "AZURE_OPENAI_API_KEY" not in os.environ:
            credential = DefaultAzureCredential()
            token_provider = get_bearer_token_provider(credential, settings.azure.bearer_token_scope)

            return AzureChatOpenAI(
                azure_endpoint=settings.azure.llm_endpoint,
                azure_deployment=settings.azure.llm_deployment_name,
                api_version=settings.azure.llm_api_version,
                azure_ad_token_provider=token_provider,
            )
        else:
            return AzureChatOpenAI(
                azure_endpoint=settings.azure.llm_endpoint,
                azure_deployment=settings.azure.llm_deployment_name,
                api_version=settings.azure.llm_api_version,
            )

    if settings.chat_backend == ChatBackend.openai:
        if not settings.openai:
            raise ValueError("OpenAI chat backend selected, but 'openai' configuration section is missing.")
        if not settings.openai.chat_model:
            raise ValueError("OpenAI configuration requires 'chat_model'.")
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError(
                "The environment variable 'OPENAI_API_KEY' is missing. Please set the variable in your '.env' file before running the script."
            )
        return ChatOpenAI(model=settings.openai.chat_model)

    if settings.chat_backend == ChatBackend.google_vertex:
        if not settings.google_vertex:
            raise ValueError(
                "Google Vertex chat backend selected, but 'google_vertex' configuration section is missing."
            )
        if (
            not settings.google_vertex.chat_model
            or not settings.google_vertex.project_id
            or not settings.google_vertex.location
        ):
            raise ValueError("Google Vertex configuration requires 'chat_model', 'project_id' and 'location'.")
        return ChatVertexAI(
            model_name=settings.google_vertex.chat_model,
            project=settings.google_vertex.project_id,
            location=settings.google_vertex.location,
        )

    if settings.chat_backend == ChatBackend.aws:
        if not settings.aws:
            raise ValueError("AWS Bedrock chat backend selected, but 'aws' configuration section is missing.")
        if not settings.aws.chat_model or not settings.aws.region_name:
            raise ValueError("AWS Bedrock configuration requires 'chat_model' and 'region'")
        return ChatBedrock(model_id=settings.aws.chat_model, region_name=settings.aws.region)

    if settings.chat_backend == ChatBackend.local:
        if not settings.local:
            raise ValueError("Local chat backend selected, but 'local' configuration section is missing.")
        if not settings.local.chat_model:
            raise ValueError("Local configuration requires 'chat_model'")
        return ChatOllama(model=settings.local.chat_model)

    if settings.chat_backend == ChatBackend.together:
        if not settings.together:
            raise ValueError("Together chat backend selected, but 'together' configuration section is missing.")
        if not settings.together.chat_model:
            raise ValueError("Together configuration requires 'chat_model'")
        return ChatTogether(model=settings.together.chat_model)

    if settings.chat_backend == ChatBackend.huggingface:
        if not settings.huggingface:
            raise ValueError("Huggingface chat backend selected, but 'huggingface' configuration section is missing.")
        if not settings.huggingface.chat_model:
            raise ValueError("Huggingface configuration requires 'chat_model'")
        llm = HuggingFacePipeline.from_model_id(
            model_id=settings.huggingface.chat_model,
            task="text-generation",
            pipeline_kwargs=dict(max_new_tokens=512, do_sample=False, repetition_penalty=1.03),
        )
        return ChatHuggingFace(llm=llm)

    if settings.chat_backend == ChatBackend.odin:
        if not settings.odin:
            raise ValueError("Odin chat backend selected, but 'odin' configuration section is missing.")
        if not settings.odin.odin_llm_endpoint or not settings.odin.odin_llm_model:
            raise ValueError("Odin configuration requires 'odin_llm_endpoint' and 'odin_llm_model'")
        return ChatOpenAI(base_url=settings.odin.odin_llm_endpoint, model=settings.odin.odin_llm_model, api_key="EMPTY")
    # This should not be reached if all Enum members are handled
    raise ValueError(f"Unknown or unhandled chat backend type: {settings.chat_backend}")


def get_embedding_model(settings: AppSettings) -> Embeddings:
    """
    Initializes and returns an embedding model based on the backend type and configuration.

    Args:
        settings: The loaded AppSettings object containing configurations.

    Returns:
        An instance of Embeddings.

    Raises:
        ValueError: If the backend type is unknown or required configuration is missing.
    """
    logger.info(f"Initializing embedding model for backend: {settings.emb_backend.value}")

    if settings.emb_backend == EmbeddingBackend.azure:
        if not settings.azure:
            raise ValueError("Azure embedding backend selected, but 'azure' configuration section is missing.")
        if (
            not settings.azure.emb_endpoint
            or not settings.azure.emb_deployment_name
            or not settings.azure.emb_api_version
        ):
            raise ValueError(
                "Azure configuration requires 'emb_endpoint', 'emb_deployment_name', and 'emb_api_version'."
            )
        if "AZURE_OPENAI_API_KEY" not in os.environ:
            credential = DefaultAzureCredential()
            token_provider = get_bearer_token_provider(credential, settings.azure.bearer_token_scope)

            return AzureOpenAIEmbeddings(
                azure_endpoint=settings.azure.emb_endpoint,
                azure_deployment=settings.azure.emb_deployment_name,
                api_version=settings.azure.emb_api_version,
                azure_ad_token_provider=token_provider,
                chunk_size=200,
            )
        else:
            return AzureOpenAIEmbeddings(
                azure_endpoint=settings.azure.emb_endpoint,
                azure_deployment=settings.azure.emb_deployment_name,
                openai_api_version=settings.azure.emb_api_version,
            )

    if settings.emb_backend == EmbeddingBackend.openai:
        if not settings.openai:
            raise ValueError("OpenAI embedding backend selected, but 'openai' configuration section is missing.")
        if not settings.openai.emb_model:
            raise ValueError("OpenAI configuration requires 'emb_model'.")
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError(
                "The environment variable 'OPENAI_API_KEY' is missing. Please set the variable in your '.env' file before running the script."
            )
        return OpenAIEmbeddings(model=settings.openai.emb_model)

    if settings.emb_backend == EmbeddingBackend.google_vertex:
        if not settings.google_vertex:
            raise ValueError(
                "Google Vertex embedding backend selected, but 'google_vertex' configuration section is missing."
            )
        if (
            not settings.google_vertex.emb_model
            or not settings.google_vertex.project_id
            or not settings.google_vertex.location
        ):
            raise ValueError("Google Vertex configuration requires 'emb_model', 'project_id', and 'location'.")
        return VertexAIEmbeddings(
            model_name=settings.google_vertex.emb_model,
            project=settings.google_vertex.project_id,
            location=settings.google_vertex.location,
        )

    if settings.emb_backend == EmbeddingBackend.aws:
        if not settings.aws:
            raise ValueError("AWS Bedrock embedding backend selected, but 'aws' configuration section is missing.")
        if not settings.aws.emb_model or not settings.aws.region:
            raise ValueError("AWS Bedrock configuration requires 'emb_model' and 'region'")
        return BedrockEmbeddings(model_id=settings.aws.emb_model, region_name=settings.aws.region)

    if settings.emb_backend == EmbeddingBackend.local:
        if not settings.local:
            raise ValueError("Local embedding backend selected, but 'local' configuration section is missing.")
        if not settings.local.emb_model:
            raise ValueError("Local configuration requires 'emb_model'")
        return OllamaEmbeddings(model=settings.local.emb_model)

    if settings.emb_backend == EmbeddingBackend.huggingface:
        if not settings.huggingface:
            raise ValueError(
                "HuggingFace embedding backend selected, but 'huggingface' configuration section is missing."
            )
        if not settings.huggingface.emb_model:
            raise ValueError("HuggingFace configuration requires 'emb_model'.")
        return HuggingFaceEmbeddings(model_name=settings.huggingface.emb_model)

    if settings.emb_backend == EmbeddingBackend.together:
        if not settings.together:
            raise ValueError("Together embedding backend selected, but 'together' configuration section is missing.")
        if not settings.together.emb_model:
            raise ValueError("Together configuration requires 'emb_model'")
        return TogetherEmbeddings(model=settings.together.emb_model)

    if settings.emb_backend == EmbeddingBackend.odin:
        if not settings.odin:
            raise ValueError("Odin embedding backend selected, but 'odin' configuration section is missing.")
        if not settings.odin.odin_emb_endpoint or not settings.odin.odin_emb_model:
            raise ValueError("Odin configuration requires 'odin_emb_endpoint' and 'odin_emb_model'")

        return OpenAIEmbeddings(
            base_url=settings.odin.odin_emb_endpoint, model=settings.odin.odin_emb_model, api_key="EMPTY"
        )

    raise ValueError(f"Unknown backend type: {settings.emb_backend}")


def get_agent_model(settings: AppSettings) -> BaseChatModel:
    """
    Initializes and returns an agent model based on the Azure agent configuration.

    Args:
        settings: The loaded AppSettings object containing configurations.

    Returns:
        An instance of BaseChatModel configured for agent use.

    Raises:
        ValueError: If Azure agent configuration is missing or incomplete.
    """
    logger.info("Initializing agent model for Azure backend")

    if not settings.azure:
        raise ValueError("Agent model requires Azure configuration section.")

    if (
        not settings.azure.agent_endpoint
        or not settings.azure.agent_deployment_name
        or not settings.azure.agent_api_version
    ):
        raise ValueError(
            "Azure agent configuration requires 'agent_endpoint', 'agent_deployment_name', and 'agent_api_version'."
        )

    if "AZURE_OPENAI_API_KEY" not in os.environ:
        credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(credential, settings.azure.bearer_token_scope)

        return AzureChatOpenAI(
            azure_endpoint=settings.azure.agent_endpoint,
            azure_deployment=settings.azure.agent_deployment_name,
            api_version=settings.azure.agent_api_version,
            azure_ad_token_provider=token_provider,
        )
    else:
        return AzureChatOpenAI(
            azure_endpoint=settings.azure.agent_endpoint,
            azure_deployment=settings.azure.agent_deployment_name,
            openai_api_version=settings.azure.agent_api_version,
        )


def get_compression_model(model_name: str, vector_store: Chroma) -> BaseRetriever:
    # TODO: Fix imports for ContextualCompressionRetriever and CrossEncoderReranker
    # base_retriever = vector_store.as_retriever(search_kwargs={"k": 20})
    # rerank_model = HuggingFaceCrossEncoder(model_name=model_name)
    # compressor = CrossEncoderReranker(model=rerank_model, top_n=4)
    # return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)
    raise NotImplementedError("Compression model temporarily disabled due to import issues")
