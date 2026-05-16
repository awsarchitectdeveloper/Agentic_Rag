"""Vector database factory and utilities."""

import logging
from typing import Any, Protocol, runtime_checkable

from generic_rag.parsers.config import AppSettings, VectorDbType

logger = logging.getLogger(__name__)


@runtime_checkable
class VectorStore(Protocol):
    """Protocol for vector store implementations."""

    def add_documents(self, documents: Any) -> None:
        """Add documents to the vector store."""
        ...

    def similarity_search(self, query: str, k: int = 4) -> Any:
        """Search for similar documents."""
        ...

    def reset_collection(self) -> None:
        """Reset/clear the vector store collection."""
        ...


def get_vector_store(
    settings: AppSettings, embedding_function: Any, collection_name: str = "generic_rag"
) -> VectorStore:
    """
    Factory function to create a vector store based on configuration.

    Args:
        settings: Application settings containing vector DB configuration
        embedding_function: The embedding model to use
        collection_name: Name of the collection/index

    Returns:
        Configured vector store instance

    Raises:
        ValueError: If unsupported vector DB type is specified
        ImportError: If required dependencies for the vector DB are not installed
    """

    db_type = settings.vector_db.type
    # Convert Path to POSIX-style string to keep tests stable across OSes
    location = getattr(settings.vector_db.location, "as_posix", None)
    if callable(location):
        location = settings.vector_db.location.as_posix()
    else:
        location = str(settings.vector_db.location)
    reset = settings.vector_db.reset

    logger.info(f"Initializing vector store: {db_type}")

    if db_type == VectorDbType.chroma:
        return _create_chroma_store(embedding_function, collection_name, location, reset)
    elif db_type == VectorDbType.ai_search:
        return _create_ai_search_store(settings, embedding_function, collection_name, location, reset)
    else:
        raise ValueError(f"Unsupported vector database type: {db_type}")


def _create_chroma_store(embedding_function: Any, collection_name: str, location: str, reset: bool) -> VectorStore:
    """Create a Chroma vector store."""
    try:
        from langchain_chroma import Chroma
    except ImportError as e:
        raise ImportError("Chroma dependencies not installed. Install with: pip install langchain-chroma") from e

    vector_store = Chroma(
        collection_name=collection_name, embedding_function=embedding_function, persist_directory=location
    )

    if reset:
        logger.warning("Resetting Chroma database as specified in configuration.")
        vector_store.reset_collection()

    return vector_store


def _create_ai_search_store(
    settings: AppSettings, embedding_function: Any, collection_name: str, location: str, reset: bool
) -> VectorStore:
    """
    Create an Azure AI Search vector store using RBAC authentication.

    This implementation uses DefaultAzureCredential for authentication, which supports:
    - Managed Identity (recommended for Azure resources)
    - Azure CLI authentication (az login)
    - Environment variables (AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID)
    - Interactive browser authentication

    Required Azure permissions:
    - Search Index Data Contributor (for read/write operations)
    - Search Service Contributor (for index management operations)

    Args:
        settings: Application settings containing Azure AI Search configuration
        embedding_function: The embedding model to use
        collection_name: Name of the search index
        location: Unused for Azure AI Search (kept for interface compatibility)
        reset: Whether to reset the index by deleting and recreating it

    Returns:
        Configured Azure AI Search vector store
    """
    try:
        from langchain_community.vectorstores.azuresearch import AzureSearch
        from azure.identity import DefaultAzureCredential
        from azure.search.documents.indexes import SearchIndexClient
        from azure.search.documents.indexes.models import (
            SearchField,
            SearchFieldDataType,
            SearchIndex,
            VectorSearch,
            HnswAlgorithmConfiguration,
            VectorSearchProfile,
            AzureOpenAIVectorizer,
            AzureOpenAIVectorizerParameters,
            SearchIndexerDataUserAssignedIdentity,
            SemanticConfiguration,
            SemanticPrioritizedFields,
            SemanticField,
            SemanticSearch,
        )
        from azure.core.exceptions import ResourceNotFoundError
    except ImportError as e:
        raise ImportError(
            "Azure AI Search dependencies not installed. Install with: pip install langchain-community azure-search-documents azure-identity"
        ) from e

    # Get Azure AI Search configuration from settings
    ai_search_settings = settings.azure_ai_search

    if not ai_search_settings:
        raise ValueError(
            "Azure AI Search configuration is required. Please add azure_ai_search section to your config."
        )

    # Get endpoint from settings (required)
    search_endpoint = ai_search_settings.endpoint
    if not search_endpoint:
        raise ValueError("Azure AI Search endpoint must be provided via azure_ai_search.endpoint in config")

    # Use collection_name or configured index name
    index_name = collection_name
    if ai_search_settings.index_name:
        index_name = ai_search_settings.index_name

    # Get vector search parameters from settings
    vector_dimensions = ai_search_settings.vector_search_dimensions
    semantic_config_name = ai_search_settings.semantic_configuration_name
    vector_profile_name = ai_search_settings.vector_search_profile_name or "default-vector-profile"

    # Get Azure OpenAI configuration for vectorizer (if available)
    azure_settings = getattr(settings, "azure", None)
    vectorizer_name = "azure-openai-vectorizer"

    logger.info(f"Azure settings found: {azure_settings is not None}")
    if azure_settings:
        logger.info(f"Azure emb_endpoint: {getattr(azure_settings, 'emb_endpoint', 'Not set')}")
        logger.info(f"Azure emb_deployment_name: {getattr(azure_settings, 'emb_deployment_name', 'Not set')}")

    # Create DefaultAzureCredential for RBAC authentication
    credential = DefaultAzureCredential()

    logger.info(f"Connecting to Azure AI Search at {search_endpoint} using RBAC authentication")

    # Create SearchIndexClient for index management operations
    index_client = SearchIndexClient(endpoint=search_endpoint, credential=credential)

    # Define the search index fields using the Azure Search SDK directly
    fields = [
        SearchField(name="id", type=SearchFieldDataType.String, key=True, filterable=True, sortable=True),
        SearchField(name="content", type=SearchFieldDataType.String, searchable=True),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=vector_dimensions,
            vector_search_profile_name=vector_profile_name,
        ),
        SearchField(name="metadata", type=SearchFieldDataType.String, searchable=True),
    ]

    # Create vector search configuration
    algorithms = [
        HnswAlgorithmConfiguration(
            name="default-hnsw-config", parameters={"m": 4, "efConstruction": 400, "efSearch": 500, "metric": "cosine"}
        )
    ]

    profiles = [VectorSearchProfile(name=vector_profile_name, algorithm_configuration_name="default-hnsw-config")]

    vectorizers = []

    # Add Azure OpenAI vectorizer if configuration is available
    if azure_settings and azure_settings.emb_endpoint and azure_settings.emb_deployment_name:
        try:
            # check if managed identity resource ID is available
            if not settings.azure_ai_search.mi_resource_id:
                logger.error("Managed identity resource ID is not configured.")
                logger.error("Please set mi_resource_id under azure_ai_search in your config.")
                logger.error("If you do not set it, AI search MCP server will not work.")
                return
            # Create identity-based authentication for the vectorizer using user-assigned managed identity
            auth_identity = SearchIndexerDataUserAssignedIdentity(resource_id=settings.azure_ai_search.mi_resource_id)

            # Configure Azure OpenAI vectorizer parameters using identity authentication
            vectorizer_params = AzureOpenAIVectorizerParameters(
                resource_url=azure_settings.emb_endpoint,
                deployment_name=azure_settings.emb_deployment_name,
                model_name=azure_settings.emb_deployment_name,  # Use deployment name as model name fallback
                auth_identity=auth_identity,  # Use managed identity instead of API key
            )

            # Create the Azure OpenAI vectorizer
            azure_openai_vectorizer = AzureOpenAIVectorizer(
                vectorizer_name=vectorizer_name, kind="azureOpenAI", parameters=vectorizer_params
            )

            vectorizers.append(azure_openai_vectorizer)

            # Update the vector search profile to use the vectorizer
            profiles[0] = VectorSearchProfile(
                name=vector_profile_name,
                algorithm_configuration_name="default-hnsw-config",
                vectorizer_name=vectorizer_name,
            )
            logger.info("✅ Successfully configured Azure OpenAI vectorizer")

        except Exception as e:
            logger.error(f"Failed to configure Azure OpenAI vectorizer: {e}")
            logger.info("Falling back to external embedding function")
    else:
        missing_config = []
        if not azure_settings:
            missing_config.append("azure settings")
        if azure_settings and not azure_settings.emb_endpoint:
            missing_config.append("emb_endpoint")
        if azure_settings and not azure_settings.emb_deployment_name:
            missing_config.append("emb_deployment_name")

        logger.info(f"Azure OpenAI vectorizer not configured. Missing: {', '.join(missing_config)}")
        logger.info("Using external embedding function instead")

    vector_search = VectorSearch(
        algorithms=algorithms, profiles=profiles, vectorizers=vectorizers if vectorizers else None
    )

    # Create semantic search configuration if specified
    semantic_search = None
    if semantic_config_name:
        semantic_search = SemanticSearch(
            configurations=[
                SemanticConfiguration(
                    name=semantic_config_name,
                    prioritized_fields=SemanticPrioritizedFields(content_fields=[SemanticField(field_name="content")]),
                )
            ]
        )

    # Create or update the search index
    index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search, semantic_search=semantic_search)

    if reset:
        logger.warning("Resetting Azure AI Search index as specified in configuration.")
        try:
            # Try to delete the existing index
            try:
                index_client.delete_index(index_name)
                logger.info(f"Successfully deleted existing index: {index_name}")
            except ResourceNotFoundError:
                logger.info(f"Index {index_name} does not exist, skipping deletion")
            except Exception as e:
                logger.warning(f"Error deleting index {index_name}: {e}")
        except Exception as e:
            logger.error(f"Could not reset Azure AI Search index: {e}")
            raise

    # Create or update the index
    try:
        index_client.create_or_update_index(index)
        logger.info(f"Successfully created/updated index: {index_name}")
    except Exception as e:
        logger.error(f"Could not create/update Azure AI Search index: {e}")
        raise

    # Create the Azure Search vector store for document operations
    vector_store = AzureSearch(
        azure_search_endpoint=search_endpoint,
        azure_search_key=None,  # Required parameter, set to None when using credential
        index_name=index_name,
        embedding_function=embedding_function,
        # Additional Azure AI Search specific parameters
        semantic_configuration_name=semantic_config_name,
        vector_search_dimensions=vector_dimensions,
        azure_credential=credential,  # Pass the credential here
    )

    return vector_store
