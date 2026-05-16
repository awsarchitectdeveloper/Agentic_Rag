"""Tests for vectordb module."""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from generic_rag.backend.vectordb import get_vector_store, VectorStore
from generic_rag.parsers.config import AppSettings, OcrSettings, VectorDbSettings, VectorDbType, AzureAISearchSettings


@pytest.fixture
def mock_azure_modules():
    """Fixture to provide mock Azure modules."""
    return {
        "langchain_community": Mock(),
        "langchain_community.vectorstores": Mock(),
        "langchain_community.vectorstores.azuresearch": Mock(),
        "azure": Mock(),
        "azure.identity": Mock(),
        "azure.search": Mock(),
        "azure.search.documents": Mock(),
        "azure.search.documents.indexes": Mock(),
        "azure.search.documents.indexes.models": Mock(),
        "azure.core": Mock(),
        "azure.core.exceptions": Mock(),
    }


@pytest.fixture
def mock_azure_ai_search_objects():
    """Fixture to provide common Azure AI Search mock objects."""
    from collections import namedtuple

    MockObjects = namedtuple("MockObjects", ["azure_search_instance", "credential", "index_client"])

    return MockObjects(azure_search_instance=Mock(), credential=Mock(), index_client=Mock())


@pytest.fixture
def mock_azure_search_patches(mock_azure_modules, mock_azure_ai_search_objects):
    """Fixture to provide common Azure AI Search patches context manager."""
    from contextlib import contextmanager

    @contextmanager
    def _patches():
        with patch.dict("sys.modules", mock_azure_modules):
            with (
                patch(
                    "langchain_community.vectorstores.azuresearch.AzureSearch",
                    return_value=mock_azure_ai_search_objects.azure_search_instance,
                ) as mock_azure_search,
                patch("azure.identity.DefaultAzureCredential", return_value=mock_azure_ai_search_objects.credential),
                patch(
                    "azure.search.documents.indexes.SearchIndexClient",
                    return_value=mock_azure_ai_search_objects.index_client,
                ),
            ):
                yield mock_azure_ai_search_objects, mock_azure_search

    return _patches


class TestGetVectorStore:
    """Test cases for get_vector_store function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_embedding_function = Mock()
        self.collection_name = "test_collection"

    def test_get_vector_store_chroma_success(self):
        """Test successful creation of Chroma vector store."""
        # Arrange
        settings = AppSettings(
            vector_db=VectorDbSettings(type=VectorDbType.chroma, location=Path("/tmp/test_chroma"), reset=False)
        )

        mock_chroma_instance = Mock(spec=VectorStore)

        with patch(
            "generic_rag.backend.vectordb._create_chroma_store", return_value=mock_chroma_instance
        ) as mock_create:
            # Act
            result = get_vector_store(settings, self.mock_embedding_function, self.collection_name)

            # Assert
            assert result == mock_chroma_instance
            mock_create.assert_called_once_with(
                self.mock_embedding_function, self.collection_name, "/tmp/test_chroma", False
            )

    def test_get_vector_store_ai_search_success(self):
        """Test successful creation of Azure AI Search vector store."""
        # Arrange
        settings = AppSettings(
            vector_db=VectorDbSettings(
                type=VectorDbType.ai_search, location=Path("/unused/for/ai/search"), reset=False
            ),
            azure_ai_search=AzureAISearchSettings(endpoint="https://test-search.search.windows.net"),
        )

        mock_ai_search_instance = Mock(spec=VectorStore)

        with patch(
            "generic_rag.backend.vectordb._create_ai_search_store", return_value=mock_ai_search_instance
        ) as mock_create:
            # Act
            result = get_vector_store(settings, self.mock_embedding_function, self.collection_name)

            # Assert
            assert result == mock_ai_search_instance
            mock_create.assert_called_once_with(
                settings, self.mock_embedding_function, self.collection_name, "/unused/for/ai/search", False
            )

    def test_get_vector_store_unsupported_type_raises_error(self):
        """Test that unsupported vector DB type raises ValueError."""
        # Arrange
        settings = AppSettings()
        # Mock an unsupported type by creating a custom enum value
        settings.vector_db.type = "unsupported_type"

        # Act & Assert
        with pytest.raises(ValueError, match="Unsupported vector database type: unsupported_type"):
            get_vector_store(settings, self.mock_embedding_function, self.collection_name)

    def test_get_vector_store_chroma_with_reset(self):
        """Test Chroma vector store creation with reset enabled."""
        # Arrange
        settings = AppSettings(
            vector_db=VectorDbSettings(type=VectorDbType.chroma, location=Path("/tmp/test_chroma"), reset=True)
        )

        mock_chroma_instance = Mock(spec=VectorStore)

        with patch(
            "generic_rag.backend.vectordb._create_chroma_store", return_value=mock_chroma_instance
        ) as mock_create:
            # Act
            result = get_vector_store(settings, self.mock_embedding_function, self.collection_name)

            # Assert
            assert result == mock_chroma_instance
            mock_create.assert_called_once_with(
                self.mock_embedding_function, self.collection_name, "/tmp/test_chroma", True
            )

    def test_get_vector_store_ai_search_with_reset(self):
        """Test Azure AI Search vector store creation with reset enabled."""
        # Arrange
        settings = AppSettings(
            vector_db=VectorDbSettings(type=VectorDbType.ai_search, location=Path("/unused/for/ai/search"), reset=True),
            azure_ai_search=AzureAISearchSettings(endpoint="https://test-search.search.windows.net"),
        )

        mock_ai_search_instance = Mock(spec=VectorStore)

        with patch(
            "generic_rag.backend.vectordb._create_ai_search_store", return_value=mock_ai_search_instance
        ) as mock_create:
            # Act
            result = get_vector_store(settings, self.mock_embedding_function, self.collection_name)

            # Assert
            assert result == mock_ai_search_instance
            mock_create.assert_called_once_with(
                settings, self.mock_embedding_function, self.collection_name, "/unused/for/ai/search", True
            )

    def test_get_vector_store_default_collection_name(self):
        """Test that default collection name is used when not specified."""
        # Arrange
        settings = AppSettings(
            vector_db=VectorDbSettings(type=VectorDbType.chroma, location=Path("/tmp/test_chroma"), reset=False)
        )

        mock_chroma_instance = Mock(spec=VectorStore)

        with patch(
            "generic_rag.backend.vectordb._create_chroma_store", return_value=mock_chroma_instance
        ) as mock_create:
            # Act
            result = get_vector_store(settings, self.mock_embedding_function)  # No collection_name provided

            # Assert
            assert result == mock_chroma_instance
            mock_create.assert_called_once_with(
                self.mock_embedding_function,
                "generic_rag",  # Default collection name
                "/tmp/test_chroma",
                False,
            )

    def test_get_vector_store_chroma_import_error_propagated(self):
        """Test that ImportError from Chroma creation is propagated."""
        # Arrange
        settings = AppSettings(
            vector_db=VectorDbSettings(type=VectorDbType.chroma, location=Path("/tmp/test_chroma"), reset=False)
        )

        import_error = ImportError("Chroma dependencies not installed")

        with patch("generic_rag.backend.vectordb._create_chroma_store", side_effect=import_error):
            # Act & Assert
            with pytest.raises(ImportError, match="Chroma dependencies not installed"):
                get_vector_store(settings, self.mock_embedding_function, self.collection_name)

    def test_get_vector_store_ai_search_import_error_propagated(self):
        """Test that ImportError from Azure AI Search creation is propagated."""
        # Arrange
        settings = AppSettings(
            vector_db=VectorDbSettings(
                type=VectorDbType.ai_search, location=Path("/unused/for/ai/search"), reset=False
            ),
            azure_ai_search=AzureAISearchSettings(endpoint="https://test-search.search.windows.net"),
        )

        import_error = ImportError("Azure AI Search dependencies not installed")

        with patch("generic_rag.backend.vectordb._create_ai_search_store", side_effect=import_error):
            # Act & Assert
            with pytest.raises(ImportError, match="Azure AI Search dependencies not installed"):
                get_vector_store(settings, self.mock_embedding_function, self.collection_name)


class TestCreateChromaStore:
    """Test cases for _create_chroma_store function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_embedding_function = Mock()
        self.collection_name = "test_collection"
        self.location = "/tmp/test_chroma"

    def test_create_chroma_store_import_error(self):
        """Test ImportError when langchain_chroma is not available."""
        from generic_rag.backend.vectordb import _create_chroma_store

        # Mock the import to raise ImportError by patching the imports at import time
        original_import = __builtins__["__import__"]

        def mock_import(name, *args, **kwargs):
            if name == "langchain_chroma":
                raise ImportError("No module named 'langchain_chroma'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            # Act & Assert
            with pytest.raises(ImportError, match="Chroma dependencies not installed"):
                _create_chroma_store(self.mock_embedding_function, self.collection_name, self.location, reset=False)

    def test_create_chroma_store_success_with_reset(self):
        """Test successful Chroma store creation with reset functionality."""
        from generic_rag.backend.vectordb import _create_chroma_store

        # Mock the Chroma class and its methods
        mock_chroma_instance = Mock()
        mock_chroma_instance.reset_collection = Mock()

        with patch.dict("sys.modules", {"langchain_chroma": Mock()}):
            with patch("langchain_chroma.Chroma", return_value=mock_chroma_instance):
                # Act
                result = _create_chroma_store(
                    self.mock_embedding_function, self.collection_name, self.location, reset=True
                )

                # Assert
                assert result == mock_chroma_instance
                mock_chroma_instance.reset_collection.assert_called_once()

    def test_create_chroma_store_success_without_reset(self):
        """Test successful Chroma store creation without reset."""
        from generic_rag.backend.vectordb import _create_chroma_store

        # Mock the Chroma class and its methods
        mock_chroma_instance = Mock()
        mock_chroma_instance.reset_collection = Mock()

        with patch.dict("sys.modules", {"langchain_chroma": Mock()}):
            with patch("langchain_chroma.Chroma", return_value=mock_chroma_instance):
                # Act
                result = _create_chroma_store(
                    self.mock_embedding_function, self.collection_name, self.location, reset=False
                )

                # Assert
                assert result == mock_chroma_instance
                mock_chroma_instance.reset_collection.assert_not_called()


class TestCreateAISearchStore:
    """Test cases for _create_ai_search_store function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_embedding_function = Mock()
        self.collection_name = "test_collection"
        self.location = "/unused/for/ai/search"

        self.settings = AppSettings(
            azure_ai_search=AzureAISearchSettings(
                endpoint="https://test-search.search.windows.net",
                index_name="test_index",
                semantic_configuration_name="test_semantic",
                vector_search_dimensions=1536,
                vector_search_profile_name="test_profile",
            ),
            ocr=OcrSettings(azure_doc_intel_endpoint="https://dummy-endpoint", azure_doc_intel_key="dummy-key"),
        )

    def test_create_ai_search_store_missing_config_raises_error(self):
        """Test that missing Azure AI Search configuration raises ValueError."""
        from generic_rag.backend.vectordb import _create_ai_search_store

        # Arrange - settings without azure_ai_search config
        settings = AppSettings()

        # Act & Assert
        with pytest.raises(ValueError, match="Azure AI Search configuration is required"):
            _create_ai_search_store(
                settings, self.mock_embedding_function, self.collection_name, self.location, reset=False
            )

    def test_create_ai_search_store_missing_endpoint_raises_error(self):
        """Test that missing endpoint raises ValueError."""
        from generic_rag.backend.vectordb import _create_ai_search_store

        # Arrange - settings with empty endpoint
        settings = AppSettings(azure_ai_search=AzureAISearchSettings(endpoint=""))

        # Act & Assert
        with pytest.raises(ValueError, match="Azure AI Search endpoint must be provided"):
            _create_ai_search_store(
                settings, self.mock_embedding_function, self.collection_name, self.location, reset=False
            )

    def test_create_ai_search_store_import_error(self):
        """Test ImportError when Azure dependencies are not available."""
        from generic_rag.backend.vectordb import _create_ai_search_store

        # Mock the import to raise ImportError by patching the imports at import time
        original_import = __builtins__["__import__"]

        def mock_import(name, *args, **kwargs):
            if name == "langchain_community.vectorstores.azuresearch":
                raise ImportError("No module named 'langchain_community'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="Azure AI Search dependencies not installed"):
                _create_ai_search_store(
                    self.settings, self.mock_embedding_function, self.collection_name, self.location, reset=False
                )

    def test_create_ai_search_store_success_with_reset(self, mock_azure_search_patches):
        """Test successful Azure AI Search store creation with reset functionality."""
        from generic_rag.backend.vectordb import _create_ai_search_store

        with mock_azure_search_patches() as (mock_objects, mock_azure_search):
            # Act
            result = _create_ai_search_store(
                self.settings, self.mock_embedding_function, self.collection_name, self.location, reset=True
            )

            # Assert
            assert result == mock_objects.azure_search_instance
            # Verify reset operations were called - the index name will be from settings.index_name
            mock_objects.index_client.delete_index.assert_called_once_with("test_index")
            mock_objects.index_client.create_or_update_index.assert_called_once()

    def test_create_ai_search_store_reset_index_not_found(self, mock_azure_search_patches):
        """Test Azure AI Search reset when index doesn't exist."""
        from generic_rag.backend.vectordb import _create_ai_search_store

        # Create a proper ResourceNotFoundError mock
        class MockResourceNotFoundError(Exception):
            pass

        with mock_azure_search_patches() as (mock_objects, mock_azure_search):
            mock_objects.index_client.delete_index.side_effect = MockResourceNotFoundError("Index not found")

            with patch("azure.core.exceptions.ResourceNotFoundError", MockResourceNotFoundError):
                # Act
                result = _create_ai_search_store(
                    self.settings, self.mock_embedding_function, self.collection_name, self.location, reset=True
                )

                # Assert
                assert result == mock_objects.azure_search_instance
                # Verify delete was attempted and create was still called
                mock_objects.index_client.delete_index.assert_called_once_with("test_index")
                mock_objects.index_client.create_or_update_index.assert_called_once()

    def test_create_ai_search_store_reset_delete_error(self, mock_azure_search_patches):
        """Test Azure AI Search reset when delete operation fails."""
        from generic_rag.backend.vectordb import _create_ai_search_store

        # Create proper exception mocks
        class MockResourceNotFoundError(Exception):
            pass

        with mock_azure_search_patches() as (mock_objects, mock_azure_search):
            # Mock a generic exception for delete operation (not ResourceNotFoundError)
            delete_error = RuntimeError("Delete operation failed")
            mock_objects.index_client.delete_index.side_effect = delete_error

            with patch("azure.core.exceptions.ResourceNotFoundError", MockResourceNotFoundError):
                # Act
                result = _create_ai_search_store(
                    self.settings, self.mock_embedding_function, self.collection_name, self.location, reset=True
                )

                # Assert
                assert result == mock_objects.azure_search_instance
                # Verify delete was attempted and create was still called despite error
                mock_objects.index_client.delete_index.assert_called_once_with("test_index")
                mock_objects.index_client.create_or_update_index.assert_called_once()

    def test_create_ai_search_store_reset_create_index_error(self, mock_azure_search_patches):
        """Test Azure AI Search reset when create index operation fails."""
        from generic_rag.backend.vectordb import _create_ai_search_store

        with mock_azure_search_patches() as (mock_objects, mock_azure_search):
            # Mock a failure in create_or_update_index operation
            create_error = Exception("Create index operation failed")
            mock_objects.index_client.create_or_update_index.side_effect = create_error

            # Act & Assert
            with pytest.raises(Exception, match="Create index operation failed"):
                _create_ai_search_store(
                    self.settings, self.mock_embedding_function, self.collection_name, self.location, reset=True
                )

    def test_create_ai_search_store_success_without_reset(self, mock_azure_search_patches):
        """Test successful Azure AI Search store creation without reset."""
        from generic_rag.backend.vectordb import _create_ai_search_store

        with mock_azure_search_patches() as (mock_objects, mock_azure_search):
            # Act
            result = _create_ai_search_store(
                self.settings, self.mock_embedding_function, self.collection_name, self.location, reset=False
            )

            # Assert
            assert result == mock_objects.azure_search_instance
            # Verify reset operations were NOT called
            mock_objects.index_client.delete_index.assert_not_called()
            mock_objects.index_client.create_or_update_index.assert_called_once()

    def test_create_ai_search_store_with_custom_index_name(self, mock_azure_search_patches):
        """Test Azure AI Search creation with custom index name from settings."""
        from generic_rag.backend.vectordb import _create_ai_search_store

        # Settings with custom index name
        settings_with_custom_index = AppSettings(
            azure_ai_search=AzureAISearchSettings(
                endpoint="https://test-search.search.windows.net",
                index_name="custom_index_name",
                semantic_configuration_name="test_semantic",
                vector_search_dimensions=512,  # Different dimensions
                vector_search_profile_name="test_profile",
            )
        )

        with mock_azure_search_patches() as (mock_objects, mock_azure_search):
            # Act
            result = _create_ai_search_store(
                settings_with_custom_index,
                self.mock_embedding_function,
                self.collection_name,
                self.location,
                reset=False,
            )

            # Assert
            assert result == mock_objects.azure_search_instance
            # The logic uses settings.index_name when available, not collection_name
            call_args = mock_azure_search.call_args
            assert call_args.kwargs["index_name"] == "custom_index_name"

    def test_create_ai_search_store_uses_collection_name_when_no_index_name(self, mock_azure_search_patches):
        """Test Azure AI Search uses collection_name when no index_name in settings."""
        from generic_rag.backend.vectordb import _create_ai_search_store

        # Settings with explicitly empty index_name to test the fallback logic
        settings_no_index_name = AppSettings(
            azure_ai_search=AzureAISearchSettings(
                endpoint="https://test-search.search.windows.net",
                index_name=None,  # Explicitly set to None to test fallback
                semantic_configuration_name="test_semantic",
                vector_search_dimensions=1536,
                vector_search_profile_name="test_profile",
            )
        )

        with mock_azure_search_patches() as (mock_objects, mock_azure_search):
            # Act
            result = _create_ai_search_store(
                settings_no_index_name, self.mock_embedding_function, self.collection_name, self.location, reset=False
            )

            # Assert
            assert result == mock_objects.azure_search_instance
            # Should use collection_name when index_name is None
            call_args = mock_azure_search.call_args
            assert call_args.kwargs["index_name"] == self.collection_name

    def test_create_ai_search_store_uses_default_index_name(self, mock_azure_search_patches):
        """Test Azure AI Search uses default index_name from settings when not overridden."""
        from generic_rag.backend.vectordb import _create_ai_search_store

        # Settings using the default index_name value
        settings_default_index = AppSettings(
            azure_ai_search=AzureAISearchSettings(
                endpoint="https://test-search.search.windows.net",
                # index_name will default to "generic_rag"
                semantic_configuration_name="test_semantic",
                vector_search_dimensions=1536,
                vector_search_profile_name="test_profile",
            )
        )

        with mock_azure_search_patches() as (mock_objects, mock_azure_search):
            # Act
            result = _create_ai_search_store(
                settings_default_index, self.mock_embedding_function, self.collection_name, self.location, reset=False
            )

            # Assert
            assert result == mock_objects.azure_search_instance
            # Should use the default index_name ("generic_rag") from settings
            call_args = mock_azure_search.call_args
            assert call_args.kwargs["index_name"] == "generic_rag"

    def test_create_ai_search_store_create_index_error_without_reset(self, mock_azure_search_patches):
        """Test Azure AI Search creation when create_or_update_index operation fails without reset."""
        from generic_rag.backend.vectordb import _create_ai_search_store

        with mock_azure_search_patches() as (mock_objects, mock_azure_search):
            # Mock a failure in create_or_update_index operation
            create_error = Exception("Create/update index operation failed")
            mock_objects.index_client.create_or_update_index.side_effect = create_error

            # Act & Assert
            with pytest.raises(Exception, match="Create/update index operation failed"):
                _create_ai_search_store(
                    self.settings, self.mock_embedding_function, self.collection_name, self.location, reset=False
                )


class TestVectorStoreIntegration:
    """Integration tests for vector store functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_embedding_function = Mock()
        self.collection_name = "test_collection"

    def test_vector_store_protocol_compliance(self):
        """Test that the VectorStore protocol is correctly defined."""
        from generic_rag.backend.vectordb import VectorStore

        # Check that VectorStore has the required methods
        assert hasattr(VectorStore, "add_documents")
        assert hasattr(VectorStore, "similarity_search")
        assert hasattr(VectorStore, "reset_collection")

    def test_chroma_vector_store_creation_with_mock_dependencies(self):
        """Test Chroma creation when dependencies are available."""
        settings = AppSettings(
            vector_db=VectorDbSettings(type=VectorDbType.chroma, location=Path("/tmp/test_chroma"), reset=False)
        )

        # Mock the Chroma class at the langchain_chroma module level
        mock_chroma_instance = Mock()
        mock_chroma_instance.add_documents = Mock()
        mock_chroma_instance.similarity_search = Mock(return_value=[])
        mock_chroma_instance.reset_collection = Mock()

        with patch.dict("sys.modules", {"langchain_chroma": Mock()}):
            with patch("langchain_chroma.Chroma", return_value=mock_chroma_instance):
                # Act
                result = get_vector_store(settings, self.mock_embedding_function, self.collection_name)

                # Assert
                assert result is not None
                # Verify it implements the VectorStore protocol
                assert hasattr(result, "add_documents")
                assert hasattr(result, "similarity_search")
                assert hasattr(result, "reset_collection")

    def test_ai_search_vector_store_creation_with_mock_dependencies(self, mock_azure_search_patches):
        """Test Azure AI Search creation when dependencies are available."""
        settings = AppSettings(
            vector_db=VectorDbSettings(
                type=VectorDbType.ai_search, location=Path("/unused/for/ai/search"), reset=False
            ),
            azure_ai_search=AzureAISearchSettings(endpoint="https://test-search.search.windows.net"),
        )

        with mock_azure_search_patches() as (mock_objects, mock_azure_search):
            # Set up the mock azure search instance with the required methods
            mock_objects.azure_search_instance.add_documents = Mock()
            mock_objects.azure_search_instance.similarity_search = Mock(return_value=[])
            mock_objects.azure_search_instance.reset_collection = Mock()

            # Act
            result = get_vector_store(settings, self.mock_embedding_function, self.collection_name)

            # Assert
            assert result is not None
            # Verify it implements the VectorStore protocol
            assert hasattr(result, "add_documents")
            assert hasattr(result, "similarity_search")
            assert hasattr(result, "reset_collection")

    def test_settings_parameter_validation(self):
        """Test that settings are properly validated and passed through."""
        # Test with minimal valid settings
        settings = AppSettings(
            vector_db=VectorDbSettings(type=VectorDbType.chroma, location=Path("/tmp/minimal_test"), reset=False)
        )

        with patch("generic_rag.backend.vectordb._create_chroma_store") as mock_create:
            mock_create.return_value = Mock()

            # Act
            get_vector_store(settings, self.mock_embedding_function, self.collection_name)

            # Assert that settings were passed correctly
            mock_create.assert_called_once_with(
                self.mock_embedding_function, self.collection_name, "/tmp/minimal_test", False
            )

    def test_location_path_conversion(self):
        """Test that Path objects are correctly converted to strings."""
        settings = AppSettings(
            vector_db=VectorDbSettings(
                type=VectorDbType.chroma, location=Path("/some/complex/path/with/nested/dirs"), reset=True
            )
        )

        with patch("generic_rag.backend.vectordb._create_chroma_store") as mock_create:
            mock_create.return_value = Mock()

            # Act
            get_vector_store(settings, self.mock_embedding_function, self.collection_name)

            # Assert that Path was converted to string
            mock_create.assert_called_once_with(
                self.mock_embedding_function,
                self.collection_name,
                "/some/complex/path/with/nested/dirs",  # Should be string, not Path
                True,
            )
