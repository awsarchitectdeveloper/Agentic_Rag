import sys
from enum import Enum
from pathlib import Path
from typing import List, Optional
import yaml
from pydantic import BaseModel, Field, ValidationError


class ChatBackend(str, Enum):
    azure = "azure"
    openai = "openai"
    google_vertex = "google_vertex"
    aws = "aws"
    local = "local"
    huggingface = "huggingface"
    together = "together"
    odin = "odin"

    def __str__(self):
        return self.value


class EmbeddingBackend(str, Enum):
    azure = "azure"
    openai = "openai"
    google_vertex = "google_vertex"
    aws = "aws"
    local = "local"
    huggingface = "huggingface"
    together = "together"
    odin = "odin"

    def __str__(self):
        return self.value


class VectorDbType(str, Enum):
    chroma = "chroma"
    ai_search = "ai_search"

    def __str__(self):
        return self.value


class ParserSelection(str, Enum):
    docling = "docling"
    legacy = "legacy"

    def __str__(self):
        return self.value


class OdinSettings(BaseModel):
    """Odin specific settings."""

    odin_llm_endpoint: Optional[str] = None
    odin_emb_endpoint: Optional[str] = None
    odin_llm_model: Optional[str] = None
    odin_emb_model: Optional[str] = None


class AzureSettings(BaseModel):
    """Azure specific settings."""

    llm_endpoint: Optional[str] = None
    llm_deployment_name: Optional[str] = None
    llm_api_version: Optional[str] = None
    emb_endpoint: Optional[str] = None
    emb_deployment_name: Optional[str] = None
    emb_api_version: Optional[str] = None
    agent_endpoint: Optional[str] = None
    agent_deployment_name: Optional[str] = None
    agent_api_version: Optional[str] = None
    stt_endpoint: Optional[str] = None
    stt_deployment_name: Optional[str] = None
    stt_api_version: Optional[str] = None
    tts_endpoint: Optional[str] = None
    tts_deployment_name: Optional[str] = None
    tts_api_version: Optional[str] = None
    bearer_token_scope: Optional[str] = None


class BlobSettings(BaseModel):
    """Azure Blob Storage specific settings."""

    storage_account_url: Optional[str] = None
    input_container_name: Optional[str] = None
    input_blob_prefix: Optional[str] = None


class OpenAISettings(BaseModel):
    """OpenAI specific settings."""

    chat_model: Optional[str] = None
    emb_model: Optional[str] = None


class GoogleVertexSettings(BaseModel):
    """Google Vertex specific settings."""

    project_id: Optional[str] = None
    location: Optional[str] = None
    chat_model: Optional[str] = None
    emb_model: Optional[str] = None


class AwsSettings(BaseModel):
    """AWS specific settings (e.g., for Bedrock)."""

    chat_model: Optional[str] = None
    emb_model: Optional[str] = None
    region: Optional[str] = None


class TogetherSettings(BaseModel):
    chat_model: Optional[str] = None
    emb_model: Optional[str] = None


class LocalSettings(BaseModel):
    """Local backend specific settings (e.g., Ollama models)."""

    chat_model: Optional[str] = None
    emb_model: Optional[str] = None


class HuggingFaceSettings(BaseModel):
    """HuggingFace specific settings (if different from local embeddings)."""

    chat_model: Optional[str] = None
    emb_model: Optional[str] = None


class PdfSettings(BaseModel):
    """PDF processing settings."""

    data: List[Path] = Field(default_factory=list)
    unstructured: bool = Field(default=False)
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    add_start_index: bool = Field(default=False)


class WebSettings(BaseModel):
    """Web data processing settings."""

    data: List[str] = Field(default_factory=list)
    chunk_size: int = Field(default=200)
    enforce_chunk_size: bool = Field(default=True)


class VectorDbSettings(BaseModel):
    """Vector Database settings."""

    type: VectorDbType = Field(default=VectorDbType.chroma)
    location: Path = Field(default=Path(".chroma_db"))
    reset: bool = Field(default=False)


class AzureAISearchSettings(BaseModel):
    """Azure AI Search specific settings."""

    endpoint: str
    index_name: Optional[str] = "generic_rag"
    semantic_configuration_name: str = Field(default="default")
    vector_search_dimensions: int = Field(default=1536)  # Default for OpenAI embeddings
    vector_search_profile_name: str = Field(default="myHnswProfile")
    mi_resource_id: str = Field(default="")


class TextSettings(BaseModel):
    """Text processing settings."""

    data: List[Path] = Field(default_factory=list)


class SharePointSettings(BaseModel):
    """SharePoint settings."""

    tenant_id: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    site_name: Optional[str] = None
    site_path: Optional[str] = None
    library_name: Optional[str] = None
    folder_path: str = Field(default="General")


class DoclingSettings(BaseModel):
    """Docling PDF processing settings."""

    do_ocr: bool = Field(default=True)
    do_table_structure: bool = Field(default=True)
    do_cell_matching: bool = Field(default=True)
    images_scale: float = Field(default=2.0)
    generate_page_images: bool = Field(default=True)
    generate_picture_images: bool = Field(default=True)
    force_full_page_ocr: bool = Field(default=True)


class OcrSettings(BaseModel):
    """OCR settings for Azure Document Intelligence."""

    azure_doc_intel_endpoint: str = Field(..., description="Azure Document Intelligence endpoint")
    enable_ocr_fallback: bool = Field(default=True, description="Enable OCR fallback for image placeholders")


class AppSettings(BaseModel):
    """
    Main application settings model.

    Loads configuration from a YAML file using the structure defined
    by the nested models.
    """

    # --- Top-level settings ---
    chat_backend: ChatBackend = Field(default=ChatBackend.local)
    emb_backend: EmbeddingBackend = Field(default=EmbeddingBackend.huggingface)
    graph_selection: str = Field(default="ret_gen")
    parser_selection: ParserSelection = Field(default=ParserSelection.docling)
    chainlit_starters: Optional[list[dict[str, str]]] = None
    use_reranker: bool = Field(default=False)
    use_summarization: bool = Field(default=False)
    voice_chat: bool = Field(default=False)

    # --- Evaluation Setting ---
    evaluation_enabled: bool = Field(default=False, description="Enable or disable LLM-based evaluation of answers")

    # --- File Processing Settings ---
    allowed_extensions: List[str] = Field(default_factory=lambda: [".pdf", ".docx", ".xlsx", ".pptx"])

    # --- Provider-specific settings ---
    azure: Optional[AzureSettings] = None
    openai: Optional[OpenAISettings] = None
    google_vertex: Optional[GoogleVertexSettings] = None
    aws: Optional[AwsSettings] = None
    local: Optional[LocalSettings] = None
    huggingface: Optional[HuggingFaceSettings] = None  # Separate HF config if needed
    together: Optional[TogetherSettings] = None
    odin: Optional[OdinSettings] = None

    # --- Storage and Search Settings ---
    azure_ai_search: Optional[AzureAISearchSettings] = None
    vector_db: VectorDbSettings = Field(default_factory=VectorDbSettings)

    # --- Data processing settings ---
    local_data: TextSettings = Field(default_factory=TextSettings)
    pdf: PdfSettings = Field(default_factory=PdfSettings)
    web: WebSettings = Field(default_factory=WebSettings)
    blob: BlobSettings = Field(default_factory=BlobSettings)
    sharepoint: SharePointSettings = Field(default_factory=SharePointSettings)
    text: TextSettings = Field(default_factory=TextSettings)
    docling: DoclingSettings = Field(default_factory=DoclingSettings)
    ocr: Optional[OcrSettings] = None


# --- Configuration Loading Function ---
def load_settings(config_path: Path = Path("config.yaml")) -> AppSettings:
    """
    Loads settings from a YAML file and validates them using Pydantic models.

    Args:
        config_path: The path to the configuration YAML file.

    Returns:
        An instance of AppSettings containing the loaded configuration.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
        ValidationError: If the data in the file doesn't match the AppSettings model.
    """
    if not config_path.is_file():
        print(f"Error: Configuration file not found at '{config_path}'", file=sys.stderr)
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    print(f"--- Loading settings from '{config_path}' ---")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
            if config_data is None:
                config_data = {}

        settings = AppSettings(**config_data)
        print("--- Settings loaded and validated successfully ---")
        return settings

    except yaml.YAMLError as e:
        print(f"Error parsing YAML file '{config_path}':\n  {e}", file=sys.stderr)
        raise
    except ValidationError as e:
        print(f"Error validating configuration from '{config_path}':\n{e}", file=sys.stderr)
        raise
    except Exception as e:
        print(f"An unexpected error occurred while loading settings from '{config_path}': {e}", file=sys.stderr)
        raise
