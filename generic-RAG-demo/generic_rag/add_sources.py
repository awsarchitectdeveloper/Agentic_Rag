import logging
import sys
from pathlib import Path

from generic_rag.backend.models import get_embedding_model
from generic_rag.backend.vectordb import get_vector_store
from generic_rag.parsers.config import AppSettings, load_settings, ParserSelection
from generic_rag.parsers.connect_sources import add_local, add_web, add_blob, add_sharepoint
from generic_rag.parsers.parser import add_pdf_files, add_urls

logger = logging.getLogger("generic_rag")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

fh = logging.FileHandler(filename="generic-rag.add_sources.log", mode="w")
fh.setLevel(logging.DEBUG)

logging.basicConfig(
    encoding="utf-8",
    level=logging.DEBUG,
    handlers=[ch, fh],
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)

# Suppress Azure HTTP logging
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE_PATH = PROJECT_ROOT / "config.yaml"


def main():
    logger.info("Adding sources to vector store database.")

    try:
        settings: AppSettings = load_settings(CONFIG_FILE_PATH)
    except (FileNotFoundError, Exception):
        logger.error(f"Failed to load configuration from {CONFIG_FILE_PATH}. Exiting.")
        sys.exit(1)

    embedding_function = get_embedding_model(settings)

    vector_store = get_vector_store(
        settings=settings, embedding_function=embedding_function, collection_name="generic_rag"
    )

    # Choose parser based on configuration
    if settings.parser_selection == ParserSelection.docling:
        logger.info("Using docling-based parsers (connect_sources.py)")

        # Add local files with error handling
        try:
            add_local(
                vector_store=vector_store,
                file_paths=settings.local_data.data,
                allowed_extensions=settings.allowed_extensions,
            )
            logger.info("Successfully processed local files.")
        except Exception as e:
            logger.error(f"Failed to add local files: {e}")
            logger.info("Continuing with other sources...")

        # Add web sources with error handling
        try:
            add_web(vector_store=vector_store, urls=settings.web.data)
            logger.info("Successfully processed web sources.")
        except Exception as e:
            logger.error(f"Failed to add web sources: {e}")
            logger.info("Continuing with other sources...")

        # Add blob storage with error handling
        try:
            add_blob(
                vector_store=vector_store,
                account_url=settings.blob.storage_account_url,
                container_name=settings.blob.input_container_name,
                folder_prefix=settings.blob.input_blob_prefix,
                allowed_extensions=settings.allowed_extensions,
            )
            logger.info("Successfully processed blob storage.")
        except Exception as e:
            logger.error(f"Failed to add blob storage sources: {e}")
            logger.info("Continuing with other sources...")

        # Add SharePoint with error handling
        try:
            add_sharepoint(
                vector_store=vector_store,
                tenant_id=settings.sharepoint.tenant_id,
                client_id=settings.sharepoint.client_id,
                client_secret=settings.sharepoint.client_secret,
                site_name=settings.sharepoint.site_name,
                site_path=settings.sharepoint.site_path,
                folder_path=settings.sharepoint.folder_path,
                library_name=settings.sharepoint.library_name,
                allowed_extensions=settings.allowed_extensions,
            )
            logger.info("Successfully processed SharePoint sources.")
        except Exception as e:
            logger.error(f"Failed to add SharePoint sources: {e}")
            logger.info("Continuing with other sources...")

    elif settings.parser_selection == ParserSelection.legacy:
        logger.info("Using legacy parsers (parser.py)")

        # Add PDF files with error handling
        try:
            add_pdf_files(
                vector_store=vector_store,
                file_paths=settings.pdf.data,
                chunk_size=settings.pdf.chunk_size,
                chunk_overlap=settings.pdf.chunk_overlap,
                add_start_index=settings.pdf.add_start_index,
                unstructerd=settings.pdf.unstructured,
            )
            logger.info("Successfully processed PDF files.")
        except Exception as e:
            logger.error(f"Failed to add PDF files: {e}")
            logger.info("Continuing with other sources...")

        # Add URLs with error handling
        try:
            add_urls(
                vector_store=vector_store,
                urls=settings.web.data,
                chunk_size=settings.web.chunk_size,
                enforce_chunk_size=settings.web.enforce_chunk_size,
            )
            logger.info("Successfully processed URLs.")
        except Exception as e:
            logger.error(f"Failed to add URLs: {e}")
            logger.info("Continuing with other sources...")

    else:
        logger.info("Blob storage and SharePoint sources are only available with docling parser. Skipping.")


if __name__ == "__main__":
    main()
