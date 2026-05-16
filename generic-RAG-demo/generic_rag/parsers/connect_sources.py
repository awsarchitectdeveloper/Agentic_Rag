import os
import requests
from azure.storage.blob import ContainerClient
from pathlib import Path
import tempfile

try:
    from msal import ConfidentialClientApplication
except Exception:  # pragma: no cover - optional dependency
    ConfidentialClientApplication = None

try:
    from azure.identity import DefaultAzureCredential
except Exception:  # pragma: no cover - optional dependency
    DefaultAzureCredential = None
from typing import Any
import logging
import time
import io
from PIL import Image, ImageEnhance
import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions

logger = logging.getLogger(__name__)


def _extract_docling_metadata(doc) -> dict:
    """Extract rich metadata from docling document."""
    metadata = {}

    # Extract document title
    if hasattr(doc, "description") and doc.description and hasattr(doc.description, "title"):
        metadata["title"] = doc.description.title
    elif hasattr(doc, "name"):
        metadata["title"] = doc.name

    # Extract page count
    if hasattr(doc, "pages") and doc.pages:
        metadata["page_count"] = len(doc.pages)
    else:
        metadata["page_count"] = 0

    # Extract table information (optional)
    if hasattr(doc, "tables") and doc.tables:
        metadata["table_count"] = len(doc.tables)
        metadata["has_tables"] = True
    else:
        metadata["table_count"] = 0
        metadata["has_tables"] = False

    # Extract image information (optional)
    if hasattr(doc, "pictures") and doc.pictures:
        metadata["image_count"] = len(doc.pictures)
        metadata["has_images"] = True
        # Extract image descriptions if available
        image_descriptions = []
        for picture in doc.pictures:
            if hasattr(picture, "description") and picture.description:
                image_descriptions.append(picture.description)
        if image_descriptions:
            metadata["image_descriptions"] = image_descriptions
    else:
        metadata["image_count"] = 0
        metadata["has_images"] = False

    return metadata


def _document_exists_in_vector_store(vector_store: Any, source_filter: dict) -> bool:
    """Check if a document with the given source filter already exists in the vector store."""
    try:
        if hasattr(vector_store, "get"):
            result = vector_store.get(where=source_filter, limit=1)
            return len(result.get("ids", [])) > 0
        else:
            logger.info(
                f"Vector store {type(vector_store).__name__} doesn't support existence checking. Processing all documents."
            )
            return False
    except Exception as e:
        logger.warning(f"Error checking document existence: {e}. Assuming document doesn't exist.")
        return False


def _group_texts_by_page(doc) -> dict[int, list[str]]:
    """Group text elements from a docling document by page number using provenance data.

    Returns:
        Dictionary mapping page numbers to lists of text strings
    """
    texts_by_page = {}

    if not hasattr(doc, "texts") or not doc.texts:
        return texts_by_page

    for text_elem in doc.texts:
        if not hasattr(text_elem, "prov") or not text_elem.prov:
            continue

        for prov in text_elem.prov:
            page_num = getattr(prov, "page_no", None)
            if page_num is None:
                continue

            try:
                page_num = int(page_num)
            except (ValueError, TypeError):
                continue

            text_content = text_elem.text if hasattr(text_elem, "text") else str(text_elem)
            if page_num not in texts_by_page:
                texts_by_page[page_num] = []
            texts_by_page[page_num].append(text_content)
            break  # Only use first provenance per text element

    return texts_by_page


def _process_docling_pages(doc, base_metadata: dict, docling_metadata: dict) -> list[Document]:
    """
    Process a docling document and extract page-specific Documents.

    Args:
        doc: Docling document object
        base_metadata: Base metadata dict (source, filetype, filename, etc.)
        docling_metadata: Rich metadata from _extract_docling_metadata

    Returns:
        List of LangChain Documents, one per page
    """
    documents = []

    # Try to extract text by page using provenance data
    texts_by_page = _group_texts_by_page(doc)

    if texts_by_page:
        logger.info(f"Document has {len(texts_by_page)} pages with provenance, processing each page separately")
        for page_num in sorted(texts_by_page.keys()):
            page_content = "\n\n".join(texts_by_page[page_num])
            page_metadata = {**base_metadata, "page": page_num, **docling_metadata}
            documents.append(Document(page_content=page_content, metadata=page_metadata))

    # Fallback: if page processing didn't work, use entire document
    if not documents:
        logger.warning("Could not process pages, using fallback")
        markdown_content = doc.export_to_markdown()

        # If we know the page count, split content proportionally and assign sequential page numbers
        if "page_count" in docling_metadata and docling_metadata["page_count"] > 1:
            logger.info(f"Document has {docling_metadata['page_count']} pages, creating sequential page assignments")
            page_count = docling_metadata["page_count"]
            content_length = len(markdown_content)
            chars_per_page = content_length // page_count

            for page_num in range(1, page_count + 1):
                start_idx = (page_num - 1) * chars_per_page
                end_idx = start_idx + chars_per_page if page_num < page_count else content_length
                page_content = markdown_content[start_idx:end_idx]

                page_metadata = {
                    **base_metadata,
                    "page": page_num,  # Sequential page number
                    **docling_metadata,
                }

                documents.append(Document(page_content=page_content, metadata=page_metadata))
        else:
            # Single page or unknown page count - use page 1
            metadata = {**base_metadata, "page": 1, **docling_metadata}

            documents.append(Document(page_content=markdown_content, metadata=metadata))

    return documents


def _get_all_local_files(local_paths: list[Path], allowed_extensions: list[str]) -> list[Path]:
    """
    Function that takes a list of local paths,
    that might contain directories paths and/or direct file paths,
    and returns a list with all file paths that have allowed extensions or any files with allowed extensions found in the directory file paths.
    This function does not scan directories recursively.
    """
    all_files = []
    for path in local_paths:
        if path.is_dir():
            # Get all files with allowed extensions from the directory
            for ext in allowed_extensions:
                # Remove the dot from extension for glob pattern
                ext_pattern = ext.lstrip(".")
                all_files.extend(list(path.glob(f"*.{ext_pattern}")))
        elif path.is_file():
            ext = path.suffix.lower()
            if ext in allowed_extensions:
                all_files.append(path)
            else:
                logger.warning(
                    f"Ignoring file {path} as it has unsupported extension {ext}. Allowed: {allowed_extensions}"
                )
        else:
            logger.warning(f"Ignoring path {path} as it is not a valid file or directory.")

    return all_files


def add_local(
    vector_store: Any,
    file_paths: list[Path],
    allowed_extensions: list[str],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    """
    Adds a list of local files as vector documents to the provided vector store.

    The local file will be parsed with docling page-by-page and split into chunks of text with the provided chunk size and overlap.
    Each chunk will have the specific page number it came from.
    """

    logger.info("Adding local files to vector store.")

    # Get all individual files from the provided paths (handles both files and directories)
    all_files = _get_all_local_files(file_paths, allowed_extensions)
    logger.info(f"Found {len(all_files)} files to process from {len(file_paths)} input paths.")

    all_documents = []
    for file in all_files:
        if _document_exists_in_vector_store(vector_store, {"source": str(file)}):
            logger.info(f"Skipping file {file}, as it is already in the database.")
            continue

        doc = docling_parser(file)

        # Extract rich docling metadata
        docling_metadata = _extract_docling_metadata(doc)

        # Process pages using helper function
        base_metadata = {"source": str(file), "filetype": "local", "filename": os.path.basename(file)}

        page_documents = _process_docling_pages(doc, base_metadata, docling_metadata)
        all_documents.extend(page_documents)

    if not all_documents:
        logger.info("No new local documents to add.")
        return

    # Split documents (this will now split pages into smaller chunks while preserving page metadata)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(all_documents)

    logger.info(f"{len(all_files)} local files split into {len(splits)} vector store documents")

    # Add to vector store
    try:
        filtered_splits = filter_complex_metadata(splits)
        logger.info(f"Adding {len(filtered_splits)} documents to vector store...")
        vector_store.add_documents(documents=filtered_splits)
        logger.info("Successfully added local documents to vector store.")
    except Exception as e:
        logger.error(f"Error adding local documents to vector store: {e}")
        raise


def add_web(vector_store: Any, urls: list[str], chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Adds a list of web URLs as vector documents to the provided vector store.

    The web page will be fetched and parsed with docling, then split into chunks of text with the provided chunk size and overlap.
    """

    logger.info("Adding web URLs to vector store.")

    # WARNING: SSL verification is disabled below.
    original_get = requests.get

    def unsafe_get(*args, **kwargs):
        kwargs["verify"] = False
        return original_get(*args, **kwargs)

    requests.get = unsafe_get

    all_documents = []
    try:
        for url in urls:
            if _document_exists_in_vector_store(vector_store, {"source": url}):
                logger.info(f"Skipping URL {url}, as it is already in the database.")
                continue

            doc = docling_parser(url)
            markdown_content = doc.export_to_markdown()

            # Extract rich docling metadata
            docling_metadata = _extract_docling_metadata(doc)

            # Create LangChain document with enriched metadata
            metadata = {
                "source": url,
                "filetype": "web",
                **docling_metadata,  # Add all docling metadata
            }

            langchain_doc = Document(page_content=markdown_content, metadata=metadata)
            all_documents.append(langchain_doc)
    finally:
        requests.get = original_get

    if not all_documents:
        logger.info("No new web documents to add.")
        return

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(all_documents)

    logger.info(f"{len(urls)} web URLs split into {len(splits)} vector store documents")

    # Add to vector store
    try:
        filtered_splits = filter_complex_metadata(splits)
        logger.info(f"Adding {len(filtered_splits)} documents to vector store...")
        vector_store.add_documents(documents=filtered_splits)
        logger.info("Successfully added web documents to vector store.")
    except Exception as e:
        logger.error(f"Error adding web documents to vector store: {e}")
        raise


def add_blob(
    vector_store: Any,
    account_url: str,
    container_name: str,
    folder_prefix: str,
    allowed_extensions: list[str],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    """
    Adds files from Azure Blob Storage as vector documents to the provided vector store.

    The blob files will be downloaded and parsed with docling, then split into chunks of text with the provided chunk size and overlap.
    """

    logger.info("Adding blob storage files to vector store.")

    container_client = ContainerClient(account_url=account_url, container_name=container_name, credential=None)

    all_documents = []
    for blob in container_client.list_blobs(name_starts_with=folder_prefix):
        ext = Path(blob.name).suffix.lower()
        if ext not in allowed_extensions:
            logger.info(f"Skipping blob {blob.name} due to unsupported extension {ext}.")
            continue

        # Create a proper blob URL for clickability
        # Format: https://storageaccount.blob.core.windows.net/container/filename
        blob_source = f"{account_url.rstrip('/')}/{container_name}/{blob.name}"

        if _document_exists_in_vector_store(vector_store, {"source": blob_source}):
            logger.info(f"Skipping blob {blob.name}, as it is already in the database.")
            continue

        logger.info(f"Processing {blob.name} - Last modified: {blob.last_modified.isoformat()}")

        blob_client = container_client.get_blob_client(blob.name)
        blob_data = blob_client.download_blob().readall()

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(blob_data)
            tmp.flush()
            tmp_path = tmp.name

        try:
            doc = docling_parser(tmp_path)

            # Extract rich docling metadata
            docling_metadata = _extract_docling_metadata(doc)

            # Group text elements by page using provenance data
            texts_by_page = _group_texts_by_page(doc)

            # Process each page separately to maintain page-level metadata
            if texts_by_page:
                for page_num in sorted(texts_by_page.keys()):
                    page_content = "\n\n".join(texts_by_page[page_num])

                    page_metadata = {
                        "source": blob_source,
                        "filetype": "blob",
                        "filename": blob.name,
                        "last_modified": blob.last_modified.isoformat(),
                        "page": page_num,
                        **docling_metadata,
                    }

                    langchain_doc = Document(page_content=page_content, metadata=page_metadata)
                    all_documents.append(langchain_doc)
            else:
                # Fallback: process entire document if pages aren't available
                logger.warning(f"No page provenance found for {blob.name}, using fallback with page=1")
                markdown_content = doc.export_to_markdown()
                metadata = {
                    "source": blob_source,
                    "filetype": "blob",
                    "filename": blob.name,
                    "last_modified": blob.last_modified.isoformat(),
                    "page": 1,  # Default page number when no provenance available
                    **docling_metadata,
                }

                langchain_doc = Document(page_content=markdown_content, metadata=metadata)
                all_documents.append(langchain_doc)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    if not all_documents:
        logger.info("No new blob documents to add.")
        return

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(all_documents)

    logger.info(f"Blob files split into {len(splits)} vector store documents")

    # Add to vector store
    try:
        filtered_splits = filter_complex_metadata(splits)
        logger.info(f"Adding {len(filtered_splits)} documents to vector store...")
        vector_store.add_documents(documents=filtered_splits)
        logger.info("Successfully added blob documents to vector store.")
    except Exception as e:
        logger.error(f"Error adding blob documents to vector store: {e}")
        raise


def add_sharepoint(
    vector_store: Any,
    tenant_id: str,
    client_id: str,
    client_secret: str,
    site_name: str,
    site_path: str,
    folder_path: str,
    library_name: str,
    allowed_extensions: list[str],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    """
    Adds files from SharePoint as vector documents to the provided vector store.

    The SharePoint files will be downloaded and parsed with docling, then split into chunks of text with the provided chunk size and overlap.
    """

    logger.info("Adding SharePoint documents to vector store.")

    def download_file(file_info, headers):
        file_name = file_info["name"]
        file_id = file_info["id"]
        download_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{file_id}/content"

        # WARNING: SSL verification is disabled below.
        response = requests.get(download_url, headers=headers, verify=False)
        if response.status_code == 200:
            with open(file_name, "wb") as f:
                f.write(response.content)
            return file_name
        else:
            logger.error(f"{response.status_code} Failed to download {file_name}")
            return None

    # Authentication
    AUTHORITY = f"https://login.microsoftonline.com/{tenant_id}"
    SCOPE = ["https://graph.microsoft.com/.default"]

    app = ConfidentialClientApplication(client_id=client_id, authority=AUTHORITY, client_credential=client_secret)

    token_response = app.acquire_token_for_client(scopes=SCOPE)
    access_token = token_response.get("access_token")

    headers = {"Authorization": f"Bearer {access_token}"}

    # Get Site ID
    site_url = f"https://graph.microsoft.com/v1.0/sites/{site_name}:{site_path}"
    site_response = requests.get(site_url, headers=headers)
    site_id = site_response.json().get("id")

    # Get Drive ID
    drive_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives"
    drive_response = requests.get(drive_url, headers=headers)
    drives = drive_response.json().get("value", [])
    drive_id = next((d["id"] for d in drives if d["name"] == library_name), None)

    # List Files
    files_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root:/{folder_path}:/children"
    files_response = requests.get(files_url, headers=headers)
    files = files_response.json().get("value", [])

    all_documents = []
    for file in files:
        ext = os.path.splitext(file["name"])[1].lower()
        if ext not in allowed_extensions:
            logger.info(f"Skipping SharePoint file {file['name']} due to unsupported extension {ext}.")
            continue

        # Use the webUrl from Graph API response for proper SharePoint URL
        # If webUrl is not available, use a fallback identifier that won't be clickable
        sharepoint_source = file.get(
            "webUrl", f"sharepoint://{site_name}{site_path}/{library_name}/{folder_path}/{file['name']}"
        )

        if _document_exists_in_vector_store(vector_store, {"source": sharepoint_source}):
            logger.info(f"Skipping SharePoint file {file['name']}, as it is already in the database.")
            continue

        local_file = download_file(file, headers)
        if local_file:
            try:
                doc = docling_parser(local_file)

                # Extract rich docling metadata
                docling_metadata = _extract_docling_metadata(doc)

                # Group text elements by page using provenance data
                texts_by_page = _group_texts_by_page(doc)

                # Process each page separately to maintain page-level metadata
                if texts_by_page:
                    for page_num in sorted(texts_by_page.keys()):
                        page_content = "\n\n".join(texts_by_page[page_num])

                        page_metadata = {
                            "source": sharepoint_source,
                            "filetype": "sharepoint",
                            "filename": file["name"],
                            "site_name": site_name,
                            "library_name": library_name,
                            "folder_path": folder_path,
                            "page": page_num,
                            **docling_metadata,
                        }

                        langchain_doc = Document(page_content=page_content, metadata=page_metadata)
                        all_documents.append(langchain_doc)
                else:
                    # Fallback: process entire document if pages aren't available
                    logger.warning(f"No page provenance found for {file['name']}, using fallback with page=1")
                    markdown_content = doc.export_to_markdown()
                    metadata = {
                        "source": sharepoint_source,
                        "filetype": "sharepoint",
                        "filename": file["name"],
                        "site_name": site_name,
                        "library_name": library_name,
                        "folder_path": folder_path,
                        "page": 1,  # Default page number when no provenance available
                        **docling_metadata,
                    }

                    langchain_doc = Document(page_content=markdown_content, metadata=metadata)
                    all_documents.append(langchain_doc)
            finally:
                os.remove(local_file)

    if not all_documents:
        logger.info("No new SharePoint documents to add.")
        return

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(all_documents)

    logger.info(f"SharePoint files split into {len(splits)} vector store documents")

    # Add to vector store
    try:
        filtered_splits = filter_complex_metadata(splits)
        logger.info(f"Adding {len(filtered_splits)} documents to vector store...")
        vector_store.add_documents(documents=filtered_splits)
        logger.info("Successfully added SharePoint documents to vector store.")
    except Exception as e:
        logger.error(f"Error adding SharePoint documents to vector store: {e}")
        raise


def preprocess_image(image_bytes):
    """Enhance image contrast for better OCR results."""
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    img = ImageEnhance.Contrast(img).enhance(2.0)
    output = io.BytesIO()
    img.save(output, format="PNG")
    return output.getvalue()


def analyze_document_intelligence(image_bytes, endpoint):
    """Send image to Azure Document Intelligence for OCR using managed identity."""
    # Get access token using managed identity
    credential = DefaultAzureCredential()
    token = credential.get_token("https://cognitiveservices.azure.com/.default").token

    headers = {"Content-Type": "application/octet-stream", "Authorization": f"Bearer {token}"}
    url = f"{endpoint}/formrecognizer/documentModels/prebuilt-layout:analyze?api-version=2023-07-31"
    response = requests.post(url, headers=headers, data=image_bytes)
    response.raise_for_status()
    operation_location = response.headers["Operation-Location"]

    # Poll for results with same token
    while True:
        result = requests.get(operation_location, headers={"Authorization": f"Bearer {token}"})
        result.raise_for_status()
        analysis = result.json()
        status = analysis.get("status")
        if status == "succeeded":
            return analysis
        elif status == "failed":
            raise Exception("Document Intelligence failed")
        time.sleep(1)


def extract_text_from_layout_result(result):
    """Extract text lines from Azure Document Intelligence result."""
    lines = []
    for page in result.get("analyzeResult", {}).get("pages", []):
        for line in page.get("lines", []):
            lines.append(line.get("content", ""))
    return lines


def extract_page_image(pdf_path, page_index, zoom=3):
    """Render a PDF page as an image."""
    pdf = fitz.open(pdf_path)
    pix = pdf[page_index].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    return pix.tobytes("png")


def extract_images_from_page(pdf_path, page_index):
    """Extract embedded images from a PDF page."""
    pdf = fitz.open(pdf_path)
    page = pdf[page_index]
    images = []
    for img in page.get_images(full=True):
        xref = img[0]
        pix = fitz.Pixmap(pdf, xref)
        if pix.n < 5:
            img_bytes = pix.tobytes("png")
        else:
            pix = fitz.Pixmap(fitz.csRGB, pix)
            img_bytes = pix.tobytes("png")
        images.append(img_bytes)
    return images


def docling_parser(file_path: str):
    """
    Parse documents with docling + OCR fallback for images.

    Args:
        file_path: Path to the document to parse
    """
    # Load settings from config file
    from generic_rag.parsers.config import load_settings

    app_settings = load_settings()
    config = app_settings.docling

    # Set up pipeline options
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = config.do_ocr
    pipeline_options.do_table_structure = config.do_table_structure
    pipeline_options.table_structure_options.do_cell_matching = config.do_cell_matching
    pipeline_options.images_scale = config.images_scale
    pipeline_options.generate_page_images = config.generate_page_images
    pipeline_options.generate_picture_images = config.generate_picture_images

    ocr_options = EasyOcrOptions(force_full_page_ocr=config.force_full_page_ocr)
    pipeline_options.ocr_options = ocr_options

    # Create converter with PDF format options
    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})

    # Convert with Docling
    doc = converter.convert(file_path).document
    docling_text = doc.export_to_markdown()

    # Check if OCR fallback is enabled and we have OCR settings
    ocr_settings = getattr(app_settings, "ocr", None)
    if not ocr_settings or not getattr(ocr_settings, "enable_ocr_fallback", False):
        # Return original docling result
        class MockDoc:
            def export_to_markdown(self):
                return docling_text

        return MockDoc()

    # OCR enhancement: replace image placeholders
    lines = docling_text.splitlines()
    placeholders = [line for line in lines if "<!-- image -->" in line]

    if not placeholders:
        # No image placeholders, return as-is
        class MockDoc:
            def export_to_markdown(self):
                return docling_text

        return MockDoc()

    logger.info(f"Found {len(placeholders)} image placeholders, replacing with OCR text...")

    # Extract images from PDF and replace placeholders
    try:
        pdf = fitz.open(str(file_path))
        all_images = []
        for i in range(len(pdf)):
            all_images.extend(extract_images_from_page(str(file_path), i))

        image_index = 0
        merged_output = []

        for line in lines:
            if "<!-- image -->" in line and image_index < len(all_images):
                img_bytes = all_images[image_index]
                image_index += 1
                preprocessed = preprocess_image(img_bytes)

                try:
                    result = analyze_document_intelligence(preprocessed, ocr_settings.azure_doc_intel_endpoint)
                    ocr_text = extract_text_from_layout_result(result)
                    ocr_text_str = " ".join(ocr_text) if ocr_text else "No text found"
                except Exception as e:
                    logger.error(f"OCR failed: {e}")
                    ocr_text_str = "OCR failed"

                new_line = line.replace("<!-- image -->", f"![Image OCR] {ocr_text_str}")
                merged_output.append(new_line)
            else:
                merged_output.append(line)

        final_text = "\n".join(merged_output)

    except Exception as e:
        logger.error(f"OCR processing failed: {e}")
        final_text = docling_text  # Fall back to original

    # Return enhanced result
    class MockDoc:
        def export_to_markdown(self):
            return final_text

    return MockDoc()
