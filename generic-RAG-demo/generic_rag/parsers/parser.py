import logging
from pathlib import Path
from typing import Any

import requests
import time
import io
from PIL import Image, ImageEnhance
import fitz  # PyMuPDF
from azure.identity import DefaultAzureCredential

from bs4 import BeautifulSoup, Tag
from docling.document_converter import DocumentConverter
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_text_splitters import HTMLSemanticPreservingSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Import modular OCR
from .modular_ocr import create_hybrid_processor

logger = logging.getLogger(__name__)


headers_to_split_on = [("h1", "Header 1"), ("h2", "Header 2")]

# Docling + OCR functions


def docling_parser(file_path: str) -> str:
    """Parse PDF using Docling and return markdown text."""
    try:
        converter = DocumentConverter()
        doc = converter.convert(file_path).document
        return doc.export_to_markdown()
    except Exception as e:
        logger.warning(f"Docling failed: {e}")
        return ""


def preprocess_image(image_bytes):
    """Enhance image contrast for better OCR results."""
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    img = ImageEnhance.Contrast(img).enhance(2.0)
    output = io.BytesIO()
    img.save(output, format="PNG")
    return output.getvalue()


def analyze_document_intelligence(image_bytes, endpoint):
    """Send image to Azure Document Intelligence for OCR using managed identity."""
    # Use managed identity for authentication
    credential = DefaultAzureCredential()
    token = credential.get_token("https://cognitiveservices.azure.com/.default")

    headers = {"Content-Type": "application/octet-stream", "Authorization": f"Bearer {token.token}"}
    url = f"{endpoint}/formrecognizer/documentModels/prebuilt-layout:analyze?api-version=2023-07-31"
    response = requests.post(url, headers=headers, data=image_bytes)
    response.raise_for_status()
    operation_location = response.headers["Operation-Location"]

    while True:
        result = requests.get(operation_location, headers={"Authorization": f"Bearer {token.token}"})
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


def extract_text_from_pdf(pdf_path: Path, settings) -> str:
    """
    Combined logic:
    1. Try Docling
    2. If Docling fails → full-page OCR
    3. If Docling has placeholders → fill with OCR text
    """
    docling_text = docling_parser(str(pdf_path))
    lines = docling_text.splitlines()
    placeholders = [line for line in lines if "<!-- image -->" in line]

    # Scenario 1: Docling failed → full-page OCR
    if docling_text.strip() == "":
        logger.info("Docling returned empty. Using full-page OCR...")
        pdf = fitz.open(str(pdf_path))
        full_text = []
        for i in range(len(pdf)):
            img_bytes = extract_page_image(str(pdf_path), i)
            preprocessed = preprocess_image(img_bytes)
            try:
                result = analyze_document_intelligence(preprocessed, settings.ocr.azure_doc_intel_endpoint)
                page_text = extract_text_from_layout_result(result)
                full_text.append("\n".join(page_text))
            except Exception as e:
                full_text.append(f"OCR failed on page {i}: {e}")
        return "\n\n".join(full_text)

    # Scenario 2: No placeholders → just Docling
    if not placeholders:
        return docling_text

    # Scenario 3: Mixed → replace placeholders with OCR text
    logger.info("Mixed content detected. Filling placeholders with OCR text...")
    pdf = fitz.open(str(pdf_path))
    all_images = []
    for i in range(len(pdf)):
        all_images.extend(extract_images_from_page(str(pdf_path), i))

    image_index = 0
    merged_output = []

    for line in lines:
        if "<!-- image -->" in line:
            if image_index < len(all_images):
                img_bytes = all_images[image_index]
                image_index += 1
                preprocessed = preprocess_image(img_bytes)
                try:
                    result = analyze_document_intelligence(preprocessed, settings.ocr.azure_doc_intel_endpoint)
                    ocr_text = extract_text_from_layout_result(result)
                    ocr_text_str = " ".join(ocr_text) if ocr_text else "No text found"
                except Exception:
                    ocr_text_str = "OCR failed"
            else:
                ocr_text_str = "No image found"

            new_line = line.replace("<!-- image -->", f"![Image OCR] {ocr_text_str}")
            merged_output.append(new_line)
        else:
            merged_output.append(line)

    return "\n".join(merged_output)


def extract_text_from_pdf_modular(pdf_path: Path, settings) -> str:
    """
    Modular version using the new architecture.
    """
    try:
        # Create hybrid processor with settings
        processor = create_hybrid_processor(azure_endpoint=settings.ocr.azure_doc_intel_endpoint)

        # Process the document
        text = processor.process_document(str(pdf_path))
        return text
    except Exception as e:
        logger.error(f"Modular OCR failed for {pdf_path}: {e}")
        # Fallback to original implementation
        return extract_text_from_pdf(pdf_path, settings)


def _document_exists_in_vector_store(vector_store: Any, source_filter: dict) -> bool:
    """
    Check if a document with the given source filter already exists in the vector store.

    This function handles different vector store implementations:
    - Chroma: Uses the .get() method
    - Other stores: Assumes documents don't exist and processes all documents

    Args:
        vector_store: The vector store instance
        source_filter: Dictionary with source metadata to filter by

    Returns:
        bool: True if document exists, False otherwise
    """
    try:
        # Try Chroma-style get method first
        if hasattr(vector_store, "get"):
            result = vector_store.get(where=source_filter, limit=1)
            return len(result.get("ids", [])) > 0

        # For Azure Search and other vector stores, we'll assume documents don't exist
        # This is safer - we might reprocess some documents, but we won't skip new ones
        else:
            logger.info(
                f"Vector store {type(vector_store).__name__} doesn't support existence checking. Processing all documents."
            )
            return False

    except Exception as e:
        logger.warning(f"Error checking document existence: {e}. Assuming document doesn't exist.")
        return False


def code_handler(element: Tag) -> str:
    """
    Custom handler for code elements.
    """
    data_lang = element.get("data-lang")
    code_format = f"<code:{data_lang}>{element.get_text()}</code>"

    return code_format


def add_urls(vector_store: Any, urls: list[str], chunk_size: int, enforce_chunk_size: bool) -> None:
    """
    Adds a list of URLs as vector documents to the provided vector store.

    The URL's will be fetched and split into chunks of text with the provided chunk size.
    """
    logger.info("Web sources to the vector store.")

    all_splits = []
    for url in urls:
        if _document_exists_in_vector_store(vector_store, {"source": url}):
            logger.info(f"Skipping URL {url}, as it is already in the database.")
            continue

        response = requests.get(url)
        html_content = response.text

        soup = BeautifulSoup(html_content, "html.parser")

        web_splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=headers_to_split_on,
            separators=["\n\n", "\n", ". ", "! ", "? "],
            max_chunk_size=chunk_size,
            preserve_images=True,
            preserve_videos=True,
            elements_to_preserve=["table", "ul", "ol", "code"],
            denylist_tags=["script", "style", "head"],
            custom_handlers={"code": code_handler},
        )

        splits = web_splitter.split_text(str(soup))

        # additional splitting to handle too large chunks caused by semantic splitter
        if enforce_chunk_size:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=50)
            splits = [split for doc in splits for split in text_splitter.split_documents([doc])]

        for split in splits:
            split.metadata["source"] = url
            split.metadata["filetype"] = "web"

        all_splits.extend(splits)

    if len(all_splits) == 0:
        logger.info("No web documents to add.")
        return

    logger.info(f"{len(urls)} web sources split in {len(all_splits)} vector store documents")
    logger.info(f"Adding {len(all_splits)} vector store documents to vector store.")

    try:
        filtered_splits = filter_complex_metadata(all_splits)
        logger.info(f"Filtered to {len(filtered_splits)} web documents. Adding to vector store...")
        vector_store.add_documents(documents=filtered_splits)
        logger.info("Successfully added web documents to vector store.")
    except Exception as e:
        logger.error(f"Error adding web documents to vector store: {e}")
        raise


def add_pdf_files(
    vector_store: Any,
    file_paths: list[Path],
    chunk_size: int,
    chunk_overlap: int,
    add_start_index: bool,
    unstructured: bool,
    settings,  # OCR-config
) -> None:
    """
    Adds a list of PDF files as vector documents to the provided vector store.

    This version uses a hybrid approach:
    - First tries Docling for structured PDF parsing.
    - Falls back to Azure Document Intelligence OCR if Docling fails or for images.
    - Splits the extracted text into chunks and adds them to the vector store.
    """

    logger.info("Adding PDF files to the vector store.")

    # 1. Find all PDF files in the given paths
    pdf_files = get_all_local_pdf_files(file_paths)
    logger.info(f"Found {len(pdf_files)} PDF files to add to the vector store.")

    # 2. Filter out PDFs that already exist in the vector store
    new_pdfs = [pdf for pdf in pdf_files if not _document_exists_in_vector_store(vector_store, {"source": str(pdf)})]

    if not new_pdfs:
        logger.info("No new PDFs to process.")
        return

    logger.info(f"{len(new_pdfs)} PDF(s) to add to the vector store.")

    # 3. Parse each PDF using OCR + Docling fallback logic
    loaded_document = []
    for file in new_pdfs:
        logger.info(f"Parsing PDF with OCR fallback: {file}")
        text = extract_text_from_pdf(file, settings)  # NEW: OCR-aware parsing
        loaded_document.append(Document(page_content=text, metadata={"source": str(file), "filetype": "pdf"}))

    logger.info(f"Loaded {len(loaded_document)} documents from PDFs")

    # 4. Split text into chunks for vector store ingestion
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=add_start_index
    )
    pdf_splits = text_splitter.split_documents(loaded_document)

    logger.info(f"{len(new_pdfs)} PDF's split in {len(pdf_splits)} vector store documents")
    logger.info(f"Adding {len(pdf_splits)} vector store documents to vector store.")

    if not pdf_splits:
        logger.warning("No documents to add to vector store after splitting.")
        return

    # 5. Filter metadata and add to vector store
    try:
        filtered_splits = filter_complex_metadata(pdf_splits)
        logger.info(f"Filtered to {len(filtered_splits)} documents. Adding to vector store...")
        vector_store.add_documents(documents=filtered_splits)
        logger.info("Successfully added PDF documents to vector store.")
    except Exception as e:
        logger.error(f"Error adding documents to vector store: {e}")
        raise


def get_all_local_pdf_files(local_paths: list[Path]) -> list[Path]:
    """
    Function that takes a list of local paths,
    that might contain directories paths and/or direct file paths,
    and returns a list with all file paths that are a PDF file or any PDF files found in the directory file paths.
    This fucntion does not scan directories recursively.
    """
    all_pdf_files = []
    for path in local_paths:
        if path.is_dir():
            all_pdf_files.extend(list(path.glob("*.pdf")))
        elif path.suffix == ".pdf":
            all_pdf_files.append(path)
        else:
            logger.warning(f"Ignoring path {path} as it is not a folder or pdf file.")

    return all_pdf_files
