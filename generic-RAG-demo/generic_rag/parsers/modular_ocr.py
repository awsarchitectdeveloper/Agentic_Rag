"""
Modular OCR processing system.
Separates concerns: OCR processing, document parsing, and hybrid logic.
"""

import io
import time
import fitz
import requests
from pathlib import Path
from PIL import Image, ImageEnhance
from azure.identity import DefaultAzureCredential
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class OCRProcessor:
    """Handles OCR operations using Azure Document Intelligence."""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.credential = DefaultAzureCredential()

    def extract_text_from_image(self, image_bytes: bytes) -> List[str]:
        """Extract text from image bytes using Azure Document Intelligence."""
        try:
            result = self._analyze_document_intelligence(image_bytes)
            return self._extract_text_from_result(result)
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return ["OCR failed"]

    def _analyze_document_intelligence(self, image_bytes: bytes) -> Dict[str, Any]:
        """Send image to Azure Document Intelligence for OCR."""
        token = self.credential.get_token("https://cognitiveservices.azure.com/.default")

        headers = {"Content-Type": "application/octet-stream", "Authorization": f"Bearer {token.token}"}
        url = f"{self.endpoint}/formrecognizer/documentModels/prebuilt-layout:analyze?api-version=2023-07-31"

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

    def _extract_text_from_result(self, result: Dict[str, Any]) -> List[str]:
        """Extract text lines from Azure Document Intelligence result."""
        lines = []
        for page in result.get("analyzeResult", {}).get("pages", []):
            for line in page.get("lines", []):
                lines.append(line.get("content", ""))
        return lines


class ImageProcessor:
    """Handles image extraction and preprocessing."""

    @staticmethod
    def preprocess_image(image_bytes: bytes) -> bytes:
        """Preprocess image for better OCR results."""
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
        img = ImageEnhance.Contrast(img).enhance(2.0)
        output = io.BytesIO()
        img.save(output, format="PNG")
        return output.getvalue()

    @staticmethod
    def extract_page_image(pdf_path: str, page_index: int, zoom: float = 2.0) -> bytes:
        """Extract full page as image from PDF."""
        pdf = fitz.open(pdf_path)
        pix = pdf[page_index].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        return pix.tobytes("png")

    @staticmethod
    def extract_embedded_images(pdf_path: str, page_index: int) -> List[bytes]:
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


class DocumentParser:
    """Handles document parsing using docling."""

    @staticmethod
    def parse_with_docling(file_path: str) -> str:
        """Parse document with docling."""
        from generic_rag.parsers.parser import docling_parser

        return docling_parser(file_path)


class HybridDocumentProcessor:
    """Orchestrates hybrid document processing: docling + OCR fallback."""

    def __init__(self, ocr_processor: OCRProcessor, image_processor: ImageProcessor, document_parser: DocumentParser):
        self.ocr_processor = ocr_processor
        self.image_processor = image_processor
        self.document_parser = document_parser

    def process_document(self, file_path: Path) -> str:
        """
        Process document with hybrid approach:
        1. Try docling first
        2. If fails -> full OCR
        3. If has placeholders -> fill with OCR
        """
        file_str = str(file_path)
        file_extension = file_path.suffix.lower()

        # Get docling result
        docling_text = self.document_parser.parse_with_docling(file_str)
        lines = docling_text.splitlines()
        placeholders = [line for line in lines if "<!-- image -->" in line]

        # Scenario 1: Docling failed -> full OCR
        if docling_text.strip() == "":
            logger.info("Docling returned empty. Using full-page OCR...")
            return self._process_full_ocr(file_str, file_extension)

        # Scenario 2: No placeholders -> just docling
        if not placeholders:
            logger.info("No image placeholders found. Using docling result.")
            return docling_text

        # Scenario 3: Mixed -> replace placeholders with OCR
        logger.info("Image placeholders detected. Filling with OCR text...")
        return self._process_mixed_content(file_str, file_extension, lines)

    def _process_full_ocr(self, file_path: str, file_extension: str) -> str:
        """Process document entirely with OCR."""
        if file_extension == ".pdf":
            return self._process_pdf_full_ocr(file_path)
        else:
            # For non-PDF files, convert to image first or use direct OCR
            return self._process_non_pdf_ocr(file_path)

    def _process_pdf_full_ocr(self, pdf_path: str) -> str:
        """Process PDF with full-page OCR."""
        pdf = fitz.open(pdf_path)
        full_text = []

        for i in range(len(pdf)):
            img_bytes = self.image_processor.extract_page_image(pdf_path, i)
            preprocessed = self.image_processor.preprocess_image(img_bytes)
            ocr_text = self.ocr_processor.extract_text_from_image(preprocessed)
            full_text.append("\n".join(ocr_text))

        return "\n\n".join(full_text)

    def _process_non_pdf_ocr(self, file_path: str) -> str:
        """Process non-PDF files with OCR (placeholder for future implementation)."""
        # For now, return empty - could implement DOCX image extraction here
        logger.warning(f"Full OCR for {file_path} not implemented for non-PDF files")
        return ""

    def _process_mixed_content(self, file_path: str, file_extension: str, lines: List[str]) -> str:
        """Replace image placeholders with OCR text."""
        if file_extension != ".pdf":
            logger.warning(f"Mixed content processing not implemented for {file_extension}")
            return "\n".join(lines)

        # Extract all images from PDF
        pdf = fitz.open(file_path)
        all_images = []
        for i in range(len(pdf)):
            all_images.extend(self.image_processor.extract_embedded_images(file_path, i))

        # Replace placeholders with OCR text
        image_index = 0
        merged_output = []

        for line in lines:
            if "<!-- image -->" in line:
                if image_index < len(all_images):
                    img_bytes = all_images[image_index]
                    image_index += 1
                    preprocessed = self.image_processor.preprocess_image(img_bytes)
                    ocr_text = self.ocr_processor.extract_text_from_image(preprocessed)
                    ocr_text_str = " ".join(ocr_text) if ocr_text else "No text found"
                else:
                    ocr_text_str = "No image found"

                new_line = line.replace("<!-- image -->", f"![Image OCR] {ocr_text_str}")
                merged_output.append(new_line)
            else:
                merged_output.append(line)

        return "\n".join(merged_output)


def create_hybrid_processor(settings=None, azure_endpoint=None) -> HybridDocumentProcessor:
    """
    Factory function to create a configured hybrid processor.

    Args:
        settings: Settings object with OCR configuration (preferred)
        azure_endpoint: Direct Azure endpoint URL (fallback)
    """
    if settings is not None:
        endpoint = settings.ocr.azure_doc_intel_endpoint
    elif azure_endpoint is not None:
        endpoint = azure_endpoint
    else:
        raise ValueError("Either settings or azure_endpoint must be provided")

    ocr_processor = OCRProcessor(endpoint)
    image_processor = ImageProcessor()
    document_parser = DocumentParser()

    return HybridDocumentProcessor(ocr_processor, image_processor, document_parser)
