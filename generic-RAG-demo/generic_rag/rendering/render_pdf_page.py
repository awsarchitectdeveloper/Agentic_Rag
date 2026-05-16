from pathlib import Path

import fitz
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
from langchain_core.documents import Document


def render_pdf_bound_box(file_path: str | Path, doc_list: list[Document], page_number: int) -> None:
    """
    Function that renders the bounding boxes of the segments on a PDF page.
    """
    pdf_page = fitz.open(file_path).load_page(page_number - 1)
    page_docs = [doc for doc in doc_list if doc.metadata.get("page_number") == page_number]
    segments = [doc.metadata for doc in page_docs]

    pix = pdf_page.get_pixmap()
    pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(pil_image)
    categories = set()
    category_to_color = {"Title": "orchid", "Image": "forestgreen", "Table": "tomato"}
    for segment in segments:
        points = segment["coordinates"]["points"]
        layout_width = segment["coordinates"]["layout_width"]
        layout_height = segment["coordinates"]["layout_height"]
        scaled_points = [(x * pix.width / layout_width, y * pix.height / layout_height) for x, y in points]
        box_color = category_to_color.get(segment["category"], "deepskyblue")
        categories.add(segment["category"])
        rect = patches.Polygon(scaled_points, linewidth=1, edgecolor=box_color, facecolor="none")
        ax.add_patch(rect)

    # Make legend
    legend_handles = [patches.Patch(color="deepskyblue", label="Text")]
    for category in ["Title", "Image", "Table"]:
        if category in categories:
            legend_handles.append(patches.Patch(color=category_to_color[category], label=category))
    ax.axis("off")
    ax.legend(handles=legend_handles, loc="upper right")
    plt.tight_layout()
    plt.savefig(f"test_{page_number}.png")
