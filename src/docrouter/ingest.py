"""Turning files on disk into `Document` objects.

The mixed scanned/digital problem is handled in one place: `extract_pdf`
reads the embedded text layer first and only rasterizes pages for OCR when
that layer is missing or is not real Arabic. Deciding per page rather than
per file matters — Saudi government PDFs are routinely a digital cover sheet
stapled to scanned attachments.
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path

from . import normalize
from .models import Document

# Below this many characters, a page's text layer is treated as absent.
MIN_CHARS_PER_PAGE = 40
# Rasterization DPI for OCR. 200 is the floor for reliable Arabic diacritics.
OCR_DPI = 200

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".docx", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class OCRUnavailable(RuntimeError):
    pass


def _ocr_images_with_claude(images: list[bytes], client=None) -> str:
    """Transcribe page images with Claude vision.

    This is the default OCR backend: it needs no external binaries (a real
    constraint on Windows) and handles Arabic ligatures and handwriting far
    better than Tesseract's `ara` model.
    """
    import anthropic

    from . import DEFAULT_MODEL

    client = client or anthropic.Anthropic()
    model = os.environ.get("DOCROUTER_MODEL", DEFAULT_MODEL)

    content: list[dict] = []
    for img in images:
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.standard_b64encode(img).decode("ascii"),
                },
            }
        )
    content.append(
        {
            "type": "text",
            "text": (
                "انسخ كل النص الظاهر في هذه الصفحات حرفياً كما هو، بالترتيب.\n"
                "- لا تترجم ولا تلخّص ولا تصحح الأخطاء الإملائية.\n"
                "- حافظ على الأسطر والعناوين وأرقام الجداول.\n"
                "- إن كانت صفحة غير مقروءة تماماً فاكتب [صفحة غير مقروءة].\n"
                "أخرج النص فقط دون أي تعليق."
            ),
        }
    )

    response = client.messages.create(
        model=model,
        max_tokens=16000,
        messages=[{"role": "user", "content": content}],
    )
    return "".join(b.text for b in response.content if b.type == "text")


def _ocr_images_with_tesseract(images: list[bytes]) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional extra
        raise OCRUnavailable(
            "tesseract backend needs `pip install 'docrouter[tesseract]'` "
            "plus the Tesseract binary with the `ara` language pack"
        ) from exc

    return "\n\n".join(
        pytesseract.image_to_string(Image.open(io.BytesIO(img)), lang="ara+eng")
        for img in images
    )


def run_ocr(images: list[bytes], backend: str = "claude", client=None) -> str:
    if not images:
        return ""
    if backend == "claude":
        return _ocr_images_with_claude(images, client=client)
    if backend == "tesseract":
        return _ocr_images_with_tesseract(images)
    raise ValueError(f"unknown OCR backend: {backend!r}")


def extract_pdf(
    path: Path, ocr_backend: str = "claude", client=None
) -> tuple[str, str, int]:
    """Return (text, source, page_count) for a PDF.

    `source` is "digital" when every page had a usable text layer, and "ocr"
    when at least one page had to be rasterized.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(path)
    try:
        page_texts: list[str] = []
        needs_ocr: list[int] = []

        for index, page in enumerate(doc):
            text = page.get_text("text") or ""
            if len(text.strip()) < MIN_CHARS_PER_PAGE or not normalize.looks_arabic(text):
                needs_ocr.append(index)
                page_texts.append("")
            else:
                page_texts.append(text)

        if needs_ocr:
            zoom = OCR_DPI / 72
            images = [
                doc[i].get_pixmap(matrix=fitz.Matrix(zoom, zoom)).tobytes("png")
                for i in needs_ocr
            ]
            ocr_text = run_ocr(images, backend=ocr_backend, client=client)
            # The OCR call transcribes the batch as one stream; attaching it to
            # the first scanned page keeps document order close enough for
            # classification, which reads the whole document anyway.
            page_texts[needs_ocr[0]] = ocr_text

        return "\n\n".join(t for t in page_texts if t), (
            "ocr" if needs_ocr else "digital"
        ), len(page_texts)
    finally:
        doc.close()


def extract_image(path: Path, ocr_backend: str = "claude", client=None) -> tuple[str, str, int]:
    return run_ocr([path.read_bytes()], backend=ocr_backend, client=client), "ocr", 1


def extract_docx(path: Path) -> tuple[str, str, int]:
    import docx

    d = docx.Document(str(path))
    parts = [p.text for p in d.paragraphs]
    for table in d.tables:
        for row in table.rows:
            parts.append("\t".join(cell.text for cell in row.cells))
    return "\n".join(parts), "digital", 1


def load_document(
    path: str | Path, ocr_backend: str = "claude", client=None
) -> Document:
    """Ingest one file, OCR-ing only what needs it."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        text, source, pages = extract_pdf(path, ocr_backend=ocr_backend, client=client)
    elif suffix == ".docx":
        text, source, pages = extract_docx(path)
    elif suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        text, source, pages = extract_image(path, ocr_backend=ocr_backend, client=client)
    elif suffix in {".txt", ".md"}:
        text, source, pages = path.read_text(encoding="utf-8"), "digital", 1
    else:
        raise ValueError(
            f"unsupported file type {suffix!r}; supported: {sorted(SUPPORTED_SUFFIXES)}"
        )

    return Document(
        doc_id=path.stem,
        text=normalize.light(text),
        source=source,
        path=str(path),
        page_count=pages,
    )


def load_directory(
    directory: str | Path, ocr_backend: str = "claude", client=None
) -> list[Document]:
    directory = Path(directory)
    files = sorted(
        p for p in directory.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    return [load_document(p, ocr_backend=ocr_backend, client=client) for p in files]
