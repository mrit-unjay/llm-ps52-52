"""
utils/pdf_handler.py
--------------------
Reads text from PDF prescriptions with PyMuPDF.

- Digital PDFs (text can be selected): text is read directly. This is
  exact, so no OCR confidence is needed.
- Scanned PDFs (each page is just a picture): the page is rendered as
  an image at 300 DPI and sent to Tesseract OCR.
"""

import pymupdf
from PIL import Image

import config
from ocr_processor import OCRError, OCRResult, ocr_image


def extract_text_from_pdf(file_bytes: bytes) -> OCRResult:
    try:
        document = pymupdf.open(stream=file_bytes, filetype="pdf")
    except Exception:
        raise OCRError("This PDF could not be opened. It may be damaged or not a real PDF.")

    if document.needs_pass:
        raise OCRError("This PDF is password-protected. Please remove the password and upload it again.")
    if document.page_count == 0:
        raise OCRError("This PDF has no pages.")

    page_texts: list[str] = []
    low_words: list[str] = []
    ocr_confidences: list[float] = []
    digital_pages = 0
    scanned_pages = 0
    pages_to_read = min(document.page_count, config.MAX_PDF_PAGES)

    for page_number in range(pages_to_read):
        page = document[page_number]
        text = page.get_text("text").strip()

        if len(text) >= config.PDF_MIN_TEXT_CHARS:
            digital_pages += 1
        else:
            # Scanned page: render to an image and OCR it.
            pixmap = page.get_pixmap(dpi=config.PDF_OCR_DPI, alpha=False)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            result = ocr_image(image)
            text = result.text
            scanned_pages += 1
            if result.average_confidence is not None:
                ocr_confidences.append(result.average_confidence)
            for word in result.low_confidence_words:
                if word not in low_words:
                    low_words.append(word)

        if pages_to_read > 1:
            text = f"--- Page {page_number + 1} ---\n{text}"
        page_texts.append(text)

    document.close()

    if scanned_pages and digital_pages:
        method = "PDF text + OCR"
    elif scanned_pages:
        method = "OCR (scanned PDF)"
    else:
        method = "PDF text"

    average = round(sum(ocr_confidences) / len(ocr_confidences), 1) if ocr_confidences else None
    return OCRResult(
        text="\n\n".join(page_texts).strip(),
        average_confidence=average,
        low_confidence_words=low_words,
        method=method,
        pages_read=pages_to_read,
    )
