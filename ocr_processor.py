"""
ocr_processor.py
----------------
Reads text from prescription images using Tesseract OCR.

Steps for an image:
  1. Open and validate the image with Pillow.
  2. Clean it up with OpenCV (grayscale, upscale, denoise, black/white).
  3. Run Tesseract and collect every word with its confidence score.
  4. Rebuild the text line by line and list the words Tesseract
     was not confident about, so the app can warn the user.
"""

import io
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageOps, UnidentifiedImageError

import config

if config.TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD


class OCRError(Exception):
    """Raised with a user-friendly message when text cannot be read."""


@dataclass
class OCRResult:
    text: str
    average_confidence: float | None = None   # 0-100, None for typed/digital text
    low_confidence_words: list[str] = field(default_factory=list)
    method: str = "OCR"                        # "OCR", "PDF text" or "PDF text + OCR"
    pages_read: int = 1


# ------------------------------------------------------------------
# Image helpers
# ------------------------------------------------------------------
def load_image(file_bytes: bytes) -> Image.Image:
    """Open image bytes safely. Raises OCRError for corrupt/fake images."""
    try:
        Image.open(io.BytesIO(file_bytes)).verify()          # integrity check
        image = Image.open(io.BytesIO(file_bytes))           # reopen after verify
        image = ImageOps.exif_transpose(image)               # fix phone rotation
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError):
        raise OCRError(
            "This file is not a readable image. Please upload a valid JPG, "
            "JPEG or PNG file."
        )


def _to_gray(image: Image.Image) -> np.ndarray:
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    # Tesseract likes capital letters about 30 px tall: upscale small photos.
    height, width = gray.shape
    if width < 2000:
        scale = min(2000 / width, 4.0)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return gray


def _remove_shadows(gray: np.ndarray) -> np.ndarray:
    """Divide by the blurred background so shadows and uneven light disappear."""
    background = cv2.medianBlur(cv2.dilate(gray, np.ones((7, 7), np.uint8)), 31)
    return cv2.divide(gray, background, scale=255)


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Straighten slightly rotated photos (up to about 10 degrees)."""
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    lines = cv2.HoughLinesP(cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((1, 25), np.uint8)),
                            1, np.pi / 720, threshold=200, minLineLength=gray.shape[1] // 4, maxLineGap=20)
    if lines is None:
        return gray
    angles = [np.degrees(np.arctan2(y2 - y1, x2 - x1)) for x1, y1, x2, y2 in lines.reshape(-1, 4)]
    angles = [a for a in angles if abs(a) < 10]
    if not angles:
        return gray
    angle = float(np.median(angles))
    if abs(angle) < 0.3:
        return gray
    h, w = gray.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)


def preprocess_variants(image: Image.Image) -> list[tuple[str, np.ndarray]]:
    """
    Make several cleaned-up versions of the photo. Different photos need
    different treatment, so we OCR each one and keep the best result.
    """
    gray = _deskew(_to_gray(image))
    flat = _remove_shadows(gray)
    _, flat_bw = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        cv2.GaussianBlur(gray, (3, 3), 0), 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 15,
    )
    return [
        ("shadow removed", flat),
        ("shadow removed, black and white", flat_bw),
        ("adaptive threshold", adaptive),
        ("grayscale", gray),
    ]


# ------------------------------------------------------------------
# OCR
# ------------------------------------------------------------------
def _run_tesseract(processed: np.ndarray, psm: int) -> dict:
    try:
        return pytesseract.image_to_data(
            processed,
            lang=config.OCR_LANGUAGE,
            config=f"--oem 3 --psm {psm}",
            output_type=pytesseract.Output.DICT,
        )
    except pytesseract.TesseractNotFoundError:
        raise OCRError(
            "Tesseract OCR could not be started. Check that it is installed "
            "and that TESSERACT_CMD in .env points to tesseract.exe."
        )
    except pytesseract.TesseractError as exc:
        raise OCRError(f"Tesseract could not read this image ({exc.message}).")


def _build_result(data: dict) -> tuple[OCRResult, float]:
    """Turn Tesseract's word table into text + a quality score."""
    lines: dict[tuple, list[str]] = {}
    confidences: list[float] = []
    low_words: list[str] = []
    score = 0.0

    for i, word in enumerate(data["text"]):
        word = (word or "").strip()
        conf = float(data["conf"][i])
        if not word or conf < 0:          # conf = -1 means "not a word"
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(word)
        confidences.append(conf)
        has_letters_or_digits = any(ch.isalnum() for ch in word)
        if has_letters_or_digits:
            score += (conf / 100) * min(len(word), 12)   # ~ "characters read confidently"
        if conf < config.OCR_LOW_CONFIDENCE and has_letters_or_digits and len(word) > 1:
            if word not in low_words:
                low_words.append(word)

    text = "\n".join(" ".join(words) for words in lines.values())
    average = round(sum(confidences) / len(confidences), 1) if confidences else None
    return OCRResult(text=text, average_confidence=average, low_confidence_words=low_words), score


def ocr_image(image: Image.Image) -> OCRResult:
    """Run Tesseract on several cleaned-up versions and keep the best one."""
    if not config.TESSERACT_CMD:
        raise OCRError(
            "Tesseract OCR was not found on this computer. Install it (see "
            "README, section 'Install Tesseract OCR') or set TESSERACT_CMD in "
            "your .env file. You can still type the prescription manually."
        )

    best, best_score, best_name = None, -1.0, ""
    variants = preprocess_variants(image)
    for name, processed in variants:
        result, score = _build_result(_run_tesseract(processed, config.OCR_PSM))
        if score > best_score:
            best, best_score, best_name, best_image = result, score, name, processed

    # Try other page layouts on the winning image (prescriptions vary a lot).
    for psm in (4, 3):
        result, score = _build_result(_run_tesseract(best_image, psm))
        if score > best_score:
            best, best_score = result, score

    best.method = f"OCR ({best_name})"
    return best


def extract_text_from_image(file_bytes: bytes) -> OCRResult:
    return ocr_image(load_image(file_bytes))


# ------------------------------------------------------------------
# Entry points used by app.py
# ------------------------------------------------------------------
def validate_upload(file_name: str, file_bytes: bytes) -> str:
    """Check size and type. Returns the file extension (e.g. 'png')."""
    if not file_bytes:
        raise OCRError("The uploaded file is empty.")

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > config.MAX_FILE_SIZE_MB:
        raise OCRError(
            f"The file is {size_mb:.1f} MB. Please upload a file smaller than "
            f"{config.MAX_FILE_SIZE_MB} MB."
        )

    extension = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
    if extension not in config.ALLOWED_EXTENSIONS:
        raise OCRError(
            f"'.{extension or '?'}' files are not supported. Please upload "
            f"{', '.join(config.ALLOWED_EXTENSIONS).upper()}."
        )
    return extension


def extract_text_from_upload(file_name: str, file_bytes: bytes) -> OCRResult:
    """Read an uploaded image or PDF with Tesseract OCR (works offline)."""
    extension = validate_upload(file_name, file_bytes)
    if extension in config.IMAGE_EXTENSIONS:
        return extract_text_from_image(file_bytes)
    from utils.pdf_handler import extract_text_from_pdf   # avoid circular import
    return extract_text_from_pdf(file_bytes)


def extract_text_with_gemini_vision(file_name: str, file_bytes: bytes) -> OCRResult:
    """
    Read an uploaded image or PDF with Gemini Vision instead of Tesseract.
    Much better for handwriting and phone photos. Needs internet + API key.
    Unreadable words come back as [unclear] instead of being guessed.
    """
    from gemini_service import transcribe_document        # imported here to keep OCR usable offline

    extension = validate_upload(file_name, file_bytes)
    if extension in config.IMAGE_EXTENSIONS:
        image = load_image(file_bytes)                     # validates + fixes phone rotation
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=92)
        data, mime_type = buffer.getvalue(), "image/jpeg"
    else:
        import pymupdf
        try:
            document = pymupdf.open(stream=file_bytes, filetype="pdf")
        except Exception:
            raise OCRError("This PDF could not be opened. It may be damaged or not a real PDF.")
        if document.needs_pass:
            raise OCRError("This PDF is password-protected. Please remove the password and upload it again.")
        document.close()
        data, mime_type = file_bytes, config.MIME_TYPES["pdf"]

    text = transcribe_document(data, mime_type)
    return OCRResult(text=text, average_confidence=None, low_confidence_words=[], method="Gemini Vision")
