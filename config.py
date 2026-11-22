"""
config.py
---------
Central place for every setting used by MediRead AI.

All secrets (the Gemini API key) are read from the .env file using
python-dotenv. Nothing secret is ever hard-coded in this file.
"""

import os
import platform
import shutil
from pathlib import Path

from dotenv import load_dotenv

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = BASE_DIR / "samples"
TEST_CASES_FILE = SAMPLES_DIR / "test_cases.json"

# Load variables from the .env file that sits next to this file.
load_dotenv(BASE_DIR / ".env")

# ------------------------------------------------------------------
# Application text
# ------------------------------------------------------------------
APP_NAME = "MediRead AI"
APP_TAGLINE = (
    "Turns a typed or photographed prescription into a clear medication "
    "schedule, and flags anything that needs checking."
)

# This exact text must appear on every page of the app.
DISCLAIMER = (
    "This tool provides general informational assistance only. It is not "
    "medical advice and does not replace consultation with a qualified doctor "
    "or pharmacist. Always verify prescription details before taking any "
    "medication."
)

# ------------------------------------------------------------------
# Gemini settings
# ------------------------------------------------------------------
_PLACEHOLDER_KEYS = {"", "your_api_key_here"}
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if GEMINI_API_KEY in _PLACEHOLDER_KEYS:
    GEMINI_API_KEY = ""

# Main Gemini model. Change it in .env (GEMINI_MODEL=...) if Google
# retires it; see https://ai.google.dev/gemini-api/docs/models
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "").strip() or "gemini-3.5-flash"
GEMINI_TIMEOUT_MS = 90_000      # 90 seconds per request
GEMINI_MAX_ATTEMPTS = 3         # tries per model for rate limits / busy server

# Backup model used automatically if the main model stays busy.
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "").strip() or "gemini-2.5-flash"

# ------------------------------------------------------------------
# Tesseract OCR settings
# ------------------------------------------------------------------
def _find_tesseract() -> str | None:
    """Return the path to tesseract.exe, or None if it cannot be found."""
    # 1. Path given in .env (most reliable on Windows)
    env_path = os.getenv("TESSERACT_CMD", "").strip().strip('"')
    if env_path and Path(env_path).exists():
        return env_path

    # 2. Tesseract is on the system PATH
    on_path = shutil.which("tesseract")
    if on_path:
        return on_path

    # 3. Default Windows install locations
    if platform.system() == "Windows":
        for candidate in (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ):
            if Path(candidate).exists():
                return candidate
    return None


TESSERACT_CMD = _find_tesseract()
OCR_LANGUAGE = "eng"
OCR_PSM = 6                  # Page segmentation mode 6 = a uniform block of text
OCR_LOW_CONFIDENCE = 60      # Words below this confidence (0-100) are flagged

# ------------------------------------------------------------------
# Upload settings
# ------------------------------------------------------------------
IMAGE_EXTENSIONS = {"jpg", "jpeg", "png"}
PDF_EXTENSIONS = {"pdf"}
ALLOWED_EXTENSIONS = sorted(IMAGE_EXTENSIONS | PDF_EXTENSIONS)
MIME_TYPES = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "pdf": "application/pdf"}
UNCLEAR_MARKER = "[unclear]"     # written by Gemini Vision for unreadable words
MAX_FILE_SIZE_MB = 10
MAX_PDF_PAGES = 10
PDF_MIN_TEXT_CHARS = 20      # Fewer characters than this = treat page as scanned
PDF_OCR_DPI = 300

# ------------------------------------------------------------------
# Values used in the extracted data
# ------------------------------------------------------------------
NOT_SPECIFIED = "Not specified"
UNCLEAR = "Unclear – Confirm with Pharmacist/Doctor"
TIMING_SLOTS = ["Morning", "Afternoon", "Night"]
