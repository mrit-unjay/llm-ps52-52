"""
app.py
------
MediRead AI – Streamlit user interface.

Run with:   streamlit run app.py

This file only handles the screen. The real work is done by:
  ocr_processor.py        -> reading text from images / PDFs
  gemini_service.py       -> asking Gemini to extract medicines
  prescription_parser.py  -> validating the answer + safety checks
  utils/data_formatter.py -> tables, explanations, CSV and report
"""

import json
import logging
from html import escape

import streamlit as st

import config
from gemini_service import GeminiServiceError, extract_prescription
from ocr_processor import OCRError, extract_text_from_upload, extract_text_with_gemini_vision
from prescription_parser import ParseError, parse_gemini_response
from utils import data_formatter as fmt

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="MediRead AI – Prescription Reader",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = ["Prescription Reader", "Test Cases", "About Project"]
READERS = ["Tesseract OCR (offline, printed text)", "Gemini Vision (handwriting and phone photos)"]

# ------------------------------------------------------------------
# Styling: white + light blue, Atkinson Hyperlegible (a typeface
# designed for readers with low vision - fitting for medicine labels).
# ------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&display=swap');

html, body, [class*="css"], .stMarkdown, button, input, textarea {
    font-family: "Atkinson Hyperlegible", "Segoe UI", Arial, sans-serif;
}
.block-container { padding-top: 1.6rem; max-width: 1150px; }

.app-header { border-bottom: 3px solid #1F6FB2; padding-bottom: 0.9rem; margin-bottom: 1rem; }
.app-header h1 { font-size: 2.3rem; color: #12456F; margin: 0; letter-spacing: -0.01em; }
.app-header p { font-size: 1.05rem; color: #3E5A73; margin: 0.25rem 0 0 0; max-width: 70ch; }

.disclaimer {
    background: #FFF7E8; border: 1px solid #E9B85A; border-radius: 6px;
    padding: 0.7rem 0.95rem; color: #5A3E05; font-size: 0.95rem; margin-bottom: 1.1rem;
}
.step-title { font-size: 1.15rem; font-weight: 700; color: #12456F; margin: 1.2rem 0 0.3rem 0; }

.verify-banner {
    background: #FDECEC; border-left: 5px solid #C0392B; padding: 0.8rem 1rem;
    border-radius: 4px; color: #6B1A12; margin: 0.6rem 0 1rem 0;
}
.ok-banner {
    background: #EAF6EE; border-left: 5px solid #2E8B57; padding: 0.8rem 1rem;
    border-radius: 4px; color: #1D5A38; margin: 0.6rem 0 1rem 0;
}

/* Medication schedule styled like a day-section pill organiser */
.schedule-wrap { overflow-x: auto; }
table.schedule { border-collapse: separate; border-spacing: 0; width: 100%; font-size: 1rem; }
table.schedule th {
    text-align: left; padding: 0.65rem 0.7rem; color: #12456F; font-weight: 700;
    border-bottom: 2px solid #1F6FB2; background: #FFFFFF;
}
table.schedule td { padding: 0.7rem; border-bottom: 1px solid #DCE8F3; vertical-align: top; }
table.schedule th.slot, table.schedule td.slot { text-align: center; width: 8.5rem; }
th.morning, td.morning { background: #F2F8FD; }
th.afternoon, td.afternoon { background: #E4F0FA; }
th.night, td.night { background: #D3E5F5; }
td.slot .take { font-size: 1.35rem; color: #1F6FB2; font-weight: 700; }
td.slot .skip { color: #9AB0C3; }
td.slot .ns { font-size: 0.82rem; color: #8A6D1C; }
td .med-name { font-weight: 700; color: #12456F; }
tr.flagged td:first-child { border-left: 4px solid #C0392B; }
td .flag { display: inline-block; font-size: 0.8rem; color: #A93226; margin-top: 0.2rem; }
.legend { color: #4A6178; font-size: 0.88rem; margin-top: 0.4rem; }

.explain li { margin-bottom: 0.45rem; line-height: 1.55; }
.footer-note { color: #6A8197; font-size: 0.85rem; margin-top: 2.5rem; border-top: 1px solid #DCE8F3; padding-top: 0.7rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Session state (remembers values between button clicks)
# ------------------------------------------------------------------
DEFAULTS = {
    "page": PAGES[0],
    "prescription_text": "",
    "ocr_info": None,
    "result": None,
    "analysed_text": "",
    "flash": None,
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def load_test_cases() -> list[dict]:
    try:
        return json.loads(config.TEST_CASES_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []


def read_sample(path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


# ------------------------------------------------------------------
# Button callbacks (run before the page is redrawn)
# ------------------------------------------------------------------
def on_read_file():
    uploaded = st.session_state.get("uploaded_file")
    if uploaded is None:
        st.session_state.flash = ("warning", "Choose a prescription file first.")
        return
    use_vision = st.session_state.get("reader") == READERS[1]
    try:
        if use_vision:
            ocr_result = extract_text_with_gemini_vision(uploaded.name, uploaded.getvalue())
        else:
            ocr_result = extract_text_from_upload(uploaded.name, uploaded.getvalue())
    except OCRError as exc:
        st.session_state.flash = ("error", str(exc))
        return
    except GeminiServiceError as exc:
        st.session_state.flash = ("error", exc.user_message)
        return
    except Exception:
        logging.exception("Unexpected OCR failure")
        st.session_state.flash = ("error", "The file could not be read. Try a clearer image or type the text.")
        return

    st.session_state.prescription_text = ocr_result.text
    st.session_state.ocr_info = ocr_result
    st.session_state.result = None
    if not ocr_result.text.strip():
        st.session_state.flash = (
            "warning",
            "No readable text was found in this file. Try a clearer, well-lit photo, "
            "or type the prescription in the text box.",
        )
    else:
        st.session_state.flash = ("success", "Text read from the file. Check it below and correct any mistakes.")


def load_text(text: str, go_to_reader: bool = False):
    st.session_state.prescription_text = text
    st.session_state.ocr_info = None
    st.session_state.result = None
    if go_to_reader:
        st.session_state.page = PAGES[0]


def clear_all():
    load_text("")
    st.session_state.flash = None


# ------------------------------------------------------------------
# Shared layout pieces
# ------------------------------------------------------------------
def render_header():
    st.markdown(
        f"""
<div class="app-header">
  <h1>{config.APP_NAME}</h1>
  <p>{escape(config.APP_TAGLINE)}</p>
</div>
<div class="disclaimer"><strong>Medical disclaimer:</strong> {escape(config.DISCLAIMER)}</div>
""",
        unsafe_allow_html=True,
    )


def render_footer():
    st.markdown(
        f'<div class="footer-note">{escape(config.DISCLAIMER)}</div>',
        unsafe_allow_html=True,
    )


def render_sidebar():
    with st.sidebar:
        st.markdown(f"### 💊 {config.APP_NAME}")
        st.radio("Go to", PAGES, key="page")
        st.divider()
        st.markdown("**System check**")
        st.markdown(("✅" if config.GEMINI_API_KEY else "❌") + " Gemini API key")
        st.markdown(("✅" if config.TESSERACT_CMD else "❌") + " Tesseract OCR")
        st.caption(f"Model: {config.GEMINI_MODEL}")
        if not config.GEMINI_API_KEY:
            st.caption("Add GEMINI_API_KEY to .env and restart.")
        if not config.TESSERACT_CMD:
            st.caption("OCR unavailable. Typed input still works.")
        st.divider()
       


# ------------------------------------------------------------------
# Result sections
# ------------------------------------------------------------------
def _slot_cell(value: str, css: str) -> str:
    if value == fmt.TAKE:
        inner = '<span class="take" aria-label="take">✓</span>'
    elif value == fmt.SKIP:
        inner = '<span class="skip" aria-label="not at this time">—</span>'
    else:
        inner = '<span class="ns">Not specified</span>'
    return f'<td class="slot {css}">{inner}</td>'


def render_schedule_table(medicines: list[dict]):
    df = fmt.build_schedule_dataframe(medicines)
    header = (
        "<tr><th>Medicine</th><th>Strength</th><th>Dosage</th>"
        '<th class="slot morning">Morning</th><th class="slot afternoon">Afternoon</th>'
        '<th class="slot night">Night</th><th>Duration</th><th>Food Instructions</th></tr>'
    )
    rows = []
    for (_, row), med in zip(df.iterrows(), medicines):
        flag = '<br><span class="flag">⚠ Verify before use</span>' if med["needs_verification"] else ""
        rows.append(
            f'<tr class="{"flagged" if med["needs_verification"] else ""}">'
            f'<td><span class="med-name">{escape(row["Medicine"])}</span>{flag}</td>'
            f'<td>{escape(row["Strength"])}</td><td>{escape(row["Dosage"])}</td>'
            + _slot_cell(row["Morning"], "morning")
            + _slot_cell(row["Afternoon"], "afternoon")
            + _slot_cell(row["Night"], "night")
            + f'<td>{escape(row["Duration"])}</td><td>{escape(row["Food Instructions"])}</td></tr>'
        )
    st.markdown(
        f'<div class="schedule-wrap"><table class="schedule"><thead>{header}</thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
        '<div class="legend">✓ take at this time · — not at this time · '
        '"Not specified" means the prescription does not say which time of day.</div>',
        unsafe_allow_html=True,
    )


def render_results(result: dict, analysed_text: str):
    medicines = result["medicines"]

    st.markdown('<div class="step-title">Results</div>', unsafe_allow_html=True)
    if analysed_text != st.session_state.prescription_text.strip():
        st.info("The text has changed since these results were made. Click Extract again to update them.")

    if result["requires_verification"]:
        st.markdown(
            '<div class="verify-banner"><strong>Check with a pharmacist or doctor before following this '
            "schedule.</strong> Some details are unclear, missing or conflicting. See the warnings below.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="ok-banner">No uncertainties were detected. AI and OCR can still make mistakes, '
            "so compare this with the original prescription.</div>",
            unsafe_allow_html=True,
        )

    if not medicines:
        st.warning("No medicines were found in this text.")
        for warning in result["general_warnings"]:
            st.write(f"• {warning}")
        return

    st.markdown("#### Medication schedule")
    render_schedule_table(medicines)

    st.markdown("#### What the prescription says, in simple English")
    items = "".join(f"<li>{escape(line)}</li>" for line in fmt.build_explanations(medicines))
    st.markdown(f'<ul class="explain">{items}</ul>', unsafe_allow_html=True)
    if result["other_instructions"]:
        st.markdown("**Other instructions written in the prescription:**")
        for instruction in result["other_instructions"]:
            st.write(f"• {instruction}")

    st.markdown("#### Medicine details")
    for index, med in enumerate(medicines, start=1):
        icon = "⚠️" if med["needs_verification"] else "💊"
        with st.expander(f"{icon} {index}. {med['medicine_name']}", expanded=False):
            left, right = st.columns(2)
            with left:
                st.markdown(f"**Medicine name:** {med['medicine_name']}")
                st.markdown(f"**Strength:** {med['strength']}")
                st.markdown(f"**Dosage:** {med['dosage']}")
                st.markdown(f"**Frequency:** {med['frequency']}")
            with right:
                st.markdown(f"**Timing:** {', '.join(med['timings']) or config.NOT_SPECIFIED}")
                st.markdown(f"**Food instructions:** {med['food_instruction']}")
                st.markdown(f"**Duration:** {med['duration']}")
            st.markdown("**Original prescription instruction:**")
            st.code(med["original_instruction"], language=None)
            for note in med["notes"]:
                st.info(note)

    st.markdown("#### Warnings and items to verify")
    any_warning = False
    for med in medicines:
        for issue in med["uncertainties"]:
            st.warning(f"**{med['medicine_name']}:** {issue}")
            any_warning = True
    for warning in result["general_warnings"]:
        st.warning(warning)
        any_warning = True
    if not any_warning:
        st.success("No uncertainties detected. Still verify every detail with your pharmacist.")

    st.markdown("#### Download")
    export_df = fmt.build_export_dataframe(medicines)
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download schedule (CSV)",
            data=fmt.to_csv_bytes(export_df),
            file_name="mediread_schedule.csv",
            mime="text/csv",
            type="primary",
        )
    with col2:
        st.download_button(
            "Download printable report (HTML)",
            data=fmt.build_html_report(result, analysed_text).encode("utf-8"),
            file_name="mediread_report.html",
            mime="text/html",
        )
    st.caption("Open the HTML report in your browser and press Ctrl+P to print or save it as a PDF.")


# ------------------------------------------------------------------
# Pages
# ------------------------------------------------------------------
def page_reader():
    flash = st.session_state.flash
    if flash:
        level, message = flash
        getattr(st, level)(message)
        st.session_state.flash = None

    st.markdown('<div class="step-title">Step 1 · Add the prescription</div>', unsafe_allow_html=True)
    upload_tab, type_tab = st.tabs(["📄 Upload a file", "⌨️ Type or paste"])

    with upload_tab:
        st.file_uploader(
            "Prescription image or PDF (JPG, JPEG, PNG, PDF · max 10 MB)",
            type=config.ALLOWED_EXTENSIONS,
            key="uploaded_file",
        )
        st.radio("Read the file with", READERS, key="reader")
        st.button("Read text from file", on_click=on_read_file)
        st.caption(
            "Tesseract works offline and suits printed prescriptions. Gemini Vision reads "
            "handwriting and difficult photos much better, but sends the image to Google. "
            "Neither is 100% accurate. Tip: photograph the page flat, in good light."
        )

    with type_tab:
        st.write("Type or paste the prescription into the box in Step 2, or load an example:")
        st.button(
            "Load sample prescription",
            on_click=load_text,
            args=(read_sample(config.SAMPLES_DIR / "sample_prescription.txt"),),
        )

    st.markdown('<div class="step-title">Step 2 · Check the text</div>', unsafe_allow_html=True)
    ocr_info = st.session_state.ocr_info
    if ocr_info is not None:
        details = f"Read using: {ocr_info.method}"
        if ocr_info.pages_read > 1:
            details += f" · {ocr_info.pages_read} pages"
        if ocr_info.average_confidence is not None:
            details += f" · average OCR confidence {ocr_info.average_confidence:.0f}%"
        st.caption(details)
        if ocr_info.average_confidence is not None and ocr_info.average_confidence < 70:
            st.warning("OCR confidence is low. Compare the text carefully with the paper prescription.")
        unclear_count = st.session_state.prescription_text.count(config.UNCLEAR_MARKER)
        if unclear_count:
            st.warning(
                f"{unclear_count} word(s) could not be read and are marked [unclear]. "
                "Replace them only if you can read them on the paper prescription."
            )
        if ocr_info.low_confidence_words:
            shown = ", ".join(f"`{w}`" for w in ocr_info.low_confidence_words[:20])
            st.warning(f"OCR was unsure about these words: {shown}. Correct them below if they are wrong.")

    st.text_area(
        "Prescription text (edit anything that was read incorrectly)",
        key="prescription_text",
        height=240,
        placeholder="Example: Tab. Paracetamol 500 mg - 1 tab twice daily after food x 5 days",
    )

    st.markdown('<div class="step-title">Step 3 · Extract</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 4])
    with col1:
        extract_clicked = st.button("Extract Prescription", type="primary")
    with col2:
        st.button("Clear", on_click=clear_all)

    if extract_clicked:
        text = st.session_state.prescription_text.strip()
        if not text:
            st.warning("The prescription text is empty. Upload a file or type the prescription first.")
        elif len(text) < 10:
            st.warning("The text is too short to be a prescription. Please add the full instructions.")
        else:
            low_words = ocr_info.low_confidence_words if ocr_info else []
            with st.spinner("Reading the prescription with Gemini…"):
                try:
                    raw = extract_prescription(text, low_words)
                    st.session_state.result = parse_gemini_response(raw, text, low_words)
                    st.session_state.analysed_text = text
                except GeminiServiceError as exc:
                    st.error(exc.user_message)
                except ParseError as exc:
                    st.error(str(exc))
                except Exception:
                    logging.exception("Unexpected extraction failure")
                    st.error("Something unexpected went wrong while extracting. Please try again.")

    if st.session_state.result is not None:
        render_results(st.session_state.result, st.session_state.analysed_text)


def page_test_cases():
    st.markdown("### Test cases")
    st.write(
        "Five fictional prescriptions that check the main behaviours. Load one, click "
        "**Extract Prescription**, and compare the result with the expected output."
    )
    cases = load_test_cases()
    if not cases:
        st.error("samples/test_cases.json is missing or invalid.")
        return
    for case in cases:
        text = read_sample(config.SAMPLES_DIR / "test_cases" / case["file"])
        with st.expander(f"{case['id']} · {case['title']}", expanded=False):
            st.markdown(f"**What it checks:** {case['checks']}")
            st.code(text or "(file missing)", language=None)
            st.markdown(f"**Expected result:** {case['expected']}")
            st.button(
                "Load into reader",
                key=f"load_{case['id']}",
                on_click=load_text,
                args=(text, True),
                disabled=not text,
            )


def page_about():
    st.markdown("### About this project")
    st.markdown(
        """
**MediRead AI** helps patients read prescriptions. It extracts what is written, lays it out
as a daily schedule and points out anything unclear. It never prescribes, recommends or
guesses.

**How it works**

1. **Input** – an image (JPG/PNG), a PDF, or typed text.
2. **OCR** – Tesseract reads images; PyMuPDF reads PDFs (scanned pages go through OCR).
   Words Tesseract is unsure of are recorded.
3. **You check the text** – the extracted text can be corrected before the AI sees it.
4. **Gemini Flash** – a strict system prompt makes it act as an *extractor, not a doctor*,
   and return JSON in a fixed format.
5. **Safety checks (no AI)** – Python code double-checks the AI: names must appear in the
   original text, missing fields stay "Not specified", frequency and timings must agree.
6. **Output** – schedule table, simple explanation, warnings, CSV and printable report.

**Technology:** Python · Streamlit · Google Gemini Flash · Tesseract OCR (pytesseract) ·
OpenCV · Pillow · PyMuPDF · Pandas · python-dotenv

**Limitations:** OCR struggles with handwriting, blur and poor lighting. AI extraction can
make mistakes. Neither is 100% accurate, which is why every result must be verified with
a pharmacist or doctor.
"""
    )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
render_sidebar()
render_header()
page = st.session_state.page
if page == "Prescription Reader":
    page_reader()
elif page == "Test Cases":
    page_test_cases()
else:
    page_about()
render_footer()
