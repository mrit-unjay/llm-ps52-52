# 💊 MediRead AI – Prescription Reader

**An AI-powered prescription reader and medication scheduler.**
Generative AI Mini Project · Problem Statement No. 52

MediRead AI reads a typed, photographed or PDF prescription and turns it into a clear daily
medication schedule. It explains only what is written in the prescription, and flags anything
unclear, missing or conflicting so the patient can confirm it with a pharmacist or doctor.

> **Disclaimer:** This tool provides general informational assistance only. It is not medical
> advice and does not replace consultation with a qualified doctor or pharmacist. Always verify
> prescription details before taking any medication.

---

## 1. Problem Statement

Medical prescriptions are often hard for patients to read. Medicine names, abbreviations
(BD, TDS, HS), dose patterns (1-0-1), timings and durations can be confusing.

The goal is an AI application that extracts medicine information from typed or image-based
prescriptions and converts it into an easy-to-understand medication schedule. It must **only**
explain information explicitly present in the prescription. It must never prescribe medicines,
recommend dosages, or guess unreadable information.

---

## 2. Features

| # | Feature | What it does |
|---|---------|--------------|
| 1 | Prescription input | Upload JPG / JPEG / PNG / PDF, or type/paste text. Sample prescription included. |
| 2 | OCR | Tesseract reads images (with shadow removal, deskewing and best-of-several preprocessing); PyMuPDF reads digital PDFs; scanned PDF pages are OCR'd automatically. Optional **Gemini Vision** reader for handwriting and difficult photos, which marks unreadable words as [unclear] instead of guessing. |
| 3 | Editable text | Extracted text is shown and can be corrected before the AI sees it. |
| 4 | AI extraction | Gemini Flash extracts name, strength, dosage, frequency, timings, food instructions, duration and the original line, as JSON. |
| 5 | Medication schedule | Table with Medicine, Strength, Dosage, Morning, Afternoon, Night, Duration, Food Instructions. |
| 6 | Simple explanation | Plain-English sentence per medicine, built by code from extracted fields (no new information can be added). |
| 7 | Uncertainty detection | Flags low-confidence OCR words, unclear names, missing fields, frequency/timing conflicts, duplicates, and names the AI may have "corrected". |
| 8 | Downloads | CSV schedule (opens in Excel) and a printable HTML report with the disclaimer. |
| 9 | Error handling | Friendly messages for bad files, empty input, OCR failure, invalid API key, rate limits, invalid JSON, network errors. |
| 10 | Disclaimer | Shown at the top and bottom of every page. |

---

## 3. Technology Stack

| Purpose | Technology |
|---------|------------|
| Language | Python 3.11 / 3.12 |
| User interface | Streamlit |
| Large language model | Google Gemini Flash (via the `google-genai` SDK) |
| OCR | Tesseract OCR + `pytesseract` |
| Image processing | Pillow, OpenCV |
| PDF processing | PyMuPDF |
| Secrets | python-dotenv |
| Data / tables | Pandas |
| Output | Structured JSON → Streamlit table, CSV, HTML report |

No LangChain or other frameworks are used: one direct API call is all this project needs.

---

## 4. System Architecture

```
                    ┌──────────────────────────────┐
                    │   Streamlit UI  (app.py)     │
                    └──────────────┬───────────────┘
                                   │
        ┌──────────────────────────┼─────────────────────────┐
        │ Image (JPG/PNG)          │ PDF                     │ Typed text
        ▼                          ▼                         │
┌──────────────────┐   ┌───────────────────────┐            │
│ ocr_processor.py │◄──│ utils/pdf_handler.py  │            │
│ OpenCV clean-up  │   │ digital text: direct  │            │
│ Tesseract OCR    │   │ scanned page: → OCR   │            │
│ + confidence     │   └───────────────────────┘            │
└────────┬─────────┘                                         │
         └──────────────► Editable text box ◄────────────────┘
                          (user corrects OCR mistakes)
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ gemini_service.py            │
                    │ strict system prompt          │
                    │ → Gemini Flash → JSON         │
                    └──────────────┬───────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │ prescription_parser.py       │
                    │ validate JSON + rule-based   │
                    │ safety checks (no AI)        │
                    └──────────────┬───────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │ utils/data_formatter.py      │
                    │ schedule table, explanation, │
                    │ CSV, printable report        │
                    └──────────────────────────────┘
```

**Two layers of safety.** Gemini is instructed not to guess (layer 1). Python code then
re-checks its answer (layer 2). For example, if the prescription says `Amox1c1l|in` and the AI
returns `Amoxicillin`, the parser notices that the name is not in the original text and flags it.
This matters because language models sometimes "helpfully" correct things they should not.

---

## 5. Project Folder Structure

```
MediRead-AI/
│
├── app.py                      # Streamlit user interface (all pages)
├── config.py                   # Settings, constants, disclaimer, .env loading
├── ocr_processor.py            # Image validation, OpenCV preprocessing, Tesseract OCR
├── gemini_service.py           # System prompt + Gemini API call + API error handling
├── prescription_parser.py      # JSON validation + rule-based safety checks
├── requirements.txt            # Python packages
├── .env.example                # Template for your API key (no real key)
├── .gitignore                  # Keeps .env and venv out of Git
├── README.md                   # This file
│
├── .streamlit/
│   └── config.toml             # White + light-blue theme, 10 MB upload limit
│
├── utils/
│   ├── __init__.py             # Makes utils a Python package (empty file)
│   ├── pdf_handler.py          # PDF reading with OCR fallback for scanned pages
│   └── data_formatter.py       # Schedule DataFrame, explanations, CSV, HTML report
│
├── samples/
│   ├── sample_prescription.txt # Demo prescription (fictional)
│   ├── test_cases.json         # Descriptions + expected results of the 5 test cases
│   └── test_cases/             # The 5 fictional test prescriptions
│       ├── tc1_single_medicine.txt
│       ├── tc2_multiple_medicines.txt
│       ├── tc3_morning_night.txt
│       ├── tc4_missing_details.txt
│       └── tc5_unclear_names.txt
│
└── tests/
    └── test_parser.py          # 13 offline unit tests (no API key needed)
```

---

## 6. Installation (Windows 11 + VS Code)

### Step 1 – Install Python

1. Download **Python 3.12** from https://www.python.org/downloads/windows/
2. Run the installer and **tick "Add python.exe to PATH"** on the first screen.
3. Check it in a new terminal:
   ```powershell
   python --version
   ```

### Step 2 – Install Tesseract OCR (Windows)

1. Open https://github.com/UB-Mannheim/tesseract/wiki and download the latest
   **64-bit Windows installer** (`tesseract-ocr-w64-setup-....exe`).
2. Run it and keep the default install folder:
   `C:\Program Files\Tesseract-OCR`
3. Add Tesseract to PATH:
   - Press **Win**, type **"environment variables"**, open **Edit the system environment variables**.
   - Click **Environment Variables…** → under *System variables* select **Path** → **Edit** → **New**.
   - Add `C:\Program Files\Tesseract-OCR` → **OK** on every window.
4. Close and reopen VS Code, then check:
   ```powershell
   tesseract --version
   ```

> If you skip step 3, the app still finds Tesseract in the default folder automatically. If you
> installed it somewhere else, set `TESSERACT_CMD` in `.env` (see step 5).

### Step 3 – Get the project and open it in VS Code

Extract the project ZIP (for example to `C:\Projects\MediRead-AI`) and open the folder:
**VS Code → File → Open Folder… → MediRead-AI**.
Open a terminal with **Terminal → New Terminal** (PowerShell).

### Step 4 – Create a virtual environment and install packages

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

You should see `(venv)` at the start of the terminal line.

> **If activation fails** with "running scripts is disabled on this system", run this once and
> then activate again:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

In VS Code, press **Ctrl+Shift+P → Python: Select Interpreter** and choose the one inside `venv`.

### Step 5 – Configure the Gemini API key

1. Go to https://aistudio.google.com/apikey, sign in with a Google account and click
   **Create API key**. Copy the key.
2. Create your `.env` file from the template:
   ```powershell
   copy .env.example .env
   ```
3. Open `.env` in VS Code and replace the placeholder:
   ```
   GEMINI_API_KEY=paste_your_real_key_here
   GEMINI_MODEL=gemini-3.5-flash
   GEMINI_FALLBACK_MODEL=gemini-2.5-flash
   ```
4. Save the file. Never share `.env` or commit it to GitHub (it is already in `.gitignore`).

**About the models:** if the main model is busy ("high demand" / 503), the app retries and then
switches to the backup model automatically. Google retires model versions regularly, so if you
see a "model not found" error, pick a current Flash model from
https://ai.google.dev/gemini-api/docs/models and put it in `.env`.

---

## 7. Running the Project

```powershell
venv\Scripts\activate
streamlit run app.py
```

The app opens at http://localhost:8501. The sidebar's **System check** shows whether the API key
and Tesseract were found. Stop the app with **Ctrl+C** in the terminal.

Run the offline unit tests:

```powershell
python -m unittest discover tests -v
```

---

## 8. How to Use

1. **Step 1 – Add the prescription:** upload an image/PDF and click **Read text from file**, or
   open the *Type or paste* tab and click **Load sample prescription**.
2. **Step 2 – Check the text:** fix any OCR mistakes. Words the OCR was unsure about are listed.
3. **Step 3 – Extract:** click **Extract Prescription**.
4. Review the schedule, the simple-English explanation, medicine details and warnings.
5. Download the **CSV** or the **printable report** (open it and press Ctrl+P to print/save as PDF).

---

## 9. Sample Input and Expected Output

**Input** (`samples/sample_prescription.txt`, fictional):

```
Rx
1. Tab. Paracetamol 500 mg - 1 tab twice daily after food x 5 days
2. Cap. Amoxicillin 250 mg - 1-0-1 after food for 7 days
3. Tab. Cetirizine 10 mg - 1 tab at night for 3 days
Advice: Drink plenty of water. Review after 1 week.
```

**Gemini JSON output (example):**

```json
{
  "medicines": [
    {
      "medicine_name": "Paracetamol",
      "strength": "500 mg",
      "dosage": "1 tablet",
      "frequency": "Twice daily",
      "timings": [],
      "food_instruction": "After food",
      "duration": "5 days",
      "original_instruction": "1. Tab. Paracetamol 500 mg - 1 tab twice daily after food x 5 days",
      "uncertainties": []
    }
  ],
  "other_instructions": ["Drink plenty of water.", "Review after 1 week."],
  "general_warnings": [],
  "requires_verification": false
}
```
*(only the first medicine is shown here)*

**Medication schedule shown in the app:**

| Medicine | Strength | Dosage | Morning | Afternoon | Night | Duration | Food Instructions |
|---|---|---|---|---|---|---|---|
| Paracetamol | 500 mg | 1 tablet | Not specified | Not specified | Not specified | 5 days | After food |
| Amoxicillin | 250 mg | 1 capsule | ✓ | — | ✓ | 7 days | After food |
| Cetirizine | 10 mg | 1 tablet | — | — | ✓ | 3 days | Not specified |

**Simple explanation:**
- Paracetamol 500 mg: Take 1 tablet, twice daily, after food, for 5 days.
- Amoxicillin 250 mg: Take 1 capsule, twice daily, in the morning and at night, after food, for 7 days.
- Cetirizine 10 mg: Take 1 tablet, once daily, at night, for 3 days.

**Why does Paracetamol show "Not specified" for the times of day?** "Twice daily" says *how
often*, not *when*. Assuming "morning and night" would be inventing information, so the app
shows "Not specified" and adds a note to ask the pharmacist. In contrast, `1-0-1` explicitly
means morning and night, so Amoxicillin gets ✓ in both columns. This is a deliberate safety
decision you can explain in the viva.

Actual AI output can vary slightly between runs.

---

## 10. Testing

### Five fictional test cases (manual, with Gemini)

Open the **Test Cases** page in the sidebar, click **Load into reader**, then **Extract Prescription**.

| ID | Test | Input (summary) | Expected result |
|----|------|-----------------|-----------------|
| TC1 | One medicine | Paracetamol 500 mg, 1 tablet twice daily after food for 5 days | 1 row, all fields filled; time slots "Not specified" with a note |
| TC2 | Several medicines | Azithromycin, Paracetamol (TDS), Ambroxol syrup, Pantoprazole before breakfast | 4 rows; TDS expanded to "Three times daily"; Pantoprazole → Morning |
| TC3 | Morning and night | Metformin `1-0-1`, Atorvastatin `0-0-1` after dinner | Metformin ✓ — ✓; Atorvastatin — — ✓ |
| TC4 | Missing dosage/duration | Cetirizine with no duration, Ambroxol with no dose, Vitamin D3 with only strength | Missing values shown as "Not specified", warnings shown, nothing invented |
| TC5 | Unclear names | `Amox1c1l\|in 5OO mg`, `Pnt0pr~zle 4O mg`, `Dol?` | Names marked "Unclear – Confirm with Pharmacist/Doctor" or flagged; red verification banner; no silent correction |

### Automated unit tests (offline)

`tests/test_parser.py` contains 13 tests that feed **mocked Gemini replies** into the parser, so
they run without an API key or internet. They check:

- all five test cases above (correct rows, timings, missing fields);
- a frequency/timing conflict is flagged (e.g. "three times daily" but only two timings);
- invalid timings such as "Evening" are rejected rather than forced into a column;
- an AI-"corrected" medicine name (`Amoxicillin` for `Amox1c1l|in`) is caught;
- low-confidence OCR words produce a warning;
- code fences in the reply are removed; non-JSON replies give a friendly error;
- empty results produce a warning; CSV and report are generated with the disclaimer.

Expected output: `Ran 13 tests ... OK`

### Error-handling checks

| Situation | Message shown |
|-----------|---------------|
| Unsupported file (e.g. .docx) | "'.docx' files are not supported…" |
| Renamed/corrupt image | "This file is not a readable image…" |
| Damaged or password-protected PDF | Clear message for each case |
| Empty text | "The prescription text is empty…" |
| Tesseract missing | Install instructions; typed input still works |
| No / invalid API key | "No Gemini API key found…" / "The Gemini API key was rejected…" |
| Rate limit (HTTP 429) | Retries once, then "Wait about a minute…" |
| Gemini server busy | Retries once, then a friendly message |
| Invalid JSON from AI | "The AI response was not valid JSON. Please click Extract again." |
| No medicines found | Warning that the text may be unreadable or not a prescription |

---

## 11. Screenshots

Add your own screenshots to a `screenshots/` folder and link them here:

```markdown
![Home page](screenshots/home.png)
![OCR text review](screenshots/ocr_review.png)
![Medication schedule](screenshots/schedule.png)
![Warnings for unclear prescription (TC5)](screenshots/warnings.png)
![Test cases page](screenshots/test_cases.png)
```

---

## 12. Limitations

- **OCR is not 100% accurate.** Tesseract works well on clear printed text but poorly on
  handwriting, blur and stamps. Gemini Vision reads handwriting much better but can still
  misread words, and it sends the image to Google. Always compare the text with the paper.
- **AI extraction is not 100% accurate.** Gemini can misread context or miss a line. The
  rule-based checks reduce, but cannot eliminate, these errors.
- The time-of-day table has only Morning, Afternoon and Night; four-times-daily or "evening"
  instructions are flagged rather than placed.
- English prescriptions only (Tesseract `eng` language pack).
- Requires an internet connection and a Gemini API key; free-tier rate limits apply.
- Prescription text is sent to Google's Gemini API. Use only fictional prescriptions for demos.

---

## 13. Future Scope

- Show Tesseract and Gemini Vision results side by side and highlight where they disagree.
- Support for Hindi and Marathi prescriptions (Tesseract language packs + multilingual prompts).
- Medication reminders (calendar export / notifications).
- Cross-checking names against an official drug database to flag unknown names.
- Bounding-box highlighting of uncertain words on the original image.
- Text-to-speech for elderly or visually impaired patients.

---

## 14. Medical Disclaimer

This tool provides general informational assistance only. It is not medical advice and does not
replace consultation with a qualified doctor or pharmacist. Always verify prescription details
before taking any medication.

All names in the sample prescriptions (patients, doctors and clinics) are fictional.
