"""
utils/data_formatter.py
-----------------------
Turns the parsed prescription data into things people can read:
  - a medication schedule (Pandas DataFrame)
  - simple English explanations (built by rules, not by AI, so nothing
    new can be added)
  - a CSV file and a printable HTML report
"""

from datetime import datetime
from html import escape

import pandas as pd

import config

NS = config.NOT_SPECIFIED
UNCLEAR = config.UNCLEAR
TAKE = "✓"
SKIP = "—"

SCHEDULE_COLUMNS = [
    "Medicine", "Strength", "Dosage", "Morning", "Afternoon", "Night",
    "Duration", "Food Instructions",
]


def _slot_value(med: dict, slot: str) -> str:
    if not med["timings"]:
        return NS                       # times of day were never written
    return TAKE if slot in med["timings"] else SKIP


def build_schedule_dataframe(medicines: list[dict]) -> pd.DataFrame:
    rows = []
    for med in medicines:
        rows.append({
            "Medicine": med["medicine_name"],
            "Strength": med["strength"],
            "Dosage": med["dosage"],
            "Morning": _slot_value(med, "Morning"),
            "Afternoon": _slot_value(med, "Afternoon"),
            "Night": _slot_value(med, "Night"),
            "Duration": med["duration"],
            "Food Instructions": med["food_instruction"],
        })
    return pd.DataFrame(rows, columns=SCHEDULE_COLUMNS)


def build_export_dataframe(medicines: list[dict]) -> pd.DataFrame:
    """Schedule plus extra columns that help the pharmacist check the data."""
    df = build_schedule_dataframe(medicines)
    df["Frequency"] = [m["frequency"] for m in medicines]
    df["Original Instruction"] = [m["original_instruction"] for m in medicines]
    df["Needs Verification"] = ["Yes" if m["needs_verification"] else "No" for m in medicines]
    df["Warnings"] = [" | ".join(m["uncertainties"]) for m in medicines]
    return df


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    # utf-8-sig lets Excel on Windows show ✓ and – correctly
    return df.to_csv(index=False).encode("utf-8-sig")


# ------------------------------------------------------------------
# Simple English explanation
# ------------------------------------------------------------------
TIMING_PHRASES = {"Morning": "in the morning", "Afternoon": "in the afternoon", "Night": "at night"}


def _join_words(items: list[str]) -> str:
    """['a', 'b', 'c'] -> 'a, b and c'"""
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def explain_medicine(med: dict) -> str:
    name = med["medicine_name"]
    if name == UNCLEAR:
        name = "Unclear medicine"
    if med["strength"] not in (NS, UNCLEAR):
        name = f"{name} {med['strength']}"

    parts = []
    missing = []

    if med["dosage"] not in (NS, UNCLEAR):
        parts.append(f"take {med['dosage']}")
    else:
        parts.append("take it")
        missing.append("how much to take")

    if med["frequency"] not in (NS, UNCLEAR):
        parts.append(med["frequency"].lower())
    else:
        missing.append("how often")

    if med["timings"]:
        parts.append(_join_words([TIMING_PHRASES[t] for t in med["timings"]]))

    if med["food_instruction"] not in (NS, UNCLEAR):
        parts.append(med["food_instruction"].lower())

    if med["duration"] not in (NS, UNCLEAR):
        parts.append(f"for {med['duration']}")
    else:
        missing.append("for how long")

    sentence = ", ".join(parts)
    sentence = sentence[0].upper() + sentence[1:] + "."

    if missing:
        sentence += f" The prescription does not say {_join_words(missing)}."
    if med["needs_verification"]:
        sentence += " Please confirm this medicine with your pharmacist or doctor."
    return f"{name}: {sentence}"


def build_explanations(medicines: list[dict]) -> list[str]:
    return [explain_medicine(m) for m in medicines]


# ------------------------------------------------------------------
# Printable HTML report
# ------------------------------------------------------------------
def build_html_report(result: dict, source_text: str) -> str:
    df = build_schedule_dataframe(result["medicines"])
    table_html = df.to_html(index=False, escape=True, border=0, classes="schedule")
    explanations = "".join(f"<li>{escape(e)}</li>" for e in build_explanations(result["medicines"]))

    warning_items = list(result["general_warnings"])
    for med in result["medicines"]:
        for issue in med["uncertainties"]:
            warning_items.append(f"{med['medicine_name']}: {issue}")
    warnings_html = (
        "".join(f"<li>{escape(w)}</li>" for w in warning_items)
        or "<li>No uncertainties were detected. Still verify every detail with your pharmacist.</li>"
    )
    others_html = "".join(f"<li>{escape(o)}</li>" for o in result.get("other_instructions", []))
    others_section = f"<h2>Other instructions in the prescription</h2><ul>{others_html}</ul>" if others_html else ""
    status = (
        "Some details need verification by a pharmacist or doctor."
        if result["requires_verification"]
        else "No uncertainties detected, but always verify."
    )
    generated = datetime.now().strftime("%d %B %Y, %I:%M %p")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MediRead AI – Medication Schedule Report</title>
<style>
  body {{ font-family: "Atkinson Hyperlegible", Arial, sans-serif; color: #16324A;
         max-width: 900px; margin: 32px auto; padding: 0 20px; line-height: 1.55; }}
  h1 {{ color: #1F6FB2; margin-bottom: 0; }}
  h2 {{ font-size: 1.1rem; margin-top: 28px; border-bottom: 2px solid #D6E6F5; padding-bottom: 4px; }}
  .meta {{ color: #4A6178; margin-top: 4px; }}
  .disclaimer {{ background: #FFF6E5; border: 1px solid #E8B44C; padding: 12px 14px; border-radius: 6px; }}
  .status {{ background: #EEF5FB; border-left: 4px solid #1F6FB2; padding: 10px 14px; }}
  table.schedule {{ border-collapse: collapse; width: 100%; font-size: 0.95rem; }}
  table.schedule th {{ background: #1F6FB2; color: white; text-align: left; padding: 8px; }}
  table.schedule td {{ border-bottom: 1px solid #D6E6F5; padding: 8px; }}
  pre {{ background: #F5F8FB; padding: 12px; white-space: pre-wrap; border-radius: 6px; }}
  @media print {{ body {{ margin: 0; }} .noprint {{ display: none; }} }}
</style>
</head>
<body>
<h1>MediRead AI – Medication Schedule</h1>
<p class="meta">Generated on {generated}. Press Ctrl+P to print or save as PDF.</p>
<p class="disclaimer"><strong>Disclaimer:</strong> {escape(config.DISCLAIMER)}</p>
<p class="status">{escape(status)}</p>
<h2>Medication schedule</h2>
{table_html}
<p class="meta">✓ = take at this time &nbsp; — = not at this time &nbsp; "Not specified" = not written in the prescription</p>
<h2>Instructions in simple English</h2>
<ul>{explanations}</ul>
{others_section}
<h2>Warnings and items to verify</h2>
<ul>{warnings_html}</ul>
<h2>Prescription text that was analysed</h2>
<pre>{escape(source_text)}</pre>
<p class="disclaimer"><strong>Disclaimer:</strong> {escape(config.DISCLAIMER)}</p>
</body>
</html>"""
