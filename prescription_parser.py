"""
prescription_parser.py
----------------------
Turns Gemini's raw reply into clean, validated data, then runs its own
safety checks. We do not blindly trust the AI:

  1. Parse the JSON (remove code fences if Gemini added them).
  2. Normalise every field: missing -> "Not specified", unclear -> the
     standard "Unclear – Confirm with Pharmacist/Doctor" text.
  3. Rule-based checks that do NOT use AI:
       - Is the medicine name really in the prescription text?
         (catches names the AI "corrected" or invented)
       - Were any OCR low-confidence words used in this medicine?
       - Are dosage, frequency or duration missing?
       - Does the frequency match the number of timings?
       - Are there duplicate medicines?
"""

import difflib
import json
import re

import config

NS = config.NOT_SPECIFIED
UNCLEAR = config.UNCLEAR

TEXT_FIELDS = ["medicine_name", "strength", "dosage", "frequency", "food_instruction", "duration"]
EMPTY_VALUES = {"", "null", "none", "n/a", "na", "not specified", "not mentioned", "not given", "-"}

# Frequency words -> number of doses per day (used for conflict checks)
FREQUENCY_COUNTS = [
    (r"\b(four times|4 times|qid|qds)\b", 4),
    (r"\b(three times|3 times|thrice|tds|tid)\b", 3),
    (r"\b(twice|two times|2 times|bd|bid)\b", 2),
    (r"\b(once|one time|1 time|od|qd)\b", 1),
]


class ParseError(Exception):
    """Raised when Gemini's reply cannot be understood."""


# ------------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------------
def _normalise(text: str) -> str:
    """Lower-case and keep only letters/digits separated by single spaces."""
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _load_json(raw: str) -> dict:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ParseError("The AI response was not valid JSON. Please click Extract again.")
        try:
            data = json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            raise ParseError("The AI response was not valid JSON. Please click Extract again.")

    if isinstance(data, list):                     # model returned only the list
        data = {"medicines": data}
    if not isinstance(data, dict):
        raise ParseError("The AI response had an unexpected structure. Please click Extract again.")
    return data


def _clean_value(value) -> str:
    if value is None:
        return NS
    if isinstance(value, (list, dict)):
        value = ", ".join(str(v) for v in value) if isinstance(value, list) else json.dumps(value)
    value = str(value).strip()
    if value.lower() in EMPTY_VALUES:
        return NS
    if value.lower().startswith("unclear"):
        return UNCLEAR
    return value


def _clean_string_list(values) -> list[str]:
    if not isinstance(values, list):
        values = [values] if values else []
    result = []
    for item in values:
        item = str(item).strip()
        if item and item.lower() not in EMPTY_VALUES and item not in result:
            result.append(item)
    return result


def _clean_timings(values, uncertainties: list[str]) -> list[str]:
    """Keep only Morning / Afternoon / Night, in that order."""
    found = set()
    for item in _clean_string_list(values):
        match = next((slot for slot in config.TIMING_SLOTS if slot.lower() == item.lower()), None)
        if match:
            found.add(match)
        else:
            uncertainties.append(
                f"Timing '{item}' does not fit the Morning/Afternoon/Night schedule. "
                "Check the original instruction."
            )
    return [slot for slot in config.TIMING_SLOTS if slot in found]


def _frequency_count(frequency: str) -> int | None:
    lowered = frequency.lower()
    if any(word in lowered for word in ("week", "month", "alternate", "needed", "sos", "prn")):
        return None
    if re.fullmatch(r"\s*\d\s*-\s*\d\s*-\s*\d\s*", lowered):          # e.g. "1-0-1"
        return sum(1 for part in lowered.split("-") if part.strip() != "0")
    for pattern, count in FREQUENCY_COUNTS:
        if re.search(pattern, lowered):
            return count
    return None


# ------------------------------------------------------------------
# Rule-based safety checks
# ------------------------------------------------------------------
def _check_name_in_text(name: str, source_text: str) -> str | None:
    """Return a warning if the name does not appear in the prescription text."""
    if name in (NS, UNCLEAR):
        return None
    name_norm, text_norm = _normalise(name), _normalise(source_text)
    if not name_norm:
        return None
    if f" {name_norm} " in f" {text_norm} ":
        return None                                    # exact match: OK

    text_words = text_norm.split()
    first_word = name_norm.split()[0]
    close = difflib.get_close_matches(first_word, text_words, n=1, cutoff=0.8)
    if close:
        return (
            f"The name '{name}' is spelled differently from the prescription text "
            f"(closest written word: '{close[0]}'). It may have been misread or changed."
        )
    return (
        f"The name '{name}' was not found in the prescription text. "
        "It may have been misread - do not rely on it."
    )


def _check_medicine(med: dict, source_text: str, low_confidence_words: list[str]) -> None:
    """Add uncertainties and notes to one medicine (changes the dict in place)."""
    issues = med["uncertainties"]
    notes = med["notes"]

    name_issue = _check_name_in_text(med["medicine_name"], source_text)
    if name_issue:
        issues.append(name_issue)

    if med["medicine_name"] == UNCLEAR:
        issues.append("Medicine name could not be read clearly.")

    # Words the reader marked as unreadable
    marker = config.UNCLEAR_MARKER
    if any(marker in str(med[f]) for f in TEXT_FIELDS + ["original_instruction"]):
        issues.append("Part of this line could not be read (marked [unclear]).")

    # Low-confidence OCR words used in this medicine's line
    medicine_words = set(_normalise(med["original_instruction"] + " " + med["medicine_name"]).split())
    for word in low_confidence_words:
        if _normalise(word) and _normalise(word) in medicine_words:
            issues.append(f"OCR was not confident about the word '{word}' in this line.")

    for field, label in (("dosage", "Dosage"), ("frequency", "Frequency"), ("duration", "Duration")):
        if med[field] == NS:
            issues.append(f"{label} is not stated in the prescription.")
        elif med[field] == UNCLEAR:
            issues.append(f"{label} could not be read clearly.")
    if med["strength"] == UNCLEAR:
        issues.append("Strength could not be read clearly.")

    # Frequency vs timings conflict
    count = _frequency_count(med["frequency"]) if med["frequency"] not in (NS, UNCLEAR) else None
    timings = med["timings"]
    if count and timings and count != len(timings):
        issues.append(
            f"Frequency ('{med['frequency']}') suggests {count} dose(s) a day, but "
            f"{len(timings)} time(s) of day were found ({', '.join(timings)})."
        )
    if count and count > 3 and timings:
        issues.append("More than three doses a day do not fit the Morning/Afternoon/Night table.")

    if med["frequency"] not in (NS, UNCLEAR) and not timings:
        notes.append(
            "The prescription says how often to take this, but not at what times of day. "
            "Ask your pharmacist when to take each dose."
        )

    # De-duplicate while keeping order
    med["uncertainties"] = list(dict.fromkeys(issues))
    med["notes"] = list(dict.fromkeys(notes))
    med["needs_verification"] = bool(med["uncertainties"]) or any(
        med[f] == UNCLEAR for f in TEXT_FIELDS
    )


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------
def parse_gemini_response(raw: str, source_text: str, low_confidence_words: list[str] | None = None) -> dict:
    low_confidence_words = [
        w for w in (low_confidence_words or []) if _normalise(w) in _normalise(source_text).split()
    ]
    data = _load_json(raw)

    raw_medicines = data.get("medicines") or []
    if not isinstance(raw_medicines, list):
        raise ParseError("The AI response did not contain a list of medicines. Please click Extract again.")

    medicines = []
    for item in raw_medicines:
        if not isinstance(item, dict):
            continue
        uncertainties = _clean_string_list(item.get("uncertainties"))
        med = {field: _clean_value(item.get(field)) for field in TEXT_FIELDS}
        med["timings"] = _clean_timings(item.get("timings"), uncertainties)
        original = item.get("original_instruction")
        med["original_instruction"] = str(original).strip() if original else NS
        med["uncertainties"] = uncertainties
        med["notes"] = []
        _check_medicine(med, source_text, low_confidence_words)
        medicines.append(med)

    general_warnings = _clean_string_list(data.get("general_warnings"))
    other_instructions = _clean_string_list(data.get("other_instructions"))

    if not medicines:
        general_warnings.append(
            "No medicines could be identified. The text may be unreadable or may not be a "
            "prescription. Check the text or ask your pharmacist."
        )

    names = [m["medicine_name"].lower() for m in medicines if m["medicine_name"] not in (NS, UNCLEAR)]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    for name in duplicates:
        general_warnings.append(
            f"'{name}' appears more than once. Confirm with your doctor whether both entries apply."
        )

    if low_confidence_words:
        general_warnings.append(
            "OCR had low confidence in some words: "
            + ", ".join(f"'{w}'" for w in low_confidence_words[:15])
            + ". Compare the text with the original prescription."
        )

    general_warnings = list(dict.fromkeys(general_warnings))
    requires_verification = (
        bool(data.get("requires_verification"))
        or any(m["needs_verification"] for m in medicines)
        or bool(general_warnings)
    )

    return {
        "medicines": medicines,
        "other_instructions": other_instructions,
        "general_warnings": general_warnings,
        "requires_verification": requires_verification,
    }
