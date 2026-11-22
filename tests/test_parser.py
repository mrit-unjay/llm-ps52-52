"""
tests/test_parser.py
--------------------
Offline tests for the safety layer. They use fake ("mocked") Gemini
replies, so they need no API key and no internet.

Run from the project folder:
    python -m unittest discover tests -v
"""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from prescription_parser import ParseError, parse_gemini_response  # noqa: E402
from utils import data_formatter as fmt  # noqa: E402

CASES = config.SAMPLES_DIR / "test_cases"
NS, UNCLEAR = config.NOT_SPECIFIED, config.UNCLEAR


def read(name: str) -> str:
    return (CASES / name).read_text(encoding="utf-8")


def med(**fields) -> dict:
    base = {
        "medicine_name": None, "strength": None, "dosage": None, "frequency": None,
        "timings": [], "food_instruction": None, "duration": None,
        "original_instruction": "", "uncertainties": [],
    }
    base.update(fields)
    return base


def reply(medicines, **extra) -> str:
    body = {"medicines": medicines, "other_instructions": [], "general_warnings": [],
            "requires_verification": False}
    body.update(extra)
    return json.dumps(body)


class TestCase1SingleMedicine(unittest.TestCase):
    def test_single_medicine_extracted(self):
        text = read("tc1_single_medicine.txt")
        raw = reply([med(
            medicine_name="Paracetamol", strength="500 mg", dosage="1 tablet",
            frequency="Twice daily", food_instruction="After food", duration="5 days",
            original_instruction="Tab. Paracetamol 500 mg - 1 tablet twice daily after food for 5 days",
        )])
        result = parse_gemini_response(raw, text)
        self.assertEqual(len(result["medicines"]), 1)
        m = result["medicines"][0]
        self.assertEqual(m["medicine_name"], "Paracetamol")
        self.assertEqual(m["uncertainties"], [])
        self.assertFalse(result["requires_verification"])
        # "Twice daily" does not say which times -> slots are Not specified, plus a note
        row = fmt.build_schedule_dataframe(result["medicines"]).iloc[0]
        self.assertEqual(row["Morning"], NS)
        self.assertTrue(m["notes"])


class TestCase2MultipleMedicines(unittest.TestCase):
    def test_all_medicines_kept(self):
        text = read("tc2_multiple_medicines.txt")
        raw = reply([
            med(medicine_name="Azithromycin", strength="500 mg", dosage="1 tablet",
                frequency="Once daily", food_instruction="Before food", duration="3 days",
                original_instruction="Tab. Azithromycin 500 mg - 1 tab once daily before food x 3 days"),
            med(medicine_name="Paracetamol", strength="650 mg", dosage="1 tablet",
                frequency="Three times daily", food_instruction="After food", duration="5 days",
                original_instruction="Tab. Paracetamol 650 mg - 1 tab TDS after food x 5 days"),
            med(medicine_name="Ambroxol", strength="15 mg/5 ml", dosage="5 ml",
                frequency="Three times daily", duration="5 days",
                original_instruction="Syp. Ambroxol 15 mg/5 ml - 5 ml thrice daily x 5 days"),
            med(medicine_name="Pantoprazole", strength="40 mg", dosage="1 tablet",
                frequency="Once daily", timings=["Morning"], food_instruction="Before breakfast",
                duration="5 days", original_instruction="Tab. Pantoprazole 40 mg - 1 tab before breakfast x 5 days"),
        ])
        result = parse_gemini_response(raw, text)
        self.assertEqual(len(result["medicines"]), 4)
        self.assertEqual(result["medicines"][3]["timings"], ["Morning"])
        self.assertEqual(len(fmt.build_schedule_dataframe(result["medicines"])), 4)


class TestCase3MorningNight(unittest.TestCase):
    def test_timings_map_to_columns(self):
        text = read("tc3_morning_night.txt")
        raw = reply([
            med(medicine_name="Metformin", strength="500 mg", dosage="1 tablet", frequency="Twice daily",
                timings=["Night", "Morning"], food_instruction="After food", duration="30 days",
                original_instruction="Tab. Metformin 500 mg 1-0-1 after food x 30 days"),
            med(medicine_name="Atorvastatin", strength="10 mg", dosage="1 tablet", frequency="Once daily",
                timings=["Night"], food_instruction="After dinner", duration="30 days",
                original_instruction="Tab. Atorvastatin 10 mg 0-0-1 after dinner x 30 days"),
        ])
        result = parse_gemini_response(raw, text)
        df = fmt.build_schedule_dataframe(result["medicines"])
        self.assertEqual(list(df.iloc[0][["Morning", "Afternoon", "Night"]]), [fmt.TAKE, fmt.SKIP, fmt.TAKE])
        self.assertEqual(list(df.iloc[1][["Morning", "Afternoon", "Night"]]), [fmt.SKIP, fmt.SKIP, fmt.TAKE])
        self.assertFalse(result["requires_verification"])

    def test_frequency_timing_conflict_flagged(self):
        raw = reply([med(medicine_name="Metformin", dosage="1 tablet", frequency="Three times daily",
                         timings=["Morning", "Night"], duration="30 days",
                         original_instruction="Tab. Metformin 500 mg 1-0-1 after food x 30 days")])
        result = parse_gemini_response(raw, read("tc3_morning_night.txt"))
        self.assertTrue(any("suggests 3 dose" in u for u in result["medicines"][0]["uncertainties"]))

    def test_invalid_timing_rejected(self):
        raw = reply([med(medicine_name="Metformin", dosage="1 tablet", frequency="Once daily",
                         timings=["Evening"], duration="30 days")])
        result = parse_gemini_response(raw, read("tc3_morning_night.txt"))
        self.assertEqual(result["medicines"][0]["timings"], [])
        self.assertTrue(result["requires_verification"])


class TestCase4MissingDetails(unittest.TestCase):
    def test_missing_fields_not_invented(self):
        text = read("tc4_missing_details.txt")
        raw = reply([
            med(medicine_name="Cetirizine", strength="10 mg", frequency="At night", timings=["Night"],
                original_instruction="Tab. Cetirizine 10 mg - at night"),
            med(medicine_name="Ambroxol", frequency="Three times daily", duration="5 days",
                original_instruction="Syp. Ambroxol - three times a day for 5 days"),
            med(medicine_name="Vitamin D3", strength="60000 IU", original_instruction="Tab. Vitamin D3 60000 IU"),
        ])
        result = parse_gemini_response(raw, text)
        cet, amb, vit = result["medicines"]
        self.assertEqual(cet["duration"], NS)
        self.assertEqual(amb["dosage"], NS)
        self.assertEqual((vit["dosage"], vit["frequency"], vit["duration"]), (NS, NS, NS))
        self.assertTrue(all(m["needs_verification"] for m in result["medicines"]))
        self.assertTrue(result["requires_verification"])
        self.assertIn("does not say", fmt.explain_medicine(vit))


class TestCase5UnclearNames(unittest.TestCase):
    def test_ai_corrected_name_is_caught(self):
        """If the AI 'fixes' Amox1c1l|in to Amoxicillin, the parser must flag it."""
        text = read("tc5_unclear_names.txt")
        raw = reply([med(medicine_name="Amoxicillin", strength="500 mg", dosage="1 capsule",
                         frequency="Twice daily", timings=["Morning", "Night"], duration="5 days",
                         original_instruction="Cap. Amox1c1l|in 5OO mg 1-0-1 aftr fd x 5 dys")])
        result = parse_gemini_response(raw, text)
        issues = " ".join(result["medicines"][0]["uncertainties"])
        self.assertTrue("spelled differently" in issues or "not found" in issues)
        self.assertTrue(result["requires_verification"])

    def test_unclear_marker_normalised(self):
        raw = reply([med(medicine_name="unclear - confirm with pharmacist", dosage="1 tablet",
                         frequency="Once daily", original_instruction="Tab. Pnt0pr~zle 4O mg 1-0-0 bfr bkfst")])
        result = parse_gemini_response(raw, read("tc5_unclear_names.txt"))
        self.assertEqual(result["medicines"][0]["medicine_name"], UNCLEAR)
        self.assertTrue(result["requires_verification"])

    def test_low_confidence_ocr_word_flagged(self):
        text = read("tc5_unclear_names.txt")
        raw = reply([med(medicine_name=UNCLEAR, original_instruction="Tab. Dol? 1 tab SOS",
                         dosage="1 tablet", frequency="Only when needed")])
        result = parse_gemini_response(raw, text, low_confidence_words=["Dol?"])
        self.assertTrue(any("OCR" in w for w in result["general_warnings"]))


class TestVisionUnclearMarker(unittest.TestCase):
    def test_unclear_marker_flagged(self):
        text = "Tab. [unclear] 500 mg 1-0-1 x 5 days"
        raw = reply([med(medicine_name=UNCLEAR, strength="500 mg", dosage="1 tablet",
                         frequency="Twice daily", timings=["Morning", "Night"], duration="5 days",
                         original_instruction=text)])
        result = parse_gemini_response(raw, text)
        self.assertTrue(any("[unclear]" in u for u in result["medicines"][0]["uncertainties"]))
        self.assertTrue(result["requires_verification"])


class TestRobustness(unittest.TestCase):
    def test_code_fences_removed(self):
        raw = "```json\n" + reply([med(medicine_name="Paracetamol", original_instruction="x")]) + "\n```"
        result = parse_gemini_response(raw, "Paracetamol 500 mg")
        self.assertEqual(result["medicines"][0]["medicine_name"], "Paracetamol")

    def test_invalid_json_raises(self):
        with self.assertRaises(ParseError):
            parse_gemini_response("Sorry, I cannot help with that.", "text")

    def test_no_medicines_warns(self):
        result = parse_gemini_response(reply([]), "Hello, this is not a prescription.")
        self.assertTrue(result["general_warnings"])
        self.assertTrue(result["requires_verification"])

    def test_csv_and_report(self):
        result = parse_gemini_response(
            reply([med(medicine_name="Paracetamol", dosage="1 tablet", frequency="Twice daily",
                       duration="5 days", original_instruction="Paracetamol 1 tab BD x 5 days")]),
            "Paracetamol 1 tab BD x 5 days",
        )
        csv_text = fmt.to_csv_bytes(fmt.build_export_dataframe(result["medicines"])).decode("utf-8-sig")
        self.assertIn("Medicine,Strength,Dosage,Morning", csv_text)
        self.assertIn(config.DISCLAIMER, fmt.build_html_report(result, "Paracetamol"))


if __name__ == "__main__":
    unittest.main()
