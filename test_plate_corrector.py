"""
test_plate_corrector.py — unit tests for plate_corrector.py

Tests cover:
  - Pass A: already-valid plates pass through unchanged
  - Pass B: structural split from mixed strings
  - Pass C: positional forced substitution
  - Pass D: all-digit recovery (313699 → ฐบ3699)
  - Pass E: sliding window in noisy strings
  - is_valid_plate / correction_score
  - correct_candidates batch helper
  - All known live-log misread examples from the v6/v7 codebase
"""

import unittest
from plate_corrector import correct_plate, is_valid_plate, correct_candidates, normalise_raw


class TestNormalise(unittest.TestCase):
    def test_strips_spaces(self):
        self.assertEqual(normalise_raw("  กข 1234  "), "กข1234")

    def test_strips_dashes(self):
        self.assertEqual(normalise_raw("กข-1234"), "กข1234")

    def test_strips_pipe(self):
        self.assertEqual(normalise_raw("กข|1234"), "กข1234")

    def test_thai_digit_glyphs_converted(self):
        self.assertEqual(normalise_raw("กข๑๒๓๔"), "กข1234")

    def test_latin_uppercased(self):
        self.assertEqual(normalise_raw("bk1234"), "BK1234")


class TestIsValidPlate(unittest.TestCase):
    def test_valid_no_leading_digit(self):
        self.assertTrue(is_valid_plate("กข1234"))

    def test_valid_with_leading_digit(self):
        self.assertTrue(is_valid_plate("1กข1234"))

    def test_invalid_wrong_consonant_count(self):
        self.assertFalse(is_valid_plate("กขค1234"))   # 3 consonants

    def test_invalid_wrong_digit_count(self):
        self.assertFalse(is_valid_plate("กข123"))     # only 3 digits

    def test_invalid_vowel_in_consonant_slot(self):
        self.assertFalse(is_valid_plate("กาข1234"))   # า is a vowel

    def test_invalid_digit_in_consonant_slot(self):
        # raw "3บ3099" — NOT a valid plate as-is
        self.assertFalse(is_valid_plate("3บ3099"))

    def test_valid_all_consonant_range(self):
        # spot-check corners of consonant set
        self.assertTrue(is_valid_plate("กก1234"))
        self.assertTrue(is_valid_plate("ฮฮ9999"))


class TestPassA(unittest.TestCase):
    """Already valid — should return unchanged with score 0.0"""

    def test_simple_valid(self):
        result, score = correct_plate("กข1234")
        self.assertEqual(result, "กข1234")
        self.assertAlmostEqual(score, 0.0)

    def test_with_leading_digit(self):
        result, score = correct_plate("1กข5678")
        self.assertEqual(result, "1กข5678")
        self.assertAlmostEqual(score, 0.0)

    def test_with_spaces_stripped(self):
        result, score = correct_plate("กข 1234")
        self.assertEqual(result, "กข1234")
        self.assertAlmostEqual(score, 0.0)


class TestPassC(unittest.TestCase):
    """Positional forced substitution — correct individual character errors."""

    def test_digit_in_first_consonant_slot(self):
        # "3บ1234"  → ฐ in slot 0, บ in slot 1
        result, score = correct_plate("3บ1234")
        self.assertEqual(result, "ฐบ1234")
        self.assertLess(score, 0.5)

    def test_both_consonant_slots_are_digits(self):
        # "311234"  → ฐ=3, บ=1, digits=1234
        result, score = correct_plate("311234")
        self.assertEqual(result, "ฐบ1234")
        self.assertLess(score, 0.5)

    def test_digit_in_digit_slot_unchanged(self):
        # "กข3ว99"  → ว=0 (from _THAI_TO_DIGIT)
        result, _ = correct_plate("กข3ว99")
        self.assertIsNotNone(result)
        self.assertTrue(is_valid_plate(result))

    def test_latin_b_in_consonant_slot(self):
        # B → บ
        result, _ = correct_plate("Bก1234")
        self.assertIsNotNone(result)
        self.assertEqual(result[1], "ก")

    def test_leading_digit_preserved(self):
        # "13บ1234"  → leading 1, ฐ=3, บ=บ, 1234
        result, _ = correct_plate("13บ1234")
        self.assertEqual(result, "1ฐบ1234")

    def test_vowel_in_digit_slot_converted(self):
        # "กขว234" → ว→0, result กข0234
        result, _ = correct_plate("กขว234")
        self.assertIsNotNone(result)
        self.assertTrue(is_valid_plate(result))

    def test_ต_in_digit_slot_converted(self):
        # ต → 1
        result, _ = correct_plate("กข1ต34")
        self.assertIsNotNone(result)
        self.assertTrue(is_valid_plate(result))


class TestPassD(unittest.TestCase):
    """All-digit recovery — the most common live failure mode."""

    def test_313699(self):
        # ฐ=3, บ=1, real digits=3699
        result, score = correct_plate("313699")
        self.assertEqual(result, "ฐบ3699")
        self.assertLess(score, 0.6)

    def test_111234(self):
        # บ=1, บ=1, real digits=1234
        result, _ = correct_plate("111234")
        self.assertEqual(result, "บบ1234")

    def test_all_digit_length_7_with_leading(self):
        # "1313699" → leading 1, ฐ=3, บ=1, 3699
        result, _ = correct_plate("1313699")
        self.assertEqual(result, "1ฐบ3699")

    def test_all_digit_unrecoverable_returns_none(self):
        # 12345 — length 5, no valid template
        result, _ = correct_plate("12345")
        self.assertIsNone(result)

    def test_351234(self):
        # ร=5 would give 5th digit slot... but ร=5 in consonant
        # 3→ฐ, 5→ร, digits=1234
        result, _ = correct_plate("351234")
        self.assertEqual(result, "ฐร1234")


class TestPassB(unittest.TestCase):
    """Structural split from strings that have runs of consonants + digits."""

    def test_consonant_run_and_digit_run(self):
        # "กข xxxx 1234" — structural split should find กข + 1234
        result, _ = correct_plate("กข1234")
        self.assertIsNotNone(result)

    def test_extra_digit_before_consonants(self):
        result, _ = correct_plate("1กข1234")
        self.assertEqual(result, "1กข1234")

    def test_consonant_run_of_3_truncated(self):
        # "กขค1234" — 3 consonants but only 2 used
        result, _ = correct_plate("กขค1234")
        # Should pick first 2 consonants
        self.assertIsNotNone(result)
        self.assertTrue(is_valid_plate(result))


class TestPassE(unittest.TestCase):
    """Sliding window in noisy / long OCR output."""

    def test_prefix_noise(self):
        # Province text bled in front: "กรงเทพกข1234"
        result, _ = correct_plate("กรงเทพกข1234")
        self.assertIsNotNone(result)
        self.assertTrue(is_valid_plate(result))

    def test_suffix_noise(self):
        result, _ = correct_plate("กข1234กรง")
        self.assertIsNotNone(result)
        self.assertTrue(is_valid_plate(result))

    def test_all_digit_noisy_longer_string(self):
        # "991313699" — sliding window finds "313699" → ฐบ3699
        result, _ = correct_plate("991313699")
        self.assertIsNotNone(result)
        self.assertTrue(is_valid_plate(result))


class TestCorrectionScore(unittest.TestCase):
    def test_zero_score_for_perfect(self):
        _, score = correct_plate("กข1234")
        self.assertAlmostEqual(score, 0.0)

    def test_nonzero_score_for_corrected(self):
        _, score = correct_plate("313699")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_none_for_garbage(self):
        result, score = correct_plate("XXXXXXXX")
        self.assertIsNone(result)


class TestLiveLogMisreads(unittest.TestCase):
    """
    Every known misread from the v6/v7 live logs.
    Each should either correct to a valid plate or return None
    (never return an invalid plate string).
    """

    known_misreads = [
        ("313699",   "ฐบ3699"),   # ฐ=3, บ=1
        ("3บ3699",   "ฐบ3699"),   # ฐ=3, บ=บ
        ("รบ3699",   "รบ3699"),   # ร=ร, บ=บ  — already valid!
        ("513699",   "รบ3699"),   # ร=5, บ=1
        ("ฐบ3699",   "ฐบ3699"),   # perfect
        ("3บ3ว99",   None),       # 3ว in digit zone — might be irrecoverable
        ("ฐบ3ว99",   None),       # ว=0? → ฐบ3099 — optional if supported
        ("ต669835",  None),       # ต bleed — too garbled
        ("313699 ",  "ฐบ3699"),   # trailing space
        (" กข 1234", "กข1234"),   # spaces in middle
    ]

    def test_all_live_misreads(self):
        for raw, expected in self.known_misreads:
            with self.subTest(raw=raw):
                result, _ = correct_plate(raw)
                if expected is None:
                    # Should be None OR a valid plate (never invalid)
                    if result is not None:
                        self.assertTrue(
                            is_valid_plate(result),
                            f"correct_plate({raw!r}) returned invalid plate {result!r}"
                        )
                else:
                    self.assertEqual(result, expected,
                                     f"correct_plate({raw!r}) = {result!r}, expected {expected!r}")


class TestCorrectCandidates(unittest.TestCase):
    def test_filters_uncorrectable(self):
        candidates = [
            ("XXXXXXXX", 0.9),
            ("313699",   0.6),
        ]
        out = correct_candidates(candidates)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][0], "ฐบ3699")

    def test_confidence_penalised(self):
        candidates = [("313699", 0.8)]
        out = correct_candidates(candidates)
        self.assertEqual(len(out), 1)
        # corrected conf must be ≤ original conf
        self.assertLessEqual(out[0][1], 0.8)

    def test_no_penalty_for_perfect(self):
        candidates = [("กข1234", 0.9)]
        out = correct_candidates(candidates)
        self.assertEqual(out[0][1], 0.9)

    def test_heavy_correction_filtered_by_max_score(self):
        # correction score for "313699" is non-trivial; with tight threshold filtered
        candidates = [("313699", 0.3)]
        out_tight = correct_candidates(candidates, max_correction_score=0.1)
        # with generous threshold kept
        out_generous = correct_candidates(candidates, max_correction_score=0.9)
        self.assertGreaterEqual(len(out_generous), len(out_tight))

    def test_batch_with_yolo_conf_tuple(self):
        # 3-tuple format
        candidates = [("กข1234", 0.9, 0.85), ("313699", 0.5, 0.70)]
        out = correct_candidates(candidates)
        texts = [t for t, _ in out]
        self.assertIn("กข1234", texts)
        self.assertIn("ฐบ3699", texts)


if __name__ == "__main__":
    unittest.main(verbosity=2)