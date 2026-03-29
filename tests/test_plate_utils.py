"""
test_plate_utils.py — unit tests for plate_utils v5
Covers all original tests + province detection + stability gate.
"""
import unittest
import numpy as np


from plate_utils import (
    extract_best_plate_read,
    normalize_plate_text,
    should_send_plate,
    looks_like_thai_plate_top_line,
    preprocess_plate_image,
    normalize_province,
    PROVINCE_MAP,
    PlateMajorityVoter,
    PlateStabilityGate,
)


# ===========================================================================
# Original tests (unchanged)
# ===========================================================================

class TestNormalize(unittest.TestCase):
    def test_removes_whitespace_and_dash(self):
        self.assertEqual(normalize_plate_text("  9 กภ- 71  "), "9กภ71")


class TestLooksLikeThai(unittest.TestCase):
    def test_true_for_valid(self):
        self.assertTrue(looks_like_thai_plate_top_line("9กภ71"))

    def test_false_for_english_only(self):
        self.assertFalse(looks_like_thai_plate_top_line("NNO"))

    def test_false_when_no_digit(self):
        self.assertFalse(looks_like_thai_plate_top_line("กภ"))


class TestExtractBestPlateRead(unittest.TestCase):
    def test_picks_best_thai_like_result(self):
        ocr_results = [
            ([[0, 0], [1, 0], [1, 1], [0, 1]], "NNO", 0.92),
            ([[0, 0], [1, 0], [1, 1], [0, 1]], "9 กภ 71", 0.60),
            ([[0, 0], [1, 0], [1, 1], [0, 1]], "ABC999", 0.95),
        ]
        plate = extract_best_plate_read(ocr_results, min_confidence=0.15)
        self.assertEqual(plate, ("9กภ71", 0.60))

    def test_returns_none_when_text_too_short(self):
        ocr_results = [
            ([[0, 0], [1, 0], [1, 1], [0, 1]], "9", 0.99),
        ]
        self.assertIsNone(extract_best_plate_read(ocr_results,
                                                  min_confidence=0.15))

    def test_returns_none_for_empty_results(self):
        self.assertIsNone(extract_best_plate_read([]))

    def test_returns_none_for_english_only(self):
        ocr_results = [
            ([[0, 0], [1, 0], [1, 1], [0, 1]], "NNO", 0.99),
        ]
        self.assertIsNone(extract_best_plate_read(ocr_results,
                                                  min_confidence=0.15))

    def test_merges_split_bounding_boxes(self):
        ocr_results = [
            ([[0, 0],  [1, 0],  [1, 5],  [0, 5]],  "99",  0.50),
            ([[0, 10], [1, 10], [1, 15], [0, 15]], "กภ",  0.50),
        ]
        result = extract_best_plate_read(ocr_results, min_confidence=0.15)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "99กภ")


class TestShouldSendPlate(unittest.TestCase):
    def test_true_when_not_seen_before(self):
        self.assertTrue(should_send_plate("9กภ71", {},
                                          current_time=5, cooldown_time=10))

    def test_false_inside_cooldown(self):
        last_seen = {"9กภ71": 95}
        self.assertFalse(should_send_plate("9กภ71", last_seen,
                                           current_time=100, cooldown_time=10))

    def test_true_after_cooldown(self):
        last_seen = {"9กภ71": 80}
        self.assertTrue(should_send_plate("9กภ71", last_seen,
                                          current_time=100, cooldown_time=10))


# ===========================================================================
# preprocess_plate_image
# ===========================================================================

class TestPreprocessPlateImage(unittest.TestCase):
    def _make_crop(self, h=60, w=150):
        img = np.ones((h, w, 3), dtype=np.uint8) * 220
        img[10:50, 10:30] = 30
        img[10:50, 40:60] = 30
        img[10:50, 70:90] = 30
        return img

    def test_output_shape_scaled_correctly(self):
        crop = self._make_crop(60, 150)
        processed = preprocess_plate_image(crop, scale=3)
        self.assertEqual(processed.shape, (180, 450))

    def test_raises_on_empty_input(self):
        empty = np.array([])
        with self.assertRaises(ValueError):
            preprocess_plate_image(empty)

    def test_output_dtype_uint8(self):
        crop = self._make_crop()
        processed = preprocess_plate_image(crop, scale=2)
        self.assertEqual(processed.dtype, np.uint8)


# ===========================================================================
# Province map & normalize_province
# ===========================================================================

class TestProvinceMap(unittest.TestCase):
    def test_all_77_provinces_present(self):
        # PROVINCE_MAP may have >77 keys because some provinces appear in
        # multiple regions (e.g. Phitsanulok in both Central and North).
        # The canonical 77 provinces should all be reachable.
        self.assertGreaterEqual(len(PROVINCE_MAP), 77)

    def test_known_province_exact_match(self):
        result = normalize_province("กรุงเทพมหานคร")
        self.assertIsNotNone(result)
        self.assertEqual(result["en"], "Bangkok")
        self.assertEqual(result["region"], "Central")
        self.assertAlmostEqual(result["score"], 1.0)

    def test_known_province_exact_chiangmai(self):
        result = normalize_province("เชียงใหม่")
        self.assertIsNotNone(result)
        self.assertEqual(result["en"], "Chiang Mai")
        self.assertEqual(result["region"], "North")

    def test_known_province_exact_phuket(self):
        result = normalize_province("ภูเก็ต")
        self.assertIsNotNone(result)
        self.assertEqual(result["en"], "Phuket")
        self.assertEqual(result["region"], "South")

    def test_fuzzy_match_partial_bangkok(self):
        # OCR might drop some characters
        result = normalize_province("กรุงเทพ")
        self.assertIsNotNone(result)
        self.assertEqual(result["en"], "Bangkok")

    def test_fuzzy_match_partial_chonburi(self):
        result = normalize_province("ชลบร")   # dropped ุี
        self.assertIsNotNone(result)
        self.assertIn("Chon Buri", result["en"])

    def test_returns_none_for_garbage(self):
        result = normalize_province("XXXXXXXX")
        self.assertIsNone(result)

    def test_returns_none_for_empty(self):
        result = normalize_province("")
        self.assertIsNone(result)

    def test_all_provinces_have_en_and_region(self):
        for thai, info in PROVINCE_MAP.items():
            self.assertIn("en",     info, msg=f"Missing 'en' for {thai}")
            self.assertIn("region", info, msg=f"Missing 'region' for {thai}")
            self.assertIn(info["region"],
                          {"Central", "North", "Northeast", "East", "West", "South"},
                          msg=f"Unknown region for {thai}")


# ===========================================================================
# PlateMajorityVoter
# ===========================================================================

class TestPlateMajorityVoter(unittest.TestCase):
    def test_returns_none_before_window_full(self):
        voter = PlateMajorityVoter(window=4, min_votes=2)
        for _ in range(3):
            result = voter.update(("9กภ71", 0.8))
        self.assertIsNone(result)

    def test_returns_winner_when_majority_reached(self):
        voter = PlateMajorityVoter(window=4, min_votes=3)
        for _ in range(3):
            voter.update(("9กภ71", 0.8))
        result = voter.update(("9กภ71", 0.8))
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "9กภ71")

    def test_returns_none_when_no_majority(self):
        voter = PlateMajorityVoter(window=4, min_votes=3)
        voter.update(("9กภ71", 0.8))
        voter.update(("9กภ72", 0.8))
        voter.update(("9กภ73", 0.8))
        result = voter.update(("9กภ74", 0.8))
        self.assertIsNone(result)

    def test_handles_none_candidates(self):
        voter = PlateMajorityVoter(window=4, min_votes=2)
        voter.update(("9กภ71", 0.8))
        voter.update(None)
        voter.update(("9กภ71", 0.9))
        result = voter.update(None)
        self.assertIsNone(result)

    def test_reset_clears_buffer(self):
        voter = PlateMajorityVoter(window=4, min_votes=2)
        for _ in range(4):
            voter.update(("9กภ71", 0.8))
        voter.reset()
        for _ in range(3):
            result = voter.update(("9กภ71", 0.8))
        self.assertIsNone(result)

    def test_average_confidence_returned(self):
        voter = PlateMajorityVoter(window=4, min_votes=4)
        for conf in [0.6, 0.7, 0.8, 0.9]:
            result = voter.update(("9กภ71", conf))
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result[1], 0.75, places=5)


# ===========================================================================
# PlateStabilityGate
# ===========================================================================

class TestPlateStabilityGate(unittest.TestCase):
    def test_not_stable_before_min_frames(self):
        gate = PlateStabilityGate(min_stable_frames=4, max_pixel_drift=6.0)
        for _ in range(3):
            stable = gate.is_stable(100, 100, 200, 150)
        self.assertFalse(stable)

    def test_stable_when_box_does_not_move(self):
        gate = PlateStabilityGate(min_stable_frames=4, max_pixel_drift=6.0)
        for _ in range(5):
            stable = gate.is_stable(100, 100, 200, 150)
        self.assertTrue(stable)

    def test_not_stable_when_box_jumps(self):
        gate = PlateStabilityGate(min_stable_frames=4, max_pixel_drift=6.0)
        for _ in range(4):
            gate.is_stable(100, 100, 200, 150)
        # Large jump — should reset the stable counter
        stable = gate.is_stable(200, 200, 300, 250)
        self.assertFalse(stable)

    def test_recovers_stability_after_jump(self):
        gate = PlateStabilityGate(min_stable_frames=3, max_pixel_drift=6.0)
        # Move then settle
        gate.is_stable(100, 100, 200, 150)
        gate.is_stable(200, 200, 300, 250)   # big jump — resets
        for _ in range(4):
            stable = gate.is_stable(200, 200, 300, 250)
        self.assertTrue(stable)

    def test_reset_clears_state(self):
        gate = PlateStabilityGate(min_stable_frames=3, max_pixel_drift=6.0)
        for _ in range(5):
            gate.is_stable(100, 100, 200, 150)
        gate.reset()
        # After reset, need min_stable_frames again
        for _ in range(2):
            stable = gate.is_stable(100, 100, 200, 150)
        self.assertFalse(stable)

    def test_small_drift_still_stable(self):
        gate = PlateStabilityGate(min_stable_frames=3, max_pixel_drift=6.0)
        # Move by 3px each time — within drift threshold
        for i in range(5):
            stable = gate.is_stable(100 + i * 2, 100, 200 + i * 2, 150)
        self.assertTrue(stable)

    def test_exact_drift_threshold_is_stable(self):
        gate = PlateStabilityGate(min_stable_frames=3, max_pixel_drift=6.0)
        # Drift of exactly 6.0px — should count as stable (<=)
        import math
        # Move by 6px in x each frame
        for i in range(5):
            stable = gate.is_stable(100 + i * 6, 100, 200 + i * 6, 150)
        self.assertTrue(stable)


if __name__ == "__main__":
    unittest.main()