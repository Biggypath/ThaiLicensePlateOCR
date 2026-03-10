import unittest

from plate_utils import extract_best_plate_read, normalize_plate_text, should_send_plate


class TestPlateUtils(unittest.TestCase):
    def test_normalize_plate_text_removes_whitespace_and_dash(self):
        self.assertEqual(normalize_plate_text("  กข- 1234  "), "กข1234")

    def test_extract_best_plate_read_picks_highest_confidence(self):
        ocr_results = [
            ([[0, 0], [1, 0], [1, 1], [0, 1]], "กข 1234", 0.60),
            ([[0, 0], [1, 0], [1, 1], [0, 1]], "AB-999", 0.92),
            ([[0, 0], [1, 0], [1, 1], [0, 1]], "XYZ111", 0.75),
        ]

        plate = extract_best_plate_read(ocr_results)
        self.assertEqual(plate, ("AB999", 0.92))

    def test_extract_best_plate_read_returns_none_when_text_too_short(self):
        ocr_results = [
            ([[0, 0], [1, 0], [1, 1], [0, 1]], "A", 0.99),
        ]

        plate = extract_best_plate_read(ocr_results)
        self.assertIsNone(plate)

    def test_extract_best_plate_read_returns_none_for_empty_results(self):
        self.assertIsNone(extract_best_plate_read([]))

    def test_should_send_plate_true_when_not_seen_before(self):
        self.assertTrue(should_send_plate("กข1234", {}, current_time=100, cooldown_time=10))

    def test_should_send_plate_false_inside_cooldown(self):
        last_seen = {"กข1234": 95}
        self.assertFalse(should_send_plate("กข1234", last_seen, current_time=100, cooldown_time=10))

    def test_should_send_plate_true_after_cooldown(self):
        last_seen = {"กข1234": 80}
        self.assertTrue(should_send_plate("กข1234", last_seen, current_time=100, cooldown_time=10))


if __name__ == "__main__":
    unittest.main()
