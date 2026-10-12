import unittest

from smi.inputs.formatter import ContextFormatter


def row(
    seg_id,
    meeting,
    position,
    speaker,
    text,
    *,
    nonspeech=False,
    empty=False,
    source_da_id=None,
):
    return {
        "seg_id": seg_id,
        "meeting": meeting,
        "position": position,
        "speaker_id": speaker,
        "text": text,
        "source_da_id": source_da_id or seg_id,
        "is_nonspeech": nonspeech,
        "is_empty": empty,
        "label": "SECRET_LABEL",
        "target_tags": ["SECRET_TAG"],
        "links": ["SECRET_LINK"],
        "split": "SECRET_SPLIT",
        "fold": 4,
    }


class ContextFormatterTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            row("m1-a", "m1", 0, "p1", "Earlier"),
            row("m1-b", "m1", 1, "p2", "Noise", nonspeech=True),
            row("m1-c", "m1", 2, "p1", "Target"),
            row("m1-d", "m1", 3, "p3", "Later"),
            row("m1-e", "m1", 4, "p2", "Empty", empty=True),
            row("m2-a", "m2", 0, "p4", "Other meeting"),
        ]
        self.formatter = ContextFormatter(reversed(self.rows))

    def test_zero_window_returns_only_target(self):
        self.assertEqual(self.formatter.select_context("m1-c", 0, 0), ["m1-c"])

    def test_windows_follow_position_and_stop_at_meeting_boundaries(self):
        self.assertEqual(
            self.formatter.select_context("m1-c", 1, 1),
            ["m1-a", "m1-c", "m1-d"],
        )
        self.assertEqual(self.formatter.select_context("m1-c", 2, 0), ["m1-a", "m1-c"])
        self.assertEqual(self.formatter.select_context("m1-c", 0, 2), ["m1-c", "m1-d"])
        self.assertEqual(self.formatter.select_context("m1-a", 2, 0), ["m1-a"])
        self.assertEqual(self.formatter.select_context("m1-e", 0, 2), ["m1-e"])

    def test_drop_skips_flagged_neighbors_but_keeps_flagged_target(self):
        self.assertEqual(
            self.formatter.select_context("m1-c", 2, 2, "drop"),
            ["m1-a", "m1-c", "m1-d"],
        )
        self.assertEqual(
            self.formatter.select_context("m1-b", 1, 1, "drop"),
            ["m1-a", "m1-b", "m1-c"],
        )

    def test_placeholder_counts_and_renders_flagged_units(self):
        expected_ids = ["m1-a", "m1-b", "m1-c", "m1-d", "m1-e"]
        actual_ids = self.formatter.select_context("m1-c", 2, 2, "placeholder")
        self.assertEqual(actual_ids, expected_ids)
        rendered = self.formatter.format("m1-c", 2, 2, "window_letters", nonspeech="placeholder")
        self.assertIn("Speaker B: [nonverbal]", rendered)
        self.assertEqual(rendered.count("[nonverbal]"), 2)
        self.assertIn("[T]Target[/T]", rendered)

    def test_speaker_modes_and_zero_window(self):
        self.assertEqual(
            self.formatter.format("m1-c", 2, 1, "raw", nonspeech="placeholder"),
            "Speaker p1: Earlier\n"
            "Speaker p2: [nonverbal]\n"
            "Speaker p1: [T]Target[/T]\n"
            "Speaker p3: Later",
        )
        self.assertIn(
            "Speaker SAME: [T]Target[/T]",
            self.formatter.format("m1-c", 2, 1, "relative", nonspeech="placeholder"),
        )
        self.assertEqual(
            self.formatter.format("m1-c", 2, 1, "window_letters", nonspeech="placeholder"),
            "Speaker A: Earlier\n"
            "Speaker B: [nonverbal]\n"
            "Speaker A: [T]Target[/T]\n"
            "Speaker C: Later",
        )
        for mode in ("raw", "relative", "window_letters"):
            self.assertEqual(self.formatter.format("m1-c", 0, 0, mode), "[T]Target[/T]")

    def test_model_text_excludes_gold_fields_and_is_stable(self):
        formatted = self.formatter.format("m1-c", 1, 1, "raw")
        self.assertNotIn("SECRET", formatted)
        self.assertEqual(formatted, self.formatter.format("m1-c", 1, 1, "raw"))

    def test_split_parts_remain_separate_context_units(self):
        formatter = ContextFormatter(
            [
                row("part-1", "m1", 0, "p1", "First part", source_da_id="same-element"),
                row("part-2", "m1", 1, "p1", "Second part", source_da_id="same-element"),
            ]
        )
        self.assertEqual(formatter.select_context("part-2", 1, 0), ["part-1", "part-2"])


if __name__ == "__main__":
    unittest.main()
