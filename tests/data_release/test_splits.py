import tempfile
import unittest
from pathlib import Path

from smi.data.splits import (
    attach_split_info,
    assign_grouped_folds,
    assign_stratified_splits,
    build_split_provenance,
    load_meeting_ids,
)


class SplitBuilderTests(unittest.TestCase):
    def test_stratified_splits_are_stable_and_assign_each_meeting_once(self):
        meeting_ids = ["Bmr001", "Bmr002", "Bmr003", "Bro001", "Bro002", "Bed001"]
        counts = {"train": 4, "val": 1, "test": 1}

        first = assign_stratified_splits(meeting_ids, counts, seed="fixture")
        second = assign_stratified_splits(reversed(meeting_ids), counts, seed="fixture")

        self.assertEqual(first, second)
        self.assertEqual(len(first["train"]), 4)
        self.assertEqual(len(first["val"]), 1)
        self.assertEqual(len(first["test"]), 1)
        assigned = first["train"] + first["val"] + first["test"]
        self.assertCountEqual(assigned, meeting_ids)
        self.assertEqual(len(assigned), len(set(assigned)))

    def test_load_meeting_ids_reads_observations(self):
        xml = """<corpus><agents><observation name='ignore'/></agents>
        <observations><observation name='Bdb001'/><observation name='Bed002'/></observations>
        </corpus>"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.xml"
            path.write_text(xml, encoding="utf-8")
            self.assertEqual(load_meeting_ids(path), ["Bdb001", "Bed002"])

    def test_fold_assignment_is_stable(self):
        first = assign_grouped_folds(["Bdb003", "Bdb001", "Bdb002"])
        second = assign_grouped_folds(["Bdb002", "Bdb003", "Bdb001"])
        self.assertEqual(first, second)
        self.assertEqual(first, {"Bdb001": 0, "Bdb002": 1, "Bdb003": 2})

    def test_provenance_excludes_test_meetings_from_folds(self):
        result = build_split_provenance(
            ["Bdb001", "Bdb002", "Bdb003", "Bdb004"],
            {"train": ["Bdb001", "Bdb002"], "val": ["Bdb003"], "test": ["Bdb004"]},
            split_source="lead Table 9 lists",
        )
        by_id = {}
        for row in result["meetings"]:
            by_id[row["meeting_id"]] = row
        self.assertIsNone(by_id["Bdb004"]["fold"])
        for meeting in ("Bdb001", "Bdb002", "Bdb003"):
            self.assertIsNotNone(by_id[meeting]["fold"])

    def test_provenance_rejects_incomplete_coverage(self):
        with self.assertRaisesRegex(ValueError, "missing from split lists"):
            build_split_provenance(
                ["Bdb001", "Bdb002"],
                {"train": ["Bdb001"], "val": [], "test": []},
                split_source="lead Table 9 lists",
            )

    def test_provenance_rejects_overlapping_splits(self):
        with self.assertRaisesRegex(ValueError, "more than one split"):
            build_split_provenance(
                ["Bdb001"],
                {"train": ["Bdb001"], "val": ["Bdb001"], "test": []},
                split_source="lead Table 9 lists",
            )

    def test_attach_split_info_adds_meeting_assignment_to_each_row(self):
        provenance = build_split_provenance(
            ["Bdb001", "Bdb002"],
            {"train": ["Bdb001"], "val": [], "test": ["Bdb002"]},
            split_source="lead Table 9 lists",
        )
        rows = [
            {"seg_id": "Bdb001-c1_0000000_0001000", "meeting": "Bdb001"},
            {"seg_id": "Bdb002-c1_0000000_0001000", "meeting": "Bdb002"},
        ]

        result = attach_split_info(rows, provenance)

        self.assertEqual(result[0]["split"], "train")
        self.assertIsInstance(result[0]["fold"], int)
        self.assertEqual(result[1]["split"], "test")
        self.assertIsNone(result[1]["fold"])
        self.assertNotIn("split", rows[0])

    def test_attach_split_info_rejects_unknown_meeting(self):
        provenance = build_split_provenance(
            ["Bdb001"],
            {"train": ["Bdb001"], "val": [], "test": []},
            split_source="lead Table 9 lists",
        )
        with self.assertRaisesRegex(ValueError, "no split assignment"):
            attach_split_info([{"meeting": "unknown"}], provenance)


if __name__ == "__main__":
    unittest.main()
