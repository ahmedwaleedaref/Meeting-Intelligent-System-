import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from smi.data.release import (
    ReleaseData,
    acceptable_responses,
    load_release,
    validate_release_content,
)
from smi.data.splits import build_split_provenance
from scripts.make_release import make_release


def make_row(
    seg_id,
    meeting,
    position,
    *,
    proposal=False,
    split="train",
    fold=0,
    quality_flags=None,
    label=None,
):
    target_tags = ["cs"] if proposal else []
    return {
        "seg_id": seg_id,
        "meeting": meeting,
        "agent": "A",
        "channel": "c1",
        "speaker_id": "speaker-1" if position == 0 else "speaker-2",
        "src_start": float(position),
        "src_end": float(position + 1),
        "start": float(position),
        "end": float(position + 1),
        "timing_method": "element",
        "position": position,
        "source_da_id": f"{meeting}.A.dialogueact{position}",
        "element_ordinal": position,
        "part_index": 0,
        "n_parts": 1,
        "raw_type": "s^cs" if proposal else "s",
        "original_type": "s^cs" if proposal else "s",
        "part_type": "s^cs" if proposal else "s",
        "adjacency_raw": None,
        "comment": None,
        "text": "text",
        "words": [],
        "quality_flags": quality_flags or [],
        "label": label if label is not None else ("cs" if proposal else "other"),
        "label_bk_merged": "cs" if proposal else "other",
        "target_tags": target_tags,
        "is_multi_target": False,
        "is_layer2_proposal": proposal,
        "general_tags": [],
        "disruption_markers": [],
        "is_nonspeech": False,
        "is_empty": False,
        "split": split,
        "fold": fold,
    }


def valid_content():
    rows = [
        make_row("m1-p", "m1", 0, proposal=True),
        make_row("m1-r", "m1", 1),
    ]
    links = [
        {
            "proposal_id": "m1-p",
            "responder_chains": [["m1-r"]],
            "has_response": True,
            "status": "ok",
            "adjacency_tokens": ["1a.1b"],
            "pair_group": "m1-pair-1",
        }
    ]
    splits = {
        "k": 5,
        "grouping": "meeting",
        "meetings": {"m1": {"split": "train", "fold": 0}},
        "source_sha256": "a" * 64,
    }
    return rows, links, splits


class Layer2ViewTests(unittest.TestCase):
    def test_acceptable_sets_for_all_policy_combinations(self):
        record = {
            "status": "ok",
            "responder_chains": [["r1", "r2"], ["r3"]],
        }
        self.assertEqual(acceptable_responses(record), {"r1", "r3"})
        self.assertEqual(acceptable_responses(record, segments="any"), {"r1", "r2", "r3"})
        self.assertEqual(acceptable_responses(record, responders="first"), {"r1"})
        self.assertEqual(
            acceptable_responses(record, responders="first", segments="any"), {"r1", "r2"}
        )

    def test_no_pair_is_empty_and_malformed_is_not_gold_none(self):
        self.assertEqual(acceptable_responses({"status": "no_pair", "responder_chains": []}), set())
        with self.assertRaisesRegex(ValueError, "not eligible"):
            acceptable_responses({"status": "malformed", "responder_chains": []})

    def test_layer2_gold_excludes_malformed_and_ambiguous_records(self):
        rows = []
        for seg_id in ("p1", "p2", "p3", "p4"):
            rows.append({"seg_id": seg_id})
        release = ReleaseData(
            rows=tuple(rows),
            links=(
                {"proposal_id": "p1", "status": "ok", "responder_chains": [["r1"]]},
                {"proposal_id": "p2", "status": "no_pair", "responder_chains": []},
                {"proposal_id": "p3", "status": "malformed", "responder_chains": []},
                {"proposal_id": "p4", "status": "ambiguous", "responder_chains": []},
            ),
            splits={},
            metadata={},
        )
        gold_ids = []
        for item in release.layer2_gold():
            gold_ids.append(item["proposal_id"])
        self.assertEqual(gold_ids, ["p1", "p2"])


class ReleaseValidationTests(unittest.TestCase):
    def test_valid_content(self):
        validate_release_content(*valid_content())

    def test_rejects_duplicate_seg_id(self):
        rows, links, splits = valid_content()
        with self.assertRaisesRegex(ValueError, "duplicate seg_id"):
            validate_release_content([rows[0], dict(rows[0]), rows[1]], links, splits)

    def test_rejects_invalid_label(self):
        rows, links, splits = valid_content()
        rows[0]["label"] = "not-a-label"
        with self.assertRaisesRegex(ValueError, "outside LABELS"):
            validate_release_content(rows, links, splits)

    def test_rejects_meeting_assigned_to_multiple_splits(self):
        rows, links, splits = valid_content()
        rows[1]["split"] = "val"
        rows[1]["fold"] = 1
        with self.assertRaisesRegex(ValueError, "multiple splits or folds"):
            validate_release_content(rows, links, splits)

    def test_rejects_missing_fold(self):
        rows, links, splits = valid_content()
        rows[0]["fold"] = None
        with self.assertRaisesRegex(ValueError, "missing a valid fold"):
            validate_release_content(rows, links, splits)

    def test_rejects_missing_link_target(self):
        rows, links, splits = valid_content()
        links[0]["responder_chains"] = [["does-not-exist"]]
        with self.assertRaisesRegex(ValueError, "missing response"):
            validate_release_content(rows, links, splits)

    def test_rejects_has_response_mismatch(self):
        rows, links, splits = valid_content()
        links[0]["has_response"] = False
        with self.assertRaisesRegex(ValueError, "has_response inconsistent"):
            validate_release_content(rows, links, splits)

    def test_rejects_error_quality_flag(self):
        rows, links, splits = valid_content()
        rows[1]["quality_flags"] = ["ref_unresolved"]
        with self.assertRaisesRegex(ValueError, "error-class quality flags"):
            validate_release_content(rows, links, splits)

    def test_allows_null_raw_type_only_with_original_type_fallback_flag(self):
        rows, links, splits = valid_content()
        rows[0]["raw_type"] = None
        rows[0]["quality_flags"] = ["type_fallback_original"]
        validate_release_content(rows, links, splits)

        rows[0]["quality_flags"] = []
        with self.assertRaisesRegex(ValueError, "null raw_type without a fallback flag"):
            validate_release_content(rows, links, splits)

    def test_rejects_proposal_without_link_and_link_without_proposal(self):
        rows, links, splits = valid_content()
        with self.assertRaisesRegex(ValueError, "proposal rows without link records"):
            validate_release_content(rows, [], splits)
        rows[0]["is_layer2_proposal"] = False
        rows[0]["target_tags"] = []
        rows[0]["label"] = "other"
        rows[0]["label_bk_merged"] = "other"
        with self.assertRaisesRegex(ValueError, "ineligible proposal"):
            validate_release_content(rows, links, splits)

    def test_loader_requires_exact_release_file_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "unexpected.txt").write_text("extra", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly five files"):
                load_release(root)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "exactly five files"):
                load_release(directory)

    def test_loader_verifies_checksums_and_loads_release(self):
        rows, links, splits = valid_content()
        metadata = {"data_version": "v1"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row_lines = []
            for row in rows:
                row_lines.append(json.dumps(row) + "\n")
            link_lines = []
            for link in links:
                link_lines.append(json.dumps(link) + "\n")
            payloads = {
                "rows.jsonl": "".join(row_lines),
                "links.jsonl": "".join(link_lines),
                "splits.json": json.dumps(splits) + "\n",
                "metadata.json": json.dumps(metadata) + "\n",
            }
            for filename, payload in payloads.items():
                (root / filename).write_text(payload, encoding="utf-8", newline="\n")
            checksum_lines = []
            for filename in sorted(payloads):
                digest = hashlib.sha256((root / filename).read_bytes()).hexdigest()
                checksum_lines.append(f"{digest}  {filename}")
            checksum_text = "\n".join(checksum_lines) + "\n"
            (root / "checksums.sha256").write_text(checksum_text, encoding="ascii")

            release = load_release(root)

            self.assertEqual(len(release.rows), 2)
            self.assertEqual(len(release.layer2_gold()), 1)
            self.assertEqual(release.layer2_gold()[0]["acceptable_responses"], {"m1-r"})


class ReleaseBuilderTests(unittest.TestCase):
    def test_build_is_deterministic_and_writes_exactly_five_files(self):
        rows, links, _ = valid_content()
        c2_rows = []
        for row in rows:
            c2_row = dict(row)
            c2_row.pop("split")
            c2_row.pop("fold")
            c2_rows.append(c2_row)

        split_provenance = build_split_provenance(
            ["m1"],
            {"train": ["m1"], "val": [], "test": []},
            split_source="fixture Table 9 lists",
        )
        source_archive = {
            "archive_name": "fixture.zip",
            "release": "fixture-1",
            "sha256": "b" * 64,
        }
        decisions = {}
        for number in range(1, 17):
            decisions[f"D{number}"] = f"fixture decision {number}"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c2_rows_path = root / "c2_rows.jsonl"
            c2_links_path = root / "c2_links.jsonl"
            split_path = root / "split_v1.json"
            source_path = root / "source_archive.json"
            decisions_path = root / "decisions.json"

            row_lines = []
            for row in c2_rows:
                row_lines.append(json.dumps(row) + "\n")
            c2_rows_path.write_text("".join(row_lines), encoding="utf-8")

            link_lines = []
            for link in links:
                link_lines.append(json.dumps(link) + "\n")
            c2_links_path.write_text("".join(link_lines), encoding="utf-8")
            split_path.write_text(json.dumps(split_provenance) + "\n", encoding="utf-8")
            source_path.write_text(json.dumps(source_archive) + "\n", encoding="utf-8")
            decisions_path.write_text(json.dumps(decisions) + "\n", encoding="utf-8")

            first_output = root / "dry-run-one"
            second_output = root / "dry-run-two"
            first_fingerprint = make_release(
                c2_rows_path,
                c2_links_path,
                split_path,
                source_path,
                decisions_path,
                first_output,
                dry_run=True,
            )
            second_fingerprint = make_release(
                c2_rows_path,
                c2_links_path,
                split_path,
                source_path,
                decisions_path,
                second_output,
                dry_run=True,
            )

            first_files = {}
            for path in first_output.iterdir():
                first_files[path.name] = path.read_bytes()
            second_files = {}
            for path in second_output.iterdir():
                second_files[path.name] = path.read_bytes()

            self.assertEqual(set(first_files), {
                "rows.jsonl",
                "links.jsonl",
                "splits.json",
                "metadata.json",
                "checksums.sha256",
            })
            self.assertEqual(first_files, second_files)
            self.assertEqual(first_fingerprint, second_fingerprint)

            final_output = root / "v1"
            provenance_path = root / "provenance" / "release_v1.json"
            release_fingerprint = make_release(
                c2_rows_path,
                c2_links_path,
                split_path,
                source_path,
                decisions_path,
                final_output,
                provenance_path=provenance_path,
            )

            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            self.assertEqual(provenance["release_fingerprint"], release_fingerprint)
            self.assertEqual(
                set(provenance["release_files_sha256"]),
                {
                    "rows.jsonl",
                    "links.jsonl",
                    "splits.json",
                    "metadata.json",
                    "checksums.sha256",
                },
            )

    def test_dry_run_cannot_target_v1_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "temporary directory"):
                make_release(
                    "rows",
                    "links",
                    "splits",
                    "source",
                    "decisions",
                    Path(directory) / "v1",
                    dry_run=True,
                )


if __name__ == "__main__":
    unittest.main()
