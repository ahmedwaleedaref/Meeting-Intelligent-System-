"""Task 2: the LABELS contract in src/smi/labels.py."""

import ast
from pathlib import Path

import pytest

from smi import labels
from smi.data.annotate import tags

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_labels_is_the_exact_seven_class_tuple() -> None:
    """LABELS is the exact D2 tuple, with other last."""
    assert labels.LABELS == ("cs", "co", "aa", "bk", "ar", "cc", "other")


def test_target_labels_drop_only_other() -> None:
    """TARGET_LABELS is LABELS without the trailing other class."""
    assert labels.TARGET_LABELS == labels.LABELS[:-1]
    assert tags.OTHER == labels.LABELS[-1] == "other"


def test_labels_are_unique() -> None:
    """No class appears twice in LABELS."""
    assert len(set(labels.LABELS)) == len(labels.LABELS)


def test_plus_variants_map_to_their_base_class() -> None:
    """aap maps to aa and arp maps to ar."""
    assert labels.RAW_TAG_TO_LABEL["aap"] == "aa"
    assert labels.RAW_TAG_TO_LABEL["arp"] == "ar"


def test_every_target_label_maps_to_itself() -> None:
    """Each target class tag is its own label."""
    for label in labels.TARGET_LABELS:
        assert labels.RAW_TAG_TO_LABEL[label] == label


def test_every_mapped_value_is_a_target_label() -> None:
    """The mapping never produces other or an unknown class."""
    assert set(labels.RAW_TAG_TO_LABEL.values()) == set(labels.TARGET_LABELS)


def test_unknown_tags_fall_back_to_other() -> None:
    """A tag outside the mapping resolves to other."""
    assert labels.get_label_from_raw_tag("df") == "other"
    assert labels.get_label_from_raw_tag("aap") == "aa"


def test_tag_mapping_is_read_only() -> None:
    """The shared mapping cannot be edited at runtime."""
    with pytest.raises(TypeError):
        labels.RAW_TAG_TO_LABEL["new"] = "cs"


def test_nothing_else_in_the_repo_redefines_labels() -> None:
    """Only src/smi/labels.py assigns the LABELS name."""
    offenders = []
    for folder in ("src", "scripts", "tests"):
        for path in (PROJECT_ROOT / folder).rglob("*.py"):
            if path == PROJECT_ROOT / "src" / "smi" / "labels.py":
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
                if any(isinstance(target, ast.Name) and target.id == "LABELS" for target in targets):
                    offenders.append(str(path.relative_to(PROJECT_ROOT)))
    assert offenders == []
