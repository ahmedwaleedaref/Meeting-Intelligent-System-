"""Label definitions for the MRDA tag -> class mapping."""

from types import MappingProxyType
from typing import Final

LABELS: Final[tuple[str, ...]] = ("cs", "co", "aa", "bk", "ar", "cc", "other")

# Difference between LABELS and TARGET_LABELS is that the latter does not include "other". Will be used for evaluation metrics.
 
TARGET_LABELS: Final[tuple[str, ...]] = ("cs", "co", "aa", "bk", "ar", "cc")

RAW_TAG_TO_LABEL: Final = MappingProxyType({
    "cs": "cs",
    "co": "co",
    "aa": "aa",
    "bk": "bk",
    "ar": "ar",
    "cc": "cc",
    "aap": "aa",
    "arp": "ar"
})


def get_label_from_raw_tag(raw_tag: str) -> str:
    return RAW_TAG_TO_LABEL.get(raw_tag, "other")
