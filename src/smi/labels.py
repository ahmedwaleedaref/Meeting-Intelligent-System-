"""The 7-class label set. Defined here once and imported by data, models and eval."""

# Fixed order: also the row/column order of the confusion matrix.
LABELS: tuple[str, ...] = ("cs", "co", "aa", "bk", "ar", "cc", "other")

# The 6 target classes; the primary metric is macro-F1 over these.
TARGET_LABELS: tuple[str, ...] = ("cs", "co", "aa", "bk", "ar", "cc")
