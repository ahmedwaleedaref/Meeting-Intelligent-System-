"""Make the shared NXT corpus builders importable for the pipeline tests."""

import sys
from pathlib import Path

NXT_TESTS = Path(__file__).resolve().parents[1] / "data_nxt"

if str(NXT_TESTS) not in sys.path:
    sys.path.insert(0, str(NXT_TESTS))
