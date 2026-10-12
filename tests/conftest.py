"""Put src/ and the repository root on sys.path so a plain pytest call works."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

for entry in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))
