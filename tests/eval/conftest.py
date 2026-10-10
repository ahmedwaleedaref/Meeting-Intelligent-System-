"""Shared pytest setup for the evaluation-harness tests (task B).

pytest runs this file automatically before the test files in this folder.
Its only job is to make `import smi` work, because the package lives in `src/`.
"""

import sys 
from pathlib import Path 

SRC_DIR = Path(__file__).resolve().parents[2] / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
