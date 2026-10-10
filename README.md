# Structured Meeting Intelligence

Graduation project on dialogue-act tagging and proposal-response linking over the ICSI MRDA corpus.

## Tasks

| ID | Task | Owner | Status |
|----|------|-------|--------|
| [B](tasks/B-eval-harness/README.md) | Evaluation harness | Ahmed | in progress |

## Setup

```bash
python3 -m venv .venv               # once
source .venv/bin/activate           # in every new terminal
pip install -r requirements.txt
```

## Running the tests

From the repo root:

```bash
python -m pytest tests -v          # every test, one line each
python -m pytest tests -q          # short summary
python -m pytest tests -v -rs      # also show why tests were skipped
```

Tests that need local data (`data/`, not in git) are skipped, not failed, when the data is missing.
