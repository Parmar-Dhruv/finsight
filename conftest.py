"""Pytest bootstrap: put the project root on sys.path so `generation` and
`retrieval` are importable regardless of the working directory."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
