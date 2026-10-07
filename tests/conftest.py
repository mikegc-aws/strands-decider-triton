"""Make `src/` importable so tests run with a bare `pytest`, no PYTHONPATH needed.

`src/` holds the `strands_decider` package (the model, prompt rendering and the engines)
and `decider_triton` (the wire format). Both are what the image ships, so the tests run
against exactly the code that is deployed.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
