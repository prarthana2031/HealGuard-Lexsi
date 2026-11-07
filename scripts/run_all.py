"""Deprecated shim. The orchestrator lives at the repo root.

The plan's layout puts `run_all.py` at the top level, next to `configs/`,
`gates/` and `artifacts/`. This file only forwards so muscle memory still works.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from run_all import main  # noqa: E402

if __name__ == "__main__":
    print("scripts/run_all.py is a shim; use `python run_all.py` from the repo root.")
    main()
