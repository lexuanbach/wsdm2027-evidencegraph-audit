from __future__ import annotations

import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1] / "artifact" / "code"
sys.path.insert(0, str(CODE_DIR))
