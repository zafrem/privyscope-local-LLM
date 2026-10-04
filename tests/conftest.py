"""Prefer the sibling working-tree core over any older installed ``privyscope``.

Runs before test modules import ``privyscope_local_llm`` (whose ``types`` imports
``privyscope``), so every test sees the same core regardless of collection order.
"""
import sys
from pathlib import Path

CORE = Path(__file__).resolve().parents[2] / "privyscope"
if (CORE / "privyscope").is_dir():
    sys.path.insert(0, str(CORE))
