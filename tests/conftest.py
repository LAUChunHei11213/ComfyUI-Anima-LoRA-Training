"""Shared pytest fixtures and sys.path setup for the
ComfyUI-Anima-LoRA-Training test suite.

The pack lives inside ComfyUI/custom_nodes/<pkg>/ so tests need the
package root on sys.path for the same `from utils.config_generator
import ...` / `from nodes.training_wizard import ...` import style the
nodes themselves use in their direct-execution fallback.
"""

import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))
