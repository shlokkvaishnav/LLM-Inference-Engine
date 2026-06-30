"""
Kaggle package isolation — runs before any test import.

The notebook kernel injects /kaggle/working/pkgs (our pinned
transformers/tokenizers/huggingface_hub) into sys.path and evicts stale
sys.modules entries.  Pytest runs in a separate subprocess that inherits
PYTHONPATH=/kaggle/working/pkgs but may still have system modules cached
from its own startup.  This conftest re-applies the same insurance so
model imports inside tests also see the pinned versions.

Safe outside Kaggle: the /kaggle/working/pkgs guard is a no-op when that
directory doesn't exist.
"""
import os
import sys

_TARGET = "/kaggle/working/pkgs"
if os.path.isdir(_TARGET) and _TARGET not in sys.path:
    sys.path.insert(0, _TARGET)

_PREFIXES = ("transformers", "tokenizers", "huggingface_hub")
for _k in list(sys.modules.keys()):
    if any(_k == p or _k.startswith(p + ".") for p in _PREFIXES):
        del sys.modules[_k]
