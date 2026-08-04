"""
SORT shim for the edge plugin.

The upstream sort/sort.py imports `skimage` and `matplotlib` at module load
for debug visualisation that is never called in the inference hot path.
Those imports make the tracker unusable in a slim edge container without
pulling in heavy dev deps.

This module stubs them out in sys.modules, then exec()s the upstream
sort/sort.py so the real tracker classes/functions are available under
the plugin's own namespace. If you need the original debug plotting,
import sort/sort.py from the repo root directly (with the full deps).
"""
import sys
import types

_mpl_stub = types.ModuleType("matplotlib")
_mpl_stub.use = lambda *a, **kw: None
_mpl_pyplot_stub = types.ModuleType("matplotlib.pyplot")
_mpl_patches_stub = types.ModuleType("matplotlib.patches")
_sk_stub = types.ModuleType("skimage")
_sk_io_stub = types.ModuleType("skimage.io")
_sk_io_stub.imread = lambda *a, **kw: None
_sk_stub.io = _sk_io_stub

sys.modules.setdefault("matplotlib", _mpl_stub)
sys.modules.setdefault("matplotlib.pyplot", _mpl_pyplot_stub)
sys.modules.setdefault("matplotlib.patches", _mpl_patches_stub)
sys.modules.setdefault("skimage", _sk_stub)
sys.modules.setdefault("skimage.io", _sk_io_stub)

from pathlib import Path

# Resolve upstream sort.py: container layout has it at /app/sort/sort.py
# (same dir as this shim). Repo-root dev layout has it at <repo>/sort/sort.py
# (one parent up). Try local-first so both layouts work without config.
_local = Path(__file__).resolve().parent / "sort" / "sort.py"
_upstream = _local if _local.exists() else Path(__file__).resolve().parents[1] / "sort" / "sort.py"
_exec_ns = {"__name__": "sort_upstream", "__file__": str(_upstream)}
exec(compile(_upstream.read_text(), str(_upstream), "exec"), _exec_ns)

Sort = _exec_ns["Sort"]
linear_assignment = _exec_ns["linear_assignment"]
iou_batch = _exec_ns["iou_batch"]

__all__ = ["Sort", "linear_assignment", "iou_batch"]
