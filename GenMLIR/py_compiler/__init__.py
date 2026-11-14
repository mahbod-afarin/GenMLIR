"""Python implementation of the GenMLIR pipeline.

Given a workload description (APSP or sequence alignment) and a hardware
configuration, it makes the tiling, mapping, tiering and mode-selection
decisions and emits the artifacts consumed by the GenDRAM simulator:

    layout.json     memory objects and their tier placement
    schedule.json   symbolic schedule (shapes, tiling, mapping rule)
    pu_programs/    per-PU summaries for inspection

Pass names match the C++/MLIR pipeline, so individual passes can be disabled
by name for ablation studies (see passes.py).
"""

from .compiler import compile_to_artifacts
from .types import CompileConfig

__all__ = ["compile_to_artifacts", "CompileConfig"]
