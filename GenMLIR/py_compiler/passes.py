"""Pass names and selection helpers for the Python GenMLIR compiler.

The names match the passes registered by genmlir-opt, so a pass can be
disabled by name, or all passes swept one at a time, without the MLIR
toolchain.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Set

ALL_PASSES: List[str] = [
    "dp-extract",
    "dp-canonicalize",
    "dp-tiling",
    "apsp-phase-formation",
    "wavefront-schedule",
    "gendram-mode-select",
    "gendram-mapping",
    "gendram-tiering",
    "gendram-comm-insert",
    "gendram-lower-to-runtime-calls",
    "gendram-emit-artifacts",
]

WORKLOAD_PASSES: Dict[str, List[str]] = {
    "apsp": [
        "dp-extract",
        "dp-canonicalize",
        "dp-tiling",
        "apsp-phase-formation",
        "gendram-mode-select",
        "gendram-mapping",
        "gendram-tiering",
        "gendram-comm-insert",
        "gendram-lower-to-runtime-calls",
        "gendram-emit-artifacts",
    ],
    "alignment": [
        "dp-extract",
        "dp-canonicalize",
        "dp-tiling",
        "wavefront-schedule",
        "gendram-mode-select",
        "gendram-mapping",
        "gendram-tiering",
        "gendram-comm-insert",
        "gendram-lower-to-runtime-calls",
        "gendram-emit-artifacts",
    ],
}

# The simulator cannot run without the emitted artifacts.
REQUIRED_PASSES: Set[str] = {"gendram-emit-artifacts"}


def normalize_pass_name(name: str) -> str:
    return str(name).strip().lower().replace("_", "-")


def is_known_pass(name: str) -> bool:
    return normalize_pass_name(name) in {normalize_pass_name(p) for p in ALL_PASSES}


def disabled_passes_set(disabled: Iterable[str]) -> Set[str]:
    return {normalize_pass_name(p) for p in disabled if str(p).strip()}


def workload_passes(workload: str) -> List[str]:
    """Passes that apply to a workload, in pipeline order."""
    w = normalize_pass_name(workload)
    return list(WORKLOAD_PASSES.get(w, ALL_PASSES))
