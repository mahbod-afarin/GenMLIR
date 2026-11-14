from __future__ import annotations

import time
from dataclasses import dataclass

from .emit import emit_layout, emit_pu_programs, emit_schedule
from .types import CompileConfig


@dataclass(frozen=True)
class CompileReport:
    out_dir: str
    layout_json: str
    schedule_json: str
    pu_programs_dir: str
    compile_time_s: float


def compile_to_artifacts(cfg: CompileConfig, out_dir: str) -> CompileReport:
    """Compile a workload configuration into GenDRAM artifacts under out_dir."""
    t0 = time.perf_counter()

    cfg.validate()
    layout_path = emit_layout(cfg, out_dir)
    schedule_path = emit_schedule(cfg, out_dir)
    pu_dir = emit_pu_programs(cfg, out_dir)

    elapsed = time.perf_counter() - t0
    return CompileReport(
        out_dir=out_dir,
        layout_json=layout_path,
        schedule_json=schedule_path,
        pu_programs_dir=pu_dir,
        compile_time_s=elapsed,
    )
