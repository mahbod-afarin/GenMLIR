from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple

from .passes import REQUIRED_PASSES, disabled_passes_set, is_known_pass

WorkloadKind = Literal["apsp", "alignment"]


@dataclass(frozen=True)
class HardwareConfig:
    """GenDRAM hardware parameters. The defaults follow the GenDRAM paper."""

    # PU partitioning.
    num_pus_total: int = 32
    num_search_pus: int = 8
    num_compute_pus: int = 24

    # Memory parallelism: channels * bank_groups_per_channel bank-groups.
    channels: int = 4
    bank_groups_per_channel: int = 8

    pe_lanes_per_pu: int = 16
    frequency_hz: float = 1.0e9

    def validate(self) -> None:
        if self.channels * self.bank_groups_per_channel != self.num_pus_total:
            raise ValueError(
                "Invalid HW config: channels*bank_groups_per_channel must equal num_pus_total "
                f"(got {self.channels}*{self.bank_groups_per_channel} != {self.num_pus_total})."
            )
        if self.num_search_pus + self.num_compute_pus != self.num_pus_total:
            raise ValueError(
                "Invalid HW config: num_search_pus + num_compute_pus must equal num_pus_total "
                f"(got {self.num_search_pus}+{self.num_compute_pus} != {self.num_pus_total})."
            )
        if self.pe_lanes_per_pu <= 0:
            raise ValueError("pe_lanes_per_pu must be > 0")
        if self.frequency_hz <= 0:
            raise ValueError("frequency_hz must be > 0")


@dataclass(frozen=True)
class APSPConfig:
    N: int
    tile_size: int = 64
    elem_bytes: int = 4
    elem_type: str = "i32"


@dataclass(frozen=True)
class AlignmentConfig:
    query_len: int
    ref_len: int
    tile_size: int = 64
    elem_bytes: int = 4
    elem_type: str = "i32"

    # Smith-Waterman scoring.
    match: int = 2
    mismatch: int = -1
    gap: int = -1
    band_width: int = 256


@dataclass(frozen=True)
class CompileConfig:
    workload: WorkloadKind
    hw: HardwareConfig
    apsp: Optional[APSPConfig] = None
    alignment: Optional[AlignmentConfig] = None

    # Passes to leave out, by name (see passes.py). Used for ablations.
    disabled_passes: Tuple[str, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        self.hw.validate()

        if self.workload == "apsp":
            if self.apsp is None:
                raise ValueError("APSP workload requires apsp config")
            if self.apsp.N <= 0:
                raise ValueError("APSP N must be > 0")
            if self.apsp.tile_size <= 0:
                raise ValueError("tile_size must be > 0")
        elif self.workload == "alignment":
            if self.alignment is None:
                raise ValueError("alignment workload requires alignment config")
            if self.alignment.query_len <= 0 or self.alignment.ref_len <= 0:
                raise ValueError("query_len/ref_len must be > 0")
            if self.alignment.tile_size <= 0:
                raise ValueError("tile_size must be > 0")

        disabled = disabled_passes_set(self.disabled_passes)
        unknown = sorted(p for p in disabled if not is_known_pass(p))
        if unknown:
            raise ValueError(
                "Unknown pass name(s) in disabled_passes: "
                + ", ".join(unknown)
                + ". Known passes include: dp-tiling, wavefront-schedule, gendram-mapping, ..."
            )

        forbidden = sorted(disabled.intersection(REQUIRED_PASSES))
        if forbidden:
            raise ValueError(
                "Cannot disable required pass(es): "
                + ", ".join(forbidden)
                + ". These are required to emit artifacts for the simulator."
            )


ScheduleJSON = Dict[str, object]
LayoutJSON = Dict[str, object]
