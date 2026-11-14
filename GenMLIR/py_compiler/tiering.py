"""Capacity-aware, latency-ranked tier placement (pass gendram-tiering).

Objects are ranked by a latency-sensitivity score sigma and placed into the
fastest tiers first, subject to per-tier capacity. The paper defines
sigma(o) = a(o) / beta(o), access intensity over reuse. The scores used here
are fixed per object class (see genomics_objects and apsp_objects).

The GenDRAM stack has 8 tiers of 4 GB each. Tier 0 is closest to the logic
die and fastest, tier 7 is slowest. The per-tier row-cycle time tRC follows
the GenDRAM timing parameters (tRCD ramp, tRP, tRAS offset).

The per-object assignment and the per-tier occupancy are written to
layout.json.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

GB = 1024 ** 3

NUM_TIERS = 8
TIER_CAPACITY_BYTES = 4 * GB

# tRCD per tier (ns) and the terms that turn it into the row-cycle time tRC.
_T_RCD_NS = [2.29, 3.92, 5.99, 8.50, 11.44, 14.82, 18.63, 22.88]
_T_RP_NS = 4.77
_T_RAS_OFFSET_NS = 27.5


def tier_trc_ns(tier: int) -> float:
    """Row-cycle time tRC = tRP + tRAS, with tRAS = tRCD + offset."""
    t = max(0, min(NUM_TIERS - 1, int(tier)))
    return float(_T_RP_NS + (_T_RCD_NS[t] + _T_RAS_OFFSET_NS))


@dataclass
class TierObject:
    name: str
    size_bytes: int
    sigma: float          # latency sensitivity; higher wants a faster tier
    category: str         # "latency_critical" or "streaming"


@dataclass
class Placement:
    name: str
    size_bytes: int
    sigma: float
    category: str
    tier_lo: int
    tier_hi: int          # inclusive; an object may span several tiers
    bytes_per_tier: Dict[int, int] = field(default_factory=dict)


def place_objects(objects: List[TierObject]) -> Tuple[List[Placement], Dict[int, Dict[str, int]]]:
    """Greedy placement, fastest tier first, bounded by tier capacity.

    Returns (placements, per_tier_occupancy), where per_tier_occupancy[t]
    maps a category to the bytes placed in tier t.
    """
    ordered = sorted(objects, key=lambda o: (-float(o.sigma), o.name))

    tier_remaining = {t: TIER_CAPACITY_BYTES for t in range(NUM_TIERS)}
    per_tier: Dict[int, Dict[str, int]] = {
        t: {"latency_critical": 0, "streaming": 0} for t in range(NUM_TIERS)
    }
    placements: List[Placement] = []

    cur_tier = 0
    for obj in ordered:
        remaining = int(obj.size_bytes)
        lo = cur_tier
        bytes_per_tier: Dict[int, int] = {}
        while remaining > 0 and cur_tier < NUM_TIERS:
            avail = tier_remaining[cur_tier]
            if avail <= 0:
                cur_tier += 1
                continue
            take = min(avail, remaining)
            tier_remaining[cur_tier] -= take
            remaining -= take
            bytes_per_tier[cur_tier] = bytes_per_tier.get(cur_tier, 0) + take
            per_tier[cur_tier][obj.category] += take
            if tier_remaining[cur_tier] == 0:
                cur_tier += 1
        used_tiers = sorted(bytes_per_tier.keys())
        placements.append(
            Placement(
                name=obj.name,
                size_bytes=int(obj.size_bytes),
                sigma=float(obj.sigma),
                category=obj.category,
                tier_lo=used_tiers[0] if used_tiers else lo,
                tier_hi=used_tiers[-1] if used_tiers else lo,
                bytes_per_tier=bytes_per_tier,
            )
        )

    return placements, per_tier


def genomics_objects(*, query_len: int, ref_len: int, elem_bytes: int = 4,
                     batch_reads: int = 1024,
                     ptr_cal_total_bytes: int = int(17 * GB)) -> List[TierObject]:
    """Memory objects of the genomics pipeline (seeding plus alignment).

    PTR/CAL hash tables   about 17 GB in total; dependent two-stage random
                          lookups, so they get the highest sigma
    DP score buffer H     (query_len + 1) x (ref_len + 1) elements
    read stream           batch_reads x query_len bytes; bandwidth-bound
    """
    ptr = int(ptr_cal_total_bytes * 0.5)
    cal = int(ptr_cal_total_bytes - ptr)
    h_bytes = int((query_len + 1) * (ref_len + 1) * elem_bytes)
    reads_bytes = int(batch_reads * query_len)  # one byte per base
    return [
        TierObject("PTR", ptr, sigma=1.00, category="latency_critical"),
        TierObject("CAL", cal, sigma=0.95, category="latency_critical"),
        TierObject("H", h_bytes, sigma=0.40, category="streaming"),
        TierObject("reads", reads_bytes, sigma=0.10, category="streaming"),
    ]


def apsp_objects(*, N: int, elem_bytes: int = 4) -> List[TierObject]:
    """APSP uses a single dense N x N distance matrix."""
    d_bytes = int(N) * int(N) * int(elem_bytes)
    return [TierObject("D", d_bytes, sigma=0.50, category="streaming")]
