from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

from .passes import disabled_passes_set
from .tiering import (
    NUM_TIERS,
    TIER_CAPACITY_BYTES,
    apsp_objects,
    genomics_objects,
    place_objects,
    tier_trc_ns,
)
from .types import AlignmentConfig, APSPConfig, CompileConfig, HardwareConfig, LayoutJSON, ScheduleJSON

SLOWEST_TIER = NUM_TIERS - 1


def _mkdir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _write_json(path: str, obj: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
        f.write("\n")


def _layout_object(name: str, mlir_type: str, *, interleave: bool) -> Dict[str, object]:
    # "tier" is assigned by the placement step in emit_layout.
    return {
        "name": name,
        "kind": "global",
        "mlir_type": mlir_type,
        "tier": SLOWEST_TIER,
        "channel": 0,
        "bank_group": 0,
        "interleave": interleave,
    }


def _hw_block(hw: HardwareConfig) -> Dict[str, object]:
    return {
        "num_pus_total": hw.num_pus_total,
        "num_search_pus": hw.num_search_pus,
        "num_compute_pus": hw.num_compute_pus,
        "pe_lanes_per_pu": hw.pe_lanes_per_pu,
        "frequency_hz": hw.frequency_hz,
    }


def emit_layout(cfg: CompileConfig, out_dir: str) -> str:
    """Write layout.json: memory objects and their tier placement."""
    _mkdir(out_dir)

    disabled = disabled_passes_set(cfg.disabled_passes)
    tiering_enabled = "gendram-tiering" not in disabled

    objects: List[Dict[str, object]] = []
    tier_objs = []

    if cfg.workload == "apsp" and cfg.apsp is not None:
        n = cfg.apsp.N
        objects.append(
            _layout_object("D", f"memref<{n}x{n}x{cfg.apsp.elem_type}>", interleave=True)
        )
        tier_objs = apsp_objects(N=n, elem_bytes=cfg.apsp.elem_bytes)

    if cfg.workload == "alignment" and cfg.alignment is not None:
        h0 = cfg.alignment.query_len + 1
        h1 = cfg.alignment.ref_len + 1
        objects.append(
            _layout_object("H", f"memref<{h0}x{h1}x{cfg.alignment.elem_type}>", interleave=True)
        )
        tier_objs = genomics_objects(
            query_len=cfg.alignment.query_len,
            ref_len=cfg.alignment.ref_len,
            elem_bytes=cfg.alignment.elem_bytes,
        )

    # The seeding tables are declared for both workloads, as in the MLIR
    # modules produced by gendsl-translate.py. This flow does not execute the
    # seeding stage.
    objects.append(_layout_object("PTR", "memref<1024xi32>", interleave=False))
    objects.append(_layout_object("CAL", "memref<4096xi32>", interleave=False))

    placements, per_tier = place_objects(tier_objs) if tier_objs else ([], {})

    # With tiering disabled (B0/B1 baselines) everything sits on the slowest
    # tier. Objects without a placement entry also default to it.
    place_by_name = {p.name: p for p in placements}
    for obj in objects:
        placed = place_by_name.get(str(obj["name"])) if tiering_enabled else None
        obj["tier"] = placed.tier_lo if placed else SLOWEST_TIER
        obj["tier_hi"] = placed.tier_hi if placed else SLOWEST_TIER

    tiering_block = {
        "enabled": tiering_enabled,
        "policy": "latency_ranked_capacity_fill",
        "num_tiers": NUM_TIERS,
        "tier_capacity_bytes": TIER_CAPACITY_BYTES,
        "objects": [
            {
                "name": p.name,
                "size_bytes": int(p.size_bytes),
                "sigma": float(p.sigma),
                "category": p.category,
                "tier_lo": p.tier_lo if tiering_enabled else SLOWEST_TIER,
                "tier_hi": p.tier_hi if tiering_enabled else SLOWEST_TIER,
                "bytes_per_tier": {str(k): int(v) for k, v in p.bytes_per_tier.items()}
                if tiering_enabled
                else {str(SLOWEST_TIER): int(p.size_bytes)},
            }
            for p in placements
        ],
        "per_tier_occupancy_bytes": {
            str(t): {
                "latency_critical": int(per_tier.get(t, {}).get("latency_critical", 0)),
                "streaming": int(per_tier.get(t, {}).get("streaming", 0)),
            }
            for t in range(NUM_TIERS)
        }
        if tiering_enabled
        else {},
        "tier_trc_ns": {str(t): round(tier_trc_ns(t), 2) for t in range(NUM_TIERS)},
    }

    layout: LayoutJSON = {
        "objects": objects,
        "tiering": tiering_block,
        "compiler": {
            "disabled_passes": sorted(disabled),
        },
    }

    path = os.path.join(out_dir, "layout.json")
    _write_json(path, layout)
    return path


def emit_schedule(cfg: CompileConfig, out_dir: str) -> str:
    """Write schedule.json.

    The schedule is symbolic. It records shapes, tiling, the mapping rule and
    the model parameters selected by the enabled passes; the simulator derives
    the tile order from them. Its size is independent of the problem size.
    """
    _mkdir(out_dir)
    cfg.validate()

    if cfg.workload == "apsp" and cfg.apsp is not None:
        return _emit_schedule_apsp(cfg, out_dir)
    if cfg.workload == "alignment" and cfg.alignment is not None:
        return _emit_schedule_alignment(cfg, out_dir)

    raise ValueError(f"Unsupported workload: {cfg.workload}")


def _emit_schedule_apsp(cfg: CompileConfig, out_dir: str) -> str:
    apsp: APSPConfig = cfg.apsp  # type: ignore[assignment]
    hw = cfg.hw
    disabled = disabled_passes_set(cfg.disabled_passes)

    N = int(apsp.N)
    B = int(apsp.tile_size)
    nb = (N + B - 1) // B

    # Without the mapping pass all tiles run on one PU. Without mode selection
    # APSP is not recognized as a broadcast workload and is confined to the
    # Compute PU pool.
    effective_num_pus = int(hw.num_pus_total)
    if "gendram-mapping" in disabled:
        effective_num_pus = 1
    elif "gendram-mode-select" in disabled and effective_num_pus > 1:
        effective_num_pus = int(hw.num_compute_pus)

    # Without tiling, row-buffer operations are charged per element instead
    # of per tile row.
    rows_per_tile = B * B if "dp-tiling" in disabled else B

    # Without phase formation, synchronization costs four times as much.
    barrier_mult = 4 if "apsp-phase-formation" in disabled else 1

    # Without synthesized broadcasts (or without broadcast mode), tiles move
    # over a path with 1/16 of the broadcast bandwidth.
    generic_movement = "gendram-comm-insert" in disabled or "gendram-mode-select" in disabled
    bcast_bw_scale = 1.0 / 16.0 if generic_movement else 1.0

    schedule: ScheduleJSON = {
        "schedule_kind": "symbolic",
        "mode": "broadcast" if "gendram-mode-select" not in disabled else "generic",
        "workload": "apsp",
        "algo": "blocked_floyd_warshall",
        "tile_size": B,
        "D_shape": [N, N],
        "elem_bytes": int(apsp.elem_bytes),
        "elem_type": apsp.elem_type,
        "mapping": {
            "policy": "modulo" if "gendram-mapping" not in disabled else "single_pu",
            "C": hw.channels,
            "G": hw.bank_groups_per_channel,
            "M": nb,
            "use_compute_pus_only": False,
        },
        "hw": _hw_block(hw),
        "comm": {
            "bcast_bw_scale": bcast_bw_scale,
        },
        "model": {
            "effective_num_pus": effective_num_pus,
            "barrier_mult": barrier_mult,
        },
        "apsp": {
            "N": N,
            "num_blocks": nb,
            "tile_size": B,
            "phases_per_k": ["pivot", "rowcol", "internal"],
            "tile_counts_per_k": {
                "pivot": 1,
                "rowcol": max(0, 2 * (nb - 1)),
                "internal": max(0, (nb - 1) * (nb - 1)),
            },
            "total_tile_updates": nb**3,
            "model": {
                "read_rows_per_tile": rows_per_tile,
                "write_rows_per_tile": rows_per_tile,
            },
        },
        "events": [],
        "compiler": {
            "disabled_passes": sorted(disabled),
        },
    }

    path = os.path.join(out_dir, "schedule.json")
    _write_json(path, schedule)
    return path


def _emit_schedule_alignment(cfg: CompileConfig, out_dir: str) -> str:
    aln: AlignmentConfig = cfg.alignment  # type: ignore[assignment]
    hw = cfg.hw
    disabled = disabled_passes_set(cfg.disabled_passes)

    qlen = int(aln.query_len)
    rlen = int(aln.ref_len)
    B = int(aln.tile_size)

    # The DP table has one extra row and column for the boundary.
    H0 = qlen + 1
    H1 = rlen + 1
    nti = (H0 + B - 1) // B
    ntj = (H1 + B - 1) // B

    # Without the wavefront schedule every tile is followed by a global
    # barrier. Without mode selection the per-diagonal barrier is a full
    # barrier. Without communication insertion each tile pays an explicit
    # memory transfer.
    per_tile_barrier = "wavefront-schedule" in disabled
    wave_barrier_scope = "wavefront_step" if "gendram-mode-select" not in disabled else "global"
    add_mem_transfer = "gendram-comm-insert" in disabled

    schedule: ScheduleJSON = {
        "schedule_kind": "symbolic",
        "mode": "pipeline" if "gendram-mode-select" not in disabled else "generic",
        "workload": "alignment",
        "algo": "smith_waterman",
        "tile_size": B,
        "H_shape": [H0, H1],
        "elem_bytes": int(aln.elem_bytes),
        "elem_type": aln.elem_type,
        "query_len": qlen,
        "ref_len": rlen,
        "match": int(aln.match),
        "mismatch": int(aln.mismatch),
        "gap": int(aln.gap),
        "band_width": int(aln.band_width),
        "mapping": {
            "policy": "modulo" if "gendram-mapping" not in disabled else "single_pu",
            "C": hw.channels,
            "G": hw.bank_groups_per_channel,
            "M": ntj,
            "use_compute_pus_only": True,
            "compute_pu_base": hw.num_search_pus,
        },
        "hw": _hw_block(hw),
        "comm": {
            "bcast_bw_scale": 1.0,
        },
        "model": {
            "per_tile_barrier": per_tile_barrier,
            "wave_barrier_scope": wave_barrier_scope,
            "add_mem_transfer": add_mem_transfer,
            "dp_tiling": "dp-tiling" not in disabled,
        },
        "alignment": {
            "tiles": [nti, ntj],
            "num_tiles": nti * ntj,
            "num_wavefront_steps": nti + ntj - 1,
        },
        "events": [],
        "compiler": {
            "disabled_passes": sorted(disabled),
            "per_tile_barrier": per_tile_barrier,
            "add_mem_transfer": add_mem_transfer,
        },
    }

    path = os.path.join(out_dir, "schedule.json")
    _write_json(path, schedule)
    return path


def emit_pu_programs(cfg: CompileConfig, out_dir: str) -> str:
    """Write one summary file per PU under pu_programs/.

    The files are for inspection only. The simulator does not read them.
    """
    pu_dir = os.path.join(out_dir, "pu_programs")
    _mkdir(pu_dir)
    hw = cfg.hw

    if cfg.workload == "apsp" and cfg.apsp is not None:
        N = int(cfg.apsp.N)
        B = int(cfg.apsp.tile_size)
        nb = (N + B - 1) // B

        for pu in range(hw.num_pus_total):
            with open(os.path.join(pu_dir, f"pu_{pu}.txt"), "w", encoding="utf-8") as f:
                f.write("PU Program (symbolic)\n")
                f.write("====================\n")
                f.write(f"pu_id={pu}\n")
                f.write("\n")
                f.write("workload=APSP (blocked Floyd-Warshall)\n")
                f.write(f"N={N} tile_size={B} num_blocks={nb}\n")
                f.write("\n")
                f.write("mapping: owner = (ti*M + tj) mod (C*G)\n")
                f.write(f"C={hw.channels} G={hw.bank_groups_per_channel} M={nb}\n")

        with open(os.path.join(pu_dir, "README.txt"), "w", encoding="utf-8") as f:
            f.write("Symbolic APSP PU programs.\n")
            f.write("Tiles are not enumerated; schedule.json encodes counts + mapping rule.\n")
        return pu_dir

    if cfg.workload == "alignment" and cfg.alignment is not None:
        H0 = int(cfg.alignment.query_len) + 1
        H1 = int(cfg.alignment.ref_len) + 1
        B = int(cfg.alignment.tile_size)
        nti = (H0 + B - 1) // B
        ntj = (H1 + B - 1) // B
        tiles_per_compute_pu = (nti * ntj) / max(hw.num_compute_pus, 1)

        for pu in range(hw.num_pus_total):
            is_search = pu < hw.num_search_pus
            with open(os.path.join(pu_dir, f"pu_{pu}.txt"), "w", encoding="utf-8") as f:
                f.write("PU Program (symbolic)\n")
                f.write("====================\n")
                f.write(f"pu_id={pu}\n")
                f.write("\n")
                if is_search:
                    f.write("role=SearchPU (idle; seeding is not executed)\n")
                else:
                    f.write("role=ComputePU\n")
                f.write("\n")
                f.write("workload=Alignment (Smith-Waterman, wavefront tiles)\n")
                f.write(f"H_shape={H0}x{H1} tile_size={B} tiles={nti}x{ntj}\n")
                if is_search:
                    f.write("assigned_tiles=0\n")
                else:
                    f.write(
                        f"assigned_tiles_estimate={tiles_per_compute_pu:.2f} (symbolic; not enumerated)\n"
                    )
                f.write("\n")
                f.write("mapping: owner = compute_pu_base + ((ti*M + tj) mod (C*G)) mod num_compute_pus\n")
                f.write(
                    f"C={hw.channels} G={hw.bank_groups_per_channel} M={ntj} "
                    f"compute_pu_base={hw.num_search_pus} num_compute_pus={hw.num_compute_pus}\n"
                )

        with open(os.path.join(pu_dir, "README.txt"), "w", encoding="utf-8") as f:
            f.write("Symbolic alignment PU program summaries.\n")
            f.write("schedule.json does NOT enumerate tiles; it encodes the mapping rule and\n")
            f.write("the simulator generates the wavefront schedule internally.\n")
        return pu_dir

    raise ValueError("emit_pu_programs called with unsupported config")
