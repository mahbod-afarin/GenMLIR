"""GenDRAM command-stream simulator.

Consumes the artifacts emitted by GenMLIR (layout.json and schedule.json) and
reports modeled cycles and time for the schedule. If workload inputs are
present in the artifact directory, it also executes the workload so that the
result can be checked against a CPU reference:

    APSP        input_D.npy                        -> output_D.npy
    alignment   input_query.txt + input_ref.txt    -> output_H.npy, output_score.txt
                input_queries.txt + input_ref.txt  -> output_H.npy, output_scores.json

Two schedule forms are accepted. An explicit schedule lists every tile and
communication event (emitted by genmlir-opt). A symbolic schedule carries
shapes, tiling and the mapping rule only (emitted by the Python compiler),
and the simulator derives the tile order from those parameters.

Timing of symbolic schedules is available in two modes:

    analytical   closed-form phase and wavefront model
    execute      discrete-event execution of the tile stream, with data
                 dependencies and PU contention

The cost of one tile is a simple model: row-buffer operations (activate,
read or write, precharge) scaled by the tier latency, plus lane-parallel DP
updates. This is not a cycle-accurate DRAM model.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    np = None

try:
    from .config import CfgArchitecture, CfgGenDRAMSim
except ImportError:  # executed as a script
    from config import CfgArchitecture, CfgGenDRAMSim

# M3D DRAM timing parameters from the GenDRAM paper (ns). Tier 0 is fastest.
_T_RCD_NS = [2.29, 3.92, 5.99, 8.50, 11.44, 14.82, 18.63, 22.88]
_T_RP_NS = 4.77
_T_RAS_OFFSET_NS = 27.5
_SLOWEST_TIER = 7

# (total cycles, cycles per PU, cycles per event kind)
Timing = Tuple[int, Dict[int, int], Dict[str, int]]


@dataclass(frozen=True)
class SimResult:
    total_cycles: int
    total_time_s: float
    per_pu_cycles: Dict[int, int]
    event_breakdown_cycles: Dict[str, int]
    output_paths: Dict[str, str]  # files written by functional execution


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _section(d: Dict[str, Any], key: str) -> Dict[str, Any]:
    """Return d[key] if it is a dict, otherwise an empty dict."""
    value = d.get(key)
    return value if isinstance(value, dict) else {}


def load_artifacts(out_dir: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load (layout, schedule) from a GenMLIR output directory."""
    layout_path = os.path.join(out_dir, "layout.json")
    schedule_path = os.path.join(out_dir, "schedule.json")
    if not os.path.isfile(layout_path):
        raise FileNotFoundError(f"Missing layout.json at: {layout_path}")
    if not os.path.isfile(schedule_path):
        raise FileNotFoundError(f"Missing schedule.json at: {schedule_path}")
    return _load_json(layout_path), _load_json(schedule_path)


def _default_tile_payload_bytes(schedule: Dict[str, Any]) -> int:
    tile_size = int(schedule.get("tile_size", 64))
    elem_bytes = int(schedule.get("elem_bytes", CfgGenDRAMSim.DEFAULT_ELEM_BYTES))
    return tile_size * tile_size * elem_bytes


def _cycle_time_s(schedule: Dict[str, Any]) -> float:
    freq = float(_section(schedule, "hw").get("frequency_hz", CfgArchitecture.LOGIC_CLK_FREQUENCY_HZ))
    if freq <= 0:
        freq = CfgArchitecture.LOGIC_CLK_FREQUENCY_HZ
    return 1.0 / freq


def _tier_latency_scale(tier: int, spread: Optional[float] = None) -> float:
    """DRAM latency of a tier relative to the slowest tier (tier 7 is 1.0).

    The latency is the row-cycle time tRC = tRP + tRAS, with
    tRAS = tRCD + 27.5 ns. If spread is given, the tRCD ramp is replaced by a
    linear ramp from the tier-0 value up to spread times that value. The
    nominal ramp has a spread of about 9.99.
    """
    if spread is not None and float(spread) > 0:
        t_min = _T_RCD_NS[0]
        top = t_min * float(spread)
        t_rcd = [t_min + (top - t_min) * (i / 7.0) for i in range(8)]
    else:
        t_rcd = _T_RCD_NS

    def t_rc(rcd: float) -> float:
        return _T_RP_NS + (rcd + _T_RAS_OFFSET_NS)

    t = max(0, min(_SLOWEST_TIER, int(tier)))
    return t_rc(t_rcd[t]) / t_rc(t_rcd[_SLOWEST_TIER])


def _tile_compute_cycles(*, dp_updates: int, read_rows: int, write_rows: int, lanes: int,
                         mem_scale: float = 1.0) -> int:
    """Cycles to process one tile: row-buffer traffic plus DP updates.

    Each row read or write costs activate + access + precharge, scaled by the
    tier latency factor mem_scale. DP updates run on `lanes` parallel lanes.
    """
    cfg = CfgGenDRAMSim
    read_cost = cfg.DRAM_ACTIVATE_CYCLES + cfg.DRAM_READ_ROWBUF_CYCLES + cfg.DRAM_PRECHARGE_CYCLES
    write_cost = cfg.DRAM_ACTIVATE_CYCLES + cfg.DRAM_WRITE_ROWBUF_CYCLES + cfg.DRAM_PRECHARGE_CYCLES
    if mem_scale <= 0:
        mem_scale = 1.0
    dram_read = math.ceil(int(read_rows) * read_cost * mem_scale)
    dram_write = math.ceil(int(write_rows) * write_cost * mem_scale)

    lanes = max(int(lanes), 1)
    compute = math.ceil(dp_updates / lanes) * max(cfg.CYCLES_PER_DP_UPDATE, 1)
    return dram_read + dram_write + compute


def _wavefront_barrier_cycles(scope: str) -> int:
    if scope == "wavefront_step":
        return CfgGenDRAMSim.WAVEFRONT_BARRIER_CYCLES
    return CfgGenDRAMSim.BARRIER_CYCLES


# ---------------------------------------------------------------------------
# Functional execution: APSP (blocked Floyd-Warshall)
# ---------------------------------------------------------------------------

def _tile_slices(D: "np.ndarray", tile_size: int, ti: int, tj: int) -> Tuple[slice, slice]:
    i0 = ti * tile_size
    j0 = tj * tile_size
    return (
        slice(i0, min(i0 + tile_size, D.shape[0])),
        slice(j0, min(j0 + tile_size, D.shape[1])),
    )


def _block_slice(D: "np.ndarray", tile_size: int, tk: int) -> slice:
    k0 = tk * tile_size
    return slice(k0, min(k0 + tile_size, D.shape[0]))


def _apsp_pivot_update(D: "np.ndarray", tile_size: int, k: int) -> None:
    """Floyd-Warshall inside the pivot tile (k, k)."""
    ks = _block_slice(D, tile_size, k)
    block = D[ks, ks]
    for kk in range(block.shape[0]):
        block = np.minimum(block, block[:, kk][:, None] + block[kk, :][None, :])
    D[ks, ks] = block


def _apsp_row_update(D: "np.ndarray", tile_size: int, k: int, j: int) -> None:
    """Update row tile (k, j) through the pivot tile."""
    ks = _block_slice(D, tile_size, k)
    cs = _tile_slices(D, tile_size, k, j)[1]
    pivot = D[ks, ks]
    tile = D[ks, cs]
    for kk in range(pivot.shape[0]):
        tile = np.minimum(tile, pivot[:, kk][:, None] + tile[kk, :][None, :])
    D[ks, cs] = tile


def _apsp_col_update(D: "np.ndarray", tile_size: int, k: int, i: int) -> None:
    """Update column tile (i, k) through the pivot tile."""
    ks = _block_slice(D, tile_size, k)
    rs = _tile_slices(D, tile_size, i, k)[0]
    pivot = D[ks, ks]
    tile = D[rs, ks]
    for kk in range(pivot.shape[0]):
        tile = np.minimum(tile, tile[:, kk][:, None] + pivot[kk, :][None, :])
    D[rs, ks] = tile


def _apsp_internal_update(D: "np.ndarray", tile_size: int, k: int, i: int, j: int) -> None:
    """Update tile (i, j) through column tile (i, k) and row tile (k, j)."""
    ks = _block_slice(D, tile_size, k)
    rs, cs = _tile_slices(D, tile_size, i, j)
    col = D[rs, ks]
    row = D[ks, cs]
    tile = D[rs, cs]
    for kk in range(col.shape[1]):
        tile = np.minimum(tile, col[:, kk][:, None] + row[kk, :][None, :])
    D[rs, cs] = tile


def _apsp_tile_order(nb: int) -> Iterator[Tuple[str, int, int, int]]:
    """Blocked Floyd-Warshall tile order as (phase, tk, ti, tj).

    For every k: the pivot tile, then the row and column tiles of block k,
    then all remaining tiles.
    """
    for k in range(nb):
        yield "pivot", k, k, k
        for j in range(nb):
            if j != k:
                yield "rowcol", k, k, j
        for i in range(nb):
            if i != k:
                yield "rowcol", k, i, k
        for i in range(nb):
            if i == k:
                continue
            for j in range(nb):
                if j != k:
                    yield "internal", k, i, j


def _run_functional_apsp(out_dir: str, schedule: Dict[str, Any], *, verbose: bool) -> Dict[str, str]:
    if np is None:
        raise RuntimeError("NumPy is required for functional APSP simulation (pip install numpy).")

    in_path = os.path.join(out_dir, "input_D.npy")
    if not os.path.isfile(in_path):
        if verbose:
            print(f"[gendram_sim] functional APSP skipped (missing {in_path})")
        return {}

    tile_size = int(schedule.get("tile_size", 64))
    D = np.load(in_path)
    if D.dtype != np.int64:
        D = D.astype(np.int64)

    # An explicit schedule lists the tile updates. A symbolic schedule has no
    # events, and the order is generated here.
    events = schedule.get("events", [])
    broadcasts = set()
    if events:
        tile_ops: List[Tuple[str, int, int, int]] = []
        for ev in events:
            op = str(ev.get("op", ""))
            ti = int(ev.get("ti", -1))
            tj = int(ev.get("tj", -1))
            tk = int(ev.get("tk", -1))
            if ti < 0 or tj < 0 or tk < 0:
                continue
            if op == "broadcast" and str(ev.get("object", "")) == "D":
                broadcasts.add((str(ev.get("phase_kind", "")), tk, ti, tj))
            elif op == "compute_tile":
                tile_ops.append((str(ev.get("phase", "")), tk, ti, tj))
    else:
        nb = (D.shape[0] + tile_size - 1) // tile_size
        tile_ops = _apsp_tile_order(nb)

    for phase, tk, ti, tj in tile_ops:
        if phase == "pivot":
            _apsp_pivot_update(D, tile_size, tk)
        elif phase == "rowcol":
            if ti == tk and tj != tk:
                _apsp_row_update(D, tile_size, tk, tj)
            elif tj == tk and ti != tk:
                _apsp_col_update(D, tile_size, tk, ti)
        elif phase == "internal":
            if ti != tk and tj != tk:
                _apsp_internal_update(D, tile_size, tk, ti, tj)

    out_path = os.path.join(out_dir, "output_D.npy")
    np.save(out_path, D)

    if verbose:
        checksum = int(np.sum(D) % 1_000_000_007)
        print(f"[gendram_sim] APSP output written: {out_path}")
        print(f"[gendram_sim] APSP output checksum (sum mod 1e9+7): {checksum}")
        print(f"[gendram_sim] broadcast events: {len(broadcasts)}")

    return {"D": out_path}


# ---------------------------------------------------------------------------
# Functional execution: alignment (Smith-Waterman)
# ---------------------------------------------------------------------------

def _alignment_tiles(schedule: Dict[str, Any], qlen: int, rlen: int) -> List[Tuple[int, ...]]:
    """Tiles of the DP table in wavefront order.

    Each tile is (step, ti, tj, i0, j0, rows, cols). Explicit schedules list
    the tiles as compute_tile events; for symbolic schedules the diagonal
    traversal is generated from the table shape and tile size.
    """
    B = int(schedule.get("tile_size", 64))
    tiles: List[Tuple[int, ...]] = []

    events = [
        ev
        for ev in schedule.get("events", [])
        if str(ev.get("op", "")) == "compute_tile" and str(ev.get("fn", "")) == "alignment"
    ]
    if events:
        for ev in events:
            ti = int(ev.get("ti", 0))
            tj = int(ev.get("tj", 0))
            step = int(ev.get("wavefront_step", ti + tj))
            view = _section(ev, "tile")
            tiles.append(
                (
                    step,
                    ti,
                    tj,
                    int(view.get("i0", -1)),
                    int(view.get("j0", -1)),
                    int(view.get("rows", 0)),
                    int(view.get("cols", 0)),
                )
            )
    else:
        hshape = schedule.get("H_shape")
        if not isinstance(hshape, list) or len(hshape) != 2:
            hshape = [qlen + 1, rlen + 1]
        H0, H1 = int(hshape[0]), int(hshape[1])
        nti = (H0 + B - 1) // B
        ntj = (H1 + B - 1) // B
        for step in range(nti + ntj - 1):
            for ti in range(nti):
                tj = step - ti
                if tj < 0 or tj >= ntj:
                    continue
                i0 = ti * B
                j0 = tj * B
                tiles.append((step, ti, tj, i0, j0, max(0, min(B, H0 - i0)), max(0, min(B, H1 - j0))))

    # Events without a tile view fall back to the view implied by (ti, tj).
    for n, (step, ti, tj, i0, j0, rows, cols) in enumerate(tiles):
        if i0 < 0 or j0 < 0 or rows <= 0 or cols <= 0:
            i0 = ti * B
            j0 = tj * B
            tiles[n] = (step, ti, tj, i0, j0, min(B, (qlen + 1) - i0), min(B, (rlen + 1) - j0))

    tiles.sort(key=lambda t: t[:3])
    return tiles


def _run_functional_alignment(out_dir: str, schedule: Dict[str, Any], *, verbose: bool) -> Dict[str, str]:
    """Run Smith-Waterman tile by tile in the order given by the schedule.

    Seeding is not executed. With a positive band width only cells within
    band_width of the main diagonal are computed. Writes the DP table of the
    first query and the best score of every query.
    """
    if np is None:
        raise RuntimeError("NumPy is required for functional alignment simulation (pip install numpy).")

    q_path = os.path.join(out_dir, "input_query.txt")
    q_multi_path = os.path.join(out_dir, "input_queries.txt")
    r_path = os.path.join(out_dir, "input_ref.txt")
    if (not os.path.isfile(q_path) and not os.path.isfile(q_multi_path)) or not os.path.isfile(r_path):
        if verbose:
            print(
                f"[gendram_sim] functional alignment skipped (missing {q_path} or {q_multi_path} or {r_path})"
            )
        return {}

    ref_full = _read_text(r_path).strip().upper()
    if os.path.isfile(q_multi_path):
        queries = [ln.strip().upper() for ln in _read_text(q_multi_path).splitlines() if ln.strip()]
    else:
        q = _read_text(q_path).strip().upper()
        queries = [q] if q else []
    if not queries:
        if verbose:
            print("[gendram_sim] functional alignment skipped (no queries)")
        return {}

    # The table is allocated at the compiled size. Queries and the reference
    # are truncated to it, and shorter queries use part of the table.
    qlen_decl = int(schedule.get("query_len", max(len(q) for q in queries)))
    rlen_decl = int(schedule.get("ref_len", len(ref_full)))
    ref = ref_full[:rlen_decl]
    rlen = len(ref)

    algo = str(schedule.get("algo", "smith_waterman"))
    match = int(schedule.get("match", 2))
    mismatch = int(schedule.get("mismatch", -1))
    gap = int(schedule.get("gap", -1))
    band = int(schedule.get("band_width", 0))
    if algo not in ("smith_waterman", "sw"):
        raise ValueError(f"Unsupported alignment algo: {algo} (expected smith_waterman)")

    H = np.zeros((qlen_decl + 1, rlen_decl + 1), dtype=np.int32)
    tiles = _alignment_tiles(schedule, qlen_decl, rlen_decl)

    scores: List[Dict[str, Any]] = []
    saved_H_path: Optional[str] = None
    saved_H_checksum = 0
    # In batch runs the first table is saved unless GENDRAM_SAVE_FIRST_H=0.
    save_first = len(queries) == 1 or os.environ.get("GENDRAM_SAVE_FIRST_H", "1") == "1"

    for qi, query_raw in enumerate(queries):
        query = query_raw[:qlen_decl]
        qlen = len(query)

        H.fill(0)
        max_score = 0
        max_pos = (0, 0)

        for _step, _ti, _tj, i0, j0, rows, cols in tiles:
            if i0 >= qlen + 1 or j0 >= rlen + 1:
                continue
            i1 = min(i0 + rows, qlen + 1)
            j1 = min(j0 + cols, rlen + 1)

            for i in range(max(1, i0), i1):
                ai = query[i - 1]
                j_lo = max(1, j0)
                j_hi = j1 - 1
                if band > 0:
                    j_lo = max(j_lo, i - band)
                    j_hi = min(j_hi, i + band)

                for j in range(j_lo, j_hi + 1):
                    sub = match if ai == ref[j - 1] else mismatch
                    val = max(
                        0,
                        int(H[i - 1, j - 1]) + sub,
                        int(H[i - 1, j]) + gap,
                        int(H[i, j - 1]) + gap,
                    )
                    H[i, j] = val
                    if val > max_score:
                        max_score = val
                        max_pos = (i, j)

        scores.append(
            {
                "read_index": qi,
                "query_len": qlen,
                "ref_len": rlen,
                "max_score": int(max_score),
                "max_pos": [int(max_pos[0]), int(max_pos[1])],
            }
        )

        if qi == 0 and save_first:
            saved_H_path = os.path.join(out_dir, "output_H.npy")
            np.save(saved_H_path, H)
            saved_H_checksum = int(np.sum(H) % 1_000_000_007)

    output_paths: Dict[str, str] = {}
    if saved_H_path:
        output_paths["H"] = saved_H_path

    if len(queries) == 1:
        s0 = scores[0]
        out_score = os.path.join(out_dir, "output_score.txt")
        with open(out_score, "w", encoding="utf-8") as f:
            f.write(f"algo={algo}\n")
            f.write(f"query_len={s0['query_len']}\n")
            f.write(f"ref_len={s0['ref_len']}\n")
            f.write(f"match={match}\n")
            f.write(f"mismatch={mismatch}\n")
            f.write(f"gap={gap}\n")
            f.write(f"band_width={band}\n")
            f.write(f"max_score={s0['max_score']}\n")
            f.write(f"max_pos={s0['max_pos'][0]},{s0['max_pos'][1]}\n")
        output_paths["score"] = out_score
    else:
        out_scores = os.path.join(out_dir, "output_scores.json")
        with open(out_scores, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "algo": algo,
                    "match": match,
                    "mismatch": mismatch,
                    "gap": gap,
                    "band_width": band,
                    "compiled_query_len": qlen_decl,
                    "compiled_ref_len": rlen_decl,
                    "scores": scores,
                },
                f,
                indent=2,
            )
            f.write("\n")
        output_paths["scores"] = out_scores

    if verbose:
        best = max(scores, key=lambda s: s["max_score"])
        print(f"[gendram_sim] alignment reads processed: {len(queries)}")
        if saved_H_path:
            print(f"[gendram_sim] alignment output written: {saved_H_path}")
            print(f"[gendram_sim] alignment checksum (sum mod 1e9+7): {saved_H_checksum}")
        print(f"[gendram_sim] best max_score: {best['max_score']} @ {best['max_pos']} (read_index={best['read_index']})")

    return output_paths


# ---------------------------------------------------------------------------
# Timing: APSP
# ---------------------------------------------------------------------------

def _apsp_shape(schedule: Dict[str, Any]) -> Tuple[int, int]:
    """Return (tile size B, number of blocks per dimension)."""
    apsp = _section(schedule, "apsp")
    dshape = schedule.get("D_shape")
    if not isinstance(dshape, list):
        dshape = [0, 0]
    N = int(apsp.get("N", dshape[0] if dshape else 0))
    B = int(apsp.get("tile_size", schedule.get("tile_size", 64)))
    nb = int(apsp.get("num_blocks", (N + B - 1) // B if N > 0 else 0))
    return B, nb


def _apsp_tile_cycles(schedule: Dict[str, Any], B: int, mem_scale: float) -> int:
    """Cycles for one full B x B tile update (B^3 DP updates)."""
    rows = _section(_section(schedule, "apsp"), "model")
    lanes = int(_section(schedule, "hw").get("pe_lanes_per_pu", CfgGenDRAMSim.PE_LANES_PER_PU))
    return _tile_compute_cycles(
        dp_updates=B * B * B,
        read_rows=int(rows.get("read_rows_per_tile", B)),
        write_rows=int(rows.get("write_rows_per_tile", B)),
        lanes=lanes,
        mem_scale=mem_scale,
    )


def _simulate_symbolic_apsp(
    schedule: Dict[str, Any],
    *,
    comm_cycles_for_bytes: Callable[[int], int],
    mem_scale: float = 1.0,
) -> Timing:
    """Closed-form timing of blocked Floyd-Warshall from a symbolic schedule.

    For every k the pivot tile runs alone, then the row/column tiles and the
    internal tiles are each spread evenly over the active PUs. Tile
    broadcasts and one barrier per phase are added in sequence.
    """
    apsp = _section(schedule, "apsp")
    tile_counts = _section(apsp, "tile_counts_per_k")
    hw = _section(schedule, "hw")
    model = _section(schedule, "model")
    B, nb = _apsp_shape(schedule)

    num_pus = int(hw.get("num_pus_total", 32))
    active_pus = max(int(model.get("effective_num_pus", num_pus)), 1)
    barrier_mult = max(int(model.get("barrier_mult", 1)), 1)

    tile_cycles = _apsp_tile_cycles(schedule, B, mem_scale)
    bcast_cycles = int(comm_cycles_for_bytes(int(schedule.get("elem_bytes", 4)) * B * B))
    barrier_cycles = CfgGenDRAMSim.BARRIER_CYCLES * barrier_mult

    pivot_tiles = int(tile_counts.get("pivot", 1))
    rowcol_tiles = int(tile_counts.get("rowcol", max(0, 2 * (nb - 1))))
    internal_tiles = int(tile_counts.get("internal", max(0, (nb - 1) * (nb - 1))))

    def parallel_phase(tile_count: int) -> int:
        if tile_count <= 0:
            return 0
        return math.ceil(tile_count / active_pus) * tile_cycles

    breakdown: Dict[str, int] = {}
    if nb > 0:
        per_k = {
            "pivot_compute": pivot_tiles * tile_cycles,
            "pivot_broadcast": bcast_cycles,
            "rowcol_compute": parallel_phase(rowcol_tiles),
            "rowcol_broadcast": rowcol_tiles * bcast_cycles,
            "internal_compute": parallel_phase(internal_tiles),
            "barrier": 3 * barrier_cycles,
        }
        breakdown = {key: nb * cycles for key, cycles in per_k.items()}
    total_cycles = sum(breakdown.values())

    breakdown["compute_tile"] = (
        breakdown.get("pivot_compute", 0)
        + breakdown.get("rowcol_compute", 0)
        + breakdown.get("internal_compute", 0)
    )
    breakdown["broadcast"] = breakdown.get("pivot_broadcast", 0) + breakdown.get("rowcol_broadcast", 0)

    # Compute load per PU, tiles divided evenly. Broadcasts and barriers are
    # not attributed to a PU.
    total_tile_updates = int(apsp.get("total_tile_updates", nb**3))
    tiles_base, tiles_rem = divmod(total_tile_updates, active_pus)
    per_pu_cycles: Dict[int, int] = {}
    for pu in range(num_pus):
        tiles = tiles_base + (1 if pu < tiles_rem else 0) if pu < active_pus else 0
        per_pu_cycles[pu] = tiles * tile_cycles

    return total_cycles, per_pu_cycles, breakdown


def _execute_apsp(
    schedule: Dict[str, Any],
    *,
    comm_cycles_for_bytes: Callable[[int], int],
    mem_scale: float = 1.0,
) -> Timing:
    """Discrete-event execution of the blocked Floyd-Warshall schedule.

    The tile stream (pivot, row/column, internal, for every k) is generated
    on the fly. Each tile update starts once its inputs are available, that
    is the previous version of the tile and the pivot, row and column tiles
    it reads, and once the PU that owns it under the modulo mapping is free.
    A broadcast adds its latency on the edge from the producing tile to its
    consumers. The result is the critical path under PU contention.
    """
    hw = _section(schedule, "hw")
    model = _section(schedule, "model")
    B, nb = _apsp_shape(schedule)

    num_pus = int(hw.get("num_pus_total", 32))
    P = max(int(model.get("effective_num_pus", num_pus)), 1)
    barrier_mult = max(1, int(model.get("barrier_mult", 1)))

    tile_cost = _apsp_tile_cycles(schedule, B, mem_scale)
    bcast = int(comm_cycles_for_bytes(int(schedule.get("elem_bytes", 4)) * B * B))
    barrier_cyc = CfgGenDRAMSim.BARRIER_CYCLES * barrier_mult

    if nb <= 0:
        return 0, {pu: 0 for pu in range(num_pus)}, {"compute_tile": 0}

    ready = [0] * (nb * nb)   # finish time of the last write to tile (i, j)
    pu_free = [0] * num_pus
    per_pu = [0] * num_pus
    compute_cycles = 0
    bcast_cycles_total = 0
    barrier_cycles_total = 0
    gsync = 0                 # time of the last global barrier

    def book(i: int, j: int, dep: int) -> int:
        nonlocal compute_cycles
        idx = i * nb + j
        pu = idx % P
        start = max(dep, pu_free[pu], ready[idx], gsync)
        fin = start + tile_cost
        pu_free[pu] = fin
        per_pu[pu] += tile_cost
        ready[idx] = fin
        compute_cycles += tile_cost
        return fin

    for k in range(nb):
        pivot_ready = book(k, k, gsync) + bcast
        bcast_cycles_total += bcast
        base_k = k * nb
        for j in range(nb):
            if j != k:
                book(k, j, pivot_ready)
        for i in range(nb):
            if i != k:
                book(i, k, pivot_ready)

        # Internal tiles wait for their row tile (k, j) and column tile
        # (i, k). This loop runs nb^3 times in total, so book() is inlined.
        for i in range(nb):
            if i == k:
                continue
            irow = i * nb
            col_ready = ready[irow + k] + bcast
            for j in range(nb):
                if j == k:
                    continue
                row_ready = ready[base_k + j] + bcast
                start = col_ready if col_ready > row_ready else row_ready
                idx = irow + j
                pu = idx % P
                pf = pu_free[pu]
                if pf > start:
                    start = pf
                rv = ready[idx]
                if rv > start:
                    start = rv
                if gsync > start:
                    start = gsync
                fin = start + tile_cost
                pu_free[pu] = fin
                per_pu[pu] += tile_cost
                ready[idx] = fin
                compute_cycles += tile_cost
        bcast_cycles_total += (2 * max(0, nb - 1)) * bcast

        # Barrier between consecutive k iterations.
        gsync = max(pu_free, default=0) + barrier_cyc
        barrier_cycles_total += barrier_cyc

    total = max(gsync, max(pu_free, default=0))
    breakdown = {
        "compute_tile": compute_cycles,
        "broadcast": bcast_cycles_total,
        "barrier": barrier_cycles_total,
    }
    return total, {pu: per_pu[pu] for pu in range(num_pus)}, breakdown


# ---------------------------------------------------------------------------
# Timing: alignment
# ---------------------------------------------------------------------------

class _AlignmentGrid:
    """Tile grid of the alignment DP table described by a schedule."""

    def __init__(self, schedule: Dict[str, Any]) -> None:
        self.B = int(schedule.get("tile_size", 64))
        hshape = schedule.get("H_shape")
        if not isinstance(hshape, list) or len(hshape) != 2:
            hshape = [0, 0]
        self.H0 = int(hshape[0])
        self.H1 = int(hshape[1])
        self.band = int(schedule.get("band_width", 0))
        self.nti = (self.H0 + self.B - 1) // self.B if self.H0 > 0 else 0
        self.ntj = (self.H1 + self.B - 1) // self.B if self.H1 > 0 else 0
        self.slope = self.H1 / self.H0 if self.H0 > 0 else 1.0

    def in_band(self, ti: int, tj: int) -> bool:
        """True if tile (ti, tj) lies within band cells of the main diagonal."""
        if self.band <= 0:
            return True
        B = self.B
        diag_lo = ti * B * self.slope - self.band
        diag_hi = (ti * B + B) * self.slope + self.band
        return not (tj * B + B < diag_lo or tj * B > diag_hi)

    def diagonal(self, step: int) -> Iterator[Tuple[int, int]]:
        """In-band tiles (ti, tj) with ti + tj == step, by increasing ti."""
        for ti in range(max(0, step - self.ntj + 1), min(self.nti, step + 1)):
            tj = step - ti
            if self.in_band(ti, tj):
                yield ti, tj

    def tile_cycles(self, ti: int, tj: int, *, lanes: int, mem_scale: float, dp_tiling: bool) -> int:
        B = self.B
        i0 = ti * B
        j0 = tj * B
        rows = max(0, min(B, self.H0 - i0))
        cols = max(0, min(B, self.H1 - j0))

        # The first row and column of the table are the boundary.
        dp_rows = max(0, rows - (1 if i0 == 0 else 0))
        dp_cols = max(0, cols - (1 if j0 == 0 else 0))
        if self.band > 0:
            dp_cols = min(dp_cols, 2 * self.band + 1)

        # Without tiling, row-buffer operations are charged per cell.
        row_ops = rows if dp_tiling else rows * cols
        return _tile_compute_cycles(
            dp_updates=dp_rows * dp_cols,
            read_rows=row_ops,
            write_rows=row_ops,
            lanes=lanes,
            mem_scale=mem_scale,
        )


def _simulate_symbolic_alignment(schedule: Dict[str, Any], *, mem_scale: float = 1.0) -> Timing:
    """Closed-form timing of tiled alignment from a symbolic schedule.

    Tiles are visited diagonal by diagonal. With the wavefront schedule and
    the PU mapping enabled, the tiles of a diagonal are balanced over the
    Compute PUs (largest tile first onto the least loaded PU) and the
    diagonal takes as long as its most loaded PU. Otherwise tiles run one
    after another. The model has no ring-bandwidth term.
    """
    hw = _section(schedule, "hw")
    num_pus = int(hw.get("num_pus_total", CfgArchitecture.NUM_PUS_TOTAL))
    num_search = int(hw.get("num_search_pus", CfgArchitecture.NUM_SEARCH_PUS))
    num_compute = int(hw.get("num_compute_pus", CfgArchitecture.NUM_COMPUTE_PUS))
    lanes = int(hw.get("pe_lanes_per_pu", CfgGenDRAMSim.PE_LANES_PER_PU))

    grid = _AlignmentGrid(schedule)
    nti, ntj = grid.nti, grid.ntj
    num_tiles = nti * ntj
    num_steps = nti + ntj - 1 if nti > 0 and ntj > 0 else 0

    model = _section(schedule, "model")
    per_tile_barrier = bool(model.get("per_tile_barrier", False))
    wave_scope = str(model.get("wave_barrier_scope", "wavefront_step"))
    add_mem_transfer = bool(model.get("add_mem_transfer", False))
    dp_tiling = bool(model.get("dp_tiling", True))

    if per_tile_barrier:
        barrier_total = num_tiles * CfgGenDRAMSim.BARRIER_CYCLES
    else:
        barrier_total = num_steps * _wavefront_barrier_cycles(wave_scope)
    mem_transfer_total = 2 * num_tiles * CfgGenDRAMSim.MEM_TRANSFER_CYCLES if add_mem_transfer else 0

    mapping = _section(schedule, "mapping")
    mapped = str(mapping.get("policy", "modulo")) == "modulo"
    C = int(mapping.get("C", hw.get("channels", 4)))
    G = int(mapping.get("G", hw.get("bank_groups_per_channel", 8)))
    M = int(mapping.get("M", ntj if ntj > 0 else 1))
    use_compute_only = bool(mapping.get("use_compute_pus_only", True))
    compute_pu_base = int(mapping.get("compute_pu_base", num_search))

    def owner_pu(ti: int, tj: int) -> int:
        logical = (ti * M + tj) % max(C * G, 1)
        if not use_compute_only:
            return logical
        return compute_pu_base + logical % max(num_compute, 1)

    def tile_cycles(ti: int, tj: int) -> int:
        return grid.tile_cycles(ti, tj, lanes=lanes, mem_scale=mem_scale, dp_tiling=dp_tiling)

    per_pu_cycles: Dict[int, int] = {pu: 0 for pu in range(num_pus)}
    compute_work = 0

    if not mapped or per_tile_barrier:
        # Serial execution: no mapping, or a global barrier after every tile.
        for step in range(num_steps):
            for ti, tj in grid.diagonal(step):
                c = tile_cycles(ti, tj)
                compute_work += c
                if mapped:
                    pu = owner_pu(ti, tj)
                    if 0 <= pu < num_pus:
                        per_pu_cycles[pu] += c
        wall = compute_work
    else:
        active_pus = max(num_compute, 1)
        wall = 0
        for step in range(num_steps):
            costs = sorted(((tile_cycles(ti, tj), ti) for ti, tj in grid.diagonal(step)), reverse=True)
            if not costs:
                continue
            compute_work += sum(c for c, _ in costs)
            loads = [0] * active_pus
            for c, _ in costs:
                loads[min(range(active_pus), key=loads.__getitem__)] += c
            for local, load in enumerate(loads):
                pu = num_search + local
                if load and 0 <= pu < num_pus:
                    per_pu_cycles[pu] += load
            wall += max(loads)

    breakdown = {"compute_tile": compute_work, "barrier": barrier_total}
    if mem_transfer_total:
        breakdown["mem_transfer"] = mem_transfer_total
    return wall + barrier_total + mem_transfer_total, per_pu_cycles, breakdown


def _execute_alignment(schedule: Dict[str, Any], *, mem_scale: float = 1.0) -> Timing:
    """Discrete-event execution of the tiled alignment schedule.

    Each in-band tile starts once its upper, left and diagonal neighbours are
    done. With the wavefront schedule and the PU mapping enabled, the ready
    tiles of a diagonal are placed, earliest first, on the Compute PU that
    becomes free first. This stands for a conflict-free mapping and is used
    instead of the static modulo owner. When a diagonal has more tiles than
    Compute PUs, the extra tiles queue. Without the wavefront schedule a
    global barrier follows every tile, which serializes execution. The model
    has no ring-bandwidth term.
    """
    hw = _section(schedule, "hw")
    num_pus = int(hw.get("num_pus_total", 32))
    num_search = int(hw.get("num_search_pus", 8))
    num_compute = int(hw.get("num_compute_pus", 24))
    lanes = int(hw.get("pe_lanes_per_pu", CfgGenDRAMSim.PE_LANES_PER_PU))

    grid = _AlignmentGrid(schedule)
    nti, ntj = grid.nti, grid.ntj

    model = _section(schedule, "model")
    per_tile_barrier = bool(model.get("per_tile_barrier", False))
    wave_scope = str(model.get("wave_barrier_scope", "wavefront_step"))
    dp_tiling = bool(model.get("dp_tiling", True))
    add_mem_transfer = bool(model.get("add_mem_transfer", False))

    mapping = _section(schedule, "mapping")
    mapped = str(mapping.get("policy", "modulo")) == "modulo"
    use_compute_only = bool(mapping.get("use_compute_pus_only", True))
    compute_pu_base = int(mapping.get("compute_pu_base", num_search))

    if per_tile_barrier:
        barrier_cyc = CfgGenDRAMSim.BARRIER_CYCLES
    else:
        barrier_cyc = _wavefront_barrier_cycles(wave_scope)
    memxfer = CfgGenDRAMSim.MEM_TRANSFER_CYCLES if add_mem_transfer else 0

    def tile_cost(ti: int, tj: int) -> int:
        return grid.tile_cycles(ti, tj, lanes=lanes, mem_scale=mem_scale, dp_tiling=dp_tiling) + memxfer

    if nti <= 0 or ntj <= 0:
        return 0, {pu: 0 for pu in range(num_pus)}, {"compute_tile": 0}

    ready = [0] * (nti * ntj)   # finish time of tile (ti, tj)
    pu_free = [0] * num_pus
    per_pu = [0] * num_pus
    compute_cycles = 0
    barrier_cycles_total = 0
    gsync = 0                   # time of the last global barrier
    pool_base = compute_pu_base if use_compute_only else 0
    pool = range(pool_base, pool_base + max(1, num_compute if use_compute_only else num_pus))

    for step in range(nti + ntj - 1):
        diag = []
        for ti, tj in grid.diagonal(step):
            dep = gsync
            if ti > 0:
                dep = max(dep, ready[(ti - 1) * ntj + tj])
            if tj > 0:
                dep = max(dep, ready[ti * ntj + (tj - 1)])
            if ti > 0 and tj > 0:
                dep = max(dep, ready[(ti - 1) * ntj + (tj - 1)])
            diag.append((dep, ti, tj))
        if not diag:
            continue

        if not mapped or per_tile_barrier:
            for dep, ti, tj in diag:
                c = tile_cost(ti, tj)
                fin = max(dep, gsync) + c
                ready[ti * ntj + tj] = fin
                compute_cycles += c
                if per_tile_barrier:
                    gsync = max(fin, max(pu_free, default=0)) + barrier_cyc
                    barrier_cycles_total += barrier_cyc
                else:
                    gsync = fin
        else:
            diag.sort()
            for dep, ti, tj in diag:
                c = tile_cost(ti, tj)
                pu = min(pool, key=pu_free.__getitem__)
                fin = max(dep, pu_free[pu]) + c
                pu_free[pu] = fin
                per_pu[pu] += c
                ready[ti * ntj + tj] = fin
                compute_cycles += c
            # One barrier per diagonal.
            gsync = max(gsync, max(pu_free, default=0)) + barrier_cyc
            barrier_cycles_total += barrier_cyc

    total = max(gsync, max(pu_free, default=0))
    breakdown = {"compute_tile": compute_cycles, "barrier": barrier_cycles_total}
    return total, {pu: per_pu[pu] for pu in range(num_pus)}, breakdown


# ---------------------------------------------------------------------------
# Timing: explicit event schedules
# ---------------------------------------------------------------------------

def _simulate_explicit_schedule(
    schedule: Dict[str, Any],
    events: List[Dict[str, Any]],
    *,
    comm_cycles_for_bytes: Callable[[int], int],
    mem_scale_for_object: Callable[[str], float],
) -> Timing:
    """Timing of an explicit event list.

    Every PU has its own timeline. A compute_tile event advances the timeline
    of its owner PU. Communication and barrier events wait for all PUs and
    then advance all timelines together.
    """
    hw = _section(schedule, "hw")
    num_pus = int(hw.get("num_pus_total", CfgArchitecture.NUM_PUS_TOTAL))
    lanes = int(hw.get("pe_lanes_per_pu", CfgGenDRAMSim.PE_LANES_PER_PU))
    default_payload = _default_tile_payload_bytes(schedule)

    pu_t: Dict[int, int] = {pu: 0 for pu in range(num_pus)}
    per_pu_cycles: Dict[int, int] = {pu: 0 for pu in range(num_pus)}
    breakdown: Dict[str, int] = {}
    global_t = 0

    for ev in events:
        op = str(ev.get("op", "unknown"))

        if op == "compute_tile":
            owner = int(ev.get("owner_pu", -1))
            rowbuf = _section(ev, "rowbuf")
            cycles = _tile_compute_cycles(
                dp_updates=int(ev.get("dp_updates", 0)),
                read_rows=int(rowbuf.get("read_rows", 0)),
                write_rows=int(rowbuf.get("write_rows", 0)),
                lanes=lanes,
                mem_scale=mem_scale_for_object(str(_section(ev, "tile").get("object", ""))),
            )
            if owner < 0:
                # A tile without an owner runs alone.
                global_t += cycles
                for pu in pu_t:
                    pu_t[pu] = max(pu_t[pu], global_t)
            else:
                pu_t[owner] = max(pu_t[owner], global_t) + cycles
                per_pu_cycles[owner] = per_pu_cycles.get(owner, 0) + cycles
            breakdown[op] = breakdown.get(op, 0) + cycles
            continue

        if op == "broadcast":
            cycles = int(comm_cycles_for_bytes(int(ev.get("payload_bytes", default_payload))))
        elif op in ("send", "recv"):
            payload = int(ev.get("payload_bytes", default_payload))
            cycles = int(CfgGenDRAMSim.SEND_RECV_CYCLES + comm_cycles_for_bytes(payload))
        elif op == "barrier":
            cycles = _wavefront_barrier_cycles(str(ev.get("scope", "")))
        elif op == "mem_transfer":
            cycles = CfgGenDRAMSim.MEM_TRANSFER_CYCLES
        else:
            cycles = 1

        global_t = max([global_t] + list(pu_t.values())) + cycles
        for pu in pu_t:
            pu_t[pu] = global_t
        breakdown[op] = breakdown.get(op, 0) + cycles

    total_cycles = max([global_t] + list(pu_t.values()))
    return total_cycles, per_pu_cycles, breakdown


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def simulate_from_artifacts(
    out_dir: str,
    *,
    verbose: bool = True,
    functional: bool = True,
    mode: str = "auto",
) -> SimResult:
    """Simulate the schedule in out_dir.

    mode="execute" runs the discrete-event executor for APSP and alignment.
    Any other value ("auto", "analytical") uses the closed-form model for
    symbolic schedules and replays explicit schedules event by event. With
    functional=True the workload is also executed if its inputs are present
    in out_dir.
    """
    layout, schedule = load_artifacts(out_dir)
    schedule_kind = str(schedule.get("schedule_kind", "explicit"))
    events: List[Dict[str, Any]] = list(schedule.get("events", []))
    workload = str(schedule.get("workload", ""))

    # Tier of every memory object, from the layout.
    tier_map: Dict[str, int] = {}
    try:
        objs = layout.get("objects", []) if isinstance(layout, dict) else []
        if isinstance(objs, list):
            for o in objs:
                if isinstance(o, dict) and "name" in o:
                    tier_map[str(o.get("name"))] = int(o.get("tier", _SLOWEST_TIER))
    except (TypeError, ValueError):
        tier_map = {}

    # Optional override of the tier latency spread (sensitivity studies).
    tier_spread = _section(schedule, "hw").get("tier_latency_spread")
    try:
        tier_spread = float(tier_spread) if tier_spread is not None else None
    except (TypeError, ValueError):
        tier_spread = None

    def mem_scale_for_object(name: str) -> float:
        if not name:
            return 1.0
        return _tier_latency_scale(tier_map.get(name, _SLOWEST_TIER), spread=tier_spread)

    # Broadcast bandwidth in bytes per cycle; the schedule may set or scale it.
    bw_bcast = float(CfgArchitecture.BW_BROADCAST_B_PER_CYCLE)
    comm = _section(schedule, "comm")
    try:
        if "bw_broadcast_B_per_cycle" in comm:
            bw_bcast = float(comm["bw_broadcast_B_per_cycle"])
        elif "bcast_bw_scale" in comm:
            bw_bcast = bw_bcast * float(comm["bcast_bw_scale"])
    except (TypeError, ValueError):
        pass
    if bw_bcast <= 0:
        bw_bcast = 1.0

    def comm_cycles_for_bytes(nbytes: int) -> int:
        return math.ceil(nbytes / bw_bcast)

    if mode == "execute" and workload == "apsp":
        timing = _execute_apsp(
            schedule,
            comm_cycles_for_bytes=comm_cycles_for_bytes,
            mem_scale=mem_scale_for_object("D"),
        )
    elif mode == "execute" and workload == "alignment":
        timing = _execute_alignment(schedule, mem_scale=mem_scale_for_object("H"))
    elif schedule_kind == "symbolic" and workload == "apsp":
        timing = _simulate_symbolic_apsp(
            schedule,
            comm_cycles_for_bytes=comm_cycles_for_bytes,
            mem_scale=mem_scale_for_object("D"),
        )
    elif schedule_kind == "symbolic" and workload == "alignment":
        timing = _simulate_symbolic_alignment(schedule, mem_scale=mem_scale_for_object("H"))
    else:
        timing = _simulate_explicit_schedule(
            schedule,
            events,
            comm_cycles_for_bytes=comm_cycles_for_bytes,
            mem_scale_for_object=mem_scale_for_object,
        )
    total_cycles, per_pu_cycles, breakdown = timing
    total_time_s = float(total_cycles) * _cycle_time_s(schedule)

    output_paths: Dict[str, str] = {}
    if functional and workload in ("apsp", "alignment"):
        run = _run_functional_apsp if workload == "apsp" else _run_functional_alignment
        label = "APSP" if workload == "apsp" else "alignment"
        try:
            output_paths = run(out_dir, schedule, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"[gendram_sim] functional {label} failed: {e}")

    per_pu_cycles = dict(sorted(per_pu_cycles.items()))
    breakdown = dict(sorted(breakdown.items()))

    if verbose:
        print("[gendram_sim] schedule_kind:", schedule_kind)
        print("[gendram_sim] schedule events:", len(events))
        print("[gendram_sim] total_cycles:", total_cycles)
        print("[gendram_sim] total_time_s:", total_time_s)
        if per_pu_cycles:
            top = sorted(per_pu_cycles.items(), key=lambda kv: kv[1], reverse=True)[:8]
            print("[gendram_sim] top per_pu cycles (id -> cycles):", top)
        print("[gendram_sim] event breakdown (cycles):", breakdown)

    return SimResult(
        total_cycles=total_cycles,
        total_time_s=total_time_s,
        per_pu_cycles=per_pu_cycles,
        event_breakdown_cycles=breakdown,
        output_paths=output_paths,
    )


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Simulate a GenMLIR artifact directory on GenDRAM")
    ap.add_argument("out_dir", help="Directory with layout.json and schedule.json")
    ap.add_argument("--quiet", action="store_true", help="Reduce logging")
    ap.add_argument(
        "--no-functional",
        action="store_true",
        help="Skip functional execution even if workload inputs are present",
    )
    args = ap.parse_args()

    simulate_from_artifacts(args.out_dir, verbose=not args.quiet, functional=not args.no_functional)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
