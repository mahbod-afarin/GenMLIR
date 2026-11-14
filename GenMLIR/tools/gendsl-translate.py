#!/usr/bin/env python3
"""GenDSL to MLIR translator.

Reads a GenDSL workload description and prints an MLIR module that uses the
GenMLIR dp dialect.

The input is a flat "key: value" file:

    workload: apsp | alignment
    N: 256
    tile_size: 64

For APSP the module contains the tile-level schedule of blocked
Floyd-Warshall. For every block index k it has a pivot phase with tile (k, k),
a rowcol phase with tiles (k, *) and (*, k), and an internal phase with the
remaining tiles. For alignment it contains the tiles of the DP table in
wavefront order.

The DP loops inside a tile are not unrolled. Each dp.tile is an anchor that
later passes annotate with an owner PU, wrap with communication ops, and turn
into schedule events.

Usage:
    python3 tools/gendsl-translate.py examples/gendsl_apsp.yaml > apsp.mlir
"""

from __future__ import annotations

import sys
from pathlib import Path


def parse_kv_yaml(path: Path) -> dict:
    cfg: dict = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        v = v.strip()
        if v.startswith(("'", '"')) and v.endswith(("'", '"')):
            v = v[1:-1]
        if v.lstrip("-").isdigit():
            cfg[k] = int(v)
        else:
            cfg[k] = v
    return cfg


def _tile(lines: list[str], ti: int, tj: int, tk: int) -> None:
    lines.append(f'      dp.tile attributes {{ ti = {ti} : i64, tj = {tj} : i64, tk = {tk} : i64 }} {{')
    lines.append("        dp.yield")
    lines.append("      }")


def emit_apsp(cfg: dict) -> str:
    """Emit the blocked Floyd-Warshall tile schedule."""
    N = int(cfg.get("N", 256))
    B = int(cfg.get("tile_size", 64))
    ntiles = (N + B - 1) // B

    lines: list[str] = []
    lines.append(
        "module attributes { "
        f'gendram.workload = "apsp", '
        f'gendram.tile_size = {B} : i64, '
        f'gendram.N = {N} : i64 '
        "} {"
    )

    lines.append(f'  memref.global "private" @D : memref<{N}x{N}xi32> = uninitialized')
    # Seeding tables of the genomics pipeline. APSP does not use them.
    lines.append('  memref.global "private" @PTR : memref<1024xi32> = uninitialized')
    lines.append('  memref.global "private" @CAL : memref<4096xi32> = uninitialized')
    lines.append("")

    lines.append("  func.func @main() {")
    lines.append(f'    %D = memref.get_global @D : memref<{N}x{N}xi32>')
    lines.append("")

    for k in range(ntiles):
        lines.append('    dp.phase attributes { kind = "pivot" } {')
        _tile(lines, k, k, k)
        lines.append("      dp.yield")
        lines.append("    }")
        lines.append("")

        lines.append('    dp.phase attributes { kind = "rowcol" } {')
        for j in range(ntiles):
            if j != k:
                _tile(lines, k, j, k)
        for i in range(ntiles):
            if i != k:
                _tile(lines, i, k, k)
        lines.append("      dp.yield")
        lines.append("    }")
        lines.append("")

        lines.append('    dp.phase attributes { kind = "internal" } {')
        for i in range(ntiles):
            if i == k:
                continue
            for j in range(ntiles):
                if j != k:
                    _tile(lines, i, j, k)
        lines.append("      dp.yield")
        lines.append("    }")
        lines.append("")

    lines.append("    func.return")
    lines.append("  }")
    lines.append("}")

    return "\n".join(lines)


def emit_alignment(cfg: dict) -> str:
    """Emit the wavefront tile schedule of Smith-Waterman local alignment.

    The DP table H has (query_len + 1) x (ref_len + 1) entries. Tiles are
    listed diagonal by diagonal, so every tile follows the tiles it depends on.
    """
    # "N" is accepted as a shorthand for a square problem.
    qlen = int(cfg.get("query_len", cfg.get("N", 256)))
    rlen = int(cfg.get("ref_len", cfg.get("N", 256)))
    B = int(cfg.get("tile_size", 64))

    algo = str(cfg.get("algo", "smith_waterman"))
    match = int(cfg.get("match", 2))
    mismatch = int(cfg.get("mismatch", -1))
    gap = int(cfg.get("gap", -1))
    band = int(cfg.get("band_width", 0))

    H0 = qlen + 1
    H1 = rlen + 1
    ntiles_i = (H0 + B - 1) // B
    ntiles_j = (H1 + B - 1) // B

    lines: list[str] = []
    lines.append(
        "module attributes { "
        f'gendram.workload = "alignment", '
        f'gendram.tile_size = {B} : i64, '
        f'gendram.query_len = {qlen} : i64, '
        f'gendram.ref_len = {rlen} : i64, '
        f'gendram.algo = "{algo}", '
        f'gendram.match = {match} : i64, '
        f'gendram.mismatch = {mismatch} : i64, '
        f'gendram.gap = {gap} : i64, '
        f'gendram.band_width = {band} : i64 '
        "} {"
    )
    # Seeding tables and the candidate list passed from seeding to alignment.
    lines.append('  memref.global "private" @PTR : memref<1024xi32> = uninitialized')
    lines.append('  memref.global "private" @CAL : memref<4096xi32> = uninitialized')
    lines.append('  memref.global "private" @candidates : memref<1024x4xi32> = uninitialized')
    # Input sequences as ASCII bytes, and the score table.
    lines.append(f'  memref.global "private" @Qseq : memref<{qlen}xi8> = uninitialized')
    lines.append(f'  memref.global "private" @Rseq : memref<{rlen}xi8> = uninitialized')
    lines.append(f'  memref.global "private" @H : memref<{H0}x{H1}xi32> = uninitialized')
    lines.append("")

    # Seeding kernel (Search PUs).
    lines.append("  func.func @seeding() {")
    lines.append("    %ptr = memref.get_global @PTR : memref<1024xi32>")
    lines.append("    %cal = memref.get_global @CAL : memref<4096xi32>")
    lines.append("    %cnd = memref.get_global @candidates : memref<1024x4xi32>")
    lines.append("    // The PTR -> CAL lookups of the seeding stage are not modeled.")
    lines.append("    func.return")
    lines.append("  }")
    lines.append("")

    # Alignment kernel (Compute PUs).
    lines.append("  func.func @alignment() {")
    lines.append("    dp.wavefront {")
    for step in range(ntiles_i + ntiles_j - 1):
        for ti in range(ntiles_i):
            tj = step - ti
            if tj < 0 or tj >= ntiles_j:
                continue
            lines.append(f'      dp.tile attributes {{ ti = {ti} : i64, tj = {tj} : i64 }} {{')
            # One dp.grid_update per tile carries the semiring.
            lines.append("        %cur0 = arith.constant 0.0 : f32")
            lines.append("        %cand0 = arith.constant 1.0 : f32")
            lines.append('        %u0 = dp.grid_update %cur0, %cand0 { semiring = "max_plus" } : f32')
            lines.append("        dp.yield")
            lines.append("      }")
    lines.append("      dp.yield")
    lines.append("    }")
    lines.append("    func.return")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines)


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: gendsl-translate.py <config.yaml>", file=sys.stderr)
        return 2

    cfg = parse_kv_yaml(Path(sys.argv[1]))
    workload = cfg.get("workload", "apsp")

    if workload == "apsp":
        print(emit_apsp(cfg))
    elif workload == "alignment":
        print(emit_alignment(cfg))
    else:
        print(f"Unknown workload: {workload}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
