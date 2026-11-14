#!/usr/bin/env python3
"""Compile a GenDSL example with genmlir-opt and simulate the result.

Steps:
  1. Translate the GenDSL example of the selected workload to MLIR (or take
     the file given with --mlir).
  2. Run the GenMLIR pass pipeline, which writes layout.json, schedule.json
     and pu_programs/ to <out-dir>/<workload>/.
  3. Generate random inputs: a distance matrix for APSP, a query and a
     reference sequence for alignment.
  4. Run the GenDRAM simulator on the artifacts, including functional
     execution.
  5. With --check, compare the simulator output with a CPU reference.

Requires a built genmlir-opt (see GenMLIR/README.md) and NumPy.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

from GenDRAM.gendram_command_sim import simulate_from_artifacts


def _gen_random_dna(length: int, *, seed: int) -> str:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, 4, size=length, dtype=np.int64)
    return "".join("ACGT"[int(i)] for i in idx)


def _gen_random_apsp_matrix(N: int, *, density: float, max_weight: int, inf: int, seed: int) -> "np.ndarray":
    """Random directed graph as a distance matrix; missing edges are `inf`."""
    rng = np.random.default_rng(seed)
    D = np.full((N, N), inf, dtype=np.int64)
    np.fill_diagonal(D, 0)

    mask = rng.random((N, N)) < density
    np.fill_diagonal(mask, False)

    weights = rng.integers(1, max_weight + 1, size=(N, N), dtype=np.int64)
    D[mask] = weights[mask]
    return D


def _cpu_apsp_reference(D_in: "np.ndarray") -> "np.ndarray":
    """Floyd-Warshall, vectorized over (i, j)."""
    D = D_in.astype(np.int64).copy()
    for k in range(D.shape[0]):
        D = np.minimum(D, D[:, k][:, None] + D[k, :][None, :])
    return D


def _cpu_sw_reference(query: str, ref: str, *, match: int, mismatch: int, gap: int, band_width: int = 0) -> int:
    """Best Smith-Waterman local alignment score, optionally banded."""
    qlen = len(query)
    rlen = len(ref)
    H = np.zeros((qlen + 1, rlen + 1), dtype=np.int32)
    best = 0
    for i in range(1, qlen + 1):
        ai = query[i - 1]
        if band_width > 0:
            j_iter = range(max(1, i - band_width), min(rlen, i + band_width) + 1)
        else:
            j_iter = range(1, rlen + 1)
        for j in j_iter:
            sub = match if ai == ref[j - 1] else mismatch
            v = max(0, int(H[i - 1, j - 1]) + sub, int(H[i - 1, j]) + gap, int(H[i, j - 1]) + gap)
            H[i, j] = v
            if v > best:
                best = v
    return int(best)


def _check_apsp(input_path: Path, out_path: str) -> str:
    if not out_path or not os.path.isfile(out_path):
        return "skipped (no output)"
    D_in = np.load(str(input_path)).astype(np.int64)
    N = D_in.shape[0]
    if N > 64:
        return f"skipped (N={N} too large; use N<=64 for --check)"
    D_out = np.load(out_path).astype(np.int64)
    return "PASS" if np.array_equal(_cpu_apsp_reference(D_in), D_out) else "FAIL"


def _check_alignment(schedule: dict, q_path: Path, r_path: Path, score_path: str) -> str:
    if not score_path or not os.path.isfile(score_path):
        return "skipped (no output_score.txt)"
    ref_score = _cpu_sw_reference(
        q_path.read_text(encoding="utf-8").strip(),
        r_path.read_text(encoding="utf-8").strip(),
        match=int(schedule.get("match", 2)),
        mismatch=int(schedule.get("mismatch", -1)),
        gap=int(schedule.get("gap", -1)),
        band_width=int(schedule.get("band_width", 0)),
    )
    sim_score = None
    for line in Path(score_path).read_text(encoding="utf-8").splitlines():
        if line.startswith("max_score="):
            sim_score = int(line.split("=", 1)[1].strip())
            break
    if sim_score is None:
        return "FAIL (could not parse score)"
    return "PASS" if sim_score == ref_score else f"FAIL (sim={sim_score}, ref={ref_score})"


def main() -> int:
    root = Path(__file__).resolve().parent
    genmlir_dir = root / "GenMLIR"

    ap = argparse.ArgumentParser(description="Compile with genmlir-opt and simulate on GenDRAM")
    ap.add_argument(
        "--workload",
        choices=["apsp", "alignment"],
        default="apsp",
        help="Workload to compile and simulate",
    )
    ap.add_argument(
        "--genmlir-opt",
        default=str(genmlir_dir / "build" / "bin" / "genmlir-opt"),
        help="Path to the built genmlir-opt binary",
    )
    ap.add_argument(
        "--mlir",
        default="",
        help="Input .mlir file (default: generated from the GenDSL example of the workload)",
    )
    ap.add_argument(
        "--out-dir",
        default=str(root / "out_e2e"),
        help="Output root directory for compilation artifacts",
    )
    ap.add_argument("--C", type=int, default=4, help="Number of channels for the mapping pass")
    ap.add_argument("--G", type=int, default=8, help="Bank-groups per channel for the mapping pass")
    ap.add_argument("--M", type=int, default=64, help="Tiles-per-row factor for the mapping pass")

    ap.add_argument("--density", type=float, default=0.20, help="(APSP) edge density of the random graph")
    ap.add_argument("--max-weight", type=int, default=20, help="(APSP) maximum edge weight")
    ap.add_argument("--inf", type=int, default=1_000_000_000, help="(APSP) value used for missing edges")
    ap.add_argument("--seed", type=int, default=0, help="Seed for the random inputs")
    ap.add_argument(
        "--regen-input",
        action="store_true",
        help="Regenerate the inputs even if they already exist",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="Compare the simulator output with a CPU reference (APSP: only if N <= 64)",
    )
    ap.add_argument("--quiet", action="store_true", help="Reduce simulator logging")
    args = ap.parse_args()

    genmlir_opt = Path(args.genmlir_opt)
    if not genmlir_opt.is_file():
        print(f"error: genmlir-opt not found at: {genmlir_opt}", file=sys.stderr)
        print(
            "Build it first (example):\n"
            "  cd GenMLIR && ./build_and_run.sh --llvm-build /path/to/llvm-build --mode apsp\n",
            file=sys.stderr,
        )
        return 2

    if np is None:
        print(
            "error: NumPy is required for functional simulation.\n"
            "Install it with:  pip install numpy\n",
            file=sys.stderr,
        )
        return 2

    out_root = Path(args.out_dir)
    out_dir = out_root / args.workload
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_root / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # MLIR input.
    if args.mlir:
        input_mlir = Path(args.mlir)
    else:
        yaml_path = genmlir_dir / "examples" / f"gendsl_{args.workload}.yaml"
        input_mlir = tmp_dir / f"{args.workload}.mlir"
        mlir_text = subprocess.check_output(
            [sys.executable, str(genmlir_dir / "tools" / "gendsl-translate.py"), str(yaml_path)],
            cwd=str(root),
        )
        input_mlir.write_bytes(mlir_text)

    if not input_mlir.is_file():
        print(f"error: input MLIR not found at: {input_mlir}", file=sys.stderr)
        return 2

    # Compile. The wall-clock time of genmlir-opt is the compile time.
    lowered_mlir = out_dir / f"{args.workload}.lowered.mlir"
    passes = [
        "--dp-canonicalize",
        "--gendram-mode-select",
        f"--gendram-mapping=C={args.C} G={args.G} M={args.M}",
        "--gendram-tiering",
        "--gendram-comm-insert",
        f"--gendram-emit-artifacts=out_dir={out_dir}",
    ]
    if args.workload == "alignment":
        passes.insert(1, "--wavefront-schedule")

    cmd = [str(genmlir_opt), str(input_mlir), *passes, "-o", str(lowered_mlir)]
    t0 = time.perf_counter()
    subprocess.run(cmd, cwd=str(root), check=True)
    compile_time_s = time.perf_counter() - t0

    # Inputs for functional execution, sized from the emitted schedule.
    input_path = out_dir / "input_D.npy"
    q_path = out_dir / "input_query.txt"
    r_path = out_dir / "input_ref.txt"
    schedule_path = out_dir / "schedule.json"
    schedule = json.loads(schedule_path.read_text(encoding="utf-8")) if schedule_path.is_file() else {}

    if args.workload == "apsp":
        if args.regen_input or not input_path.is_file():
            if not schedule.get("D_shape"):
                print("error: schedule.json did not contain D_shape; cannot generate input.", file=sys.stderr)
                return 2
            N = int(schedule["D_shape"][0])
            D = _gen_random_apsp_matrix(
                N,
                density=float(args.density),
                max_weight=int(args.max_weight),
                inf=int(args.inf),
                seed=int(args.seed),
            )
            np.save(str(input_path), D)
            if not args.quiet:
                print(f"[e2e] wrote APSP input: {input_path}  (N={N}, density={args.density})")
    else:
        qlen = int(schedule.get("query_len", 128))
        rlen = int(schedule.get("ref_len", 160))
        if args.regen_input or not q_path.is_file() or not r_path.is_file():
            query = _gen_random_dna(qlen, seed=int(args.seed))
            ref = _gen_random_dna(rlen, seed=int(args.seed) + 1)
            q_path.write_text(query + "\n", encoding="utf-8")
            r_path.write_text(ref + "\n", encoding="utf-8")
            if not args.quiet:
                print(f"[e2e] wrote alignment inputs: {q_path} (len={qlen}), {r_path} (len={rlen})")

    # Simulate.
    t1 = time.perf_counter()
    sim_res = simulate_from_artifacts(str(out_dir), verbose=not args.quiet, functional=True)
    sim_wall_s = time.perf_counter() - t1

    check_ok = None
    if args.check:
        if args.workload == "apsp":
            check_ok = _check_apsp(input_path, sim_res.output_paths.get("D", ""))
        else:
            check_ok = _check_alignment(schedule, q_path, r_path, sim_res.output_paths.get("score", ""))

    accel_time_s = float(sim_res.total_time_s)
    total_modeled = float(compile_time_s) + accel_time_s
    total_wall = float(compile_time_s) + float(sim_wall_s)
    compiler_fraction = compile_time_s / total_modeled if total_modeled > 0 else 0.0
    execution_fraction = accel_time_s / total_modeled if total_modeled > 0 else 0.0

    print("\n=== End-to-end summary ===")
    print(f"workload                : {args.workload}")
    print(f"compile_time_s          : {compile_time_s:.6f}")
    print(f"accelerator_cycles      : {sim_res.total_cycles}")
    print(f"accelerator_time_s      : {accel_time_s:.9f}  (simulated @ 1GHz)")
    if total_modeled > 0:
        print(f"total_modeled_time_s    : {total_modeled:.9f}  (compile + simulated)")
        print(f"compiler_fraction       : {compiler_fraction:.6f}")
        print(f"execution_fraction      : {execution_fraction:.6f}")
    if total_wall > 0:
        print(f"total_wall_time_s       : {total_wall:.6f}  (compile + simulator wallclock)")
        print(f"compiler_fraction_wall  : {compile_time_s / total_wall:.6f}")
        print(f"simulator_fraction_wall : {sim_wall_s / total_wall:.6f}")
    print(f"simulator_wallclock_s   : {sim_wall_s:.6f}")
    print(f"artifacts_out_dir       : {out_dir}")
    print(f"lowered_mlir            : {lowered_mlir}")

    report_lines = [
        f"workload={args.workload}",
        f"compile_time_s={compile_time_s:.9f}",
        f"accelerator_cycles={sim_res.total_cycles}",
        f"accelerator_time_s={accel_time_s:.12f}",
        f"simulator_wallclock_s={sim_wall_s:.9f}",
        f"total_modeled_time_s={total_modeled:.12f}",
        f"compiler_fraction={compiler_fraction:.6f}",
        f"execution_fraction={execution_fraction:.6f}",
    ]

    if args.workload == "apsp":
        print(f"input_D.npy             : {input_path}")
        report_lines.append(f"input_D_npy={input_path}")
        if sim_res.output_paths.get("D"):
            print(f"output_D.npy            : {sim_res.output_paths['D']}")
            report_lines.append(f"output_D_npy={sim_res.output_paths['D']}")
    else:
        print(f"input_query.txt         : {q_path}")
        print(f"input_ref.txt           : {r_path}")
        report_lines.append(f"input_query_txt={q_path}")
        report_lines.append(f"input_ref_txt={r_path}")
        if sim_res.output_paths.get("score"):
            print(f"output_score.txt        : {sim_res.output_paths['score']}")
            report_lines.append(f"output_score_txt={sim_res.output_paths['score']}")
        if sim_res.output_paths.get("H"):
            print(f"output_H.npy            : {sim_res.output_paths['H']}")
            report_lines.append(f"output_H_npy={sim_res.output_paths['H']}")
    if check_ok is not None:
        print(f"correctness_check       : {check_ok}")
        report_lines.append(f"correctness_check={check_ok}")

    (out_dir / "e2e_report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
