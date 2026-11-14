# GenMLIR compiler

This directory holds the compiler. The C++ sources form an out-of-tree MLIR project with two dialects, the pass pipeline and the `genmlir-opt` driver. `py_compiler/` is a Python implementation of the same pipeline that needs no LLVM build. The top-level README explains how the two relate.

## Dialects

The `dp` dialect keeps the intent of a dynamic-programming kernel explicit. `dp.grid_update` is one semi-ring update (`min_plus` or `max_plus`). `dp.tile` marks the computation of one tile and carries its coordinates and, after mapping, its owner PU. `dp.phase` groups the tiles of an APSP phase (pivot, rowcol, internal) and `dp.wavefront` groups the tiles of an alignment kernel. `dp.yield` terminates their regions.

The `gendram` dialect makes device actions explicit. It has `gendram.broadcast`, `gendram.send`, `gendram.recv`, `gendram.barrier` and `gendram.mem_transfer`.

## Passes

| Pass | Scope | Action |
|---|---|---|
| `dp-extract` | function | rewrites `select(cmp(candidate, current), candidate, current)` into `dp.grid_update` with the matching semiring |
| `dp-canonicalize` | function | sets a missing or unknown semiring attribute to `min_plus` |
| `dp-tiling` | function | wraps the body in a single `dp.tile` if the function has none |
| `apsp-phase-formation` | function | wraps the body in a single `dp.phase` if the function has none |
| `wavefront-schedule` | function | annotates every `dp.tile` with `wavefront_step = ti + tj` |
| `gendram-mode-select` | module | sets `gendram.mode` to `broadcast` or `pipeline` |
| `gendram-mapping` | function | annotates every `dp.tile` with `owner_pu = (ti*M + tj) mod (C*G)` |
| `gendram-tiering` | module | attaches tier, channel, bank-group and interleaving to memory objects |
| `gendram-comm-insert` | module | inserts broadcasts and barriers (broadcast mode) or send and recv (pipeline mode) |
| `gendram-lower-to-runtime-calls` | module | lowers `gendram` ops to calls of the runtime API in `runtime/` |
| `gendram-emit-artifacts` | module | writes `layout.json`, `schedule.json` and `pu_programs/` |

The GenDSL translator emits `dp.phase`, `dp.wavefront` and `dp.tile` ops directly, so `dp-tiling` and `apsp-phase-formation` leave its output unchanged. `gendram-mapping` takes the options `C`, `G` and `M`. `M` is the number of tiles per row of the problem. `gendram-emit-artifacts` takes `out_dir`. The runtime API takes f32 payloads, and `gendram-lower-to-runtime-calls` leaves an op with another payload element type in place.

## Building

The build needs CMake 3.20 or newer, Ninja, a C++17 compiler and an LLVM build with MLIR enabled. We tested with LLVM 18.1.

```bash
mkdir -p build && cd build
cmake -G Ninja .. \
  -DMLIR_DIR=/path/to/llvm-build/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm-build/lib/cmake/llvm
ninja genmlir-opt
```

`build_and_run.sh --llvm-build /path/to/llvm-build` runs these steps and then both example pipelines. `build_and_run.sh --help` lists its options.

## Usage

Translate a GenDSL description to MLIR, then run the pipeline.

```bash
python3 tools/gendsl-translate.py examples/gendsl_apsp.yaml > /tmp/apsp.mlir

./build/bin/genmlir-opt /tmp/apsp.mlir \
  --dp-canonicalize \
  --gendram-mode-select \
  --gendram-mapping="C=4 G=8 M=4" \
  --gendram-tiering \
  --gendram-comm-insert \
  --gendram-emit-artifacts="out_dir=/tmp/genmlir_apsp" \
  -o /tmp/apsp.lowered.mlir
```

For alignment, use `examples/gendsl_alignment.yaml` and add `--wavefront-schedule` after `--dp-canonicalize`. The artifacts appear in the directory given as `out_dir`. `examples/apsp.mlir` and `examples/alignment.mlir` are the translator output for the two GenDSL examples.

`run_e2e.py` in the repository root runs these steps and simulates the result.

## Layout

```
include/genmlir/
  Dialect/DPGrid/       dp dialect (ODS)
  Dialect/GenDRAM/      gendram dialect (ODS)
  Passes/               pass declarations (ODS)
lib/
  Dialect/              dialect registration
  Passes/               pass implementations
tools/
  genmlir-opt/          driver
  gendsl-translate.py   GenDSL to MLIR translator
runtime/                runtime API called by the lowered IR
examples/               GenDSL inputs and generated MLIR
py_compiler/            Python implementation of the pipeline
```
