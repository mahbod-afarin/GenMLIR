# GenMLIR

GenMLIR is an MLIR-based compiler for semi-ring dynamic programming on GenDRAM, a tier-aware 3D-DRAM processing-in-memory accelerator [arXiv'26 GenDRAM]. Starting from a short description of the recurrence, it decides the tiling, the tile schedule, the assignment of tiles to processing units (PUs), the tier placement of the memory objects and the communication between PUs. Its output is a memory layout, a schedule and per-PU programs for a GenDRAM device or simulator. Two workloads are supported, all-pairs shortest path (APSP, blocked Floyd-Warshall, min-plus) and sequence alignment (Smith-Waterman, max-plus).

This repository accompanies our ASP-DAC 2027 paper. It contains the compiler, a command-stream simulator of GenDRAM, the datasets and the scripts that produce the evaluation results.

## Repository layout

```
GenMLIR/              compiler
  include/, lib/      dp and gendram dialects and the passes (C++/MLIR)
  tools/              genmlir-opt driver and the GenDSL translator
  runtime/            runtime API targeted by the lowering pass
  examples/           GenDSL inputs and the MLIR generated from them
  py_compiler/        Python implementation of the pipeline
GenDRAM/              command-stream simulator
datasets/             graphs and long-read sets
evaluation/           experiment scripts and reference results
run_datasets.py       Python compiler and simulator on the datasets
run_e2e.py            genmlir-opt and simulator on the GenDSL examples
```

## Requirements

Python 3.9 or newer with NumPy, plus Matplotlib for the evaluation figures. The Python compiler, the simulator and the evaluation need nothing else.

Building the C++ compiler additionally needs CMake 3.20 or newer, Ninja, a C++17 compiler and an LLVM build with MLIR enabled. We tested with LLVM 18.1.

## Quick start

```bash
pip install numpy matplotlib
python3 evaluation/run_all.py
```

This regenerates every figure and table in `evaluation/results/` in about a minute. `evaluation/README.md` lists the experiments and states what is measured and what is modeled.

To compile and simulate a single dataset workload, use `run_datasets.py`.

```bash
python3 run_datasets.py --workload apsp --graph Graphs/CA-GrQc.txt
python3 run_datasets.py --workload alignment \
    --fastq Alignment/ont2d_6000_0001.fastq --ref Alignment/ont2d_6000_0001.ref
```

A run writes its artifacts to `out/<workload>/` and prints the compile time and the simulated accelerator time. Alignment runs also execute the DP for the selected read and write its score. APSP runs on the dataset graphs simulate timing only. `run_datasets.py` defaults to a tile size of 64 and the closed-form timing model, whereas the evaluation scripts use a tile size of 256 and the discrete-event executor, so the cycle counts of the two differ.

## Compiler pipeline

The pipeline has four pass groups and a back end.

| Group | Passes | Role |
|---|---|---|
| G1, DP recognition | `dp-extract`, `dp-canonicalize` | recognize semi-ring updates and normalize them |
| G2, tiling and schedule | `dp-tiling`, `apsp-phase-formation`, `wavefront-schedule` | tiles, APSP phases, wavefront order |
| G3, mapping and placement | `gendram-mode-select`, `gendram-mapping`, `gendram-tiering` | execution mode, tile-to-PU mapping, tier placement |
| G4, communication | `gendram-comm-insert` | broadcasts, queues and barriers |
| Back end | `gendram-lower-to-runtime-calls`, `gendram-emit-artifacts` | runtime calls, artifact emission |

The repository has two implementations of this pipeline.

The C++/MLIR implementation (`GenMLIR/include`, `GenMLIR/lib`, `GenMLIR/tools`) defines the `dp` and `gendram` dialects and the passes, and provides the `genmlir-opt` driver. Its input is the MLIR module that `tools/gendsl-translate.py` generates from a GenDSL description. That module already carries the tile-level schedule as `dp.phase`, `dp.wavefront` and `dp.tile` ops, which the passes annotate and extend. The emitted schedule lists every tile and communication event. `GenMLIR/README.md` describes the dialects, the passes and the build.

The Python implementation (`GenMLIR/py_compiler`) needs no LLVM build and is the one used by `run_datasets.py` and by the evaluation. It does not transform IR. It computes the layout and a symbolic schedule directly from the workload parameters. The symbolic schedule records shapes, tiling, the mapping rule and the model parameters selected by the enabled passes, and the simulator derives the tile order from them, so its size does not grow with the problem. Tier placement follows the capacity-aware, latency-ranked policy of the paper. The C++ tiering pass uses a simpler rule that puts the seeding tables on tier 0 and all other objects on tier 1.

### Disabling passes

Passes can be disabled by name to form baselines and to measure the effect of a pass.

```bash
python3 run_datasets.py --workload apsp --list-passes
python3 run_datasets.py --workload apsp --graph Graphs/CA-GrQc.txt --disable-pass gendram-mapping
python3 run_datasets.py --workload alignment --sweep-disable-each-pass --no-functional
```

The sweep runs the full pipeline and then one configuration per pass with that pass disabled, and writes `out/<workload>/pass_sweep/pass_sweep.json`.

In the Python implementation a disabled pass changes the part of the layout or schedule that the pass is responsible for.

| Disabled pass | Effect |
|---|---|
| `dp-tiling` | row-buffer operations are charged per element instead of per tile row |
| `apsp-phase-formation` | barriers cost four times as much |
| `wavefront-schedule` | a global barrier follows every tile, so tiles run one after another |
| `gendram-mode-select` | APSP uses only the 24 Compute PUs and 1/16 of the broadcast bandwidth; alignment pays a full barrier per diagonal instead of the cheaper wavefront barrier |
| `gendram-mapping` | all tiles run on one PU |
| `gendram-tiering` | every object is placed on the slowest tier |
| `gendram-comm-insert` | APSP uses 1/16 of the broadcast bandwidth; alignment pays a memory transfer per tile |
| `dp-extract`, `dp-canonicalize`, `gendram-lower-to-runtime-calls` | none |
| `gendram-emit-artifacts` | cannot be disabled |

## Building and running the C++ compiler

```bash
cd GenMLIR
./build_and_run.sh --llvm-build /path/to/llvm-build
```

The script builds `GenMLIR/build/bin/genmlir-opt` and runs the APSP and alignment example pipelines, which write their artifacts to `GenMLIR/out/`. The manual CMake commands are in `GenMLIR/README.md`.

`run_e2e.py` then drives the whole flow from the repository root. It translates the GenDSL example of the workload, compiles it with `genmlir-opt`, generates random inputs, and runs the simulator including functional execution.

```bash
python3 run_e2e.py --workload apsp
python3 run_e2e.py --workload alignment --check
```

With `--check` the simulator output is compared with a CPU reference. For APSP the comparison runs only for N up to 64, and the shipped example has N = 256. To check APSP, translate a smaller description and pass it with `--mlir`.

```bash
printf 'workload: apsp\nN: 48\ntile_size: 16\n' > /tmp/apsp_small.yaml
python3 GenMLIR/tools/gendsl-translate.py /tmp/apsp_small.yaml > /tmp/apsp_small.mlir
python3 run_e2e.py --workload apsp --mlir /tmp/apsp_small.mlir --check
```

## Artifacts

Both implementations write three artifacts.

`layout.json` lists the memory objects with their tier, channel, bank-group and interleaving. The Python implementation adds the placement of every object across tiers and the occupancy of each tier.

`schedule.json` is the input of the simulator. The C++ implementation writes one event per tile computation, broadcast, queue operation and barrier. The Python implementation writes the symbolic form described above.

`pu_programs/` has one text file per PU for inspection. The simulator does not read it.

## Datasets

`datasets/Graphs` holds three edge lists. CA-GrQc (5,242 nodes) and p2p-Gnutella08 (6,301 nodes) come from the SNAP collection [SNAP'14 Datasets]. `osm_graph_65536_nodes.txt` has 65,536 nodes and 138,510 edges.

`datasets/Alignment` holds ten sets of simulated long reads, one per technology profile (`pacbio`, `ont2d`) and read length (2, 4, 6, 8 and 10 kbp). Each set has a FASTQ file with 172 reads, the 5.15 Mbp reference in FASTA format (`.ref`, the same sequence in all ten sets) and a MAF file with the true alignment of every read.

## Scope of the timing model

The simulator is a cost model. It charges every tile for its row-buffer operations, scaled by the latency of the tier that holds the data, and for its lane-parallel DP updates, and it resolves data dependencies and PU contention in a discrete-event executor. It is not a cycle-accurate DRAM model. `evaluation/README.md` lists the assumptions behind each experiment.

## License

MIT. See `LICENSE`.