# GenDRAM command-stream simulator

`gendram_command_sim.py` takes the artifacts emitted by GenMLIR and reports the modeled cycle count and time of the schedule on the GenDRAM accelerator [arXiv'26 GenDRAM]. It can also execute the workload, so that the result of a schedule can be compared with a CPU reference.

## Usage

```python
from GenDRAM.gendram_command_sim import simulate_from_artifacts

res = simulate_from_artifacts("out/apsp", verbose=False, functional=False, mode="execute")
print(res.total_cycles, res.total_time_s)
```

```bash
python3 GenDRAM/gendram_command_sim.py out/apsp
```

The directory must contain `layout.json` and `schedule.json`. The result holds the total cycles, the time at the clock frequency of the schedule, the cycles per PU and the cycles per event kind.

## Schedules and timing modes

An explicit schedule, as written by `genmlir-opt`, lists every tile computation, broadcast, queue operation and barrier. It is replayed event by event. Every PU has its own timeline, and communication and barrier events synchronize all PUs.

A symbolic schedule, as written by the Python compiler, gives shapes, tiling, the mapping rule and model parameters. The simulator derives the tile order from them and offers two timing models.

The closed-form model (`mode="auto"` or `"analytical"`) sums the cost of the phases. For APSP it spreads the row/column tiles and the internal tiles of each block index evenly over the PUs. For alignment it balances the tiles of every wavefront diagonal over the Compute PUs.

The executor (`mode="execute"`) generates the tile stream and starts every tile once its inputs are available and its PU is free. For APSP a tile waits for its previous version and for the pivot, row and column tiles it reads, and runs on the PU given by the modulo mapping. For alignment a tile waits for its upper, left and diagonal neighbours and is placed on the Compute PU that becomes free first. The reported cycle count is the resulting critical path.

## Cost model

A tile costs its row-buffer operations plus its DP updates. A row read or write is an activate, a row-buffer access and a precharge. The row-buffer cost is scaled by the row-cycle time of the tier that holds the object relative to the slowest tier, using the M3D DRAM timing of GenDRAM. DP updates run on the parallel lanes of a PU. The parameters are in `config.py`.

| Parameter | Value |
|---|---|
| activate, precharge | 12 cycles each |
| row-buffer read, write | 16 cycles each |
| DP update | 1 cycle per lane |
| lanes per PU | 16 |
| barrier | 8 cycles |
| wavefront barrier | 4 cycles |
| send, recv | 32 cycles plus payload transfer |
| memory transfer | 64 cycles |
| broadcast bandwidth | 1 TB/s (1000 bytes per cycle at 1 GHz) |
| clock | 1 GHz |

This is a cost model of the command stream. It is not a cycle-accurate DRAM model, and it does not model energy or area.

## Functional execution

With `functional=True` the simulator executes the workload if its inputs are in the artifact directory.

| Workload | Inputs | Outputs |
|---|---|---|
| APSP | `input_D.npy` | `output_D.npy` |
| alignment, one read | `input_query.txt`, `input_ref.txt` | `output_H.npy`, `output_score.txt` |
| alignment, batch | `input_queries.txt`, `input_ref.txt` | `output_H.npy` (first read), `output_scores.json` |

APSP runs blocked Floyd-Warshall in the tile order of the schedule. Alignment runs Smith-Waterman tile by tile in wavefront order, banded if the schedule has a positive band width. The seeding stage of the genomics pipeline is not executed.
