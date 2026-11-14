"""Architecture and timing parameters of the GenDRAM command-stream simulator."""


class CfgArchitecture:
    """Hardware parameters. schedule.json may override the PU counts and clock."""

    LOGIC_CLK_FREQUENCY_HZ = 1e9
    CYCLE_TIME_S = 1 / LOGIC_CLK_FREQUENCY_HZ

    # Broadcast network bandwidth, converted to bytes per logic cycle.
    BW_BROADCAST_TB_S = 1.0
    BW_BROADCAST_B_PER_CYCLE = (BW_BROADCAST_TB_S * 1e12) * CYCLE_TIME_S

    # One PU per bank-group: 8 Search PUs and 24 Compute PUs.
    NUM_PUS_TOTAL = 32
    NUM_SEARCH_PUS = 8
    NUM_COMPUTE_PUS = 24


class CfgGenDRAMSim:
    """Per-operation costs in logic cycles.

    These describe a simple cost model for the command stream, not
    cycle-accurate DRAM timing.
    """

    # Row-buffer operations.
    DRAM_ACTIVATE_CYCLES = 12
    DRAM_READ_ROWBUF_CYCLES = 16
    DRAM_WRITE_ROWBUF_CYCLES = 16
    DRAM_PRECHARGE_CYCLES = 12

    # PU execution.
    PE_LANES_PER_PU = 16
    CYCLES_PER_DP_UPDATE = 1

    # Synchronization and messaging. The per-diagonal barrier of the
    # alignment wavefront is cheaper than a full barrier.
    WAVEFRONT_BARRIER_CYCLES = 4
    BARRIER_CYCLES = 8
    SEND_RECV_CYCLES = 32
    MEM_TRANSFER_CYCLES = 64

    # Element size assumed when a schedule event gives no payload size.
    DEFAULT_ELEM_BYTES = 4
