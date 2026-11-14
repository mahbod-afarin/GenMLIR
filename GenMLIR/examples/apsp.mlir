module attributes { gendram.workload = "apsp", gendram.tile_size = 64 : i64, gendram.N = 256 : i64 } {
  memref.global "private" @D : memref<256x256xi32> = uninitialized
  memref.global "private" @PTR : memref<1024xi32> = uninitialized
  memref.global "private" @CAL : memref<4096xi32> = uninitialized

  func.func @main() {
    %D = memref.get_global @D : memref<256x256xi32>

    dp.phase attributes { kind = "pivot" } {
      dp.tile attributes { ti = 0 : i64, tj = 0 : i64, tk = 0 : i64 } {
        dp.yield
      }
      dp.yield
    }

    dp.phase attributes { kind = "rowcol" } {
      dp.tile attributes { ti = 0 : i64, tj = 1 : i64, tk = 0 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 0 : i64, tj = 2 : i64, tk = 0 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 0 : i64, tj = 3 : i64, tk = 0 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 1 : i64, tj = 0 : i64, tk = 0 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 2 : i64, tj = 0 : i64, tk = 0 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 3 : i64, tj = 0 : i64, tk = 0 : i64 } {
        dp.yield
      }
      dp.yield
    }

    dp.phase attributes { kind = "internal" } {
      dp.tile attributes { ti = 1 : i64, tj = 1 : i64, tk = 0 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 1 : i64, tj = 2 : i64, tk = 0 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 1 : i64, tj = 3 : i64, tk = 0 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 2 : i64, tj = 1 : i64, tk = 0 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 2 : i64, tj = 2 : i64, tk = 0 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 2 : i64, tj = 3 : i64, tk = 0 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 3 : i64, tj = 1 : i64, tk = 0 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 3 : i64, tj = 2 : i64, tk = 0 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 3 : i64, tj = 3 : i64, tk = 0 : i64 } {
        dp.yield
      }
      dp.yield
    }

    dp.phase attributes { kind = "pivot" } {
      dp.tile attributes { ti = 1 : i64, tj = 1 : i64, tk = 1 : i64 } {
        dp.yield
      }
      dp.yield
    }

    dp.phase attributes { kind = "rowcol" } {
      dp.tile attributes { ti = 1 : i64, tj = 0 : i64, tk = 1 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 1 : i64, tj = 2 : i64, tk = 1 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 1 : i64, tj = 3 : i64, tk = 1 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 0 : i64, tj = 1 : i64, tk = 1 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 2 : i64, tj = 1 : i64, tk = 1 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 3 : i64, tj = 1 : i64, tk = 1 : i64 } {
        dp.yield
      }
      dp.yield
    }

    dp.phase attributes { kind = "internal" } {
      dp.tile attributes { ti = 0 : i64, tj = 0 : i64, tk = 1 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 0 : i64, tj = 2 : i64, tk = 1 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 0 : i64, tj = 3 : i64, tk = 1 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 2 : i64, tj = 0 : i64, tk = 1 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 2 : i64, tj = 2 : i64, tk = 1 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 2 : i64, tj = 3 : i64, tk = 1 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 3 : i64, tj = 0 : i64, tk = 1 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 3 : i64, tj = 2 : i64, tk = 1 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 3 : i64, tj = 3 : i64, tk = 1 : i64 } {
        dp.yield
      }
      dp.yield
    }

    dp.phase attributes { kind = "pivot" } {
      dp.tile attributes { ti = 2 : i64, tj = 2 : i64, tk = 2 : i64 } {
        dp.yield
      }
      dp.yield
    }

    dp.phase attributes { kind = "rowcol" } {
      dp.tile attributes { ti = 2 : i64, tj = 0 : i64, tk = 2 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 2 : i64, tj = 1 : i64, tk = 2 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 2 : i64, tj = 3 : i64, tk = 2 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 0 : i64, tj = 2 : i64, tk = 2 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 1 : i64, tj = 2 : i64, tk = 2 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 3 : i64, tj = 2 : i64, tk = 2 : i64 } {
        dp.yield
      }
      dp.yield
    }

    dp.phase attributes { kind = "internal" } {
      dp.tile attributes { ti = 0 : i64, tj = 0 : i64, tk = 2 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 0 : i64, tj = 1 : i64, tk = 2 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 0 : i64, tj = 3 : i64, tk = 2 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 1 : i64, tj = 0 : i64, tk = 2 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 1 : i64, tj = 1 : i64, tk = 2 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 1 : i64, tj = 3 : i64, tk = 2 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 3 : i64, tj = 0 : i64, tk = 2 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 3 : i64, tj = 1 : i64, tk = 2 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 3 : i64, tj = 3 : i64, tk = 2 : i64 } {
        dp.yield
      }
      dp.yield
    }

    dp.phase attributes { kind = "pivot" } {
      dp.tile attributes { ti = 3 : i64, tj = 3 : i64, tk = 3 : i64 } {
        dp.yield
      }
      dp.yield
    }

    dp.phase attributes { kind = "rowcol" } {
      dp.tile attributes { ti = 3 : i64, tj = 0 : i64, tk = 3 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 3 : i64, tj = 1 : i64, tk = 3 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 3 : i64, tj = 2 : i64, tk = 3 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 0 : i64, tj = 3 : i64, tk = 3 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 1 : i64, tj = 3 : i64, tk = 3 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 2 : i64, tj = 3 : i64, tk = 3 : i64 } {
        dp.yield
      }
      dp.yield
    }

    dp.phase attributes { kind = "internal" } {
      dp.tile attributes { ti = 0 : i64, tj = 0 : i64, tk = 3 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 0 : i64, tj = 1 : i64, tk = 3 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 0 : i64, tj = 2 : i64, tk = 3 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 1 : i64, tj = 0 : i64, tk = 3 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 1 : i64, tj = 1 : i64, tk = 3 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 1 : i64, tj = 2 : i64, tk = 3 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 2 : i64, tj = 0 : i64, tk = 3 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 2 : i64, tj = 1 : i64, tk = 3 : i64 } {
        dp.yield
      }
      dp.tile attributes { ti = 2 : i64, tj = 2 : i64, tk = 3 : i64 } {
        dp.yield
      }
      dp.yield
    }

    func.return
  }
}
