module attributes { gendram.workload = "alignment", gendram.tile_size = 32 : i64, gendram.query_len = 128 : i64, gendram.ref_len = 160 : i64, gendram.algo = "smith_waterman", gendram.match = 2 : i64, gendram.mismatch = -1 : i64, gendram.gap = -1 : i64, gendram.band_width = 0 : i64 } {
  memref.global "private" @PTR : memref<1024xi32> = uninitialized
  memref.global "private" @CAL : memref<4096xi32> = uninitialized
  memref.global "private" @candidates : memref<1024x4xi32> = uninitialized
  memref.global "private" @Qseq : memref<128xi8> = uninitialized
  memref.global "private" @Rseq : memref<160xi8> = uninitialized
  memref.global "private" @H : memref<129x161xi32> = uninitialized

  func.func @seeding() {
    %ptr = memref.get_global @PTR : memref<1024xi32>
    %cal = memref.get_global @CAL : memref<4096xi32>
    %cnd = memref.get_global @candidates : memref<1024x4xi32>
    // The PTR -> CAL lookups of the seeding stage are not modeled.
    func.return
  }

  func.func @alignment() {
    dp.wavefront {
      dp.tile attributes { ti = 0 : i64, tj = 0 : i64 } {
        %cur0 = arith.constant 0.0 : f32
        %cand0 = arith.constant 1.0 : f32
        %u0 = dp.grid_update %cur0, %cand0 { semiring = "max_plus" } : f32
        dp.yield
      }
      dp.tile attributes { ti = 0 : i64, tj = 1 : i64 } {
        %cur0 = arith.constant 0.0 : f32
        %cand0 = arith.constant 1.0 : f32
        %u0 = dp.grid_update %cur0, %cand0 { semiring = "max_plus" } : f32
        dp.yield
      }
      dp.tile attributes { ti = 1 : i64, tj = 0 : i64 } {
        %cur0 = arith.constant 0.0 : f32
        %cand0 = arith.constant 1.0 : f32
        %u0 = dp.grid_update %cur0, %cand0 { semiring = "max_plus" } : f32
        dp.yield
      }
      dp.tile attributes { ti = 0 : i64, tj = 2 : i64 } {
        %cur0 = arith.constant 0.0 : f32
        %cand0 = arith.constant 1.0 : f32
        %u0 = dp.grid_update %cur0, %cand0 { semiring = "max_plus" } : f32
        dp.yield
      }
      dp.tile attributes { ti = 1 : i64, tj = 1 : i64 } {
        %cur0 = arith.constant 0.0 : f32
        %cand0 = arith.constant 1.0 : f32
        %u0 = dp.grid_update %cur0, %cand0 { semiring = "max_plus" } : f32
        dp.yield
      }
      dp.tile attributes { ti = 2 : i64, tj = 0 : i64 } {
        %cur0 = arith.constant 0.0 : f32
        %cand0 = arith.constant 1.0 : f32
        %u0 = dp.grid_update %cur0, %cand0 { semiring = "max_plus" } : f32
        dp.yield
      }
      dp.tile attributes { ti = 0 : i64, tj = 3 : i64 } {
        %cur0 = arith.constant 0.0 : f32
        %cand0 = arith.constant 1.0 : f32
        %u0 = dp.grid_update %cur0, %cand0 { semiring = "max_plus" } : f32
        dp.yield
      }
      dp.tile attributes { ti = 1 : i64, tj = 2 : i64 } {
        %cur0 = arith.constant 0.0 : f32
        %cand0 = arith.constant 1.0 : f32
        %u0 = dp.grid_update %cur0, %cand0 { semiring = "max_plus" } : f32
        dp.yield
      }
      dp.tile attributes { ti = 2 : i64, tj = 1 : i64 } {
        %cur0 = arith.constant 0.0 : f32
        %cand0 = arith.constant 1.0 : f32
        %u0 = dp.grid_update %cur0, %cand0 { semiring = "max_plus" } : f32
        dp.yield
      }
      dp.tile attributes { ti = 3 : i64, tj = 0 : i64 } {
        %cur0 = arith.constant 0.0 : f32
        %cand0 = arith.constant 1.0 : f32
        %u0 = dp.grid_update %cur0, %cand0 { semiring = "max_plus" } : f32
        dp.yield
      }
      dp.tile attributes { ti = 0 : i64, tj = 4 : i64 } {
        %cur0 = arith.constant 0.0 : f32
        %cand0 = arith.constant 1.0 : f32
        %u0 = dp.grid_update %cur0, %cand0 { semiring = "max_plus" } : f32
        dp.yield
      }
      dp.tile attributes { ti = 1 : i64, tj = 3 : i64 } {
        %cur0 = arith.constant 0.0 : f32
        %cand0 = arith.constant 1.0 : f32
        %u0 = dp.grid_update %cur0, %cand0 { semiring = "max_plus" } : f32
        dp.yield
      }
      dp.tile attributes { ti = 2 : i64, tj = 2 : i64 } {
        %cur0 = arith.constant 0.0 : f32
        %cand0 = arith.constant 1.0 : f32
        %u0 = dp.grid_update %cur0, %cand0 { semiring = "max_plus" } : f32
        dp.yield
      }
      dp.tile attributes { ti = 3 : i64, tj = 1 : i64 } {
        %cur0 = arith.constant 0.0 : f32
        %cand0 = arith.constant 1.0 : f32
        %u0 = dp.grid_update %cur0, %cand0 { semiring = "max_plus" } : f32
        dp.yield
      }
      dp.tile attributes { ti = 4 : i64, tj = 0 : i64 } {
        %cur0 = arith.constant 0.0 : f32
        %cand0 = arith.constant 1.0 : f32
        %u0 = dp.grid_update %cur0, %cand0 { semiring = "max_plus" } : f32
        dp.yield
      }
      dp.tile attributes { ti = 0 : i64, tj = 5 : i64 } {
        %cur0 = arith.constant 0.0 : f32
        %cand0 = arith.constant 1.0 : f32
        %u0 = dp.grid_update %cur0, %cand0 { semiring = "max_plus" } : f32
        dp.yield
      }
      dp.tile attributes { ti = 1 : i64, tj = 4 : i64 } {
        %cur0 = arith.constant 0.0 : f32
        %cand0 = arith.constant 1.0 : f32
        %u0 = dp.grid_update %cur0, %cand0 { semiring = "max_plus" } : f32
        dp.yield
      }
      dp.tile attributes { ti = 2 : i64, tj = 3 : i64 } {
        %cur0 = arith.constant 0.0 : f32
        %cand0 = arith.constant 1.0 : f32
        %u0 = dp.grid_update %cur0, %cand0 { semiring = "max_plus" } : f32
        dp.yield
      }
      dp.tile attributes { ti = 3 : i64, tj = 2 : i64 } {
        %cur0 = arith.constant 0.0 : f32
        %cand0 = arith.constant 1.0 : f32
        %u0 = dp.grid_update %cur0, %cand0 { semiring = "max_plus" } : f32
        dp.yield
      }
      dp.tile attributes { ti = 4 : i64, tj = 1 : i64 } {
        %cur0 = arith.constant 0.0 : f32
        %cand0 = arith.constant 1.0 : f32
        %u0 = dp.grid_update %cur0, %cand0 { semiring = "max_plus" } : f32
        dp.yield
      }
      dp.tile attributes { ti = 1 : i64, tj = 5 : i64 } {
        %cur0 = arith.constant 0.0 : f32
        %cand0 = arith.constant 1.0 : f32
        %u0 = dp.grid_update %cur0, %cand0 { semiring = "max_plus" } : f32
        dp.yield
      }
      dp.tile attributes { ti = 2 : i64, tj = 4 : i64 } {
        %cur0 = arith.constant 0.0 : f32
        %cand0 = arith.constant 1.0 : f32
        %u0 = dp.grid_update %cur0, %cand0 { semiring = "max_plus" } : f32
        dp.yield
      }
      dp.tile attributes { ti = 3 : i64, tj = 3 : i64 } {
        %cur0 = arith.constant 0.0 : f32
        %cand0 = arith.constant 1.0 : f32
        %u0 = dp.grid_update %cur0, %cand0 { semiring = "max_plus" } : f32
        dp.yield
      }
      dp.tile attributes { ti = 4 : i64, tj = 2 : i64 } {
        %cur0 = arith.constant 0.0 : f32
        %cand0 = arith.constant 1.0 : f32
        %u0 = dp.grid_update %cur0, %cand0 { semiring = "max_plus" } : f32
        dp.yield
      }
      dp.tile attributes { ti = 2 : i64, tj = 5 : i64 } {
        %cur0 = arith.constant 0.0 : f32
        %cand0 = arith.constant 1.0 : f32
        %u0 = dp.grid_update %cur0, %cand0 { semiring = "max_plus" } : f32
        dp.yield
      }
      dp.tile attributes { ti = 3 : i64, tj = 4 : i64 } {
        %cur0 = arith.constant 0.0 : f32
        %cand0 = arith.constant 1.0 : f32
        %u0 = dp.grid_update %cur0, %cand0 { semiring = "max_plus" } : f32
        dp.yield
      }
      dp.tile attributes { ti = 4 : i64, tj = 3 : i64 } {
        %cur0 = arith.constant 0.0 : f32
        %cand0 = arith.constant 1.0 : f32
        %u0 = dp.grid_update %cur0, %cand0 { semiring = "max_plus" } : f32
        dp.yield
      }
      dp.tile attributes { ti = 3 : i64, tj = 5 : i64 } {
        %cur0 = arith.constant 0.0 : f32
        %cand0 = arith.constant 1.0 : f32
        %u0 = dp.grid_update %cur0, %cand0 { semiring = "max_plus" } : f32
        dp.yield
      }
      dp.tile attributes { ti = 4 : i64, tj = 4 : i64 } {
        %cur0 = arith.constant 0.0 : f32
        %cand0 = arith.constant 1.0 : f32
        %u0 = dp.grid_update %cur0, %cand0 { semiring = "max_plus" } : f32
        dp.yield
      }
      dp.tile attributes { ti = 4 : i64, tj = 5 : i64 } {
        %cur0 = arith.constant 0.0 : f32
        %cand0 = arith.constant 1.0 : f32
        %u0 = dp.grid_update %cur0, %cand0 { semiring = "max_plus" } : f32
        dp.yield
      }
      dp.yield
    }
    func.return
  }
}
