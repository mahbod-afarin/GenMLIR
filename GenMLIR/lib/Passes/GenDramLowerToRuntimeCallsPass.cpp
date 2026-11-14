//===- GenDramLowerToRuntimeCallsPass.cpp -------------------------------===//
//
// Back-end pass: lower gendram dialect ops to runtime API calls.
//
// The ops become calls to the functions declared in runtime/gendram_runtime.h,
// which a simulator or trace runner can implement:
//
//   gendram.broadcast    -> call @gendram_broadcast(payload)
//   gendram.send         -> call @gendram_send(qid, payload)
//   gendram.recv         -> call @gendram_recv(qid)
//   gendram.barrier      -> call @gendram_barrier()
//   gendram.mem_transfer -> call @gendram_mem_transfer(src, dst)
//
// The runtime API takes payloads as unranked f32 memrefs, so that memrefs of
// any rank go through one signature. An op whose payload is not an f32 memref
// cannot be cast to that type and is left in place.
//
//===----------------------------------------------------------------------===//

#include "genmlir/Dialect/GenDRAM/GenDRAMOps.h"
#include "genmlir/Passes/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"

using namespace mlir;

namespace genmlir {
namespace {

#define GEN_PASS_DEF_GENDRAMLOWERTORUNTIMECALLS
#include "genmlir/Passes/Passes.h.inc"

static func::FuncOp getOrInsertRuntimeFn(ModuleOp module, StringRef name,
                                        FunctionType type) {
  if (auto fn = module.lookupSymbol<func::FuncOp>(name))
    return fn;

  OpBuilder builder(module.getBodyRegion());
  builder.setInsertionPointToStart(module.getBody());

  auto fn = builder.create<func::FuncOp>(module.getLoc(), name, type);
  fn.setPrivate();
  return fn;
}

static Value castToUnrankedMemRef(OpBuilder &builder, Location loc, Value v,
                                  UnrankedMemRefType unrankedTy) {
  // If already unranked, nothing to do.
  if (v.getType() == unrankedTy)
    return v;

  // memref.cast goes from ranked to unranked only if the element types match.
  auto rankedTy = dyn_cast<MemRefType>(v.getType());
  if (!rankedTy || rankedTy.getElementType() != unrankedTy.getElementType())
    return Value();

  return builder.create<memref::CastOp>(loc, unrankedTy, v).getResult();
}

struct GenDramLowerToRuntimeCallsPass
    : public impl::GenDramLowerToRuntimeCallsBase<GenDramLowerToRuntimeCallsPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    // The runtime API takes payloads as memref<*xf32>.
    Type f32 = Float32Type::get(ctx);
    auto unrankedF32 = UnrankedMemRefType::get(f32, /*memorySpace=*/0);

    // Declare runtime functions.
    auto fnBroadcast = getOrInsertRuntimeFn(
        module, "gendram_broadcast",
        FunctionType::get(ctx, {unrankedF32}, {}));

    auto fnSend = getOrInsertRuntimeFn(
        module, "gendram_send",
        FunctionType::get(ctx, {IntegerType::get(ctx, 64), unrankedF32}, {}));

    auto fnRecv = getOrInsertRuntimeFn(
        module, "gendram_recv",
        FunctionType::get(ctx, {IntegerType::get(ctx, 64)}, {}));

    auto fnBarrier = getOrInsertRuntimeFn(
        module, "gendram_barrier",
        FunctionType::get(ctx, {}, {}));

    auto fnMemTransfer = getOrInsertRuntimeFn(
        module, "gendram_mem_transfer",
        FunctionType::get(ctx, {unrankedF32, unrankedF32}, {}));

    // Walk all functions and rewrite gendram ops.
    module.walk([&](func::FuncOp fn) {
      for (Block &b : fn.getBody()) {
        for (Operation &op : llvm::make_early_inc_range(b.getOperations())) {
          OpBuilder builder(&op);

          if (auto bc = dyn_cast<genmlir::gendram::BroadcastOp>(&op)) {
            Value payload = bc.getPayload();
            Value casted = castToUnrankedMemRef(builder, bc.getLoc(), payload, unrankedF32);
            if (!casted)
              continue; // payload cannot be cast; keep the op

            builder.create<func::CallOp>(bc.getLoc(), fnBroadcast, ValueRange{casted});
            bc.erase();
            continue;
          }

          if (auto s = dyn_cast<genmlir::gendram::SendOp>(&op)) {
            Value payload = s.getPayload();
            Value casted = castToUnrankedMemRef(builder, s.getLoc(), payload, unrankedF32);
            if (!casted)
              continue;

            int64_t qid = s.getQid();
            Value qidC = builder.create<arith::ConstantIntOp>(s.getLoc(), qid, 64);
            builder.create<func::CallOp>(s.getLoc(), fnSend, ValueRange{qidC, casted});
            s.erase();
            continue;
          }

          if (auto r = dyn_cast<genmlir::gendram::RecvOp>(&op)) {
            int64_t qid = r.getQid();
            Value qidC = builder.create<arith::ConstantIntOp>(r.getLoc(), qid, 64);
            builder.create<func::CallOp>(r.getLoc(), fnRecv, ValueRange{qidC});
            r.erase();
            continue;
          }

          if (auto bar = dyn_cast<genmlir::gendram::BarrierOp>(&op)) {
            builder.create<func::CallOp>(bar.getLoc(), fnBarrier, ValueRange{});
            bar.erase();
            continue;
          }

          if (auto mt = dyn_cast<genmlir::gendram::MemTransferOp>(&op)) {
            Value src = castToUnrankedMemRef(builder, mt.getLoc(), mt.getSrc(), unrankedF32);
            Value dst = castToUnrankedMemRef(builder, mt.getLoc(), mt.getDst(), unrankedF32);
            if (!src || !dst)
              continue;

            builder.create<func::CallOp>(mt.getLoc(), fnMemTransfer, ValueRange{src, dst});
            mt.erase();
            continue;
          }
        }
      }
    });
  }
};

} // namespace

std::unique_ptr<Pass> createGenDramLowerToRuntimeCalls() {
  return std::make_unique<GenDramLowerToRuntimeCallsPass>();
}

} // namespace genmlir
