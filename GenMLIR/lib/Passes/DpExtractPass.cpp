//===- DpExtractPass.cpp ------------------------------------------------===//
//
// Group 1 pass: DP recognition.
//
// Rewrites the compare-and-select form of a DP update into dp.grid_update, so
// that later passes match one op instead of arith.cmp/arith.select sequences.
//
// Matched pattern:
//   updated = select(cmp(candidate, current), candidate, current)
//     - min_plus if the comparison is "candidate < current"
//     - max_plus if the comparison is "candidate > current"
//
// The arith.select is replaced by dp.grid_update(current, candidate). The
// store that consumes the result is left as it is.
//
//===----------------------------------------------------------------------===//

#include "genmlir/Dialect/DPGrid/DPGridOps.h"
#include "genmlir/Passes/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace genmlir {
namespace {

#define GEN_PASS_DEF_DPEXTRACT
#include "genmlir/Passes/Passes.h.inc"

/// Pattern that rewrites arith.select(cmp(candidate, current), candidate, current)
/// into dp.grid_update(current, candidate) with the right semiring attribute.
struct SelectToGridUpdate : public OpRewritePattern<arith::SelectOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SelectOp sel,
                                PatternRewriter &rewriter) const override {
    // We only handle selects where the true/false values are the compared values.
    Value trueV = sel.getTrueValue();
    Value falseV = sel.getFalseValue();

    // Match floating-point comparisons.
    if (auto cmpf = sel.getCondition().getDefiningOp<arith::CmpFOp>()) {
      Value lhs = cmpf.getLhs();
      Value rhs = cmpf.getRhs();

      // We want select(cmp(candidate, current), candidate, current).
      // Here, candidate = lhs, current = rhs in our convention.
      if (trueV != lhs || falseV != rhs)
        return failure();

      auto pred = cmpf.getPredicate();
      StringRef semiring;

      // candidate < current  ==> min_plus update
      if (pred == arith::CmpFPredicate::OLT || pred == arith::CmpFPredicate::ULT)
        semiring = "min_plus";
      // candidate > current  ==> max_plus update
      else if (pred == arith::CmpFPredicate::OGT ||
               pred == arith::CmpFPredicate::UGT)
        semiring = "max_plus";
      else
        return failure();

      auto upd = rewriter.create<genmlir::dp::GridUpdateOp>(
          sel.getLoc(), rhs.getType(), /*current=*/rhs, /*candidate=*/lhs,
          rewriter.getStringAttr(semiring));
      rewriter.replaceOp(sel, upd.getResult());
      return success();
    }

    // Match integer comparisons.
    if (auto cmpi = sel.getCondition().getDefiningOp<arith::CmpIOp>()) {
      Value lhs = cmpi.getLhs();
      Value rhs = cmpi.getRhs();
      if (trueV != lhs || falseV != rhs)
        return failure();

      auto pred = cmpi.getPredicate();
      StringRef semiring;

      if (pred == arith::CmpIPredicate::slt || pred == arith::CmpIPredicate::ult)
        semiring = "min_plus";
      else if (pred == arith::CmpIPredicate::sgt || pred == arith::CmpIPredicate::ugt)
        semiring = "max_plus";
      else
        return failure();

      auto upd = rewriter.create<genmlir::dp::GridUpdateOp>(
          sel.getLoc(), rhs.getType(), /*current=*/rhs, /*candidate=*/lhs,
          rewriter.getStringAttr(semiring));
      rewriter.replaceOp(sel, upd.getResult());
      return success();
    }

    return failure();
  }
};

struct DpExtractPass : public impl::DpExtractBase<DpExtractPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<SelectToGridUpdate>(ctx);

    // Apply patterns greedily. This is safe here because dp.grid_update is pure.
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createDpExtract() {
  return std::make_unique<DpExtractPass>();
}

} // namespace genmlir
