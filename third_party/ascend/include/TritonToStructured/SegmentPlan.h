// Segmented structured memory accesses for incomplete div/rem periods.
#ifndef TRITON_TO_STRUCTURED_SEGMENT_PLAN_H
#define TRITON_TO_STRUCTURED_SEGMENT_PLAN_H
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
namespace TritonToStructured {
// Failure means unsupported or already handled; no IR is changed on failure.
// Generated accesses contain no tensor div/rem and therefore cannot re-enter.
mlir::LogicalResult tryRewriteSegmentedLoad(
    mlir::triton::LoadOp op, mlir::PatternRewriter &rewriter,
    bool optimizeDynamicOffset, mlir::Value *rewrittenMask = nullptr,
    bool allowAlignedFallback = false);
mlir::LogicalResult tryRewriteSegmentedStore(mlir::triton::StoreOp op,
                                             mlir::PatternRewriter &rewriter,
                                             bool optimizeDynamicOffset,
                                             bool allowAlignedFallback = false);
} // namespace TritonToStructured
#endif
