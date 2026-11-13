// Lift one-axis segmented memory rewrites through pure broadcasting.
#ifndef TRITON_TO_STRUCTURED_SEGMENT_LIFT_H
#define TRITON_TO_STRUCTURED_SEGMENT_LIFT_H
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
namespace TritonToStructured {
// Unsupported/aligned inputs leave the IR unchanged. Loads may enumerate a
// tiny outer axis within an eight-leaf budget. Repeated stores with nonuniform
// values are deliberately not collapsed.
mlir::LogicalResult tryRewriteSegmentedBroadcastLoad(
    mlir::triton::LoadOp op, mlir::PatternRewriter &rewriter,
    bool optimizeDynamicOffset);
mlir::LogicalResult tryRewriteSegmentedBroadcastStore(
    mlir::triton::StoreOp op, mlir::PatternRewriter &rewriter,
    bool optimizeDynamicOffset);
}
#endif
