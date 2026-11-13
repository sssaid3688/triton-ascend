// Project a purely broadcast memory access onto its single active axis,
// run the one-dimensional segmented rewrite transactionally, then restore the
// original shape. No speculative operations enter the caller on failure.
#include "TritonToStructured/SegmentLift.h"
#include "TritonToStructured/SegmentPlan.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"
#include <optional>
#include <type_traits>

using namespace mlir;
namespace TritonToStructured {
namespace {

bool projectableElementwise(Operation *op) {
  return isa<triton::AddPtrOp, arith::AddIOp, arith::SubIOp,
             arith::MulIOp, arith::DivSIOp, arith::RemSIOp,
             arith::CmpIOp, arith::AndIOp, arith::ExtSIOp,
             arith::IndexCastOp, arith::AddFOp, arith::SubFOp,
             arith::MulFOp, arith::ExtFOp, arith::TruncFOp>(op);
}

// -1 denotes a value constant along every axis. Other values denote the sole
// varying axis in that value's own shape. Absence means nonprojectable.
struct Projection {
  int64_t extent;
  int64_t pinnedExtent = 0, pinnedIndex = 0;
  explicit Projection(int64_t extent, int64_t pinnedExtent = 0,
                      int64_t pinnedIndex = 0)
      : extent(extent), pinnedExtent(pinnedExtent), pinnedIndex(pinnedIndex) {}
  llvm::DenseMap<Value, std::optional<int64_t>> axes;
  llvm::DenseMap<Value, Value> projected;

  std::optional<int64_t> axis(Value value) {
    if (!value)
      return std::nullopt;
    if (auto it = axes.find(value); it != axes.end())
      return it->second;
    auto result = analyze(value);
    axes[value] = result;
    return result;
  }

  std::optional<int64_t> analyze(Value value) {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    if (!type)
      return -1;
    if (!type.hasStaticShape() || type.getEncoding())
      return std::nullopt;
    if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
      auto dense = dyn_cast<DenseElementsAttr>(constant.getValue());
      if (dense && dense.isSplat())
        return -1;
      if (type.getRank() == 1 && type.getDimSize(0) == extent)
        return 0;
      return std::nullopt;
    }
    if (value.getDefiningOp<triton::SplatOp>())
      return -1;
    // A pre-existing vector is an atomic projection leaf regardless of how it
    // was computed. Pure expand/broadcast paths prove equality on removed axes.
    if (type.getRank() == 1 && type.getDimSize(0) == extent)
      return 0;
    if (auto range = value.getDefiningOp<triton::MakeRangeOp>()) {
      if (type.getRank() != 1)
        return std::nullopt;
      if (type.getDimSize(0) == extent)
        return 0;
      if (pinnedExtent && type.getDimSize(0) == pinnedExtent)
        return -1;
      return std::nullopt;
    }
    if (auto expand = value.getDefiningOp<triton::ExpandDimsOp>()) {
      auto a = axis(expand.getSrc());
      if (!a || *a < 0)
        return a;
      return *a >= expand.getAxis() ? *a + 1 : *a;
    }
    if (auto broadcast = value.getDefiningOp<triton::BroadcastOp>()) {
      auto a = axis(broadcast.getSrc());
      if (!a || *a < 0)
        return a;
      if (type.getDimSize(*a) != extent)
        return std::nullopt;
      return a;
    }
    if (auto trans = value.getDefiningOp<triton::TransOp>()) {
      auto a = axis(trans.getSrc());
      if (!a || *a < 0)
        return a;
      for (auto [i, inputAxis] : llvm::enumerate(trans.getOrder()))
        if (inputAxis == *a)
          return i;
      return std::nullopt;
    }
    auto *op = value.getDefiningOp();
    if (!op) {
      if (type.getRank() == 1 && type.getDimSize(0) == extent)
        return 0;
      return std::nullopt;
    }
    if (!projectableElementwise(op) || op->getNumResults() != 1)
      return std::nullopt;
    int64_t active = -1;
    for (Value operand : op->getOperands()) {
      auto a = axis(operand);
      if (!a || (*a >= 0 && active >= 0 && *a != active))
        return std::nullopt;
      if (*a >= 0)
        active = *a;
    }
    if (active >= 0 && type.getDimSize(active) != extent)
      return std::nullopt;
    return active;
  }

  Value project(Value value, PatternRewriter &rewriter) {
    if (!value)
      return {};
    auto type = dyn_cast<RankedTensorType>(value.getType());
    if (!type)
      return value;
    if (auto it = projected.find(value); it != projected.end())
      return it->second;
    if (!axis(value))
      return {};
    if (type.getRank() == 1 && type.getDimSize(0) == extent)
      return value;
    auto target = RankedTensorType::get({extent}, type.getElementType());
    auto *op = value.getDefiningOp();
    Value result;
    if (auto expand = dyn_cast_or_null<triton::ExpandDimsOp>(op))
      result = project(expand.getSrc(), rewriter);
    else if (auto broadcast = dyn_cast_or_null<triton::BroadcastOp>(op))
      result = project(broadcast.getSrc(), rewriter);
    else if (auto trans = dyn_cast_or_null<triton::TransOp>(op))
      result = project(trans.getSrc(), rewriter);
    else if (auto splat = dyn_cast_or_null<triton::SplatOp>(op))
      result = rewriter.create<triton::SplatOp>(op->getLoc(), target,
                                                splat.getSrc());
    else if (auto range = dyn_cast_or_null<triton::MakeRangeOp>(op)) {
      auto attr = DenseElementsAttr::get(target,
          rewriter.getI32IntegerAttr(range.getStart() + pinnedIndex));
      result = rewriter.create<arith::ConstantOp>(op->getLoc(), attr);
    } else if (auto constant = dyn_cast_or_null<arith::ConstantOp>(op)) {
      auto dense = cast<DenseElementsAttr>(constant.getValue());
      auto attr = DenseElementsAttr::get(target, dense.getSplatValue<Attribute>());
      result = rewriter.create<arith::ConstantOp>(op->getLoc(), attr);
    } else {
      IRMapping mapping;
      for (Value operand : op->getOperands()) {
        Value mapped = project(operand, rewriter);
        if (!mapped)
          return {};
        mapping.map(operand, mapped);
      }
      Operation *clone = rewriter.clone(*op, mapping);
      clone->getResult(0).setType(target);
      // Per-axis frontend hints belong to the old rank; do not misapply them
      // to the projected vector.
      clone->removeAttr("tt.divisibility");
      clone->removeAttr("tt.contiguity");
      clone->removeAttr("tt.constancy");
      SmallVector<Value> folded;
      if (succeeded(rewriter.tryFold(clone, folded)) && folded.size() == 1) {
        result = folded.front();
        rewriter.eraseOp(clone);
      } else {
        result = clone->getResult(0);
      }
    }
    projected[value] = result;
    return result;
  }
};

Value restore(Value vector, RankedTensorType type, int64_t active,
              PatternRewriter &rewriter, Location loc) {
  Value expanded = vector;
  for (int64_t i = 0; i < type.getRank(); ++i)
    if (i != active)
      expanded = rewriter.create<triton::ExpandDimsOp>(loc, expanded, i);
  auto output = RankedTensorType::get(type.getShape(),
      cast<RankedTensorType>(vector.getType()).getElementType());
  if (expanded.getType() == output)
    return expanded;
  return rewriter.create<triton::BroadcastOp>(loc, output, expanded);
}

template <typename Op>
LogicalResult lift(Op op, PatternRewriter &rewriter, bool optimize) {
  constexpr bool isLoad = std::is_same_v<Op, triton::LoadOp>;
  auto ptrType = dyn_cast<RankedTensorType>(op.getPtr().getType());
  if (!ptrType || !ptrType.hasStaticShape() || ptrType.getEncoding() ||
      ptrType.getRank() < 2 || ptrType.getRank() > 3)
    return failure();
  if constexpr (isLoad)
    if (op.getIsVolatile())
      return failure();

  // Determine the active axis without constructing any operations. Try each
  // final axis as the one-dimensional extent; the dependency analysis rejects
  // genuinely independent varying axes even when their extents are equal.
  for (int64_t active = 0; active < ptrType.getRank(); ++active) {
    int64_t extent = ptrType.getDimSize(active);
    if (extent <= 0)
      continue;
    Projection projection(extent);
    auto ptrAxis = projection.axis(op.getPtr());
    if (!ptrAxis || (*ptrAxis >= 0 && *ptrAxis != active))
      continue;
    auto maskAxis = op.getMask() ? projection.axis(op.getMask())
                                 : std::optional<int64_t>(-1);
    if (!maskAxis || (*maskAxis >= 0 && *maskAxis != active))
      continue;
    if (*ptrAxis < 0 && *maskAxis < 0)
      continue;

    bool fullOther = false;
    if constexpr (isLoad) {
      if (op.getOther()) {
        auto otherAxis = projection.axis(op.getOther());
        fullOther = !otherAxis || (*otherAxis >= 0 && *otherAxis != active);
      }
    } else {
      auto valueAxis = projection.axis(op.getValue());
      // Removing repeated stores is valid only when both their values and
      // predicates are identical along every axis being removed.
      if (!valueAxis || (*valueAxis >= 0 && *valueAxis != active))
        continue;
    }

    Block scratch;
    PatternRewriter local(rewriter.getContext());
    local.setInsertionPointToEnd(&scratch);
    Value ptr = projection.project(op.getPtr(), local);
    Value mask = op.getMask() ? projection.project(op.getMask(), local) : Value();
    if (!ptr || (op.getMask() && !mask))
      continue;
    if constexpr (isLoad) {
      Value other = op.getOther() && !fullOther
                        ? projection.project(op.getOther(), local) : Value();
      if (op.getOther() && !fullOther && !other)
        continue;
      auto projected = local.create<triton::LoadOp>(
          op.getLoc(), ptr, mask, other, op.getCache(), op.getEvict(),
          op.getIsVolatile());
      auto sink = local.create<UnrealizedConversionCastOp>(
          op.getLoc(), TypeRange{projected.getType()},
          ValueRange{projected.getResult()});
      local.setInsertionPoint(projected);
      Value rewrittenMask;
      if (failed(tryRewriteSegmentedLoad(projected, local, optimize,
                                         fullOther ? &rewrittenMask : nullptr)))
        continue;
      Value result = sink.getInputs().front();
      local.eraseOp(sink);
      local.setInsertionPointToEnd(&scratch);
      result = restore(result, ptrType, active, local, op.getLoc());
      if (fullOther) {
        Value fullMask = restore(rewrittenMask, ptrType, active, local, op.getLoc());
        result = local.create<arith::SelectOp>(op.getLoc(), fullMask, result,
                                               op.getOther());
      }
      while (!scratch.empty())
        rewriter.moveOpBefore(&scratch.front(), op);
      rewriter.replaceOp(op, result);
    } else {
      Value value = projection.project(op.getValue(), local);
      if (!value)
        continue;
      auto projected = local.create<triton::StoreOp>(
          op.getLoc(), ptr, value, mask, op.getBoundaryCheck(),
          op.getCache(), op.getEvict());
      local.setInsertionPoint(projected);
      if (failed(tryRewriteSegmentedStore(projected, local, optimize)))
        continue;
      while (!scratch.empty())
        rewriter.moveOpBefore(&scratch.front(), op);
      rewriter.eraseOp(op);
    }
    return success();
  }
  return failure();
}

// Enumerate at most two independent outer lanes at compile time. Every large
// axis still lowers as rectangles. Count actual emitted loads against a hard
// eight-leaf budget before committing the transaction.
LogicalResult liftSmallOuterLoad(triton::LoadOp op, PatternRewriter &rewriter,
                                bool optimize) {
  auto type = dyn_cast<RankedTensorType>(op.getPtr().getType());
  if (!type || type.getRank() != 2 || !type.hasStaticShape() ||
      type.getEncoding() || op.getIsVolatile())
    return failure();
  for (int64_t active = 0; active < 2; ++active) {
    int64_t outer = 1 - active;
    int64_t rows = type.getDimSize(outer), width = type.getDimSize(active);
    if (rows <= 1 || rows > 2 || width <= rows)
      continue;
    Block scratch;
    PatternRewriter local(rewriter.getContext());
    local.setInsertionPointToEnd(&scratch);
    auto outputType = cast<RankedTensorType>(op.getType());
    Value result = local.create<tensor::EmptyOp>(
        op.getLoc(), outputType.getShape(), outputType.getElementType());
    bool supported = true;
    for (int64_t row = 0; row < rows; ++row) {
      Projection projection(width, rows, row);
      auto ptrAxis = projection.axis(op.getPtr());
      auto maskAxis = op.getMask() ? projection.axis(op.getMask())
                                   : std::optional<int64_t>(-1);
      if (!ptrAxis || (*ptrAxis >= 0 && *ptrAxis != active) ||
          !maskAxis || (*maskAxis >= 0 && *maskAxis != active)) {
        supported = false;
        break;
      }
      Value ptr = projection.project(op.getPtr(), local);
      Value mask = op.getMask() ? projection.project(op.getMask(), local) : Value();
      if (!ptr || (op.getMask() && !mask)) {
        supported = false;
        break;
      }
      SmallVector<OpFoldResult> offsets{local.getIndexAttr(0),
                                       local.getIndexAttr(0)};
      offsets[outer] = local.getIndexAttr(row);
      SmallVector<OpFoldResult> sizes{local.getIndexAttr(type.getDimSize(0)),
                                     local.getIndexAttr(type.getDimSize(1))};
      sizes[outer] = local.getIndexAttr(1);
      SmallVector<OpFoldResult> strides{local.getIndexAttr(1),
                                       local.getIndexAttr(1)};
      Value other;
      if (op.getOther()) {
        auto otherAxis = projection.axis(op.getOther());
        if (otherAxis && (*otherAxis < 0 || *otherAxis == active))
          other = projection.project(op.getOther(), local);
        else
          other = local.create<tensor::ExtractSliceOp>(
              op.getLoc(),
              RankedTensorType::get({width}, outputType.getElementType()),
              op.getOther(), offsets, sizes, strides);
      }
      auto projected = local.create<triton::LoadOp>(
          op.getLoc(), ptr, mask, other, op.getCache(), op.getEvict(),
          op.getIsVolatile());
      auto sink = local.create<UnrealizedConversionCastOp>(
          op.getLoc(), TypeRange{projected.getType()},
          ValueRange{projected.getResult()});
      local.setInsertionPoint(projected);
      if (failed(tryRewriteSegmentedLoad(projected, local, optimize))) {
        supported = false;
        break;
      }
      Value vector = sink.getInputs().front();
      local.eraseOp(sink);
      local.setInsertionPointToEnd(&scratch);
      result = local.create<tensor::InsertSliceOp>(
          op.getLoc(), vector, result, offsets, sizes, strides);
      unsigned leaves = 0;
      scratch.walk([&](triton::LoadOp) { ++leaves; });
      if (leaves > 8) {
        supported = false;
        break;
      }
    }
    if (!supported)
      continue;
    while (!scratch.empty())
      rewriter.moveOpBefore(&scratch.front(), op);
    rewriter.replaceOp(op, result);
    return success();
  }
  return failure();
}

} // namespace
LogicalResult tryRewriteSegmentedBroadcastLoad(
    triton::LoadOp op, PatternRewriter &rewriter, bool optimizeDynamicOffset) {
  if (succeeded(lift(op, rewriter, optimizeDynamicOffset)))
    return success();
  return liftSmallOuterLoad(op, rewriter, optimizeDynamicOffset);
}
LogicalResult tryRewriteSegmentedBroadcastStore(
    triton::StoreOp op, PatternRewriter &rewriter, bool optimizeDynamicOffset) {
  return lift(op, rewriter, optimizeDynamicOffset);
}
} // namespace TritonToStructured
