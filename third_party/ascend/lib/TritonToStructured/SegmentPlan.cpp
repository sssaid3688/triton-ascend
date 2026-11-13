// Segmented structured lowering. No memory is accessed outside the original
// mask: partial periods are represented by bounded, disjoint tensor slices.
#include "TritonToStructured/SegmentPlan.h"
#include "TritonToStructured/PtrAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseMap.h"
#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>

using namespace mlir;
namespace TritonToStructured {
namespace {
template <typename Op> Op markSegmented(Op op) {
  op->setAttr("tts.segmented", UnitAttr::get(op->getContext()));
  return op;
}
struct Expr;
using E = std::shared_ptr<Expr>;
struct Expr {
  enum Kind { Constant, Scalar, Lane, Add, Sub, Mul, Div, Rem, Cast } kind;
  int64_t number = 0;
  Value value;
  Type type;
  E lhs, rhs;
};
E constant(int64_t n, Type t) {
  auto e = std::make_shared<Expr>();
  e->kind = Expr::Constant;
  e->number = n;
  e->type = t;
  return e;
}
E binary(Expr::Kind k, E a, E b, Type t) {
  auto e = std::make_shared<Expr>();
  e->kind = k;
  e->lhs = a;
  e->rhs = b;
  e->type = t;
  return e;
}
bool scalarOnly(E e) {
  if (!e)
    return false;
  if (e->kind == Expr::Lane || e->kind == Expr::Div || e->kind == Expr::Rem)
    return false;
  return (!e->lhs || scalarOnly(e->lhs)) && (!e->rhs || scalarOnly(e->rhs));
}
std::optional<int64_t> staticValue(E e) {
  if (!e)
    return {};
  if (e->kind == Expr::Constant)
    return e->number;
  if (e->kind == Expr::Cast)
    return staticValue(e->lhs);
  auto a = staticValue(e->lhs), b = staticValue(e->rhs);
  if (!a || !b)
    return {};
  __int128 n = 0;
  if (e->kind == Expr::Add)
    n = (__int128)*a + *b;
  else if (e->kind == Expr::Sub)
    n = (__int128)*a - *b;
  else if (e->kind == Expr::Mul)
    n = (__int128)*a * *b;
  else
    return {};
  if (n < std::numeric_limits<int64_t>::min() ||
      n > std::numeric_limits<int64_t>::max())
    return {};
  return (int64_t)n;
}
bool sameExpr(E a, E b) {
  if (!a || !b)
    return !a && !b;
  if (a == b)
    return true;
  auto av = staticValue(a), bv = staticValue(b);
  if (av && bv)
    return av == bv;
  return a->kind == b->kind && a->value == b->value && a->number == b->number &&
         sameExpr(a->lhs, b->lhs) && sameExpr(a->rhs, b->rhs);
}
bool nonNegativeValue(Value v, unsigned depth = 0) {
  if (!v || depth > 12)
    return false;
  APInt n;
  if (matchPattern(v, m_ConstantInt(&n)))
    return !n.isNegative();
  if (v.getDefiningOp<triton::GetProgramIdOp>())
    return true;
  if (auto a = v.getDefiningOp<arith::AddIOp>())
    return nonNegativeValue(a.getLhs(), depth + 1) &&
           nonNegativeValue(a.getRhs(), depth + 1);
  if (auto a = v.getDefiningOp<arith::MulIOp>())
    return nonNegativeValue(a.getLhs(), depth + 1) &&
           nonNegativeValue(a.getRhs(), depth + 1);
  if (auto a = v.getDefiningOp<arith::ExtSIOp>())
    return nonNegativeValue(a.getIn(), depth + 1);
  if (auto a = v.getDefiningOp<arith::IndexCastOp>())
    return nonNegativeValue(a.getIn(), depth + 1);
  return false;
}
bool nonNegative(E e) {
  if (auto n = staticValue(e))
    return *n >= 0;
  if (e->kind == Expr::Scalar)
    return nonNegativeValue(e->value);
  if (e->kind == Expr::Cast)
    return nonNegative(e->lhs);
  if (e->kind == Expr::Add || e->kind == Expr::Mul)
    return nonNegative(e->lhs) && nonNegative(e->rhs);
  return false;
}
struct Linear {
  E offset;
  int64_t stride = 0;
  bool valid = false;
};
Linear linear(E e) {
  if (!e)
    return {};
  if (scalarOnly(e))
    return {e, 0, true};
  if (e->kind == Expr::Lane)
    return {constant(0, e->type), 1, true};
  if (e->kind == Expr::Cast) {
    auto a = linear(e->lhs);
    if (a.valid)
      a.offset = binary(Expr::Add, a.offset, constant(0, e->type), e->type);
    return a;
  }
  auto a = linear(e->lhs), b = linear(e->rhs);
  if (!a.valid || !b.valid)
    return {};
  if (e->kind == Expr::Add || e->kind == Expr::Sub) {
    // Remove neutral additions so separately parsed x/range expressions share
    // a stable scalar origin without relying on SSA CSE.
    E off;
    auto az = staticValue(a.offset), bz = staticValue(b.offset);
    if (bz && *bz == 0)
      off = a.offset;
    else if (az && *az == 0 && e->kind == Expr::Add)
      off = b.offset;
    else
      off = binary(e->kind, a.offset, b.offset, e->type);
    return {off,
            e->kind == Expr::Add ? a.stride + b.stride : a.stride - b.stride,
            true};
  }
  if (e->kind == Expr::Mul) {
    auto ac = staticValue(e->lhs), bc = staticValue(e->rhs);
    if (ac)
      return {binary(Expr::Mul, constant(*ac, e->type), b.offset, e->type),
              *ac * b.stride, true};
    if (bc)
      return {binary(Expr::Mul, a.offset, constant(*bc, e->type), e->type),
              *bc * a.stride, true};
  }
  return {};
}
struct Bound {
  E value;
  int sign = 1;
  int64_t scale = 1;
  int64_t bias = 0;
  E origin;
  int64_t divideBy = 1;
};
struct Plan {
  int64_t block = 0, divisor = 0, inputStride = 1, period = 0;
  Type element;
  E anchorOffset, pointer;
  Value source, oldMask;
  SmallVector<Bound> lowers, uppers, remLowers, remUppers;
  llvm::DenseMap<Value, E> cache;
  bool bad = false, sawDivRem = false, hasLinearMask = false,
       generalStride = false;
  E parse(Value v) {
    if (auto it = cache.find(v); it != cache.end())
      return it->second;
    E e;
    auto ty = dyn_cast<ShapedType>(v.getType());
    if (auto ranked = dyn_cast<RankedTensorType>(v.getType());
        ranked && ranked.getEncoding())
      return {};
    Type t = ty ? ty.getElementType() : v.getType();
    if (!t.isIntOrIndex())
      return {};
    if (!ty) {
      APInt n;
      if (matchPattern(v, m_ConstantInt(&n)))
        e = constant(n.getSExtValue(), t);
      else {
        e = std::make_shared<Expr>();
        e->kind = Expr::Scalar;
        e->value = v;
        e->type = t;
      }
    } else if (auto c = v.getDefiningOp<arith::ConstantOp>()) {
      auto d = dyn_cast<DenseIntElementsAttr>(c.getValue());
      if (d && d.isSplat())
        e = constant(d.getSplatValue<APInt>().getSExtValue(), t);
    } else if (auto s = v.getDefiningOp<triton::SplatOp>())
      e = parse(s.getSrc());
    else if (auto r = v.getDefiningOp<triton::MakeRangeOp>()) {
      auto rt = cast<RankedTensorType>(r.getType());
      if (rt.getRank() != 1 || rt.getNumElements() != block)
        return {};
      e = std::make_shared<Expr>();
      e->kind = Expr::Lane;
      e->type = t;
      if (r.getStart() != 0)
        e = binary(Expr::Add, e, constant(r.getStart(), t), t);
    } else if (auto a = v.getDefiningOp<arith::AddIOp>())
      e = binary(Expr::Add, parse(a.getLhs()), parse(a.getRhs()), t);
    else if (auto a = v.getDefiningOp<arith::SubIOp>())
      e = binary(Expr::Sub, parse(a.getLhs()), parse(a.getRhs()), t);
    else if (auto a = v.getDefiningOp<arith::MulIOp>())
      e = binary(Expr::Mul, parse(a.getLhs()), parse(a.getRhs()), t);
    else if (auto a = v.getDefiningOp<arith::DivSIOp>())
      e = binary(Expr::Div, parse(a.getLhs()), parse(a.getRhs()), t);
    else if (auto a = v.getDefiningOp<arith::RemSIOp>())
      e = binary(Expr::Rem, parse(a.getLhs()), parse(a.getRhs()), t);
    else if (auto a = v.getDefiningOp<arith::ExtSIOp>())
      e = binary(Expr::Cast, parse(a.getIn()), {}, t);
    else if (auto a = v.getDefiningOp<arith::IndexCastOp>())
      e = binary(Expr::Cast, parse(a.getIn()), {}, t);
    if (!e ||
        ((e->kind == Expr::Add || e->kind == Expr::Sub ||
          e->kind == Expr::Mul || e->kind == Expr::Div ||
          e->kind == Expr::Rem) &&
         (!e->lhs || !e->rhs)) ||
        (e->kind == Expr::Cast && !e->lhs))
      return {};
    cache[v] = e;
    return e;
  }
  bool observe(E e) {
    if (!e)
      return false;
    if (e->kind == Expr::Div || e->kind == Expr::Rem) {
      auto d = staticValue(e->rhs);
      auto l = linear(e->lhs);
      if (!d || *d <= 0 || *d > INT32_MAX || !l.valid || l.stride <= 0 ||
          l.stride > INT32_MAX || !nonNegative(l.offset))
        return false;
      bool general = *d % l.stride != 0 && l.stride % *d != 0;
      if (general && !staticValue(l.offset))
        return false;
      if (divisor && (divisor != *d || inputStride != l.stride ||
                      !sameExpr(anchorOffset, l.offset)))
        return false;
      divisor = *d;
      inputStride = l.stride;
      period = divisor / std::gcd(divisor, inputStride);
      generalStride = general;
      anchorOffset = l.offset;
      sawDivRem = true;
    }
    return (!e->lhs || observe(e->lhs)) && (!e->rhs || observe(e->rhs));
  }
  bool ptr(Value v) {
    if (isa<triton::PointerType>(v.getType())) {
      if (source)
        return false;
      source = v;
      pointer = constant(0, IntegerType::get(v.getContext(), 32));
      return true;
    }
    if (auto s = v.getDefiningOp<triton::SplatOp>())
      return ptr(s.getSrc());
    if (auto a = v.getDefiningOp<triton::AddPtrOp>()) {
      if (!ptr(a.getPtr()))
        return false;
      auto e = parse(a.getOffset());
      if (!e || !observe(e))
        return false;
      pointer = binary(Expr::Add, pointer, e, e->type);
      return true;
    }
    return false;
  }
  bool mask(Value v) {
    if (!v)
      return true;
    if (auto a = v.getDefiningOp<arith::AndIOp>())
      return mask(a.getLhs()) && mask(a.getRhs());
    if (auto c = v.getDefiningOp<arith::ConstantOp>()) {
      auto d = dyn_cast<DenseIntElementsAttr>(c.getValue());
      if (!d || !d.isSplat())
        return false;
      if (d.getSplatValue<APInt>().isZero())
        uppers.push_back({constant(0, IntegerType::get(v.getContext(), 32))});
      return true;
    }
    auto cmp = v.getDefiningOp<arith::CmpIOp>();
    if (!cmp)
      return false;
    auto lhs = parse(cmp.getLhs()), rhs = parse(cmp.getRhs());
    if (!lhs || !rhs || !scalarOnly(rhs) || !observe(lhs))
      return false;
    auto pred = cmp.getPredicate();
    bool upper =
        pred == arith::CmpIPredicate::slt || pred == arith::CmpIPredicate::sle;
    bool lower =
        pred == arith::CmpIPredicate::sge || pred == arith::CmpIPredicate::sgt;
    if (!upper && !lower)
      return false;
    int64_t bias =
        (pred == arith::CmpIPredicate::sle || pred == arith::CmpIPredicate::sgt)
            ? 1
            : 0;
    auto &bounds = upper ? uppers : lowers;
    auto l = linear(lhs);
    if (l.valid && l.stride > 0 && nonNegative(l.offset)) {
      hasLinearMask = true;
      bounds.push_back({rhs, 1, 1, bias, l.offset, l.stride});
      return true;
    }
    if (lhs->kind == Expr::Div) {
      bounds.push_back(
          {rhs, 1, divisor, bias * divisor, anchorOffset, inputStride});
      return true;
    }
    if (lhs->kind == Expr::Rem) {
      (upper ? remUppers : remLowers).push_back({rhs, 1, 1, bias, {}});
      return true;
    }
    return false;
  }
};

class Emitter {
  PatternRewriter &b;
  Location loc;
  Plan &p;
  llvm::DenseMap<const Expr *, Value> scalarCache;

public:
  Emitter(PatternRewriter &b, Location loc, Plan &p) : b(b), loc(loc), p(p) {}
  Value c(int64_t n) { return b.create<arith::ConstantIndexOp>(loc, n); }
  Value castTo(Value v, Type t) {
    if (v.getType() == t)
      return v;
    if (v.getType().isIndex() || t.isIndex())
      return b.create<arith::IndexCastOp>(loc, t, v);
    auto from = cast<IntegerType>(v.getType()), to = cast<IntegerType>(t);
    if (from.getWidth() < to.getWidth())
      return b.create<arith::ExtSIOp>(loc, t, v);
    return b.create<arith::TruncIOp>(loc, t, v);
  }
  Value scalar(E e) {
    if (auto it = scalarCache.find(e.get()); it != scalarCache.end())
      return it->second;
    Value v;
    if (e->kind == Expr::Constant)
      v = c(e->number);
    else if (e->kind == Expr::Scalar)
      v = castTo(e->value, b.getIndexType());
    else if (e->kind == Expr::Cast)
      v = scalar(e->lhs);
    else {
      auto a = scalar(e->lhs), bb = scalar(e->rhs);
      if (e->kind == Expr::Add)
        v = add(a, bb);
      else if (e->kind == Expr::Sub)
        v = sub(a, bb);
      else
        v = mul(a, bb);
    }
    scalarCache[e.get()] = v;
    return v;
  }
  Value add(Value a, Value bb) {
    return b.createOrFold<arith::AddIOp>(loc, a, bb);
  }
  Value sub(Value a, Value bb) {
    return b.createOrFold<arith::SubIOp>(loc, a, bb);
  }
  Value mul(Value a, Value bb) {
    return b.createOrFold<arith::MulIOp>(loc, a, bb);
  }
  Value min(Value a, Value bb) {
    return b.createOrFold<arith::MinSIOp>(loc, a, bb);
  }
  Value max(Value a, Value bb) {
    return b.createOrFold<arith::MaxSIOp>(loc, a, bb);
  }
  Value div(Value a, Value bb) {
    return b.createOrFold<arith::DivSIOp>(loc, a, bb);
  }
  Value rem(Value a, Value bb) {
    return b.createOrFold<arith::RemSIOp>(loc, a, bb);
  }
  Value bound(const Bound &x) {
    auto v = scalar(x.value);
    bool wide = x.value->type.isIndex() ||
                (isa<IntegerType>(x.value->type) &&
                 cast<IntegerType>(x.value->type).getWidth() > 32);
    if (x.origin)
      wide = wide || x.origin->type.isIndex() ||
             (isa<IntegerType>(x.origin->type) &&
              cast<IntegerType>(x.origin->type).getWidth() > 32);
    Value negative, overflow;
    int64_t cap = x.origin ? p.block : p.divisor;
    if (wide) {
      negative = b.createOrFold<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                               v, c(0));
      int64_t safeMax = (INT64_MAX - x.bias) / x.scale;
      overflow = b.createOrFold<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                               v, c(safeMax));
      v = min(max(v, c(0)), c(safeMax));
    }
    if (x.scale != 1)
      v = mul(v, c(x.scale));
    if (x.bias)
      v = add(v, c(x.bias));
    if (x.origin)
      v = sub(v, scalar(x.origin));
    if (x.divideBy != 1)
      v = b.createOrFold<arith::CeilDivSIOp>(loc, v, c(x.divideBy));
    // The only users are lane/column bounds. Saturating before index-width
    // conversion also handles <= INT32_MAX without truncating to negative i32.
    v = min(max(v, c(0)), c(cap));
    if (wide) {
      v = b.createOrFold<arith::SelectOp>(loc, negative, c(0), v);
      v = b.createOrFold<arith::SelectOp>(loc, overflow, c(cap), v);
    }
    return v;
  }
  OpFoldResult mixed(Value v) {
    APInt n;
    if (matchPattern(v, m_ConstantInt(&n)))
      return b.getIndexAttr(n.getSExtValue());
    return v;
  }
  SmallVector<OpFoldResult> sizes(Value n) { return {mixed(n)}; }
  Value fill(Value v, ArrayRef<int64_t> shape) {
    return b.create<triton::SplatOp>(
        loc, RankedTensorType::get(shape, v.getType()), v);
  }
  Value range(int64_t n) {
    return b.create<triton::MakeRangeOp>(
        loc, RankedTensorType::get({n}, b.getI32Type()), 0, n);
  }
  Value axis(int64_t n, unsigned axis, ArrayRef<int64_t> shape) {
    Value v = range(n);
    for (unsigned j = 0; j < shape.size(); ++j)
      if (j != axis)
        v = b.create<triton::ExpandDimsOp>(loc, v, j);
    if (shape.size() > 1)
      v = b.create<triton::BroadcastOp>(
          loc, RankedTensorType::get(shape, b.getI32Type()), v);
    return v;
  }
  Value reshape(Value v, ArrayRef<int64_t> shape) {
    auto t = RankedTensorType::get(
        shape, cast<ShapedType>(v.getType()).getElementType());
    if (shape.size() == 2)
      return b.create<tensor::ExpandShapeOp>(
          loc, t, v, ArrayRef<ReassociationIndices>{{0, 1}});
    return v;
  }
  Value flatten(Value v) {
    if (cast<ShapedType>(v.getType()).getRank() == 1)
      return v;
    return b.create<tensor::CollapseShapeOp>(
        loc, v, ArrayRef<ReassociationIndices>{{0, 1}});
  }
  Value extract(Value v, Value start, Value n) {
    return b.create<tensor::ExtractSliceOp>(
        loc, v, SmallVector<OpFoldResult>{mixed(start)}, sizes(n),
        SmallVector<OpFoldResult>{b.getIndexAttr(1)});
  }
  Value insert(Value source, Value dest, Value start, Value n) {
    return b.create<tensor::InsertSliceOp>(
        loc, source, dest, SmallVector<OpFoldResult>{mixed(start)}, sizes(n),
        SmallVector<OpFoldResult>{b.getIndexAttr(1)});
  }
  Value scalarAt(E e, Value start) {
    if (scalarOnly(e))
      return castTo(scalar(e), e->type);
    if (e->kind == Expr::Lane)
      return castTo(start, e->type);
    auto a = scalarAt(e->lhs, start);
    if (e->kind == Expr::Cast)
      return castTo(a, e->type);
    auto bb = scalarAt(e->rhs, start);
    a = castTo(a, e->type);
    bb = castTo(bb, e->type);
    if (e->kind == Expr::Add)
      return add(a, bb);
    if (e->kind == Expr::Sub)
      return sub(a, bb);
    if (e->kind == Expr::Mul)
      return mul(a, bb);
    if (e->kind == Expr::Div)
      return div(a, bb);
    return rem(a, bb);
  }
  Value anyActive(Value start, Value count) {
    auto zero = c(0);
    Value lo = zero, hi = count;
    auto origin = add(scalar(p.anchorOffset), mul(start, c(p.inputStride)));
    auto phase = rem(origin, c(p.divisor));
    if (p.inputStride % p.divisor == 0) {
      Value active = b.createOrFold<arith::CmpIOp>(
          loc, arith::CmpIPredicate::sgt, count, zero);
      for (auto &x : p.remLowers) {
        auto t = b.createOrFold<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                               phase, bound(x));
        active = b.createOrFold<arith::AndIOp>(loc, active, t);
      }
      for (auto &x : p.remUppers) {
        auto t = b.createOrFold<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                               phase, bound(x));
        active = b.createOrFold<arith::AndIOp>(loc, active, t);
      }
      return active;
    }
    for (auto &x : p.remLowers) {
      auto first = b.createOrFold<arith::CeilDivSIOp>(loc, sub(bound(x), phase),
                                                      c(p.inputStride));
      lo = max(lo, first);
    }
    for (auto &x : p.remUppers) {
      auto last = b.createOrFold<arith::CeilDivSIOp>(loc, sub(bound(x), phase),
                                                     c(p.inputStride));
      hi = min(hi, last);
    }
    return b.createOrFold<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, hi,
                                         lo);
  }
  Value tensorExpr(E e, Value start, ArrayRef<int64_t> shape,
                   bool rowOnly = false) {
    Type t = e->type;
    auto splat = [&](Value v) { return fill(castTo(v, t), shape); };
    if (scalarOnly(e))
      return splat(scalar(e));
    unsigned colAxis = shape.size() - 1;
    auto col = axis(shape.back(), colAxis, shape);
    auto convertTensor = [&](Value v, Type target) -> Value {
      auto old = cast<ShapedType>(v.getType()).getElementType();
      if (old == target)
        return v;
      auto ty = RankedTensorType::get(shape, target);
      if (old.isIndex() || target.isIndex())
        return b.create<arith::IndexCastOp>(loc, ty, v);
      if (cast<IntegerType>(old).getWidth() <
          cast<IntegerType>(target).getWidth())
        return b.create<arith::ExtSIOp>(loc, ty, v);
      return b.create<arith::TruncIOp>(loc, ty, v);
    };
    col = rowOnly ? splat(c(0)) : convertTensor(col, t);
    auto row = [&]() -> Value {
      if (shape.size() == 1 && !rowOnly)
        return splat(c(0));
      return convertTensor(axis(shape[0], 0, shape), t);
    };
    if (e->kind == Expr::Lane) {
      Value v = b.create<arith::AddIOp>(loc, col, splat(start));
      if (shape.size() == 2 || rowOnly)
        v = b.create<arith::AddIOp>(
            loc, v, b.create<arith::MulIOp>(loc, row(), splat(c(p.period))));
      return v;
    }
    if (e->kind == Expr::Div || e->kind == Expr::Rem) {
      auto origin = add(scalar(p.anchorOffset), mul(start, c(p.inputStride)));
      auto base = e->kind == Expr::Div ? div(origin, c(p.divisor))
                                       : rem(origin, c(p.divisor));
      if (e->kind == Expr::Div) {
        Value increment = b.create<arith::MulIOp>(
            loc, row(), splat(c(p.inputStride * p.period / p.divisor)));
        if (p.generalStride && p.inputStride / p.divisor)
          increment = b.create<arith::AddIOp>(
              loc, increment,
              b.create<arith::MulIOp>(loc, col,
                                      splat(c(p.inputStride / p.divisor))));
        return b.create<arith::AddIOp>(loc, splat(base), increment);
      }
      if (p.inputStride % p.divisor == 0)
        return splat(base);
      auto increment = b.create<arith::MulIOp>(
          loc, col,
          splat(
              c(p.generalStride ? p.inputStride % p.divisor : p.inputStride)));
      return b.create<arith::AddIOp>(loc, splat(base), increment);
    }
    auto a = tensorExpr(e->lhs, start, shape, rowOnly);
    if (e->kind == Expr::Cast)
      return convertTensor(a, t);
    auto bb = tensorExpr(e->rhs, start, shape, rowOnly);
    a = convertTensor(a, t);
    bb = convertTensor(bb, t);
    if (e->kind == Expr::Add)
      return b.create<arith::AddIOp>(loc, a, bb);
    if (e->kind == Expr::Sub)
      return b.create<arith::SubIOp>(loc, a, bb);
    return b.create<arith::MulIOp>(loc, a, bb);
  }
  Value comparison(Value coordinate, Value rhs, arith::CmpIPredicate pred,
                   unsigned axisId, ArrayRef<int64_t> shape) {
    auto st = cast<ShapedType>(coordinate.getType()).getElementType();
    auto ty = RankedTensorType::get({shape[axisId]}, st);
    auto rhsT = b.create<triton::SplatOp>(loc, ty, castTo(rhs, st));
    Value v = b.create<arith::CmpIOp>(loc, pred, coordinate, rhsT);
    for (unsigned j = 0; j < shape.size(); ++j)
      if (j != axisId)
        v = b.create<triton::ExpandDimsOp>(loc, v, j);
    if (shape.size() > 1)
      v = b.create<triton::BroadcastOp>(
          loc, RankedTensorType::get(shape, b.getI1Type()), v);
    return v;
  }
  Value intervalMask(Value lo, Value hi, int64_t extent) {
    APInt l, h;
    bool lowerZero = matchPattern(lo, m_ConstantInt(&l)) && l.isZero();
    bool upperFull =
        matchPattern(hi, m_ConstantInt(&h)) && h.getSExtValue() == extent;
    if (lowerZero && upperFull) {
      auto t = b.create<arith::ConstantOp>(loc, b.getBoolAttr(true));
      return fill(t, {extent});
    }
    Value r = range(extent), result;
    if (!lowerZero)
      result = comparison(r, lo, arith::CmpIPredicate::sge, 0, {extent});
    if (!upperFull) {
      auto upper = comparison(r, hi, arith::CmpIPredicate::slt, 0, {extent});
      result = result ? b.create<arith::AndIOp>(loc, result, upper).getResult()
                      : upper;
    }
    return result;
  }
  Value mask(Value start, Value valid, ArrayRef<int64_t> shape) {
    APInt count;
    if (p.remLowers.empty() && p.remUppers.empty() &&
        matchPattern(valid, m_ConstantInt(&count)) &&
        count.getSExtValue() == shape[0])
      return {};
    unsigned colAxis = shape.size() - 1;
    Value active = shape.size() == 1
                       ? comparison(range(shape[0]), valid,
                                    arith::CmpIPredicate::slt, 0, shape)
                       : comparison(range(shape[0]), valid,
                                    arith::CmpIPredicate::slt, 0, shape);
    auto origin = add(scalar(p.anchorOffset), mul(start, c(p.inputStride)));
    auto phase = rem(origin, c(p.divisor));
    auto r = range(shape.back());
    Value localOrigin = fill(castTo(phase, b.getI32Type()), {shape.back()});
    Value col;
    if (p.inputStride % p.divisor == 0)
      col = localOrigin;
    else {
      auto step = fill(
          castTo(c(p.generalStride ? p.inputStride % p.divisor : p.inputStride),
                 b.getI32Type()),
          {shape.back()});
      auto scaled = b.create<arith::MulIOp>(loc, r, step);
      col = b.create<arith::AddIOp>(loc, scaled, localOrigin);
    }
    for (const auto &x : p.remLowers) {
      auto m =
          comparison(col, bound(x), arith::CmpIPredicate::sge, colAxis, shape);
      active = b.create<arith::AndIOp>(loc, active, m);
    }
    for (const auto &x : p.remUppers) {
      auto m =
          comparison(col, bound(x), arith::CmpIPredicate::slt, colAxis, shape);
      active = b.create<arith::AndIOp>(loc, active, m);
    }
    return active;
  }
};

bool constantWithinPiece(E e, const Plan &p) {
  if (scalarOnly(e))
    return true;
  if (e->kind == Expr::Div)
    return !p.generalStride && p.inputStride < p.divisor;
  if (e->kind == Expr::Rem)
    return p.inputStride % p.divisor == 0;
  if (e->kind == Expr::Lane)
    return false;
  return (!e->lhs || constantWithinPiece(e->lhs, p)) &&
         (!e->rhs || constantWithinPiece(e->rhs, p));
}
bool invariantInPeriodRows(E e) {
  if (scalarOnly(e) || e->kind == Expr::Rem)
    return true;
  if (e->kind == Expr::Lane || e->kind == Expr::Div)
    return false;
  return (!e->lhs || invariantInPeriodRows(e->lhs)) &&
         (!e->rhs || invariantInPeriodRows(e->rhs));
}
std::optional<int64_t> staticInnerStride(E e, const Plan &p) {
  if (scalarOnly(e))
    return 0;
  if (e->kind == Expr::Lane)
    return 1;
  if (e->kind == Expr::Div)
    return p.inputStride < p.divisor ? 0 : p.inputStride / p.divisor;
  if (e->kind == Expr::Rem)
    return p.inputStride % p.divisor == 0 ? 0 : p.inputStride;
  if (e->kind == Expr::Cast)
    return staticInnerStride(e->lhs, p);
  auto a = staticInnerStride(e->lhs, p), bb = staticInnerStride(e->rhs, p);
  if (!a || !bb)
    return {};
  if (e->kind == Expr::Add)
    return *a + *bb;
  if (e->kind == Expr::Sub)
    return *a - *bb;
  if (e->kind == Expr::Mul) {
    if (auto k = staticValue(e->lhs))
      return *k * *bb;
    if (auto k = staticValue(e->rhs))
      return *k * *a;
  }
  return {};
}
bool storeInjective(E e, int64_t divisor, int64_t stride, int64_t period);
Value createPieceLoad(triton::LoadOp op, Plan &p, Emitter &em, Value start,
                      Value count, ArrayRef<int64_t> shape, Value pointer,
                      Value mask, PatternRewriter &b) {
  Type element =
      cast<triton::PointerType>(
          cast<RankedTensorType>(op.getPtr().getType()).getElementType())
          .getPointeeType();
  // A non-aligned local row makes the NPU copy helper fall back to scalar
  // GM reads. Read an aligned prefix and the exact remainder into contiguous
  // subtiles, retaining the original period for the row address increment.
  auto innerStride = staticInnerStride(p.pointer, p);
  if (shape.size() == 2 && shape.front() > 1 && !p.generalStride &&
      p.remLowers.empty() && p.remUppers.empty() && element.isIntOrFloat() &&
      innerStride && *innerStride == 1 &&
      storeInjective(p.pointer, p.divisor, p.inputStride, p.period)) {
    int64_t bits = element.getIntOrFloatBitWidth();
    int64_t bytes = bits % 8 == 0 ? bits / 8 : 0;
    if (bytes > 0 && bytes <= 32 && 32 % bytes == 0) {
      int64_t alignedColumns = 32 / bytes;
      int64_t main = shape.back() / alignedColumns * alignedColumns;
      if (main > 0 && main < shape.back()) {
        Value result = b.create<tensor::EmptyOp>(op.getLoc(), shape, element);
        for (auto [delta, width] :
             {std::pair<int64_t, int64_t>{0, main},
              std::pair<int64_t, int64_t>{main, shape.back() - main}}) {
          SmallVector<int64_t> subShape{shape.front(), width};
          auto subStart = em.add(start, em.c(delta));
          auto offset = em.tensorExpr(p.pointer, subStart, subShape);
          auto base = em.fill(p.source, subShape);
          auto ptr = b.create<triton::AddPtrOp>(op.getLoc(), base.getType(),
                                                base, offset);
          auto subMask = em.mask(subStart, count, subShape);
          Value load = markSegmented(b.create<triton::LoadOp>(
              op.getLoc(), ptr, subMask, Value(), op.getCache(), op.getEvict(),
              op.getIsVolatile()));
          result = b.create<tensor::InsertSliceOp>(
              op.getLoc(), load, result,
              SmallVector<OpFoldResult>{b.getIndexAttr(0),
                                        b.getIndexAttr(delta)},
              SmallVector<OpFoldResult>{b.getIndexAttr(shape.front()),
                                        b.getIndexAttr(width)},
              SmallVector<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});
        }
        return result;
      }
    }
  }
  // Removing a broadcast pointer axis is legal only when the corresponding
  // masked domain is nonempty. Otherwise an all-false rectangle could turn
  // into a live load when the legacy factorization drops a zero-stride axis.
  if (shape.size() == 2 && element.isIntOrFloat() &&
      !storeInjective(p.pointer, p.divisor, p.inputStride, p.period)) {
    auto rowsActive = b.createOrFold<arith::CmpIOp>(
        op.getLoc(), arith::CmpIPredicate::sgt, count, em.c(0));
    auto columnsActive = em.anyActive(start, em.c(shape.back()));
    auto active =
        b.createOrFold<arith::AndIOp>(op.getLoc(), rowsActive, columnsActive);
    auto load = [&]() -> Value {
      bool rowsInvariant = invariantInPeriodRows(p.pointer);
      bool columnsInvariant = constantWithinPiece(p.pointer, p);
      if (rowsInvariant || columnsInvariant) {
        int64_t n = rowsInvariant ? shape.back() : shape.front();
        SmallVector<int64_t> reducedShape{n};
        auto off = em.tensorExpr(p.pointer, start, reducedShape,
                                 /*rowOnly=*/!rowsInvariant);
        auto base = em.fill(p.source, reducedShape);
        auto reducedPtr =
            b.create<triton::AddPtrOp>(op.getLoc(), base.getType(), base, off);
        Value reducedMask;
        // The enclosing nonempty-domain guard proves the only lane active
        // when the reduced axis has extent one. Do not materialize a
        // tensor<1> comparison mask: downstream mask parsing has no varying
        // comparison dimension in that case.
        if (rowsInvariant && n > 1)
          reducedMask = em.mask(start, em.c(n), reducedShape);
        else if (!rowsInvariant && n > 1) {
          APInt rows;
          if (!matchPattern(count, m_ConstantInt(&rows)) ||
              rows.getSExtValue() != n)
            reducedMask = em.comparison(
                em.range(n), count, arith::CmpIPredicate::slt, 0, reducedShape);
        }
        Value value = markSegmented(b.create<triton::LoadOp>(
            op.getLoc(), reducedPtr, reducedMask, Value(), op.getCache(),
            op.getEvict(), op.getIsVolatile()));
        value = b.create<triton::ExpandDimsOp>(op.getLoc(), value,
                                               rowsInvariant ? 0 : 1);
        return b.create<triton::BroadcastOp>(
            op.getLoc(), RankedTensorType::get(shape, element), value);
      }
      return markSegmented(b.create<triton::LoadOp>(
          op.getLoc(), pointer, mask, Value(), op.getCache(), op.getEvict(),
          op.getIsVolatile()));
    };
    auto zero = [&]() -> Value {
      auto z = b.create<arith::ConstantOp>(op.getLoc(), b.getZeroAttr(element));
      return em.fill(z, shape);
    };
    APInt flag;
    if (matchPattern(active, m_ConstantInt(&flag)))
      return flag.isZero() ? zero() : load();
    auto resultType = RankedTensorType::get(shape, element);
    auto branch =
        b.create<scf::IfOp>(op.getLoc(), TypeRange{resultType}, active, true);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(&branch.getThenRegion().front());
      auto v = load();
      b.create<scf::YieldOp>(op.getLoc(), v);
      b.setInsertionPointToStart(&branch.getElseRegion().front());
      auto v0 = zero();
      b.create<scf::YieldOp>(op.getLoc(), v0);
    }
    return branch.getResult(0);
  }
  if (shape.size() != 1 || shape[0] <= 1 || !element.isIntOrFloat() ||
      !constantWithinPiece(p.pointer, p))
    return markSegmented(b.create<triton::LoadOp>(
        op.getLoc(), pointer, mask, Value(), op.getCache(), op.getEvict(),
        op.getIsVolatile()));
  auto condition = em.anyActive(start, count);
  auto doLoad = [&]() -> Value {
    auto offset = em.scalarAt(p.pointer, start);
    auto ptr = b.create<triton::AddPtrOp>(op.getLoc(), p.source.getType(),
                                          p.source, offset);
    return markSegmented(b.create<triton::LoadOp>(
        op.getLoc(), ptr, Value(), Value(), op.getCache(), op.getEvict(),
        op.getIsVolatile()));
  };
  APInt c;
  Value loaded;
  if (matchPattern(condition, m_ConstantInt(&c))) {
    if (!c.isZero())
      loaded = doLoad();
    else
      loaded = b.create<arith::ConstantOp>(op.getLoc(), b.getZeroAttr(element));
  } else {
    auto branch =
        b.create<scf::IfOp>(op.getLoc(), TypeRange{element}, condition, true);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(&branch.getThenRegion().front());
      auto v = doLoad();
      b.create<scf::YieldOp>(op.getLoc(), v);
      b.setInsertionPointToStart(&branch.getElseRegion().front());
      auto zero =
          b.create<arith::ConstantOp>(op.getLoc(), b.getZeroAttr(element));
      b.create<scf::YieldOp>(op.getLoc(), zero.getResult());
    }
    loaded = branch.getResult(0);
  }
  return em.fill(loaded, shape);
}

// Carry validity as an ordinary integer tensor while slicing/assembling.
// Live TT memory masks remain i1; convert back only after reconstruction.
Value maskCarrier(Value mask, PatternRewriter &b, Location loc) {
  auto type = cast<RankedTensorType>(mask.getType());
  if (type.getElementType().isInteger(32))
    return mask;
  return b.create<arith::ExtUIOp>(
      loc, RankedTensorType::get(type.getShape(), b.getI32Type()), mask);
}
Value restoredMask(Value carrier, Emitter &em, PatternRewriter &b,
                   Location loc) {
  if (!carrier)
    return {};
  auto type = cast<RankedTensorType>(carrier.getType());
  auto zero = b.create<arith::ConstantOp>(loc, b.getI32IntegerAttr(0));
  auto zeros = em.fill(zero, type.getShape());
  return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, carrier, zeros);
}

bool storeInjective(E e, int64_t divisor, int64_t stride, int64_t period) {
  // Conservative affine proof: positive inner stride and disjoint row spans.
  std::function<std::optional<std::array<int64_t, 3>>(E)> coeff =
      [&](E n) -> std::optional<std::array<int64_t, 3>> {
    if (scalarOnly(n))
      return std::array<int64_t, 3>{0, 0, 0};
    if (n->kind == Expr::Lane)
      return std::array<int64_t, 3>{1, 0, 0};
    if (n->kind == Expr::Div)
      return std::array<int64_t, 3>{0, 1, 0};
    if (n->kind == Expr::Rem)
      return std::array<int64_t, 3>{0, 0, 1};
    if (n->kind == Expr::Cast)
      return coeff(n->lhs);
    auto a = coeff(n->lhs), b = coeff(n->rhs);
    if (!a || !b)
      return {};
    if (n->kind == Expr::Add || n->kind == Expr::Sub) {
      for (unsigned j = 0; j < 3; ++j)
        (*a)[j] += n->kind == Expr::Add ? (*b)[j] : -(*b)[j];
      return a;
    }
    auto ac = staticValue(n->lhs), bc = staticValue(n->rhs);
    if (!ac && !bc)
      return {};
    auto r = ac ? *b : *a;
    auto factor = ac ? *ac : *bc;
    for (auto &v : r)
      v *= factor;
    return r;
  };
  auto c = coeff(e);
  if (!c)
    return false;
  int64_t inner = (*c)[0] + (*c)[2] * stride;
  int64_t advance = (*c)[0] * period + (*c)[1] * (stride * period / divisor);
  return period == 1 ? advance > 0 : inner > 0 && advance >= inner * period;
}

// A static strided lattice is split at its remainder discontinuities. The
// repeated affine rectangles share a gcd period, rather than unrolling lanes.
template <typename Op>
LogicalResult rewriteGeneral(Op op, PatternRewriter &b, Plan &p,
                             Value *outputMask) {
  constexpr bool isLoad = std::is_same_v<Op, triton::LoadOp>;
  auto evalBound = [](const Bound &x) -> std::optional<int64_t> {
    auto a = staticValue(x.value);
    auto o = x.origin ? staticValue(x.origin) : std::optional<int64_t>(0);
    if (!a || !o)
      return {};
    __int128 v = (__int128)*a * x.scale + x.bias - *o;
    if (x.divideBy != 1) {
      auto d = x.divideBy;
      v = v >= 0 ? (v + d - 1) / d : v / d;
    }
    if (v < INT64_MIN || v > INT64_MAX)
      return {};
    return (int64_t)v;
  };
  int64_t lo = 0, hi = p.block;
  for (auto &x : p.lowers) {
    auto n = evalBound(x);
    if (!n)
      return failure();
    lo = std::max(lo, *n);
  }
  for (auto &x : p.uppers) {
    auto n = evalBound(x);
    if (!n)
      return failure();
    hi = std::min(hi, *n);
  }
  lo = std::clamp(lo, int64_t(0), p.block);
  hi = std::clamp(hi, lo, p.block);
  auto o = staticValue(p.anchorOffset);
  if (!o || (__int128)*o + (__int128)p.inputStride * p.block > INT64_MAX)
    return failure();
  // Each leaf is affine in the period row and its contiguous column interval.
  // The final body+tail memory-piece budget is eight.
  struct Leaf {
    int64_t start, width;
  };
  SmallVector<Leaf> leaves;
  int64_t width = std::min(p.period, hi - lo);
  auto remainder = [&](int64_t i) {
    return (*o + p.inputStride * (lo + i)) % p.divisor;
  };
  for (int64_t c = 0; c < width;) {
    auto step = p.inputStride % p.divisor;
    int64_t run = (p.divisor - 1 - remainder(c)) / step + 1;
    int64_t end = std::min(width, c + run);
    leaves.push_back({c, end - c});
    if (leaves.size() > 8)
      return failure();
    c = end;
  }
  int64_t rows = (hi - lo) / p.period, tail = (hi - lo) % p.period;
  size_t pieces = rows ? leaves.size() : 0;
  if (tail)
    pieces += llvm::count_if(
        leaves, [&](const Leaf &leaf) { return leaf.start < tail; });
  if (pieces > 8)
    return failure();
  Emitter em(b, op.getLoc(), p);
  auto element =
      cast<triton::PointerType>(
          cast<RankedTensorType>(op.getPtr().getType()).getElementType())
          .getPointeeType();
  bool simpleInterval = p.remLowers.empty() && p.remUppers.empty();
  Value result, maskResult;
  auto falseValue =
      b.create<arith::ConstantOp>(op.getLoc(), b.getI32IntegerAttr(0));
  if constexpr (isLoad) {
    if (simpleInterval && op.getOther())
      result = op.getOther();
    else
      result = b.create<tensor::EmptyOp>(op.getLoc(),
                                         ArrayRef<int64_t>{p.block}, element);
    if (!simpleInterval && (op.getOther() || outputMask))
      maskResult = em.fill(falseValue, {p.block});
  }
  auto rectExtract = [&](Value v, int64_t row, int64_t col, int64_t rows,
                         int64_t cols) -> Value {
    return b.create<tensor::ExtractSliceOp>(
        op.getLoc(), v,
        SmallVector<OpFoldResult>{b.getIndexAttr(row), b.getIndexAttr(col)},
        SmallVector<OpFoldResult>{b.getIndexAttr(rows), b.getIndexAttr(cols)},
        SmallVector<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});
  };
  auto rectInsert = [&](Value v, Value dst, int64_t c, int64_t rows,
                        int64_t cols) -> Value {
    return b.create<tensor::InsertSliceOp>(
        op.getLoc(), v, dst,
        SmallVector<OpFoldResult>{b.getIndexAttr(0), b.getIndexAttr(c)},
        SmallVector<OpFoldResult>{b.getIndexAttr(rows), b.getIndexAttr(cols)},
        SmallVector<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});
  };
  auto emit = [&](int64_t start, ArrayRef<int64_t> shape,
                  Value value) -> std::pair<Value, Value> {
    auto off = em.tensorExpr(p.pointer, em.c(start), shape);
    auto ptrs = em.fill(p.source, shape);
    auto ptr =
        b.create<triton::AddPtrOp>(op.getLoc(), ptrs.getType(), ptrs, off);
    auto mask = em.mask(em.c(start), em.c(shape[0]), shape);
    if constexpr (isLoad) {
      Value load = createPieceLoad(op, p, em, em.c(start), em.c(shape[0]),
                                   shape, ptr, mask, b);
      return {load, mask};
    } else {
      markSegmented(b.create<triton::StoreOp>(op.getLoc(), ptr, value, mask,
                                              op.getBoundaryCheck(),
                                              op.getCache(), op.getEvict()));
      return {{}, {}};
    }
  };

  if (rows) {
    SmallVector<int64_t> bodyShape{rows, p.period};
    Value body, bodyMask;
    if constexpr (isLoad) {
      body = b.create<tensor::EmptyOp>(op.getLoc(), bodyShape, element);
      if (maskResult)
        bodyMask = em.fill(falseValue, bodyShape);
    } else {
      body =
          em.reshape(em.extract(op.getValue(), em.c(lo), em.c(rows * p.period)),
                     bodyShape);
    }
    for (auto leaf : leaves) {
      Value value;
      if constexpr (!isLoad)
        value = rectExtract(body, 0, leaf.start, rows, leaf.width);
      auto [v, m] =
          emit(lo + leaf.start, SmallVector<int64_t>{rows, leaf.width}, value);
      if constexpr (isLoad)
        if (bodyMask && !m) {
          auto t =
              b.create<arith::ConstantOp>(op.getLoc(), b.getBoolAttr(true));
          m = em.fill(t, SmallVector<int64_t>{rows, leaf.width});
        }
      if constexpr (isLoad) {
        body = rectInsert(v, body, leaf.start, rows, leaf.width);
        if (bodyMask)
          bodyMask = rectInsert(maskCarrier(m, b, op.getLoc()), bodyMask,
                                leaf.start, rows, leaf.width);
      }
    }
    if constexpr (isLoad) {
      result =
          em.insert(em.flatten(body), result, em.c(lo), em.c(rows * p.period));
      if (bodyMask)
        maskResult = em.insert(em.flatten(bodyMask), maskResult, em.c(lo),
                               em.c(rows * p.period));
    }
  }
  if (tail) {
    for (auto leaf : leaves) {
      if (leaf.start >= tail)
        break;
      auto n = std::min(leaf.width, tail - leaf.start);
      auto start = lo + rows * p.period + leaf.start;
      Value value;
      if constexpr (!isLoad)
        value = em.extract(op.getValue(), em.c(start), em.c(n));
      auto [v, m] = emit(start, SmallVector<int64_t>{n}, value);
      if constexpr (isLoad)
        if (maskResult && !m) {
          auto t =
              b.create<arith::ConstantOp>(op.getLoc(), b.getBoolAttr(true));
          m = em.fill(t, SmallVector<int64_t>{n});
        }
      if constexpr (isLoad) {
        result = em.insert(v, result, em.c(start), em.c(n));
        if (maskResult)
          maskResult = em.insert(maskCarrier(m, b, op.getLoc()), maskResult,
                                 em.c(start), em.c(n));
      }
    }
  }
  if constexpr (isLoad) {
    if (simpleInterval && outputMask)
      maskResult = em.intervalMask(em.c(lo), em.c(hi), p.block);
    else if (!simpleInterval)
      maskResult = restoredMask(maskResult, em, b, op.getLoc());
    if (outputMask)
      *outputMask = maskResult;
    if (!simpleInterval && op.getOther())
      result = b.create<arith::SelectOp>(op.getLoc(), maskResult, result,
                                         op.getOther());
    b.replaceOp(op, result);
  } else
    b.eraseOp(op);
  return success();
}

template <typename Op>
LogicalResult rewrite(Op op, PatternRewriter &b, bool optimize,
                      Value *outputMask = nullptr,
                      bool allowAlignedFallback = false) {
  constexpr bool isLoad = std::is_same_v<Op, triton::LoadOp>;
  auto ty = dyn_cast<RankedTensorType>(op.getPtr().getType());
  if (!ty || ty.getEncoding() || ty.getRank() != 1 || !ty.hasStaticShape() ||
      ty.getNumElements() <= 0 || ty.getNumElements() > INT32_MAX)
    return failure();
  if (op.getMask() &&
      cast<RankedTensorType>(op.getMask().getType()).getEncoding())
    return failure();
  if constexpr (isLoad) {
    if (cast<RankedTensorType>(op.getType()).getEncoding())
      return failure();
  } else if (cast<RankedTensorType>(op.getValue().getType()).getEncoding()) {
    return failure();
  }
  if constexpr (isLoad)
    if (op.getIsVolatile())
      return failure();
  Plan p;
  p.block = ty.getNumElements();
  p.oldMask = op.getMask();
  if (!p.ptr(op.getPtr()) || !p.mask(op.getMask()) || !p.sawDivRem ||
      !p.anchorOffset)
    return failure();
  // Keep the true aligned fast path in the existing factorization converter.
  auto offset = staticValue(p.anchorOffset);
  bool alignedOrigin = offset && *offset % p.divisor == 0;
  if (!alignedOrigin && p.anchorOffset->kind == Expr::Scalar)
    alignedOrigin =
        isKnownNonNegativeMultiple(p.anchorOffset->value, p.divisor);
  if (!allowAlignedFallback && !p.generalStride && alignedOrigin &&
      (p.block % p.period == 0 || p.period % p.block == 0) && !p.hasLinearMask)
    return failure();
  if (!offset && !optimize)
    return failure();
  if constexpr (!isLoad)
    if (!storeInjective(p.pointer, p.divisor, p.inputStride, p.period))
      return failure();
  if (p.generalStride)
    return rewriteGeneral(op, b, p, outputMask);
  Emitter em(b, op.getLoc(), p);
  auto zero = em.c(0), block = em.c(p.block), D = em.c(p.divisor),
       P = em.c(p.period), K = em.c(p.inputStride);
  Value lo = zero, hi = block;
  for (auto &x : p.lowers)
    lo = em.max(lo, em.bound(x));
  for (auto &x : p.uppers)
    hi = em.min(hi, em.bound(x));
  lo = em.min(block, em.max(zero, lo));
  hi = em.min(block, em.max(lo, hi));
  auto length = em.sub(hi, lo);
  auto origin = em.add(em.scalar(p.anchorOffset), em.mul(lo, K));
  auto phase = em.rem(origin, D);
  auto residue = em.rem(phase, K);
  auto nonzero = b.createOrFold<arith::CmpIOp>(
      op.getLoc(), arith::CmpIPredicate::ne, phase, residue);
  auto distance = em.sub(D, phase);
  auto headCount = b.createOrFold<arith::CeilDivSIOp>(op.getLoc(), distance, K);
  auto headCandidate = em.min(length, headCount);
  auto head = b.createOrFold<arith::SelectOp>(op.getLoc(), nonzero,
                                              headCandidate, zero);
  auto bodyStart = em.add(lo, head);
  auto bodyRows = em.div(em.sub(length, head), P);
  auto bodyLength = em.mul(bodyRows, P);
  auto tailStart = em.add(bodyStart, bodyLength);
  auto tail = em.sub(hi, tailStart);
  auto element =
      cast<triton::PointerType>(ty.getElementType()).getPointeeType();
  bool simpleInterval = p.remLowers.empty() && p.remUppers.empty();
  Value result, assembledMask;
  if constexpr (isLoad) {
    if (simpleInterval && op.getOther())
      result = op.getOther();
    else
      result = b.create<tensor::EmptyOp>(op.getLoc(),
                                         ArrayRef<int64_t>{p.block}, element);
    if (!simpleInterval && (op.getOther() || outputMask)) {
      auto f = b.create<arith::ConstantOp>(op.getLoc(), b.getI32IntegerAttr(0));
      assembledMask = em.fill(f, {p.block});
    }
  }
  auto emitPiece = [&](Value start, Value count, int64_t capacity, bool body) {
    auto n = em.mixed(count);
    if (auto a = dyn_cast<Attribute>(n);
        a && cast<IntegerAttr>(a).getInt() == 0)
      return;
    SmallVector<int64_t> shape = body ? SmallVector<int64_t>{capacity, p.period}
                                      : SmallVector<int64_t>{capacity};
    auto validLength = body ? em.mul(count, P) : count;
    auto offsets = em.tensorExpr(p.pointer, start, shape);
    auto ptrs = em.fill(p.source, shape);
    auto ptr =
        b.create<triton::AddPtrOp>(op.getLoc(), ptrs.getType(), ptrs, offsets);
    auto mask = em.mask(start, count, shape);
    if constexpr (isLoad) {
      auto load = createPieceLoad(op, p, em, start, count, shape, ptr, mask, b);
      auto flat = em.flatten(load);
      auto slice = em.extract(flat, zero, validLength);
      result = em.insert(slice, result, start, validLength);
      if (assembledMask) {
        if (!mask) {
          auto t =
              b.create<arith::ConstantOp>(op.getLoc(), b.getBoolAttr(true));
          mask = em.fill(t, shape);
        }
        auto flatMask = em.flatten(maskCarrier(mask, b, op.getLoc()));
        auto maskSlice = em.extract(flatMask, zero, validLength);
        assembledMask = em.insert(maskSlice, assembledMask, start, validLength);
      }
    } else {
      auto slice = em.extract(op.getValue(), start, validLength);
      int64_t flatCapacity = body ? capacity * p.period : capacity;
      Value carrier = b.create<tensor::EmptyOp>(
          op.getLoc(), ArrayRef<int64_t>{flatCapacity}, element);
      carrier = em.insert(slice, carrier, zero, validLength);
      carrier = em.reshape(carrier, shape);
      markSegmented(b.create<triton::StoreOp>(op.getLoc(), ptr, carrier, mask,
                                              op.getBoundaryCheck(),
                                              op.getCache(), op.getEvict()));
    }
  };
  int64_t cap = std::min(p.block, p.period);
  emitPiece(lo, head, cap, false);
  if (p.block / p.period)
    emitPiece(bodyStart, bodyRows, p.block / p.period, true);
  emitPiece(tailStart, tail, cap, false);
  if constexpr (isLoad) {
    if (simpleInterval && outputMask)
      assembledMask = em.intervalMask(lo, hi, p.block);
    else if (!simpleInterval)
      assembledMask = restoredMask(assembledMask, em, b, op.getLoc());
    if (outputMask)
      *outputMask = assembledMask;
    if (!simpleInterval && assembledMask && op.getOther())
      result = b.create<arith::SelectOp>(op.getLoc(), assembledMask, result,
                                         op.getOther());
    b.replaceOp(op, result);
  } else
    b.eraseOp(op);
  return success();
}
} // namespace
LogicalResult tryRewriteSegmentedLoad(triton::LoadOp op,
                                      PatternRewriter &rewriter,
                                      bool optimizeDynamicOffset,
                                      Value *rewrittenMask,
                                      bool allowAlignedFallback) {
  return rewrite(op, rewriter, optimizeDynamicOffset, rewrittenMask,
                 allowAlignedFallback);
}
LogicalResult tryRewriteSegmentedStore(triton::StoreOp op,
                                       PatternRewriter &rewriter,
                                       bool optimizeDynamicOffset,
                                       bool allowAlignedFallback) {
  return rewrite(op, rewriter, optimizeDynamicOffset, nullptr,
                 allowAlignedFallback);
}
} // namespace TritonToStructured
