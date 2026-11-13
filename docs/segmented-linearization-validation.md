# Segmented structured linearization

This change extends the existing `TritonToStructured` pass for incomplete
quotient/remainder periods and nonzero starting phases. It preserves the
rectangular fast path when the shape and origin are provably aligned.

The implementation plans a head, a full-period body, and a tail from structured
shape, stride, phase, and mask information. It reconstructs the original lane
order and `other` values without unmasked fallback accesses. Repeated loads read
one period or the valid rows before broadcasting. Stores require an injective
address proof; volatile loads retain their original access structure.

The new internal segment plan also handles rank-2/rank-3 single-axis broadcasts,
transposes, and bounded independent outer-axis combinations. Ordinary-stride
and lifted plans are limited to eight actual memory fragments. No public dialect
operation is added. Dynamic origins use the existing launch option:

```python
kernel[grid](..., optimize_dynamic_offset=True)
```

## Validation snapshot

Measured on A5 with the existing CANN 9.2.0 installation on 2026-11-13, against
baseline commit `0a5378d28abf6bbcd8d8916e03d397e9ed886a55`. Hardware numerical
checks primarily cover FP32 data and signed i32/i64 indices.

| Test population | Result |
| --- | --- |
| Complete IR/address/numerical matrix | 1,464 executed; 882 actually converted; zero numerical, address-safety, evaluator, or compiler-exit failures |
| Provably nonnegative, unit-stride supported population | 768/768 converted |
| Non-divisible shapes within that supported population | 529/529 converted |
| All non-divisible-period cases, including conservative fallbacks | 553/958 converted |
| Program-id-derived dynamic origins | 120/120 converted |
| Seeded random and interval-mask cases | 304/304 converted, numerically correct, and address-safe; checked with both mask-fallback settings |
| IR regression commands | 15 FileCheck runs passed |
| Previously working aligned IR | 24/24 byte-identical to baseline |
| Actual NPU execution | 57 one-dimensional/boundary plus 6 multidimensional cases passed; output guards intact |

Conversion success means the target address/memory-mask tensor div/rem was
removed, or an entirely inactive access was eliminated. Keeping an expression
for the existing lowering is a fallback, not a successful conversion. The 529
cases are a subset of the 958 cases, not an additional test population. These
counts describe this matrix; they are not a universal success-rate claim.

The device cases include empty and single-element valid regions, NaN and
per-element `other`, nonzero phases, periodic masks, transposes, and two bounded
independent axes. A successful structured pass does not imply an entirely SIMD
final kernel: some validated cases retain mixed backend paths.

## Performance

The actual old/new adapters for aligned rem, div, and packed kernels are
identical after removing debug locations. Five interleaved baseline/control/new
rounds did not show a definite added cost; shared-device noise and wide
confidence intervals preclude a universal fixed-percentage guarantee.

For the following nonaligned examples (`B=1024`, `N=262131`), three paired rounds
against correct handwritten code gave:

| Access | New median (us) | Handwritten median (us) | Median paired new/handwritten ratio |
| --- | ---: | ---: | ---: |
| Remainder, D=16 | 5.5785 | 5.7630 | 0.9680 |
| Remainder, D=24 | 9.7505 | 10.1570 | 0.9611 |
| Packed, D=65 | 11.4625 | 10.4580 | 1.1004 |
| Packed, D=65, periodic mask | 8.5650 | 8.8815 | 0.9611 |

These examples range from about -3.9% to +10.0% relative to handwritten code;
this is not a claim that every template accelerates. Some original SIMT fallback
kernels remain faster. Explicit period-load reuse and 32-byte-aligned column
splitting avoid the large repeated/scalar GM-transfer regressions found during
validation.

Current CANN requires an aligned UB staging/assembly copy for some irregular
packed shapes. A dynamic two-dimensional result-view fusion was rejected by
CANN's `PropagateReshape` pass and is not part of this change. The implementation
does not claim zero additional copying. The no-periodic-mask path initializes
from `other` and avoids a final whole-result recovery copy.

## Reproduction and limits

Build `triton-opt` using the repository installation instructions. The CPU
runners accept explicit paths and produce JSON reports:

```bash
python third_party/ascend/unittest/test_segmented_linearization.py \
  --triton-opt /path/to/triton-opt --report /path/to/matrix.json
python third_party/ascend/unittest/test_segmented_fuzz.py \
  --triton-opt /path/to/triton-opt --report /path/to/fuzz.json
```

The `.mlir` files under
`third_party/ascend/unittest/Conversion/General/TritonToStructured` contain their
FileCheck `RUN` commands. The two `*_npu.py` runners are explicit command-line
hardware experiments, skipped during normal test collection. They retain the
original experiment root `/home/oyhj/triton_cann` and expect its baseline/cache
layout; adapt that root and the CANN environment for another machine. Device
experiments should run serially when collecting performance measurements.

Unknown signed origins without a nonnegativity proof, negative indices, runtime
divisors, encoded tensors, unsafe repeated stores, unsupported mixed anchors,
ordinary strides with dynamic lane bounds, and over-budget plans conservatively
keep their existing representation. Index arithmetic is assumed not to overflow
within the supported domain. This validation does not cover every data dtype.
