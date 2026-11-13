// RUN: triton-opt %s '--triton-to-structured=enable-mask-fallback-conversion=false optimize-dynamic-offset=true' --split-input-file | FileCheck %s
// Ordinary repeated-address stores have lane-dependent values. They must retain
// their original memory semantics, not be compressed like repeated loads.
// CHECK-LABEL: tt.func public @repeat_rem_unaligned
// CHECK: arith.remsi
// CHECK-NOT: tensor.extract_slice
// CHECK: tt.store
module {
  tt.func public @repeat_rem_unaligned(%p: !tt.ptr<f32>) {
    %r = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %d = arith.constant dense<3> : tensor<128xi32>
    %i = arith.remsi %r, %d : tensor<128xi32>
    %value = arith.sitofp %r : tensor<128xi32> to tensor<128xf32>
    %ps = tt.splat %p : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %ptr = tt.addptr %ps, %i : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    tt.store %ptr, %value : tensor<128x!tt.ptr<f32>>
    tt.return
  }
}
// -----
// CHECK-LABEL: tt.func public @repeat_div_aligned
// CHECK: arith.divsi
// CHECK-NOT: tensor.extract_slice
// CHECK: tt.store
module {
  tt.func public @repeat_div_aligned(%p: !tt.ptr<f32>) {
    %r = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %d = arith.constant dense<16> : tensor<128xi32>
    %i = arith.divsi %r, %d : tensor<128xi32>
    %value = arith.sitofp %r : tensor<128xi32> to tensor<128xf32>
    %ps = tt.splat %p : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %ptr = tt.addptr %ps, %i : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    tt.store %ptr, %value : tensor<128x!tt.ptr<f32>>
    tt.return
  }
}
