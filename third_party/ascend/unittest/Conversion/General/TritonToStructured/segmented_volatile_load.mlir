// RUN: triton-opt %s '--triton-to-structured=enable-mask-fallback-conversion=false optimize-dynamic-offset=true' --canonicalize --split-input-file | FileCheck %s
// Volatile repeated reads must also bypass the legacy rectangular converter.
// CHECK-LABEL: tt.func @volatile_period(
// CHECK: arith.divsi
// CHECK: tt.load {{.*}}isVolatile = true
// CHECK-SAME: tensor<128x!tt.ptr<f32>>
// CHECK: tt.return
module {
  tt.func @volatile_period(%p: !tt.ptr<f32>) -> tensor<128xf32> {
    %r = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %d = arith.constant dense<16> : tensor<128xi32>
    %q = arith.divsi %r, %d : tensor<128xi32>
    %base = tt.splat %p : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %ptr = tt.addptr %base, %q : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %v = tt.load %ptr {isVolatile = true} : tensor<128x!tt.ptr<f32>>
    tt.return %v : tensor<128xf32>
  }
}

// -----

// CHECK-LABEL: tt.func @volatile_broadcast(
// CHECK: tt.broadcast
// CHECK: tt.load {{.*}}isVolatile = true
// CHECK-SAME: tensor<2x8x!tt.ptr<f32>>
// CHECK: tt.return
module {
  tt.func @volatile_broadcast(%p: !tt.ptr<f32>) -> tensor<2x8xf32> {
    %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %base = tt.splat %p : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %ptr = tt.addptr %base, %r : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %expanded = tt.expand_dims %ptr {axis = 0 : i32} : tensor<8x!tt.ptr<f32>> -> tensor<1x8x!tt.ptr<f32>>
    %broadcast = tt.broadcast %expanded : tensor<1x8x!tt.ptr<f32>> -> tensor<2x8x!tt.ptr<f32>>
    %v = tt.load %broadcast {isVolatile = true} : tensor<2x8x!tt.ptr<f32>>
    tt.return %v : tensor<2x8xf32>
  }
}
