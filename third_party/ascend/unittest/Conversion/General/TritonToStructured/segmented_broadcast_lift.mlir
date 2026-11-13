// RUN: triton-opt '--triton-to-structured=enable-mask-fallback-conversion=false optimize-dynamic-offset=true' --split-input-file %s | FileCheck %s

// A varying full-rank other must be applied after restoring the projected load.
tt.func @rank2_full_other(%p: !tt.ptr<f32>, %other: tensor<2x8xf32>) -> tensor<2x8xf32> {
  %i = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %e = tt.expand_dims %i {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
  %b = tt.broadcast %e : tensor<1x8xi32> -> tensor<2x8xi32>
  %d = arith.constant dense<3> : tensor<2x8xi32>
  %limit = arith.constant dense<2> : tensor<2x8xi32>
  %r = arith.remsi %b, %d : tensor<2x8xi32>
  %mask = arith.cmpi slt, %r, %limit : tensor<2x8xi32>
  %base = tt.splat %p : !tt.ptr<f32> -> tensor<2x8x!tt.ptr<f32>>
  %ptr = tt.addptr %base, %r : tensor<2x8x!tt.ptr<f32>>, tensor<2x8xi32>
  %v = tt.load %ptr, %mask, %other : tensor<2x8x!tt.ptr<f32>>
  tt.return %v : tensor<2x8xf32>
}
// CHECK-LABEL: @rank2_full_other(
// CHECK-NOT: arith.remsi
// CHECK: tt.load
// CHECK-NOT: arith.remsi
// CHECK: tt.broadcast {{.*}} -> tensor<2x8xi1>
// CHECK: arith.select {{.*}} : tensor<2x8xi1>, tensor<2x8xf32>
// CHECK: tt.return

// -----

tt.func @rank3_middle_axis(%p: !tt.ptr<f32>) -> tensor<2x8x3xf32> {
  %i = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %e0 = tt.expand_dims %i {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
  %e1 = tt.expand_dims %e0 {axis = 2 : i32} : tensor<1x8xi32> -> tensor<1x8x1xi32>
  %b = tt.broadcast %e1 : tensor<1x8x1xi32> -> tensor<2x8x3xi32>
  %d = arith.constant dense<3> : tensor<2x8x3xi32>
  %r = arith.divsi %b, %d : tensor<2x8x3xi32>
  %base = tt.splat %p : !tt.ptr<f32> -> tensor<2x8x3x!tt.ptr<f32>>
  %ptr = tt.addptr %base, %r : tensor<2x8x3x!tt.ptr<f32>>, tensor<2x8x3xi32>
  %v = tt.load %ptr : tensor<2x8x3x!tt.ptr<f32>>
  tt.return %v : tensor<2x8x3xf32>
}
// CHECK-LABEL: @rank3_middle_axis(
// CHECK-NOT: arith.divsi {{.*}} : tensor
// CHECK: tt.load
// CHECK: tt.broadcast {{.*}} -> tensor<2x8x3xf32>
// CHECK: tt.return

// -----

tt.func @transposed_axis(%p: !tt.ptr<f32>, %other: tensor<8x2xf32>) -> tensor<8x2xf32> {
  %i = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %e = tt.expand_dims %i {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
  %b = tt.broadcast %e : tensor<1x8xi32> -> tensor<2x8xi32>
  %d = arith.constant dense<3> : tensor<2x8xi32>
  %limit = arith.constant dense<2> : tensor<2x8xi32>
  %r = arith.remsi %b, %d : tensor<2x8xi32>
  %m = arith.cmpi slt, %r, %limit : tensor<2x8xi32>
  %mask = tt.trans %m {order = array<i32: 1, 0>} : tensor<2x8xi1> -> tensor<8x2xi1>
  %base = tt.splat %p : !tt.ptr<f32> -> tensor<2x8x!tt.ptr<f32>>
  %ptr0 = tt.addptr %base, %r : tensor<2x8x!tt.ptr<f32>>, tensor<2x8xi32>
  %ptr = tt.trans %ptr0 {order = array<i32: 1, 0>} : tensor<2x8x!tt.ptr<f32>> -> tensor<8x2x!tt.ptr<f32>>
  %v = tt.load %ptr, %mask, %other : tensor<8x2x!tt.ptr<f32>>
  tt.return %v : tensor<8x2xf32>
}
// CHECK-LABEL: @transposed_axis(
// CHECK-NOT: arith.remsi
// CHECK: tt.load
// CHECK: tt.broadcast {{.*}} -> tensor<8x2xf32>
// CHECK: arith.select {{.*}} : tensor<8x2xi1>, tensor<8x2xf32>
// CHECK: tt.return

// -----

// Orthogonal repeated stores can be removed only for equal values and masks.
tt.func @uniform_broadcast_store(%p: !tt.ptr<f32>, %value: tensor<8xf32>) {
  %i = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %e = tt.expand_dims %i {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
  %b = tt.broadcast %e : tensor<1x8xi32> -> tensor<2x8xi32>
  %d = arith.constant dense<3> : tensor<2x8xi32>
  %scale = arith.constant dense<4> : tensor<2x8xi32>
  %q = arith.divsi %b, %d : tensor<2x8xi32>
  %rows = arith.muli %q, %scale : tensor<2x8xi32>
  %offset = arith.addi %rows, %b : tensor<2x8xi32>
  %v0 = tt.expand_dims %value {axis = 0 : i32} : tensor<8xf32> -> tensor<1x8xf32>
  %v = tt.broadcast %v0 : tensor<1x8xf32> -> tensor<2x8xf32>
  %base = tt.splat %p : !tt.ptr<f32> -> tensor<2x8x!tt.ptr<f32>>
  %ptr = tt.addptr %base, %offset : tensor<2x8x!tt.ptr<f32>>, tensor<2x8xi32>
  tt.store %ptr, %v : tensor<2x8x!tt.ptr<f32>>
  tt.return
}
// CHECK-LABEL: @uniform_broadcast_store(
// CHECK-NOT: arith.divsi {{.*}} : tensor
// CHECK: tt.store
// CHECK-NOT: tensor<2x8x!tt.ptr<f32>>
// CHECK: tt.return

// -----

tt.func @nonuniform_broadcast_store_kept(%p: !tt.ptr<f32>, %value: tensor<2x8xf32>) {
  %i = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %e = tt.expand_dims %i {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
  %b = tt.broadcast %e : tensor<1x8xi32> -> tensor<2x8xi32>
  %d = arith.constant dense<3> : tensor<2x8xi32>
  %q = arith.divsi %b, %d : tensor<2x8xi32>
  %base = tt.splat %p : !tt.ptr<f32> -> tensor<2x8x!tt.ptr<f32>>
  %ptr = tt.addptr %base, %q : tensor<2x8x!tt.ptr<f32>>, tensor<2x8xi32>
  tt.store %ptr, %value : tensor<2x8x!tt.ptr<f32>>
  tt.return
}
// CHECK-LABEL: @nonuniform_broadcast_store_kept(
// CHECK: arith.divsi {{.*}} : tensor<2x8xi32>
// CHECK: tt.store {{.*}} : tensor<2x8x!tt.ptr<f32>>
// CHECK: tt.return

// -----

tt.func @small_independent_axes(%p: !tt.ptr<f32>) -> tensor<2x8xf32> {
  %i = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %e = tt.expand_dims %i {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
  %b = tt.broadcast %e : tensor<1x8xi32> -> tensor<2x8xi32>
  %j = tt.make_range {start = 0 : i32, end = 2 : i32} : tensor<2xi32>
  %j0 = tt.expand_dims %j {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %j1 = tt.broadcast %j0 : tensor<2x1xi32> -> tensor<2x8xi32>
  %d = arith.constant dense<3> : tensor<2x8xi32>
  %q = arith.divsi %b, %d : tensor<2x8xi32>
  %offset = arith.addi %q, %j1 : tensor<2x8xi32>
  %base = tt.splat %p : !tt.ptr<f32> -> tensor<2x8x!tt.ptr<f32>>
  %ptr = tt.addptr %base, %offset : tensor<2x8x!tt.ptr<f32>>, tensor<2x8xi32>
  %v = tt.load %ptr : tensor<2x8x!tt.ptr<f32>>
  tt.return %v : tensor<2x8xf32>
}
// CHECK-LABEL: @small_independent_axes(
// CHECK-NOT: arith.divsi {{.*}} : tensor
// CHECK: tt.load
// CHECK: tensor.insert_slice
// CHECK: tt.return


// -----

// A second independent divisor can be folded after pinning the tiny outer axis.
tt.func @two_period_axes(%p: !tt.ptr<f32>) -> tensor<2x8xf32> {
  %i = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %e = tt.expand_dims %i {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
  %b = tt.broadcast %e : tensor<1x8xi32> -> tensor<2x8xi32>
  %j = tt.make_range {start = 0 : i32, end = 2 : i32} : tensor<2xi32>
  %j0 = tt.expand_dims %j {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %j1 = tt.broadcast %j0 : tensor<2x1xi32> -> tensor<2x8xi32>
  %one = arith.constant dense<1> : tensor<2x8xi32>
  %outerd = arith.constant dense<2> : tensor<2x8xi32>
  %d = arith.constant dense<3> : tensor<2x8xi32>
  %shift = arith.addi %j1, %one : tensor<2x8xi32>
  %outerq = arith.divsi %shift, %outerd : tensor<2x8xi32>
  %q = arith.divsi %b, %d : tensor<2x8xi32>
  %offset = arith.addi %q, %outerq : tensor<2x8xi32>
  %base = tt.splat %p : !tt.ptr<f32> -> tensor<2x8x!tt.ptr<f32>>
  %ptr = tt.addptr %base, %offset : tensor<2x8x!tt.ptr<f32>>, tensor<2x8xi32>
  %v = tt.load %ptr : tensor<2x8x!tt.ptr<f32>>
  tt.return %v : tensor<2x8xf32>
}
// CHECK-LABEL: @two_period_axes(
// CHECK-NOT: arith.divsi {{.*}} : tensor
// CHECK: tt.load
// CHECK: tensor.insert_slice
// CHECK: tt.return


// -----

// Eight independent outer lanes would exceed the rectangle budget.
tt.func @large_independent_axes_kept(%p: !tt.ptr<f32>) -> tensor<8x16xf32> {
  %i = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
  %e = tt.expand_dims %i {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %b = tt.broadcast %e : tensor<1x16xi32> -> tensor<8x16xi32>
  %j = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %j0 = tt.expand_dims %j {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
  %j1 = tt.broadcast %j0 : tensor<8x1xi32> -> tensor<8x16xi32>
  %d = arith.constant dense<3> : tensor<8x16xi32>
  %q = arith.divsi %b, %d : tensor<8x16xi32>
  %offset = arith.addi %q, %j1 : tensor<8x16xi32>
  %base = tt.splat %p : !tt.ptr<f32> -> tensor<8x16x!tt.ptr<f32>>
  %ptr = tt.addptr %base, %offset : tensor<8x16x!tt.ptr<f32>>, tensor<8x16xi32>
  %v = tt.load %ptr : tensor<8x16x!tt.ptr<f32>>
  tt.return %v : tensor<8x16xf32>
}
// CHECK-LABEL: @large_independent_axes_kept(
// CHECK: arith.divsi {{.*}} : tensor<8x16xi32>
// CHECK: tt.load {{.*}} : tensor<8x16x!tt.ptr<f32>>
// CHECK: tt.return


// -----

// The broadcast value is a real loaded vector, not an artificial tensor argument.
tt.func @loaded_vector_broadcast_store(%p: !tt.ptr<f32>, %valuep: !tt.ptr<f32>) {
  %vi = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %vb = tt.splat %valuep : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %vp = tt.addptr %vb, %vi : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
  %value = tt.load %vp : tensor<8x!tt.ptr<f32>>

  %i = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %e = tt.expand_dims %i {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
  %b = tt.broadcast %e : tensor<1x8xi32> -> tensor<2x8xi32>
  %d = arith.constant dense<3> : tensor<2x8xi32>
  %scale = arith.constant dense<4> : tensor<2x8xi32>
  %q = arith.divsi %b, %d : tensor<2x8xi32>
  %rows = arith.muli %q, %scale : tensor<2x8xi32>
  %offset = arith.addi %rows, %b : tensor<2x8xi32>
  %v0 = tt.expand_dims %value {axis = 0 : i32} : tensor<8xf32> -> tensor<1x8xf32>
  %v = tt.broadcast %v0 : tensor<1x8xf32> -> tensor<2x8xf32>
  %base = tt.splat %p : !tt.ptr<f32> -> tensor<2x8x!tt.ptr<f32>>
  %ptr = tt.addptr %base, %offset : tensor<2x8x!tt.ptr<f32>>, tensor<2x8xi32>
  tt.store %ptr, %v : tensor<2x8x!tt.ptr<f32>>
  tt.return
}
// CHECK-LABEL: @loaded_vector_broadcast_store(
// CHECK-NOT: arith.divsi {{.*}} : tensor
// CHECK: tt.load
// CHECK: tt.store
// CHECK-NOT: tensor<2x8x!tt.ptr<f32>>
// CHECK: tt.return
