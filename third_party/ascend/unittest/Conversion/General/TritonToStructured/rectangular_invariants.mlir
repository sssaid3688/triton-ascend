// RUN: triton-opt '--triton-to-structured=enable-mask-fallback-conversion=false optimize-dynamic-offset=true' --split-input-file %s | FileCheck %s
// RUN: triton-opt '--triton-to-structured=enable-mask-fallback-conversion=false optimize-dynamic-offset=false' --split-input-file %s | FileCheck %s
// Rectangular fast-path invariants. All dynamic program-id arithmetic below
// follows the supported nonnegative, no-index-overflow indexing domain.

tt.func public @aligned_rem_mask_nonsquare(%p: !tt.ptr<f32>) -> tensor<8xf32> {
  %i = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %d = arith.constant dense<4> : tensor<8xi32>
  %x = arith.remsi %i, %d : tensor<8xi32>
  %p0 = tt.splat %p : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %p1 = tt.addptr %p0, %x : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
  %limit = arith.constant dense<2> : tensor<8xi32>
  %mask = arith.cmpi slt, %x, %limit : tensor<8xi32>
  %other = arith.constant dense<0.0> : tensor<8xf32>
  %v = tt.load %p1, %mask, %other : tensor<8x!tt.ptr<f32>>
  tt.return %v : tensor<8xf32>
}

// CHECK-LABEL: @aligned_rem_mask_nonsquare(
// CHECK-NOT: arith.remsi
// CHECK: tt.load {{.*}} : tensor<4x!tt.ptr<f32>>
// CHECK: linalg.broadcast {{.*}}tensor<2x4xf32>
// CHECK: tt.return

// -----

tt.func public @aligned_rem_mask_large(%p: !tt.ptr<f32>) -> tensor<256xf32> {
  %i = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
  %d = arith.constant dense<128> : tensor<256xi32>
  %x = arith.remsi %i, %d : tensor<256xi32>
  %p0 = tt.splat %p : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
  %p1 = tt.addptr %p0, %x : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
  %limit = arith.constant dense<2> : tensor<256xi32>
  %mask = arith.cmpi slt, %x, %limit : tensor<256xi32>
  %other = arith.constant dense<0.0> : tensor<256xf32>
  %v = tt.load %p1, %mask, %other : tensor<256x!tt.ptr<f32>>
  tt.return %v : tensor<256xf32>
}

// CHECK-LABEL: @aligned_rem_mask_large(
// CHECK-NOT: arith.remsi
// CHECK: tt.load {{.*}} : tensor<128x!tt.ptr<f32>>
// CHECK: linalg.broadcast {{.*}}tensor<2x128xf32>
// CHECK: tt.return

// -----

tt.func public @negative_offset_div(%p: !tt.ptr<f32>) -> tensor<8xf32> {
  %i = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %d = arith.constant dense<4> : tensor<8xi32>
  %shift = arith.constant dense<-4> : tensor<8xi32>
  %shifted = arith.addi %i, %shift : tensor<8xi32>
  %x = arith.divsi %shifted, %d : tensor<8xi32>
  %p0 = tt.splat %p : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %p1 = tt.addptr %p0, %x : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
  %v = tt.load %p1 : tensor<8x!tt.ptr<f32>>
  tt.return %v : tensor<8xf32>
}

// CHECK-LABEL: @negative_offset_div(
// CHECK: arith.divsi {{.*}} : tensor<8xi32>
// CHECK: tt.load {{.*}} : tensor<8x!tt.ptr<f32>>
// CHECK: tt.return

// -----

tt.func public @unknown_signed_hint_div(%p: !tt.ptr<f32>, %offset: i32) -> tensor<8xf32> {
  %i = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %d = arith.constant dense<4> : tensor<8xi32>
  %one = arith.constant 1 : i32
  %hinted = arith.muli %offset, %one {tt.divisibility = dense<4> : tensor<1xi32>} : i32
  %bases = tt.splat %hinted : i32 -> tensor<8xi32>
  %shifted = arith.addi %i, %bases : tensor<8xi32>
  %x = arith.divsi %shifted, %d : tensor<8xi32>
  %p0 = tt.splat %p : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %p1 = tt.addptr %p0, %x : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
  %v = tt.load %p1 : tensor<8x!tt.ptr<f32>>
  tt.return %v : tensor<8xf32>
}

// CHECK-LABEL: @unknown_signed_hint_div(
// CHECK: arith.divsi {{.*}} : tensor<8xi32>
// CHECK: tt.load {{.*}} : tensor<8x!tt.ptr<f32>>
// CHECK: tt.return

// -----

tt.func public @negative_offset_rem(%p: !tt.ptr<f32>) -> tensor<8xf32> {
  %i = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %d = arith.constant dense<4> : tensor<8xi32>
  %shift = arith.constant dense<-4> : tensor<8xi32>
  %shifted = arith.addi %i, %shift : tensor<8xi32>
  %x = arith.remsi %shifted, %d : tensor<8xi32>
  %p0 = tt.splat %p : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %p1 = tt.addptr %p0, %x : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
  %v = tt.load %p1 : tensor<8x!tt.ptr<f32>>
  tt.return %v : tensor<8xf32>
}

// CHECK-LABEL: @negative_offset_rem(
// CHECK: arith.remsi {{.*}} : tensor<8xi32>
// CHECK: tt.load {{.*}} : tensor<8x!tt.ptr<f32>>
// CHECK: tt.return

// -----

tt.func public @unknown_signed_hint_rem(%p: !tt.ptr<f32>, %offset: i32) -> tensor<8xf32> {
  %i = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %d = arith.constant dense<4> : tensor<8xi32>
  %one = arith.constant 1 : i32
  %hinted = arith.muli %offset, %one {tt.divisibility = dense<4> : tensor<1xi32>} : i32
  %bases = tt.splat %hinted : i32 -> tensor<8xi32>
  %shifted = arith.addi %i, %bases : tensor<8xi32>
  %x = arith.remsi %shifted, %d : tensor<8xi32>
  %p0 = tt.splat %p : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %p1 = tt.addptr %p0, %x : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
  %v = tt.load %p1 : tensor<8x!tt.ptr<f32>>
  tt.return %v : tensor<8xf32>
}

// CHECK-LABEL: @unknown_signed_hint_rem(
// CHECK: arith.remsi {{.*}} : tensor<8xi32>
// CHECK: tt.load {{.*}} : tensor<8x!tt.ptr<f32>>
// CHECK: tt.return

// -----

tt.func public @program_id_aligned_div(%p: !tt.ptr<f32>) -> tensor<8xf32> {
  %i = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %d = arith.constant dense<4> : tensor<8xi32>
  %block = arith.constant 8 : i32
  %pid = tt.get_program_id x : i32
  %base = arith.muli %pid, %block : i32
  %bases = tt.splat %base : i32 -> tensor<8xi32>
  %shifted = arith.addi %i, %bases : tensor<8xi32>
  %x = arith.divsi %shifted, %d : tensor<8xi32>
  %p0 = tt.splat %p : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %p1 = tt.addptr %p0, %x : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
  %v = tt.load %p1 : tensor<8x!tt.ptr<f32>>
  tt.return %v : tensor<8xf32>
}

// CHECK-LABEL: @program_id_aligned_div(
// CHECK: tt.load {{.*}} : tensor<2x!tt.ptr<f32>>
// CHECK: linalg.broadcast {{.*}}tensor<2x4xf32>
// CHECK: tt.return
