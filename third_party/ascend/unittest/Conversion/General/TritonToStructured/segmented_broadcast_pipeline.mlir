// RUN: triton-opt --pass-pipeline="builtin.module(triton-to-structured{enable-mask-fallback-conversion=false optimize-dynamic-offset=true},triton-to-annotation,triton-to-unstructure,triton-to-hivm,triton-to-hfusion,triton-to-llvm,bubble-up-operation,triton-to-structured{enable-mask-fallback-conversion=false optimize-dynamic-offset=true},triton-to-linalg{global-kernel=false})" --split-input-file %s -o /dev/null
// Exercise the real lowering order; direct structure-to-linalg misses required preparation.

tt.func @rank2_full_other(%p: !tt.ptr<f32>, %otherp: !tt.ptr<f32>, %outputp: !tt.ptr<f32>) {
  %other_i = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
  %other_b = tt.splat %otherp : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %other_p = tt.addptr %other_b, %other_i : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  %other_flat = tt.load %other_p : tensor<16x!tt.ptr<f32>>
  %other_shape = arith.constant dense<[2, 8]> : tensor<2xi64>
  %other = tensor.reshape %other_flat(%other_shape) : (tensor<16xf32>, tensor<2xi64>) -> tensor<2x8xf32>
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
  %output_shape = arith.constant dense<[16]> : tensor<1xi64>
  %output_flat = tensor.reshape %v(%output_shape) : (tensor<2x8xf32>, tensor<1xi64>) -> tensor<16xf32>
  %output_i = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
  %output_b = tt.splat %outputp : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %output_p = tt.addptr %output_b, %output_i : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  tt.store %output_p, %output_flat : tensor<16x!tt.ptr<f32>>
  tt.return
}


// -----



tt.func @rank3_middle_axis(%p: !tt.ptr<f32>, %outputp: !tt.ptr<f32>) {
  %i = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %e0 = tt.expand_dims %i {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
  %e1 = tt.expand_dims %e0 {axis = 2 : i32} : tensor<1x8xi32> -> tensor<1x8x1xi32>
  %b = tt.broadcast %e1 : tensor<1x8x1xi32> -> tensor<2x8x3xi32>
  %d = arith.constant dense<3> : tensor<2x8x3xi32>
  %r = arith.divsi %b, %d : tensor<2x8x3xi32>
  %base = tt.splat %p : !tt.ptr<f32> -> tensor<2x8x3x!tt.ptr<f32>>
  %ptr = tt.addptr %base, %r : tensor<2x8x3x!tt.ptr<f32>>, tensor<2x8x3xi32>
  %v = tt.load %ptr : tensor<2x8x3x!tt.ptr<f32>>
  %output_shape = arith.constant dense<[48]> : tensor<1xi64>
  %output_flat = tensor.reshape %v(%output_shape) : (tensor<2x8x3xf32>, tensor<1xi64>) -> tensor<48xf32>
  %output_i = tt.make_range {start = 0 : i32, end = 48 : i32} : tensor<48xi32>
  %output_b = tt.splat %outputp : !tt.ptr<f32> -> tensor<48x!tt.ptr<f32>>
  %output_p = tt.addptr %output_b, %output_i : tensor<48x!tt.ptr<f32>>, tensor<48xi32>
  tt.store %output_p, %output_flat : tensor<48x!tt.ptr<f32>>
  tt.return
}


// -----



tt.func @transposed_axis(%p: !tt.ptr<f32>, %otherp: !tt.ptr<f32>, %outputp: !tt.ptr<f32>) {
  %other_i = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
  %other_b = tt.splat %otherp : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %other_p = tt.addptr %other_b, %other_i : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  %other_flat = tt.load %other_p : tensor<16x!tt.ptr<f32>>
  %other_shape = arith.constant dense<[8, 2]> : tensor<2xi64>
  %other = tensor.reshape %other_flat(%other_shape) : (tensor<16xf32>, tensor<2xi64>) -> tensor<8x2xf32>
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
  %output_shape = arith.constant dense<[16]> : tensor<1xi64>
  %output_flat = tensor.reshape %v(%output_shape) : (tensor<8x2xf32>, tensor<1xi64>) -> tensor<16xf32>
  %output_i = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
  %output_b = tt.splat %outputp : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %output_p = tt.addptr %output_b, %output_i : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  tt.store %output_p, %output_flat : tensor<16x!tt.ptr<f32>>
  tt.return
}


// -----



tt.func @uniform_broadcast_store(%p: !tt.ptr<f32>, %valuep: !tt.ptr<f32>) {
  %value_i = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %value_b = tt.splat %valuep : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %value_p = tt.addptr %value_b, %value_i : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
  %value = tt.load %value_p : tensor<8x!tt.ptr<f32>>
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

// -----
// Dynamic bounds + full-rank other used to bubble dynamic slices into illegal TT splats.
#loc = loc("/home/oyhj/triton_cann/src/triton-ascend/third_party/ascend/unittest/test_segmented_broadcast_npu.py":16:0)
#loc24 = loc("P"(#loc))
#loc25 = loc("O"(#loc))
#loc26 = loc("N"(#loc))
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @broadcast2(%P: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("P"(#loc)), %O: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("O"(#loc)), %N: i32 loc("N"(#loc))) attributes {noinline = false} {
    %other = arith.constant dense<2.500000e-01> : tensor<1x8xf32> loc(#loc27)
    %active = arith.constant dense<1> : tensor<1x8xi32> loc(#loc28)
    %cst = arith.constant dense<3> : tensor<1x8xi32> loc(#loc3)
    %outidx = arith.constant dense<8> : tensor<2x1xi32> loc(#loc29)
    %other_0 = arith.constant dense<-1.700000e+01> : tensor<2x1xf32> loc(#loc30)
    %other_1 = arith.constant dense<5.000000e-01> : tensor<2x1xf32> loc(#loc31)
    %c = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32> loc(#loc32)
    %c_2 = tt.expand_dims %c {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32> loc(#loc33)
    %row = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32> loc(#loc34)
    %row_3 = tt.expand_dims %row {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32> loc(#loc35)
    %logical = tt.broadcast %c_2 : tensor<1x8xi32> -> tensor<2x8xi32> loc(#loc36)
    %address = arith.remsi %c_2, %cst : tensor<1x8xi32> loc(#loc37)
    %active_4 = tt.splat %N : i32 -> tensor<1x8xi32> loc(#loc38)
    %active_5 = arith.cmpi slt, %c_2, %active_4 : tensor<1x8xi32> loc(#loc38)
    %active_6 = arith.cmpi slt, %address, %active : tensor<1x8xi32> loc(#loc28)
    %active_7 = arith.andi %active_5, %active_6 : tensor<1x8xi1> loc(#loc39)
    %active_8 = tt.broadcast %active_7 : tensor<1x8xi1> -> tensor<2x8xi1> loc(#loc39)
    %other_9 = arith.sitofp %row_3 : tensor<2x1xi32> to tensor<2x1xf32> loc(#loc40)
    %other_10 = arith.mulf %other_9, %other_1 : tensor<2x1xf32> loc(#loc31)
    %other_11 = arith.addf %other_10, %other_0 : tensor<2x1xf32> loc(#loc30)
    %other_12 = arith.sitofp %c_2 : tensor<1x8xi32> to tensor<1x8xf32> loc(#loc41)
    %other_13 = arith.mulf %other_12, %other : tensor<1x8xf32> loc(#loc27)
    %other_14 = tt.broadcast %other_13 : tensor<1x8xf32> -> tensor<2x8xf32> loc(#loc27)
    %other_15 = tt.broadcast %other_11 : tensor<2x1xf32> -> tensor<2x8xf32> loc(#loc42)
    %other_16 = arith.addf %other_15, %other_14 : tensor<2x8xf32> loc(#loc42)
    %outidx_17 = arith.muli %row_3, %outidx : tensor<2x1xi32> loc(#loc29)
    %outidx_18 = tt.broadcast %outidx_17 : tensor<2x1xi32> -> tensor<2x8xi32> loc(#loc43)
    %outidx_19 = arith.addi %outidx_18, %logical : tensor<2x8xi32> loc(#loc43)
    %v = tt.splat %P : !tt.ptr<f32> -> tensor<1x8x!tt.ptr<f32>> loc(#loc44)
    %v_20 = tt.addptr %v, %address : tensor<1x8x!tt.ptr<f32>>, tensor<1x8xi32> loc(#loc44)
    %v_21 = tt.broadcast %v_20 : tensor<1x8x!tt.ptr<f32>> -> tensor<2x8x!tt.ptr<f32>> loc(#loc44)
    %v_22 = tt.load %v_21, %active_8, %other_16 : tensor<2x8x!tt.ptr<f32>> loc(#loc45)
    %0 = tt.splat %O : !tt.ptr<f32> -> tensor<2x8x!tt.ptr<f32>> loc(#loc21)
    %1 = tt.addptr %0, %outidx_19 : tensor<2x8x!tt.ptr<f32>>, tensor<2x8xi32> loc(#loc21)
    tt.store %1, %v_22 : tensor<2x8x!tt.ptr<f32>> loc(#loc22)
    tt.return loc(#loc23)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("/home/oyhj/triton_cann/src/triton-ascend/third_party/ascend/unittest/test_segmented_broadcast_npu.py":31:70)
#loc2 = loc("/home/oyhj/triton_cann/src/triton-ascend/third_party/ascend/unittest/test_segmented_broadcast_npu.py":29:31)
#loc3 = loc(unknown)
#loc4 = loc("/home/oyhj/triton_cann/src/triton-ascend/third_party/ascend/unittest/test_segmented_broadcast_npu.py":32:15)
#loc5 = loc("/home/oyhj/triton_cann/src/triton-ascend/third_party/ascend/unittest/test_segmented_broadcast_npu.py":31:24)
#loc6 = loc("/home/oyhj/triton_cann/src/triton-ascend/third_party/ascend/unittest/test_segmented_broadcast_npu.py":31:43)
#loc7 = loc("/home/oyhj/triton_cann/src/triton-ascend/third_party/ascend/unittest/test_segmented_broadcast_npu.py":19:18)
#loc8 = loc("/home/oyhj/triton_cann/src/triton-ascend/third_party/ascend/unittest/test_segmented_broadcast_npu.py":19:21)
#loc9 = loc("/home/oyhj/triton_cann/src/triton-ascend/third_party/ascend/unittest/test_segmented_broadcast_npu.py":20:20)
#loc10 = loc("/home/oyhj/triton_cann/src/triton-ascend/third_party/ascend/unittest/test_segmented_broadcast_npu.py":20:23)
#loc11 = loc("/home/oyhj/triton_cann/src/triton-ascend/third_party/ascend/unittest/test_segmented_broadcast_npu.py":21:30)
#loc12 = loc("/home/oyhj/triton_cann/src/triton-ascend/third_party/ascend/unittest/test_segmented_broadcast_npu.py":23:14)
#loc13 = loc("/home/oyhj/triton_cann/src/triton-ascend/third_party/ascend/unittest/test_segmented_broadcast_npu.py":27:35)
#loc14 = loc("/home/oyhj/triton_cann/src/triton-ascend/third_party/ascend/unittest/test_segmented_broadcast_npu.py":29:25)
#loc15 = loc("/home/oyhj/triton_cann/src/triton-ascend/third_party/ascend/unittest/test_segmented_broadcast_npu.py":31:31)
#loc16 = loc("/home/oyhj/triton_cann/src/triton-ascend/third_party/ascend/unittest/test_segmented_broadcast_npu.py":31:58)
#loc17 = loc("/home/oyhj/triton_cann/src/triton-ascend/third_party/ascend/unittest/test_segmented_broadcast_npu.py":31:47)
#loc18 = loc("/home/oyhj/triton_cann/src/triton-ascend/third_party/ascend/unittest/test_segmented_broadcast_npu.py":32:17)
#loc19 = loc("/home/oyhj/triton_cann/src/triton-ascend/third_party/ascend/unittest/test_segmented_broadcast_npu.py":38:16)
#loc20 = loc("/home/oyhj/triton_cann/src/triton-ascend/third_party/ascend/unittest/test_segmented_broadcast_npu.py":38:24)
#loc21 = loc("/home/oyhj/triton_cann/src/triton-ascend/third_party/ascend/unittest/test_segmented_broadcast_npu.py":39:15)
#loc22 = loc("/home/oyhj/triton_cann/src/triton-ascend/third_party/ascend/unittest/test_segmented_broadcast_npu.py":39:22)
#loc23 = loc("/home/oyhj/triton_cann/src/triton-ascend/third_party/ascend/unittest/test_segmented_broadcast_npu.py":39:4)
#loc27 = loc("other"(#loc1))
#loc28 = loc("active"(#loc2))
#loc29 = loc("outidx"(#loc4))
#loc30 = loc("other"(#loc5))
#loc31 = loc("other"(#loc6))
#loc32 = loc("c"(#loc7))
#loc33 = loc("c"(#loc8))
#loc34 = loc("row"(#loc9))
#loc35 = loc("row"(#loc10))
#loc36 = loc("logical"(#loc11))
#loc37 = loc("address"(#loc12))
#loc38 = loc("active"(#loc13))
#loc39 = loc("active"(#loc14))
#loc40 = loc("other"(#loc15))
#loc41 = loc("other"(#loc16))
#loc42 = loc("other"(#loc17))
#loc43 = loc("outidx"(#loc18))
#loc44 = loc("v"(#loc19))
#loc45 = loc("v"(#loc20))
