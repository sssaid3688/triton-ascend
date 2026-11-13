// RUN: triton-opt %s --split-input-file --triton-to-annotation --triton-to-unstructure --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation --triton-to-linalg --canonicalize | FileCheck %s
// Regression: the masked store observes only the inserted prefix; the fixed
// empty carrier must not survive final lowering or force a carrier allocation.
// CHECK-LABEL: func.func @test(
// CHECK-NOT: tensor.empty
// CHECK-NOT: tensor.insert_slice
// CHECK-NOT: tensor<126xf32>
// CHECK: bufferization.materialize_in_destination
// CHECK-NOT: tensor.empty
// CHECK-NOT: tensor.insert_slice
// CHECK-NOT: tensor<126xf32>
// CHECK: return

module {
 tt.func public @test(%p: !tt.ptr<f32>, %o: !tt.ptr<f32>, %start: i32, %count: i32) {
  %c3 = arith.constant 3 : index
  %c126 = arith.constant 126 : index
  %c0 = arith.constant 0.0 : f32
  %s = arith.index_cast %start : i32 to index
  %q = arith.index_cast %count : i32 to index
  %n = arith.muli %q, %c3 : index
  %padn = arith.subi %c126, %n : index
  %ir = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %ips = tt.splat %p : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  %ip = tt.addptr %ips, %ir : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  %values = tt.load %ip : tensor<128x!tt.ptr<f32>>
  %part = tensor.extract_slice %values[%s] [%n] [1] : tensor<128xf32> to tensor<?xf32>
  %segempty = tensor.empty() : tensor<126xf32>
  %padded = tensor.insert_slice %part into %segempty[0] [%n] [1] : tensor<?xf32> into tensor<126xf32>
  %value2 = tensor.expand_shape %padded [[0, 1]] output_shape [42, 3] : tensor<126xf32> into tensor<42x3xf32>
  %rr = tt.make_range {end = 42 : i32, start = 0 : i32} : tensor<42xi32>
  %cc = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %r = tt.expand_dims %rr {axis = 1 : i32} : tensor<42xi32> -> tensor<42x1xi32>
  %c = tt.expand_dims %cc {axis = 0 : i32} : tensor<3xi32> -> tensor<1x3xi32>
  %rb = tt.broadcast %r : tensor<42x1xi32> -> tensor<42x3xi32>
  %cb = tt.broadcast %c : tensor<1x3xi32> -> tensor<42x3xi32>
  %stride = arith.constant dense<17> : tensor<42x3xi32>
  %rowoffset = arith.muli %rb, %stride : tensor<42x3xi32>
  %offset = arith.addi %rowoffset, %cb : tensor<42x3xi32>
  %bound = tt.splat %count : i32 -> tensor<42x1xi32>
  %rowmask = arith.cmpi slt, %r, %bound : tensor<42x1xi32>
  %mask = tt.broadcast %rowmask : tensor<42x1xi1> -> tensor<42x3xi1>
  %ops = tt.splat %o : !tt.ptr<f32> -> tensor<42x3x!tt.ptr<f32>>
  %optr = tt.addptr %ops, %offset : tensor<42x3x!tt.ptr<f32>>, tensor<42x3xi32>
  tt.store %optr, %value2, %mask : tensor<42x3x!tt.ptr<f32>>
  tt.return
 }
}

// -----

// CHECK-LABEL: func.func @column_interval(
// CHECK-NOT: tensor.empty
// CHECK-NOT: tensor.insert_slice
// CHECK-NOT: tensor<126xf32>
// CHECK: tensor.extract_slice {{.*}}[0, 1]
// CHECK: bufferization.materialize_in_destination
// CHECK: return

module {
 tt.func public @column_interval(%p: !tt.ptr<f32>, %o: !tt.ptr<f32>, %start: i32, %count: i32) {
  %c3 = arith.constant 3 : index
  %c126 = arith.constant 126 : index
  %c0 = arith.constant 0.0 : f32
  %s = arith.index_cast %start : i32 to index
  %q = arith.index_cast %count : i32 to index
  %n = arith.muli %q, %c3 : index
  %padn = arith.subi %c126, %n : index
  %ir = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %ips = tt.splat %p : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  %ip = tt.addptr %ips, %ir : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  %values = tt.load %ip : tensor<128x!tt.ptr<f32>>
  %part = tensor.extract_slice %values[%s] [%n] [1] : tensor<128xf32> to tensor<?xf32>
  %segempty = tensor.empty() : tensor<126xf32>
  %padded = tensor.insert_slice %part into %segempty[0] [%n] [1] : tensor<?xf32> into tensor<126xf32>
  %value2 = tensor.expand_shape %padded [[0, 1]] output_shape [42, 3] : tensor<126xf32> into tensor<42x3xf32>
  %rr = tt.make_range {end = 42 : i32, start = 0 : i32} : tensor<42xi32>
  %cc = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %r = tt.expand_dims %rr {axis = 1 : i32} : tensor<42xi32> -> tensor<42x1xi32>
  %c = tt.expand_dims %cc {axis = 0 : i32} : tensor<3xi32> -> tensor<1x3xi32>
  %rb = tt.broadcast %r : tensor<42x1xi32> -> tensor<42x3xi32>
  %cb = tt.broadcast %c : tensor<1x3xi32> -> tensor<42x3xi32>
  %stride = arith.constant dense<17> : tensor<42x3xi32>
  %rowoffset = arith.muli %rb, %stride : tensor<42x3xi32>
  %offset = arith.addi %rowoffset, %cb : tensor<42x3xi32>
  %bound = tt.splat %count : i32 -> tensor<42x1xi32>
  %rowmask = arith.cmpi slt, %r, %bound : tensor<42x1xi32>
  %rowbmask = tt.broadcast %rowmask : tensor<42x1xi1> -> tensor<42x3xi1>
  %one = arith.constant dense<1> : tensor<1x3xi32>
  %cm = arith.cmpi sge, %c, %one : tensor<1x3xi32>
  %colmask = tt.broadcast %cm : tensor<1x3xi1> -> tensor<42x3xi1>
  %mask = arith.andi %rowbmask, %colmask : tensor<42x3xi1>
  %ops = tt.splat %o : !tt.ptr<f32> -> tensor<42x3x!tt.ptr<f32>>
  %optr = tt.addptr %ops, %offset : tensor<42x3x!tt.ptr<f32>>, tensor<42x3xi32>
  tt.store %optr, %value2, %mask : tensor<42x3x!tt.ptr<f32>>
  tt.return
 }
}
