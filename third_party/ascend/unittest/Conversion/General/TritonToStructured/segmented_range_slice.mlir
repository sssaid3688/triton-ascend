// RUN: triton-opt %s --bubble-up-operation --canonicalize | FileCheck %s
// A static range slice must retain progression, start, and stride.
// CHECK-LABEL: tt.func @strided_range_slice(
// CHECK-DAG: arith.constant dense<2> : tensor<3xi32>
// CHECK-DAG: arith.constant dense<7> : tensor<3xi32>
// CHECK: tt.make_range {end = 3 : i32, start = 0 : i32}
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: tt.return
// CHECK-LABEL: tt.func @dynamic_range_slice(
// CHECK-NOT: tt.splat
// CHECK: tensor.extract_slice
// CHECK: tt.return
// CHECK-LABEL: tt.func @dynamic_same_type(
// CHECK: tensor.extract_slice
// CHECK: tt.return

module {
  tt.func @strided_range_slice() -> tensor<3xi32> {
    %range = tt.make_range {start = 5 : i32, end = 13 : i32} : tensor<8xi32>
    %slice = tensor.extract_slice %range[2] [3] [2] : tensor<8xi32> to tensor<3xi32>
    tt.return %slice : tensor<3xi32>
  }
  tt.func @dynamic_range_slice(%count: index) -> tensor<?xi32> {
    %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %slice = tensor.extract_slice %range[0] [%count] [1] : tensor<8xi32> to tensor<?xi32>
    tt.return %slice : tensor<?xi32>
  }
  tt.func @dynamic_same_type(%source: tensor<?xi32>, %offset: index, %count: index) -> tensor<?xi32> {
    %slice = tensor.extract_slice %source[%offset] [%count] [1] : tensor<?xi32> to tensor<?xi32>
    tt.return %slice : tensor<?xi32>
  }
}
