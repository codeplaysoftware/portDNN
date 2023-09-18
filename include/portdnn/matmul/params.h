/*
 * Copyright Codeplay Software Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef PORTDNN_INCLUDE_MATMUL_PARAMS_H_
#define PORTDNN_INCLUDE_MATMUL_PARAMS_H_
#include "portdnn/batch_format.h"

/**
 * \file
 * Defines the \ref sycldnn::matmul::MatmulParams struct,
 * which contains the values used in a matmul operation.
 */
namespace sycldnn {
namespace matmul {

/** Struct that contains values used in a matmul op. */
struct MatmulParams {
  /** The type of the params is int, providing a decent
   * upper bound on the tensor sizes.*/
  using Index = int;

  /**The number of matrices in each tensor. Must be a positive value.*/
  Index batches;

  /**The number of rows (columns if TransposeLHS) in the left hand matrix. Must
   * be a positive value.*/
  Index m;

  /** The number of columns (rows if TransposeLHS) in the left hand matrix and
   * the number of rows (columns if TransposeRHS) in the right hand matrix. Must
   * be a positive value.*/
  Index k;

  /** The number of columns (rows if TransposeRHS) in the right hand matrix.
   * Must be a positive value. */
  Index n;

  /**A scalar value to scale the output tensor.*/
  float beta;

  /** Specifies how the batches are strided in the tensor*/
  sycldnn::BatchFormat batch_type = sycldnn::BatchFormat::STRIDED;
};

}  // namespace matmul
}  // namespace sycldnn
#endif  // PORTDNN_INCLUDE_MATMUL_PARAMS_H_
