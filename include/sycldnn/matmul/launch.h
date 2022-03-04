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
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef SYCLDNN_INCLUDE_MATMUL_LAUNCH_H_
#define SYCLDNN_INCLUDE_MATMUL_LAUNCH_H_

/**
 * \file
 * Implements the \ref sycldnn::matmul::launch() function, which asynchronously
 * dispatches the SYCL kernels required to perform a matrix multiply.
 */
#include "sycldnn/mem_object.h"
#include "sycldnn/status.h"

#include "sycldnn/helpers/macros.h"
#include "sycldnn/internal/matmul/launch.h"

namespace sycldnn {
namespace matmul {
/**
 * Launch a batched matrix multiplication.
 *
 * Will compute: output[i] = beta * output[i] + op(lhs[i]) * op(rhs[i])
 * where i ranges over the number of batches and op(X) is either X or X^T if
 * TransposeX is true.
 *
 * \param lhs A pointer to the memory representing the left hand matrix.
 * \param rhs A pointer to the memory representing the right hand matrix.
 * \param output A pointer to the memory representing the output tensor.
 * \param batches The number of matrices in each tensor. Must be a positive
 *                value.
 * \param m The number of rows (columns if TransposeLHS) in the left hand
 *          matrix. Must be a positive value.
 * \param k The number of columns (rows if TransposeLHS) in the left hand
 *          matrix and the number of rows (columns if TransposeRHS) in the
 *          right hand matrix. Must be a positive value.
 * \param n The number of columns (rows if TransposeRHS) in the right hand
 *          matrix. Must be a positive value.
 * \param beta A scalar value to scale the output tensor.
 * \param backend The backend implementation, used to map between pointer
 *                representations.
 * \return Returns an SNNStatus containing the SYCL event tied to the kernel
 *         launches and a StatusCode enum showing if the launch was OK or
 *         whether it encountered some problem.
 */
template <typename T, bool TransposeLHS, bool TransposeRHS, typename Backend>
SNNStatus launch(typename Backend::template pointer_type<T const> lhs,
                 typename Backend::template pointer_type<T const> rhs,
                 typename Backend::template pointer_type<T> output, int batches,
                 int m, int k, int n, T beta, Backend& backend) {
  SNN_VALIDATE_PARAM(batches > 0, "The number of batches must be positive.");
  SNN_VALIDATE_PARAM(m > 0, "The value of m must be positive.");
  SNN_VALIDATE_PARAM(k > 0, "The value of k must be positive.");
  SNN_VALIDATE_PARAM(n > 0, "The value of n must be positive.");

  size_t lhs_size = batches * m * k;
  size_t rhs_size = batches * k * n;
  size_t out_size = batches * m * n;

  auto lhs_acc = backend.get_mem_object(lhs, lhs_size);
  auto rhs_acc = backend.get_mem_object(rhs, rhs_size);
  auto out_acc = backend.get_mem_object(output, out_size);

  auto sycl_queue = backend.get_queue();

  return internal::launch<T, TransposeLHS, TransposeRHS>(
      lhs_acc, rhs_acc, out_acc, batches, m, k, n, beta, sycl_queue);
}
}  // namespace matmul
}  // namespace sycldnn
#endif  // SYCLDNN_INCLUDE_MATMUL_LAUNCH_H_
