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

#ifndef SYCLDNN_INCLUDE_BINARYOP_LAUNCH_H_
#define SYCLDNN_INCLUDE_BINARYOP_LAUNCH_H_

/**
 * \file
 * Implements the \ref sycldnn::binaryop::launch() function, which
 * asynchronously dispatches the SYCL kernels to compute a binary elementwise
 * operation.
 */

#include "sycldnn/mem_object.h"
#include "sycldnn/status.h"

#include "sycldnn/helpers/dims.h"
#include "sycldnn/helpers/macros.h"

#include "sycldnn/binaryop/params.h"

#include "sycldnn/internal/binaryop/launch.h"

namespace sycldnn {
/** Namespace containing all binary elementwise operations. */
namespace binaryop {

/**
 * Launch the binary operation kernel.
 *
 * \tparam T         The data type of the input tensor.
 * \tparam Op        The type of the BinaryOp.
 * \tparam Backend   The type of the Backend.
 *
 * \param [in]  lhs      A pointer to the first input tensor.
 * \param [in]  rhs      A pointer to the second input tensor.
 * \param [out] out      A pointer to the output tensor.
 * \param [in]  params   The parameters of the binary operation.
 * \param [in]  backend  The backend that provides access to the SYCL buffers
 *                       corresponding to the input and output pointers.
 * \return An \ref SNNStatus containing the SYCL event tied to the kernel
 *         launches and a \ref StatusCode enum showing if the launch was OK or
 *         whether it encountered some problem.
 */
template <typename T, typename Op, typename Backend>
SNNStatus launch(typename Backend::template pointer_type<T const> lhs,
                 typename Backend::template pointer_type<T const> rhs,
                 typename Backend::template pointer_type<T> out,
                 const BinaryParams& params, Backend& backend) {
  auto lhs_dims = params.lhs_dims;
  auto rhs_dims = params.rhs_dims;
  SNN_VALIDATE_PARAM(lhs_dims.size() <= MAX_DIMS,
                     "Left operand exceeds the maximum number of dimensions");
  SNN_VALIDATE_PARAM(rhs_dims.size() <= MAX_DIMS,
                     "Right operand exceeds the maximum number of dimensions");

  // Empty dimensions may be used to represent scalars.
  if (lhs_dims.size() == 0) {
    lhs_dims.push_back(1);
  }
  if (rhs_dims.size() == 0) {
    rhs_dims.push_back(1);
  }

  size_t lhs_size = helpers::get_total_size(lhs_dims);
  size_t rhs_size = helpers::get_total_size(rhs_dims);
  SNN_VALIDATE_PARAM(lhs_size > 0, "Left operand cannot be zero.");
  SNN_VALIDATE_PARAM(rhs_size > 0, "Right operand cannot be zero.");

  std::vector<int> out_dims;
  auto status = internal::compute_out_dims(lhs_dims, rhs_dims, out_dims);
  if (status.status != StatusCode::OK) {
    return status;
  }
  size_t out_size = helpers::get_total_size(out_dims);

  auto lhs_mem = backend.get_mem_object(lhs, lhs_size);
  auto rhs_mem = backend.get_mem_object(rhs, rhs_size);
  auto out_mem = backend.get_mem_object(out, out_size);
  auto queue = backend.get_queue();
  return internal::launch_binaryop<Op>(lhs_mem, rhs_mem, out_mem, lhs_dims,
                                       rhs_dims, out_dims, queue);
}

}  // namespace binaryop
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_BINARYOP_LAUNCH_H_
