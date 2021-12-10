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

#include "sycldnn/helpers/macros.h"

#include "sycldnn/binaryop/params.h"

#include "sycldnn/internal/binaryop/launch.h"

namespace sycldnn {
/** Namespace containing all binary elementwise operations. */
namespace binaryop {
/** Namespace containing internal implementation details for binary operation.
 */
namespace internal {

/**
 * Validate that the user provided binaryop parameters are consistent with what
 * is expected by SYCL-DNN.
 *
 * If compiled with asserts, any invalid parameter will fail an assert.
 * Otherwise a status code \ref StatusCode::InvalidParameter will be returned.
 *
 * \param [in] params User provided parameters to validate
 * \return An SNNStatus obejct containing either \ref StatusCode::OK if all
 *         parameters are valid, or \ref StatusCode::InvalidParameter otherwise.
 */

SNNStatus inline validate_params(BinaryParams const& params) {
  SNN_VALIDATE_PARAM(params.lhs_items > 0, "The number of items in 1st input.");
  SNN_VALIDATE_PARAM(params.rhs_items > 0, "The number of items in 2nd input.");
  return SNNStatus{{}, StatusCode::OK};
}

}  // namespace internal

/**
 * Launch the binary operation kernel.
 *
 * \tparam T         The data type of the input tensor.
 * \tparam Op        The type of the BinaryOp.
 * \tparam Backend   The type of the Backend.
 *
 * \param [in]  lhs   A pointer to the first input tensor.
 * \param [in]  rhs   A pointer to the second input tensor.
 * \param [out] output   A pointer to the output tensor.
 * \param [in]  params       The parameters of the binary operation.
 * \param [in]  backend  The backend that provides access to the SYCL buffers
 *                       corresponding to the input and output pointers.
 * \return An \ref SNNStatus containing the SYCL event tied to the kernel
 *         launches and a \ref StatusCode enum showing if the launch was OK or
 *         whether it encountered some problem.
 */
template <typename T, typename Op, typename Backend>
SNNStatus launch(typename Backend::template pointer_type<T const> lhs,
                 typename Backend::template pointer_type<T const> rhs,
                 typename Backend::template pointer_type<T> output,
                 const BinaryParams& params, Backend& backend) {
  auto validation_status = internal::validate_params(params);
  if (validation_status.status != StatusCode::OK) {
    return validation_status;
  }

  auto inp1_mem =
      backend.get_mem_object(lhs, static_cast<size_t>(params.lhs_items));
  auto inp2_mem =
      backend.get_mem_object(rhs, static_cast<size_t>(params.rhs_items));
  auto outp_mem =
      backend.get_mem_object(output, static_cast<size_t>(params.lhs_items));
  auto queue = backend.get_queue();
  return internal::launch_binaryop<T, Op>(inp1_mem, inp2_mem, outp_mem,
                                          params.rhs_items, queue);
}

}  // namespace binaryop
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_BINARYOP_LAUNCH_H_
