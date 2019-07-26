/*
 * Copyright 2018 Codeplay Software Ltd
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

#ifndef SYCLDNN_INCLUDE_POINTWISE_LAUNCH_H_
#define SYCLDNN_INCLUDE_POINTWISE_LAUNCH_H_

/**
 * \file
 * Implements the \ref sycldnn::pointwise::launch() function, which
 * asynchronously dispatches the SYCL kernels to compute a pointwise operation.
 */

#include "sycldnn/mem_object.h"
#include "sycldnn/status.h"

#include "sycldnn/helpers/macros.h"

#include "sycldnn/pointwise/direction.h"

#include "sycldnn/pointwise/operators.h"

#include "sycldnn/internal/pointwise/launch_internal.h"

namespace sycldnn {
/** Namespace containing all pointwise operations. */
namespace pointwise {

/**
 * Launch the pointwise operation kernel.
 *
 * \tparam T              The data type of the input tensor.
 * \tparam PointwiseType  The type of pointwise operation used.
 * \tparam Direction      Whether the pointwise operation computed should
 *                        be a Forward, Gradient, or GradGrad pass.
 * \tparam Backend        The type of the Backend.
 *
 * \param [in]  input     A pointer to the input tensor.
 * \param [out] output    A pointer to the output tensor.
 * \param [in]  n_items   The number of items in the input tensor.
 * \param [in]  backend   The backend providing access to the SYCL buffers
 *                        corresponding to the input and output pointers.
 *
 * \return An \ref SNNStatus containing the SYCL event tied to the kernel
 *         launches and a \ref StatusCode enum showing if the launch was OK or
 *         whether it encountered some problem.
 */
template <typename T, template <typename> class PointwiseType,
          typename Direction, typename Backend,
          typename = internal::DisableIfGradient<Direction>>
SNNStatus launch(typename Backend::template pointer_type<T const> input,
                 typename Backend::template pointer_type<T> output,
                 size_t const n_items, Backend& backend) {
  SNN_VALIDATE_PARAM(n_items > 0, "The number of items must be positive.");

  auto inp_access = backend.get_mem_object(input, n_items);
  auto outp_access = backend.get_mem_object(output, n_items);

  auto queue = backend.get_queue();
  return internal::launch_pointwise<T, PointwiseType, Direction>(
      inp_access, outp_access, n_items, queue);
}

/**
 * Launch the pointwise gradient kernel.
 *
 * \tparam T                       The data type of the input tensor.
 * \tparam PointwiseType           The type of pointwise operation used.
 * \tparam Direction               Whether the pointwise operation computed
 *                                 should be a Forward, Gradient, or GradGrad
 *                                 pass.
 * \tparam Backend                 The type of the Backend.
 *
 * \param [in]  input_forward      A pointer to the forward input tensor.
 * \param [in]  input_backprop     A pointer to the backprop input tensor.
 * \param [out] output_backprop    A pointer to the output tensor.
 * \param [in]  n_items            The number of items in the input tensor.
 * \param [in]  backend            The backend providing access to the SYCL
 *                                 buffers corresponding to the input and
 *                                 output pointers.
 *
 * \return An \ref SNNStatus containing the SYCL event tied to the kernel
 *         launches and a \ref StatusCode enum showing if the launch was OK or
 *         whether it encountered some problem.
 */
template <typename T, template <typename> class PointwiseType,
          typename Direction, typename Backend,
          typename = internal::EnableIfGradient<Direction>>
SNNStatus launch(
    typename Backend::template pointer_type<T const> input_forward,
    typename Backend::template pointer_type<T const> input_backprop,
    typename Backend::template pointer_type<T> output_backprop,
    size_t const n_items, Backend& backend) {
  SNN_VALIDATE_PARAM(n_items > 0, "The number of items must be positive.");

  auto inp_fwd_access = backend.get_mem_object(input_forward, n_items);
  auto inp_bk_access = backend.get_mem_object(input_backprop, n_items);
  auto out_bk_access = backend.get_mem_object(output_backprop, n_items);

  auto queue = backend.get_queue();
  return internal::launch_pointwise<T, PointwiseType, Direction>(
      inp_fwd_access, inp_bk_access, out_bk_access, n_items, queue);
}

}  // namespace pointwise
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_POINTWISE_LAUNCH_H_
