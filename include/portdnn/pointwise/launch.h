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

#ifndef PORTDNN_INCLUDE_POINTWISE_LAUNCH_H_
#define PORTDNN_INCLUDE_POINTWISE_LAUNCH_H_

/**
 * \file
 * Implements the \ref sycldnn::pointwise::launch() function, which
 * asynchronously dispatches the SYCL kernels to compute a pointwise operation.
 */

#include "portdnn/mem_object.h"
#include "portdnn/status.h"

#include "portdnn/backend/backend_helpers.h"
#include "portdnn/helpers/macros.h"

#include "portdnn/pointwise/direction.h"

#include "portdnn/pointwise/operators.h"

#include "portdnn/internal/pointwise/launch_internal.h"

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
          typename = internal::DisableIfGradient<Direction>,
          typename = typename std::enable_if<
              sycldnn::backend::is_buffer_backend_v<Backend>>::type>
SNNStatus launch(typename Backend::template pointer_type<T const> input,
                 typename Backend::template pointer_type<T> output,
                 size_t const n_items, Backend& backend) {
  return internal::sublaunch<T, PointwiseType, Direction, Backend>(
      input, output, n_items, backend, {});
}

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
 * \param [in]  events    Events which should be completed before the operation.
 *
 * \return An \ref SNNStatus containing the SYCL event tied to the kernel
 *         launches and a \ref StatusCode enum showing if the launch was OK or
 *         whether it encountered some problem.
 */
template <typename T, template <typename> class PointwiseType,
          typename Direction, typename Backend,
          typename = internal::DisableIfGradient<Direction>,
          typename = typename std::enable_if<
              sycldnn::backend::is_usm_backend_v<Backend>>::type>
SNNStatus launch(typename Backend::template pointer_type<T const> input,
                 typename Backend::template pointer_type<T> output,
                 size_t const n_items, Backend& backend,
                 const std::vector<cl::sycl::event>& events = {}) {
  return internal::sublaunch<T, PointwiseType, Direction, Backend>(
      input, output, n_items, backend, events);
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
          typename = internal::EnableIfGradient<Direction>,
          typename = typename std::enable_if<
              sycldnn::backend::is_buffer_backend_v<Backend>>::type>
SNNStatus launch(
    typename Backend::template pointer_type<T const> input_forward,
    typename Backend::template pointer_type<T const> input_backprop,
    typename Backend::template pointer_type<T> output_backprop,
    size_t const n_items, Backend& backend) {
  return internal::sublaunch<T, PointwiseType, Direction, Backend>(
      input_forward, input_backprop, output_backprop, n_items, backend, {});
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
 * \param [in]  events             Events which should be completed before the
 * operation.
 *
 * \return An \ref SNNStatus containing the SYCL event tied to the kernel
 *         launches and a \ref StatusCode enum showing if the launch was OK or
 *         whether it encountered some problem.
 */
template <typename T, template <typename> class PointwiseType,
          typename Direction, typename Backend,
          typename = internal::EnableIfGradient<Direction>,
          typename = typename std::enable_if<
              sycldnn::backend::is_usm_backend_v<Backend>>::type>
SNNStatus launch(
    typename Backend::template pointer_type<T const> input_forward,
    typename Backend::template pointer_type<T const> input_backprop,
    typename Backend::template pointer_type<T> output_backprop,
    size_t const n_items, Backend& backend,
    const std::vector<cl::sycl::event>& events = {}) {
  return internal::sublaunch<T, PointwiseType, Direction, Backend>(
      input_forward, input_backprop, output_backprop, n_items, backend, events);
}

}  // namespace pointwise
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_POINTWISE_LAUNCH_H_
