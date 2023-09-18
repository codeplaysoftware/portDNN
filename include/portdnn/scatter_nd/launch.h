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
#ifndef PORTDNN_INCLUDE_SCATTER_ND_LAUNCH_H_
#define PORTDNN_INCLUDE_SCATTER_ND_LAUNCH_H_

/**
 * \file
 * Implements the \ref sycldnn::scatter_nd::launch() function, which
 * asynchronously dispatches a SYCL kernel to compute a scatter_nd operation
 * along a single dimension of a N-dimensional tensor.
 */
#include "portdnn/status.h"

#include "portdnn/scatter_nd/operators.h"
#include "portdnn/scatter_nd/sizes.h"

#include "portdnn/internal/scatter_nd/launch_internal.h"

#include "portdnn/backend/backend_helpers.h"
#include "portdnn/helpers/macros.h"

namespace sycldnn {
/** Namespace containing the scatter_nd operator. */
namespace scatter_nd {
/** Namespace containing internal implementation details for scatter_nd. */

/**
 * Launch the scatter_nd operation kernel.
 *
 * \tparam T           The data type of the input tensor.
 * \tparam Indices     The data type of the indices tensor.
 * \tparam ScatterNDType The update operator used, such as Assign, Add, Mul etc.
 * \tparam Backend      The type of backend.
 * \param input         A pointer to the memory representing the input tensor.
 * \param indices       A pointer to the memory representing the indices tensor.
 * \param update        A pointer to the memory representing the updates tensor.
 * \param output        A pointer to the memory representing the output tensor.
 * \param params        The scatter_nd parameters, which describe the tensor
 * shape and layout.
 * \param backend       The backend implementation, used to
 * map between pointer representations.
 * \return Returns a SNNStatus containing
 * the SYCL event tied to the kernel launches and a StatusCode enum showing if
 * the launch was OK or whether it encountered some problem.
 */
template <typename T, typename Index, typename ScatterNDType, typename Backend,
          typename = typename std::enable_if<
              sycldnn::backend::is_buffer_backend_v<Backend>>::type>
SNNStatus launch(typename Backend::template pointer_type<T const> input,
                 typename Backend::template pointer_type<Index const> indices,
                 typename Backend::template pointer_type<T const> update,
                 typename Backend::template pointer_type<T> output,
                 ScatterNDParams const& params, Backend& backend) {
  return internal::sublaunch<T, Index, ScatterNDType, Backend>(
      input, indices, update, output, params, backend, {});
}

/**
 * Launch the scatter_nd operation kernel.
 *
 * \tparam T           The data type of the input tensor.
 * \tparam Indices     The data type of the indices tensor.
 * \tparam ScatterNDType The update operator used, such as Assign, Add, Mul etc.
 * \tparam Backend      The type of backend.
 * \param input         A pointer to the memory representing the input tensor.
 * \param indices       A pointer to the memory representing the indices tensor.
 * \param update        A pointer to the memory representing the updates tensor.
 * \param output        A pointer to the memory representing the output tensor.
 * \param params        The scatter_nd parameters, which describe the tensor
 * shape and layout.
 * \param events Events which should be completed before the operation.
 * \param backend       The backend implementation, used to
 * map between pointer representations.
 * \return Returns a SNNStatus containing
 * the SYCL event tied to the kernel launches and a StatusCode enum showing if
 * the launch was OK or whether it encountered some problem.
 */
template <typename T, typename Index, typename ScatterNDType, typename Backend,
          typename = typename std::enable_if<
              sycldnn::backend::is_usm_backend_v<Backend>>::type>
SNNStatus launch(typename Backend::template pointer_type<T const> input,
                 typename Backend::template pointer_type<Index const> indices,
                 typename Backend::template pointer_type<T const> update,
                 typename Backend::template pointer_type<T> output,
                 ScatterNDParams const& params, Backend& backend,
                 const std::vector<cl::sycl::event>& events = {}) {
  return internal::sublaunch<T, Index, ScatterNDType, Backend>(
      input, indices, update, output, params, backend, events);
}

}  // namespace scatter_nd
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_SCATTER_ND_LAUNCH_H_
