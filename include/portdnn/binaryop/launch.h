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

#ifndef PORTDNN_INCLUDE_BINARYOP_LAUNCH_H_
#define PORTDNN_INCLUDE_BINARYOP_LAUNCH_H_

/**
 * \file
 * Implements the \ref sycldnn::binaryop::launch() function, which
 * asynchronously dispatches the SYCL kernels to compute a binary elementwise
 * operation.
 */

#include "portdnn/mem_object.h"
#include "portdnn/status.h"

#include "portdnn/backend/backend_helpers.h"
#include "portdnn/helpers/macros.h"

#include "portdnn/binaryop/params.h"

#include "portdnn/internal/binaryop/launch.h"

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
template <typename T, typename Op, typename Backend,
          typename = typename std::enable_if<
              sycldnn::backend::is_buffer_backend_v<Backend>>::type>
SNNStatus launch(typename Backend::template pointer_type<T const> lhs,
                 typename Backend::template pointer_type<T const> rhs,
                 typename Backend::template pointer_type<T> out,
                 const BinaryParams& params, Backend& backend) {
  return internal::sublaunch<T, Op, Backend>(lhs, rhs, out, params, backend,
                                             {});
}

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
 * \param [in]  events    Events which should be completed before the operation.
 * \return An \ref SNNStatus containing the SYCL event tied to the kernel
 *         launches and a \ref StatusCode enum showing if the launch was OK or
 *         whether it encountered some problem.
 */
template <typename T, typename Op, typename Backend,
          typename = typename std::enable_if<
              sycldnn::backend::is_usm_backend_v<Backend>>::type>
SNNStatus launch(typename Backend::template pointer_type<T const> lhs,
                 typename Backend::template pointer_type<T const> rhs,
                 typename Backend::template pointer_type<T> out,
                 const BinaryParams& params, Backend& backend,
                 std::vector<cl::sycl::event> events = {}) {
  return internal::sublaunch<T, Op, Backend>(lhs, rhs, out, params, backend,
                                             events);
}

}  // namespace binaryop
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_BINARYOP_LAUNCH_H_
