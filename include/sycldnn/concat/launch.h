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
#ifndef SYCLDNN_INCLUDE_CONCAT_LAUNCH_H_
#define SYCLDNN_INCLUDE_CONCAT_LAUNCH_H_

#include <vector>

#include "sycldnn/status.h"

#include "sycldnn/concat/params.h"

#include "sycldnn/internal/concat/launch_internal.h"

/**
 * \file
 * Implements the \ref sycldnn::concat::launch() function, which asynchronously
 * dispatches the SYCL kernels to concatenate tensors.
 */

namespace sycldnn {
namespace concat {

/**
 * Launch the concat kernel (Buffers).
 *
 * \tparam T              The data type of the input tensors.
 * \tparam Backend        The type of the Backend.
 * \tparam TArrayMaxSize  The maximum number of input tensors that can be stored
 * in a single TArray
 *
 * \param [in]  input     A vector of pointers to the input tensors.
 * \param [out] output    A pointer to the output tensor.
 * \param [in]  params    The parameters of the concat operation.
 * \param [in]  backend   The backend that provides access to the SYCL buffers
 * corresponding to the input and output pointers.
 *
 * \return An \ref sycldnn::SNNStatus containing the SYCL event tied to the
 * kernel launches and a \ref sycldnn::StatusCode enum showing if the launch was
 * OK or whether it encountered some problem.
 */
template <typename T, typename Backend, int TArrayMaxSize = 6,
          typename = typename std::enable_if<
              sycldnn::backend::is_buffer_backend_v<Backend>>::type>
SNNStatus launch(
    std::vector<typename Backend::template pointer_type<T const>>& input,
    typename Backend::template pointer_type<T> output, ConcatParams& params,
    Backend& backend) {
  return internal::sublaunch<T, Backend, TArrayMaxSize>(input, output, params,
                                                        backend, {});
}

/**
 * Launch the concat kernel (USM).
 *
 * \tparam T              The data type of the input tensors.
 * \tparam Backend        The type of the Backend.
 * \tparam TArrayMaxSize  The maximum number of input tensors that can be stored
 * in a single TArray
 *
 * \param [in]  input     A vector of pointers to the input tensors.
 * \param [out] output    A pointer to the output tensor.
 * \param [in]  params    The parameters of the concat operation.
 * \param [in]  backend   The backend that provides access to the SYCL buffers
 * corresponding to the input and output pointers.
 * \param [in]  events    A vector of events to wait on before starting the
 *
 * \return An \ref sycldnn::SNNStatus containing the SYCL event tied to the
 * kernel launches and a \ref sycldnn::StatusCode enum showing if the launch was
 * OK or whether it encountered some problem.
 */
template <typename T, typename Backend, int TArrayMaxSize = 6,
          typename = typename std::enable_if<
              sycldnn::backend::is_usm_backend_v<Backend>>::type>
SNNStatus launch(
    std::vector<typename Backend::template pointer_type<T const>>& input,
    typename Backend::template pointer_type<T> output, ConcatParams& params,
    Backend& backend, const std::vector<cl::sycl::event>& events = {}) {
  return internal::sublaunch<T, Backend, TArrayMaxSize>(input, output, params,
                                                        backend, events);
}

}  // namespace concat
}  // namespace sycldnn

#endif