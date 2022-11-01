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
#ifndef SYCLDNN_INCLUDE_BACKEND_SNN_USM_REDUCE_PROVIDER_H_
#define SYCLDNN_INCLUDE_BACKEND_SNN_USM_REDUCE_PROVIDER_H_

/**
 * \file
 * Contains the implementation of \ref sycldnn::backend::SNNReduceProvider,
 * which provides a reduce implementation using the internal
 * SYCL-DNN reduce kernels.
 */

#include "sycldnn/backend/backend_traits.h"
#include "sycldnn/backend/internal_backend.h"
#include "sycldnn/reduce/launch.h"

namespace sycldnn {
namespace backend {

/**
 * CRTP module to provide a reduce implementation using the
 * internal SYCL-DNN kernels.
 */
template <typename Backend>
struct SNNUSMReduceProvider {
 private:
  /** The pointer representation required by the internal handler. */
  template <typename T>
  using internal_pointer_type =
      typename BackendTraits<Backend>::template internal_pointer_type<T>;

 public:
  /**
   * A wrapper around a call to reduce.
   *
   * Perform a reduction using Op on the outer axis from an input: [batch,
   * outer, inner].
   *
   * \param [in]  input  Pointer to a buffer containing the input tensor.
   * \param [out] output Pointer to a buffer containing the output tensor.
   * \param [in]  batch  Batch size.
   * \param [in]  outer  Outer size.
   * \param [in]  inner  Inner size.
   * \param [in]  events Events which should be completed before the operation.
   *
   * \return A SYCL event corresponding to the reduce kernel launch.
   */
  template <typename Op, typename T, typename Index>
  cl::sycl::event reduce(internal_pointer_type<const T> const input,
                         internal_pointer_type<T> const output,
                         Index const batch, Index const outer,
                         Index const inner,
                         const std::vector<cl::sycl::event> events = {}) {
    auto& underlying_backend = static_cast<Backend&>(*this);
    internal::InternalBackend<Backend> internal_backend{underlying_backend};
    auto status = reduce::launch<T, Op>(input, output, batch, outer, inner,
                                        internal_backend, events);
    SNN_ASSERT(status.status == StatusCode::OK,
               "Error launching reduce kernel.");
    return status.event;
  }
};

}  // namespace backend
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_BACKEND_SNN_USM_REDUCE_PROVIDER_H_