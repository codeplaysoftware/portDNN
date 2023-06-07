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

#ifndef SYCLDNN_SRC_POINTWISE_QUEUE_POINTWISE_GRAD_H_
#define SYCLDNN_SRC_POINTWISE_QUEUE_POINTWISE_GRAD_H_

#include "sycldnn/mem_object.h"
#include "sycldnn/status.h"

#include <CL/sycl.hpp>

namespace sycldnn {
namespace pointwise {
namespace internal {

/**
 * Queue a pointwise operation on the SYCL queue queue.
 */
template <typename T, typename Index, template <typename> class PointwiseType,
          typename Direction, int VectorWidth, int BlockSize,
          template <typename> class MemObj>
SNNStatus queue_pointwise(MemObj<T const>& in_forward_mem,
                          MemObj<T const>& in_backprop_mem,
                          MemObj<T>& out_backprop_mem, Index const n_items,
                          cl::sycl::queue& queue,
                          const std::vector<cl::sycl::event>& events);
}  // namespace internal
}  // namespace pointwise
}  // namespace sycldnn

#endif  // SYCLDNN_SRC_POINTWISE_QUEUE_POINTWISE_GRAD_H_
