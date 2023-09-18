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
#ifndef PORTDNN_SRC_POOLING_QUEUE_MAX_GRAD_KERNEL_H_
#define PORTDNN_SRC_POOLING_QUEUE_MAX_GRAD_KERNEL_H_

#include "portdnn/mem_object.h"
#include "portdnn/status.h"

#include "portdnn/pooling/params.h"

#include "src/pooling/kernels.h"

#include <CL/sycl.hpp>

namespace sycldnn {
namespace pooling {
namespace internal {

template <typename T, typename Index, template <typename U> class PoolType,
          typename Direction, int VectorWidth, bool UseFastDiv, typename Format,
          template <typename> class MemObj>
SNNStatus queue_max_grad_pooling(MemObj<T const>& input_mem,
                                 MemObj<T const>& output_mem,
                                 MemObj<T const>& input_backprop_mem,
                                 MemObj<T>& output_backprop_mem,
                                 const PoolingParams& pp, size_t threads,
                                 cl::sycl::queue& queue,
                                 const std::vector<cl::sycl::event>& events);

}  // namespace internal
}  // namespace pooling
}  // namespace sycldnn

#endif  // PORTDNN_SRC_POOLING_QUEUE_MAX_GRAD_KERNEL_H_
