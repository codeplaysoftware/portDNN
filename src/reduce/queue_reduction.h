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
#ifndef PORTDNN_SRC_REDUCE_QUEUE_KERNEL_H_
#define PORTDNN_SRC_REDUCE_QUEUE_KERNEL_H_

#include <unordered_map>

#include "portdnn/internal/helpers/types.h"
#include "portdnn/mem_object.h"
#include "portdnn/status.h"

namespace sycldnn {
namespace reduce {
namespace internal {

/** Add a reduce kernel to the provided SYCL queue. */
template <typename T, typename Index, typename Op,
          template <typename> class MemObj>
SNNStatus queue_default_kernel(MemObj<T const>& input, MemObj<T>& output,
                               int batches, int outer, int inner,
                               int finalizeParam, cl::sycl::queue& queue,
                               const std::vector<cl::sycl::event>& events);

#ifndef SNN_DISABLE_SYCL_PROGRAM
template <typename T, typename Index, typename Op,
          template <typename> class MemObj>
SNNStatus queue_subgroup_kernel(
    MemObj<T const>& input_mem, MemObj<T>& output_mem, int batches, int outer,
    int inner, cl::sycl::queue& queue, cl::sycl::program& program,
    sycldnn::internal::types::KernelSubgroupSizesMap&
        max_kernel_sub_group_sizes,
    const std::vector<cl::sycl::event>& events);
#endif
}  // namespace internal
}  // namespace reduce
}  // namespace sycldnn

#endif  // PORTDNN_SRC_REDUCE_QUEUE_KERNEL_H_
