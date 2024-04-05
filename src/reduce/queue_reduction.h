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
#ifndef SYCLDNN_SRC_REDUCE_QUEUE_KERNEL_H_
#define SYCLDNN_SRC_REDUCE_QUEUE_KERNEL_H_

#include <algorithm>
#include <cstddef>
#include <unordered_map>
#include <vector>

#include "sycldnn/internal/reduce/internal_params.h"
#include "sycldnn/mem_object.h"
#include "sycldnn/status.h"

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

/** Add a lm reduce kernel to the provided SYCL queue. */
template <typename T, typename Index, typename Op, int NumSGReductions,
          bool Transpose, bool BankOffset, template <typename> class MemObj>
SNNStatus queue_lm_kernel(MemObj<T const>& input_mem, MemObj<T>& output_mem,
                          internal::ReduceParams const& params, Index row_wg,
                          Index col_wg, cl::sycl::queue queue,
                          const std::vector<cl::sycl::event>& events);

}  // namespace internal
}  // namespace reduce
}  // namespace sycldnn

#endif  // SYCLDNN_SRC_REDUCE_QUEUE_KERNEL_H_
