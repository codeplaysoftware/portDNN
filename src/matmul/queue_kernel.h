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
#ifndef SYCLDNN_SRC_MATMUL_QUEUE_KERNEL_H_
#define SYCLDNN_SRC_MATMUL_QUEUE_KERNEL_H_

#include "sycldnn/matmul/params.h"

#include "sycldnn/mem_object.h"
#include "sycldnn/status.h"

namespace sycldnn {
namespace matmul {
namespace internal {

/** Add a matrix multiply kernel to the provided SYCL queue. */
template <typename T, typename Index, bool TransposeLHS, bool TransposeRHS,
          int RowTile, int AccTile, int ColTile, bool CheckBounds,
          template <typename> class MemObj>
SNNStatus queue_kernel(MemObj<T const>& lhs, MemObj<T const>& rhs,
                       MemObj<T>& output, MatmulParams const& params,
                       cl::sycl::queue& queue, size_t wg_row, size_t wg_col,
                       size_t wg_batch,
                       const std::vector<cl::sycl::event>& events);

}  // namespace internal
}  // namespace matmul
}  // namespace sycldnn

#endif  // SYCLDNN_SRC_MATMUL_QUEUE_KERNEL_H_
