/*
 * Copyright 2018 Codeplay Software Ltd
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
#ifndef SYCLDNN_SRC_MATMUL_QUEUE_KERNEL_IMPL_H_
#define SYCLDNN_SRC_MATMUL_QUEUE_KERNEL_IMPL_H_

#include "sycldnn/mem_object.h"
#include "sycldnn/status.h"

#include "sycldnn/helpers/ratio.h"

#include "src/matmul/kernels.h"
#include "src/matmul/queue_kernel.h"

namespace sycldnn {
namespace matmul {
namespace internal {

template <typename T, typename Index, bool TransposeLHS, bool TransposeRHS,
          int RowTile, int AccTile, int ColTile, bool CheckBounds>
SNNStatus queue_kernel(BaseMemObject<T const>& lhs_mem,
                       BaseMemObject<T const>& rhs_mem,
                       BaseMemObject<T>& output_mem, int batches, int m, int k,
                       int n, T beta, cl::sycl::queue& queue, size_t wg_row,
                       size_t wg_col, size_t wg_batch) {
  Index const output_size_row = helpers::round_ratio_up(m, RowTile);
  Index const output_size_col = helpers::round_ratio_up(n, ColTile);
  size_t const n_row_threads =
      helpers::round_up_to_nearest_multiple(output_size_row, wg_row);
  size_t const n_col_threads =
      helpers::round_up_to_nearest_multiple(output_size_col, wg_col);
  size_t const n_batch_threads =
      helpers::round_up_to_nearest_multiple(batches, wg_batch);

  auto event = queue.submit([&](cl::sycl::handler& cgh) {
    auto lhs = lhs_mem.read_accessor(cgh);
    auto rhs = rhs_mem.read_accessor(cgh);
    auto output = output_mem.read_write_accessor(cgh);

    using Functor = MatmulKernel<T, Index, TransposeLHS, TransposeRHS, RowTile,
                                 AccTile, ColTile, CheckBounds>;

    Functor functor{lhs, rhs, output, batches, m, k, n, beta};

    cgh.parallel_for(
        cl::sycl::nd_range<3>{
            cl::sycl::range<3>{n_batch_threads, n_row_threads, n_col_threads},
            cl::sycl::range<3>{std::min(wg_batch, n_batch_threads),
                               std::min(wg_row, n_row_threads),
                               std::min(wg_col, n_col_threads)},
        },
        functor);
  });
  return {event, StatusCode::OK};
}

}  // namespace internal
}  // namespace matmul
}  // namespace sycldnn
#endif  // SYCLDNN_SRC_MATMUL_QUEUE_KERNEL_IMPL_H_
