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
#ifndef SYCLDNN_SRC_REDUCE_QUEUE_REDUCTION_IMPL_H_
#define SYCLDNN_SRC_REDUCE_QUEUE_REDUCTION_IMPL_H_

#include <algorithm>
#include <limits>
#include <numeric>
#include <type_traits>

#include "sycldnn/helpers/event_handling.h"
#include "sycldnn/helpers/mem_utils.h"
#include "sycldnn/helpers/ratio.h"

#include "sycldnn/mem_object.h"
#include "sycldnn/status.h"

#include "src/helpers/local_mem_tile.h"
#include "src/helpers/math.h"

#include "src/reduce/queue_reduction.h"

#include "src/reduce/default_kernel.h"
#include "src/reduce/local_mem_kernel.h"

namespace sycldnn {
namespace reduce {
namespace internal {

template <typename T, typename Index, typename Op,
          template <typename> class MemObj>
SNNStatus queue_default_kernel(MemObj<T const>& input_mem,
                               MemObj<T>& output_mem, int batches, int outer,
                               int inner, int finalizeParam,
                               cl::sycl::queue& queue,
                               const std::vector<cl::sycl::event>& events) {
  constexpr bool is_usm = is_usm_obj_v<MemObj<T>, T>;
  auto event = queue.submit([&](cl::sycl::handler& cgh) {
    cgh.depends_on(events);
    auto input = input_mem.read_mem(cgh);
    auto output = output_mem.write_mem(cgh);

    ReduceKernel<T, Index, Op, is_usm> functor{
        input,
        output,
        batches,
        outer,
        inner,
        finalizeParam,
        internal::Identity<T, Op>::value};

    cgh.parallel_for(cl::sycl::range<2>(batches, inner), functor);
  });
  return {event, StatusCode::OK};
}

template <typename T, typename Index, typename Op, bool Transpose,
          int NumSGReductions, bool GlobalAtomicReduce, bool BankOffset,
          template <typename> class MemObj>
SNNStatus lm_parallel_for(MemObj<T const>& input_mem, MemObj<T>& output_mem,
                          cl::sycl::nd_range<3> const& nd_range,
                          Index batch_size, Index reduce_size,
                          Index stride_size, Index batch_groups, Index tile_row,
                          Index tile_col, cl::sycl::queue queue,
                          std::vector<cl::sycl::event> const& events) {
  using LocalMem = sycl::local_accessor<T, 1>;
  constexpr bool IsUSM = is_usm_obj_v<MemObj<T>, T>;
  auto event = queue.submit([&](cl::sycl::handler& cgh) {
    cgh.depends_on(events);
    auto in_mem = input_mem.read_mem(cgh);
    auto out_mem = output_mem.read_write_mem(cgh);
    LocalMem local_mem(
        sycl::range<1>((tile_row + BankOffset) * (tile_col + BankOffset)), cgh);
    LocalMemReduceKernel<T, Index, Op, Transpose, NumSGReductions,
                         GlobalAtomicReduce, BankOffset, IsUSM, LocalMem>
        functor(in_mem, out_mem, local_mem, batch_size, reduce_size,
                stride_size, tile_row, tile_col, batch_groups);
    cgh.parallel_for(nd_range, functor);
  });
  return {event, StatusCode::OK};
}

template <typename T, typename Index, typename Op, int NumSGReductions,
          bool Transpose, bool BankOffset, template <typename> class MemObj>
SNNStatus queue_lm_kernel(MemObj<T const>& input_mem, MemObj<T>& output_mem,
                          internal::ReduceParams const& params,
                          Index local_row_wg, Index local_col_wg,
                          cl::sycl::queue queue,
                          std::vector<cl::sycl::event> const& events) {
  auto needs_atomic_reduction = params.batch_groups != -1;
  auto const batch_size = params.batch_size;
  auto const stride_size = params.stride_size;
  auto const reduce_size = params.reduce_size;
  auto const batch_groups =
      params.batch_groups == -1 ? batch_size : params.batch_groups;
  auto const max_wg_size = params.max_wg_size;
  auto const max_sg_size = params.max_sg_size;
  // Calculate tile sizes and local range
  Index tile_col = helpers::internal::get_max_col_tile_dim(
      stride_size, (Index)max_wg_size, (Index)max_sg_size);
  tile_col = std::min(tile_col, local_col_wg);
  Index tile_row = helpers::internal::get_max_row_tile_dim(
      reduce_size, tile_col, (Index)max_wg_size, (Index)max_sg_size);
  tile_row = std::min(tile_row, local_row_wg);
  cl::sycl::range<3> local_range(1, tile_row, tile_col);

  // Calculate global range
  auto global_col =
      helpers::round_up_to_nearest_multiple(stride_size, tile_col);
  auto global_row =
      helpers::round_up_to_nearest_multiple(reduce_size, tile_row);
  cl::sycl::range<3> global_range(batch_size, global_row, global_col);
  needs_atomic_reduction =
      needs_atomic_reduction ||
      (global_row >
       std::min(tile_row, static_cast<Index>(NumSGReductions * max_sg_size)));

  // Set the nd-range
  cl::sycl::nd_range<3> nd_range(global_range, local_range);

  if (needs_atomic_reduction) {
    return lm_parallel_for<T, Index, Op, true, NumSGReductions, true,
                           BankOffset>(
        input_mem, output_mem, nd_range, batch_size, reduce_size, stride_size,
        batch_groups, tile_row, tile_col, queue, events);
  } else {
    return lm_parallel_for<T, Index, Op, true, NumSGReductions, false,
                           BankOffset>(
        input_mem, output_mem, nd_range, batch_size, reduce_size, stride_size,
        batch_groups, tile_row, tile_col, queue, events);
  }
}

}  // namespace internal
}  // namespace reduce
}  // namespace sycldnn
#endif  // SYCLDNN_SRC_REDUCE_QUEUE_REDUCTION_IMPL_H_
