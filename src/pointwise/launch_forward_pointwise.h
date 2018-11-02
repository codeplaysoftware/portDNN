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

#ifndef SYCLDNN_SRC_POINTWISE_LAUNCH_FORWARD_POINTWISE_H_
#define SYCLDNN_SRC_POINTWISE_LAUNCH_FORWARD_POINTWISE_H_

#include "sycldnn/internal/pointwise/launch_internal.h"

#include "sycldnn/pointwise/direction.h"

#include "sycldnn/pointwise/operators.h"

#include "src/pointwise/kernels.h"

#define SNN_INSTANTIATE_LAUNCH_POINTWISE_KERNEL(DTYPE, OP, DIRECTION)         \
  template SNNStatus launch_pointwise<DTYPE, OP, DIRECTION>(                  \
      ReadAccessor<DTYPE const> inp_access, WriteAccessor<DTYPE> outp_access, \
      size_t const n_items, cl::sycl::queue& queue);

namespace sycldnn {
namespace pointwise {
namespace internal {

/**
 * Queue a pointwise operation with a number of threads equal to the number of
 * work items.
 */
template <typename T, typename Index, template <typename> class PointwiseType,
          typename Direction>
SNNStatus queue_pointwise(ReadAccessor<T const> input, WriteAccessor<T> output,
                          Index const n_items, Index const input_offset,
                          Index const output_offset, cl::sycl::queue& queue) {
  size_t const threads = n_items;
  auto event = queue.submit([&](cl::sycl::handler& cgh) {
    cgh.require(input);
    cgh.require(output);
    PointwiseOp<T, Index, PointwiseType, Direction> pointwise_op(
        input, output, n_items, input_offset, output_offset);

    cgh.parallel_for(cl::sycl::range<1>{threads}, pointwise_op);
  });

  return {event, StatusCode::OK};
}

/**
 * Queue a pointwise operation with a thread per element if supported,
 * otherwise return an SNNStatus error code.
 */
template <typename T, template <typename> class PointwiseType,
          typename Direction, typename EnableIf>
SNNStatus launch_pointwise(ReadAccessor<T const> input, WriteAccessor<T> output,
                           size_t const n_items, cl::sycl::queue& queue) {
  auto const input_offset = input.get_offset().get(0);
  auto const output_offset = output.get_offset().get(0);
  if (n_items > std::numeric_limits<int64_t>::max()) {
    SNNStatus index_too_large;
    index_too_large.status = StatusCode::IndexExceeded;
    return index_too_large;
  } else if (n_items > std::numeric_limits<int32_t>::max()) {
#ifdef SNN_USE_INT64
    return queue_pointwise<T, int64_t, PointwiseType, Direction>(
        input, output, n_items, static_cast<int64_t>(input_offset),
        static_cast<int64_t>(output_offset), queue);
#else
    SNNStatus index_too_large;
    index_too_large.status = StatusCode::IndexExceeded;
    return index_too_large;
#endif  // SNN_USE_INT64
  } else {
    return queue_pointwise<T, int32_t, PointwiseType, Direction>(
        input, output, n_items, static_cast<int32_t>(input_offset),
        static_cast<int32_t>(output_offset), queue);
  }
}

}  // namespace internal
}  // namespace pointwise
}  // namespace sycldnn
#endif  // SYCLDNN_SRC_POINTWISE_LAUNCH_FORWARD_POINTWISE_H_
