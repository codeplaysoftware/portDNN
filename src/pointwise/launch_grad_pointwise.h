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

#ifndef SYCLDNN_SRC_POINTWISE_LAUNCH_GRAD_POINTWISE_H_
#define SYCLDNN_SRC_POINTWISE_LAUNCH_GRAD_POINTWISE_H_

#include "sycldnn/internal/pointwise/launch_internal.h"

#include "sycldnn/pointwise/direction.h"

#include "sycldnn/pointwise/operators.h"

#include "src/pointwise/kernels.h"

#define SNN_INSTANTIATE_LAUNCH_POINTWISE_GRADIENT_KERNEL(DTYPE, OP) \
  template SNNStatus                                                \
  launch_pointwise<DTYPE, OP, sycldnn::pointwise::Gradient>(        \
      ReadAccessor<DTYPE const> inp_fwd_access,                     \
      ReadAccessor<DTYPE const> inp_bk_access,                      \
      WriteAccessor<DTYPE> outp_access, size_t const n_items,       \
      cl::sycl::queue& queue);

namespace sycldnn {
namespace pointwise {
namespace internal {

/**
 * Queue a pointwise operation with a number of threads equal to the number of
 * work items.
 */
template <typename T, typename Index, template <typename> class PointwiseType,
          typename Direction, int VectorWidth>
SNNStatus queue_pointwise(ReadAccessor<T const> input_forward,
                          ReadAccessor<T const> input_backprop,
                          WriteAccessor<T> output_backprop, Index const n_items,
                          Index const input_fwd_offset,
                          Index const input_bk_offset,
                          Index const output_bk_offset,
                          cl::sycl::queue& queue) {
  auto event = queue.submit([&](cl::sycl::handler& cgh) {
    cgh.require(input_forward);
    cgh.require(input_backprop);
    cgh.require(output_backprop);
    Index const n_vecs = n_items / VectorWidth;
    // TODO(jwlawson): Should this be rounded to a multiple of a power of 2?
    size_t const n_threads = n_vecs;
    PointwiseOp<T, Index, PointwiseType, Direction, VectorWidth> pointwise_op{
        input_forward,    input_backprop,  output_backprop, n_vecs,
        input_fwd_offset, input_bk_offset, output_bk_offset};

    cgh.parallel_for(cl::sycl::range<1>{n_threads}, pointwise_op);
  });

  return {event, StatusCode::OK};
}

template <typename T, typename Index, template <typename> class PointwiseType,
          typename Direction>
SNNStatus launch_vector_pointwise(
    ReadAccessor<T const> input_forward, ReadAccessor<T const> input_backprop,
    WriteAccessor<T> output_backprop, Index const n_items,
    Index const input_fwd_offset, Index const input_bk_offset,
    Index const output_bk_offset, cl::sycl::queue& queue) {
  if (n_items % 4 == 0) {
    return queue_pointwise<T, Index, PointwiseType, Direction, 4>(
        input_forward, input_backprop, output_backprop, n_items,
        input_fwd_offset, input_bk_offset, output_bk_offset, queue);
  } else if (n_items % 2 == 0) {
    return queue_pointwise<T, Index, PointwiseType, Direction, 2>(
        input_forward, input_backprop, output_backprop, n_items,
        input_fwd_offset, input_bk_offset, output_bk_offset, queue);
  } else {
    return queue_pointwise<T, Index, PointwiseType, Direction, 1>(
        input_forward, input_backprop, output_backprop, n_items,
        input_fwd_offset, input_bk_offset, output_bk_offset, queue);
  }
}

/**
 * Queue a pointwise operation with a thread per element if supported,
 * otherwise return an SNNStatus error code.
 */
template <typename T, template <typename> class PointwiseType,
          typename Direction, typename EnableIf>
SNNStatus launch_pointwise(ReadAccessor<T const> input_forward,
                           ReadAccessor<T const> input_backprop,
                           WriteAccessor<T> output_backprop,
                           size_t const n_items, cl::sycl::queue& queue) {
  auto const input_fwd_offset = input_forward.get_offset().get(0);
  auto const input_bk_offset = input_backprop.get_offset().get(0);
  auto const output_bk_offset = output_backprop.get_offset().get(0);
  if (n_items > std::numeric_limits<int64_t>::max()) {
    SNNStatus index_too_large;
    index_too_large.status = StatusCode::IndexExceeded;
    return index_too_large;
  } else if (n_items > std::numeric_limits<int32_t>::max()) {
#ifdef SNN_USE_INT64
    return launch_vector_pointwise<T, int64_t, PointwiseType, Direction>(
        input_forward, input_backprop, output_backprop, n_items,
        static_cast<int64_t>(input_fwd_offset),
        static_cast<int64_t>(input_bk_offset),
        static_cast<int64_t>(output_bk_offset), queue);
#else
    SNNStatus index_too_large;
    index_too_large.status = StatusCode::IndexExceeded;
    return index_too_large;
#endif  // SNN_USE_INT64
  } else {
    return launch_vector_pointwise<T, int32_t, PointwiseType, Direction>(
        input_forward, input_backprop, output_backprop, n_items,
        static_cast<int32_t>(input_fwd_offset),
        static_cast<int32_t>(input_bk_offset),
        static_cast<int32_t>(output_bk_offset), queue);
  }
}

}  // namespace internal
}  // namespace pointwise
}  // namespace sycldnn
#endif  // SYCLDNN_SRC_POINTWISE_LAUNCH_GRAD_POINTWISE_H_
