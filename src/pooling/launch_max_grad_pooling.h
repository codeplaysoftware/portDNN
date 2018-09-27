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

#ifndef SYCLDNN_SRC_POOLING_LAUNCH_MAX_GRAD_POOLING_H_
#define SYCLDNN_SRC_POOLING_LAUNCH_MAX_GRAD_POOLING_H_

#include "sycldnn/pooling/params.h"
#include "sycldnn/pooling/sizes.h"

#include "sycldnn/internal/pooling/launch_internal.h"

#include "src/pooling/kernels.h"

namespace sycldnn {
namespace pooling {
namespace internal {

template <typename T, typename Index, template <typename U> class PoolType,
          typename Direction>
SNNStatus queue_pooling(ReadAccessor<T const> input_data,
                        ReadAccessor<T const> output_data,
                        ReadAccessor<T const> input_backprop,
                        WriteAccessor<T> output_backprop,
                        const PoolingParams& pp, size_t threads,
                        cl::sycl::queue& queue) {
  auto event = queue.submit([&](cl::sycl::handler& cgh) {
    cgh.require(input_data);
    cgh.require(output_data);
    cgh.require(input_backprop);
    cgh.require(output_backprop);
    PoolingOp<T, Index, PoolType, Direction> pool(
        input_data, output_data, input_backprop, output_backprop, pp);

    cgh.parallel_for(cl::sycl::range<1>{threads}, pool);
  });

  return {event, StatusCode::OK};
}

template <typename T, template <typename> class PoolType, typename Direction,
          typename EnableIf>
SNNStatus launch_pooling(ReadAccessor<T const> inp_data,
                         ReadAccessor<T const> outp_data,
                         ReadAccessor<T const> inp_backprop,
                         WriteAccessor<T> outp_backprop,
                         const PoolingParams& pp, cl::sycl::queue& queue) {
  auto sizes = get_sizes<Direction>(pp);
  size_t threads = sizes.output_size;
  if (threads > std::numeric_limits<int32_t>::max()) {
#ifdef SNN_USE_INT64
    return queue_pooling<T, int64_t, PoolType, Direction>(input, output, pp,
                                                          threads, queue);
#else
    SNNStatus index_too_large;
    index_too_large.status = StatusCode::IndexExceeded;
    return index_too_large;
#endif  // SNN_USE_INT64
  } else {
    return queue_pooling<T, int32_t, PoolType, Direction>(
        inp_data, outp_data, inp_backprop, outp_backprop, pp, threads, queue);
  }
}

}  // namespace internal
}  // namespace pooling
}  // namespace sycldnn

#endif  // SYCLDNN_SRC_POOLING_LAUNCH_MAX_GRAD_POOLING_H_
