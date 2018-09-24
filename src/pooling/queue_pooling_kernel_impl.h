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

#ifndef SYCLDNN_SRC_POOLING_QUEUE_IMPL_H_
#define SYCLDNN_SRC_POOLING_QUEUE_IMPL_H_

#include "sycldnn/pooling/params.h"

#include "src/pooling/kernels.h"
#include "src/pooling/queue_pooling_kernel.h"

#define SNN_INSTANTIATE_LAUNCH_POOLING_KERNEL(DTYPE, OP, DIRECTION)           \
  template SNNStatus launch_pooling<DTYPE, OP, DIRECTION>(                    \
      ReadAccessor<DTYPE const> inp_access, WriteAccessor<DTYPE> outp_access, \
      const PoolingParams& pp, cl::sycl::queue& queue);

namespace sycldnn {
namespace pooling {
namespace internal {

template <typename T, typename Index, template <typename U> class PoolType,
          typename Direction>
SNNStatus queue_pooling(ReadAccessor<T const> input, WriteAccessor<T> output,
                        const PoolingParams& pp, size_t threads,
                        cl::sycl::queue& queue) {
  auto event = queue.submit([&](cl::sycl::handler& cgh) {
    cgh.require(input);
    cgh.require(output);
    PoolingOp<T, Index, PoolType, Direction> pool(input, output, pp);

    cgh.parallel_for(cl::sycl::range<1>{threads}, pool);
  });

  return {event, StatusCode::OK};
}

template <typename T, template <typename U> class PoolType, typename Direction>
SNNStatus launch_pooling(ReadAccessor<T const> input, WriteAccessor<T> output,
                         const PoolingParams& pp, cl::sycl::queue& queue) {
  size_t threads = pp.batch * pp.in_rows * pp.in_cols * pp.channels;
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
    return queue_pooling<T, int32_t, PoolType, Direction>(input, output, pp,
                                                          threads, queue);
  }
}

}  // namespace internal
}  // namespace pooling
}  // namespace sycldnn
#endif  // SYCLDNN_SRC_POOLING_QUEUE_IMPL_H_
