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
#include "sycldnn/pooling/operators.h"

#include "src/pooling/launch_max_grad_pooling.h"
#include "src/pooling/queue_pooling_kernel_impl.h"

namespace sycldnn {
namespace pooling {
namespace internal {

SNN_INSTANTIATE_LAUNCH_POOLING_KERNEL(int32_t, Max, Forward);
#ifdef SNN_USE_INT64
SNN_INSTANTIATE_LAUNCH_POOLING_KERNEL(int64_t, Max, Forward);
#endif  // SNN_USE_INT64
SNN_INSTANTIATE_LAUNCH_POOLING_KERNEL(float, Max, Forward);
#ifdef SNN_USE_HALF
SNN_INSTANTIATE_LAUNCH_POOLING_KERNEL(cl::sycl::half, Max, Forward);
#endif  // SNN_USE_HALF
#ifdef SNN_USE_DOUBLE
SNN_INSTANTIATE_LAUNCH_POOLING_KERNEL(double, Max, Forward);
#endif  // SNN_USE_DOUBLE

#define SNN_INSTANTIATE_LAUNCH_MAX_GRAD_POOLING_KERNEL(DTYPE)         \
  template SNNStatus launch_pooling<DTYPE, sycldnn::pooling::Max,     \
                                    sycldnn::pooling::Backpropagate>( \
      ReadAccessor<DTYPE const> input_data,                           \
      ReadAccessor<DTYPE const> output_data,                          \
      ReadAccessor<DTYPE const> input_backprop,                       \
      WriteAccessor<DTYPE> outp_backprop, const PoolingParams& pp,    \
      cl::sycl::queue& queue)

SNN_INSTANTIATE_LAUNCH_MAX_GRAD_POOLING_KERNEL(int32_t);
#ifdef SNN_USE_INT64
SNN_INSTANTIATE_LAUNCH_MAX_GRAD_POOLING_KERNEL(int64_t);
#endif  // SNN_USE_INT64
SNN_INSTANTIATE_LAUNCH_MAX_GRAD_POOLING_KERNEL(float);
#ifdef SNN_USE_HALF
SNN_INSTANTIATE_LAUNCH_MAX_GRAD_POOLING_KERNEL(cl::sycl::half);
#endif  // SNN_USE_HALF
#ifdef SNN_USE_DOUBLE
SNN_INSTANTIATE_LAUNCH_MAX_GRAD_POOLING_KERNEL(double);
#endif  // SNN_USE_DOUBLE

}  // namespace internal
}  // namespace pooling
}  // namespace sycldnn
