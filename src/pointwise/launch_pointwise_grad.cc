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

#include "sycldnn/internal/pointwise/launch_internal.h"

#include "sycldnn/pointwise/direction.h"
#include "sycldnn/pointwise/operators.h"

#include "src/pointwise/queue_pointwise_grad.h"

#include <CL/sycl.hpp>

#include "sycldnn/export.h"

namespace sycldnn {
namespace pointwise {
namespace internal {

template <typename T, typename Index, template <typename> class PointwiseType,
          typename Direction>
SNNStatus launch_vector_pointwise(BaseMemObject<T const>& input_forward,
                                  BaseMemObject<T const>& input_backprop,
                                  BaseMemObject<T>& output_backprop,
                                  Index const n_items, cl::sycl::queue& queue) {
  if (n_items % 4 == 0) {
    return queue_pointwise<T, Index, PointwiseType, Direction, 4>(
        input_forward, input_backprop, output_backprop, n_items, queue);
  } else if (n_items % 2 == 0) {
    return queue_pointwise<T, Index, PointwiseType, Direction, 2>(
        input_forward, input_backprop, output_backprop, n_items, queue);
  } else {
    return queue_pointwise<T, Index, PointwiseType, Direction, 1>(
        input_forward, input_backprop, output_backprop, n_items, queue);
  }
}

/**
 * Queue a pointwise operation with a thread per element if supported,
 * otherwise return an SNNStatus error code.
 */
template <typename T, template <typename> class PointwiseType,
          typename Direction, typename EnableIf>
SNNStatus launch_pointwise(BaseMemObject<T const>& input_forward,
                           BaseMemObject<T const>& input_backprop,
                           BaseMemObject<T>& output_backprop,
                           size_t const n_items, cl::sycl::queue& queue) {
  if (n_items > std::numeric_limits<int64_t>::max()) {
    SNNStatus index_too_large;
    index_too_large.status = StatusCode::IndexExceeded;
    return index_too_large;
  } else if (n_items > std::numeric_limits<int32_t>::max()) {
#ifdef SNN_USE_INT64
    return launch_vector_pointwise<T, int64_t, PointwiseType, Direction>(
        input_forward, input_backprop, output_backprop, n_items, queue);
#else
    SNNStatus index_too_large;
    index_too_large.status = StatusCode::IndexExceeded;
    return index_too_large;
#endif  // SNN_USE_INT64
  } else {
    return launch_vector_pointwise<T, int32_t, PointwiseType, Direction>(
        input_forward, input_backprop, output_backprop, n_items, queue);
  }
}

#define SNN_INSTANTIATE_LAUNCH_POINTWISE_GRADIENT_KERNEL(DTYPE, OP)    \
  template SNN_EXPORT SNNStatus launch_pointwise<DTYPE, OP, Gradient>( \
      BaseMemObject<DTYPE const> & inp_fwd_access,                     \
      BaseMemObject<DTYPE const> & inp_bk_access,                      \
      BaseMemObject<DTYPE> & outp_access, size_t const n_items,        \
      cl::sycl::queue& queue);

SNN_INSTANTIATE_LAUNCH_POINTWISE_GRADIENT_KERNEL(float, Relu)
SNN_INSTANTIATE_LAUNCH_POINTWISE_GRADIENT_KERNEL(float, Tanh)
SNN_INSTANTIATE_LAUNCH_POINTWISE_GRADIENT_KERNEL(float, Exp)
#ifdef SNN_USE_HALF
SNN_INSTANTIATE_LAUNCH_POINTWISE_GRADIENT_KERNEL(cl::sycl::half, Relu)
SNN_INSTANTIATE_LAUNCH_POINTWISE_GRADIENT_KERNEL(cl::sycl::half, Tanh)
SNN_INSTANTIATE_LAUNCH_POINTWISE_GRADIENT_KERNEL(cl::sycl::half, Exp)
#endif  // SNN_USE_HALF
#ifdef SNN_USE_DOUBLE
SNN_INSTANTIATE_LAUNCH_POINTWISE_GRADIENT_KERNEL(double, Relu)
SNN_INSTANTIATE_LAUNCH_POINTWISE_GRADIENT_KERNEL(double, Tanh)
SNN_INSTANTIATE_LAUNCH_POINTWISE_GRADIENT_KERNEL(double, Exp)
#endif  // SNN_USE_DOUBLE

}  // namespace internal
}  // namespace pointwise
}  // namespace sycldnn
