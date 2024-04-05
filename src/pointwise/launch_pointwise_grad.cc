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
          typename Direction, template <typename> class MemObj>
SNNStatus launch_vector_pointwise(MemObj<T const>& input_forward,
                                  MemObj<T const>& input_backprop,
                                  MemObj<T>& output_backprop,
                                  Index const n_items, cl::sycl::queue& queue,
                                  const std::vector<cl::sycl::event>& events) {
  constexpr int block_size = 4;
  const auto remainder = n_items % block_size;
  auto is_divisible = [](int n, int div) { return n % div == 0; };
  if (is_divisible(block_size, 4) && is_divisible(remainder, 4)) {
    return queue_pointwise<T, Index, PointwiseType, Direction, 4, block_size>(
        input_forward, input_backprop, output_backprop, n_items, queue, events);
  } else if (is_divisible(block_size, 2) && is_divisible(remainder, 2)) {
    return queue_pointwise<T, Index, PointwiseType, Direction, 2, block_size>(
        input_forward, input_backprop, output_backprop, n_items, queue, events);
  } else {
    return queue_pointwise<T, Index, PointwiseType, Direction, 1, block_size>(
        input_forward, input_backprop, output_backprop, n_items, queue, events);
  }
}

/**
 * Queue a pointwise operation with a thread per element if supported,
 * otherwise return an SNNStatus error code.
 */
template <template <typename> class PointwiseType, typename T,
          typename Direction, template <typename> class MemObj,
          typename EnableIf>
SNNStatus launch_pointwise(MemObj<T const>& input_forward,
                           MemObj<T const>& input_backprop,
                           MemObj<T>& output_backprop, size_t const n_items,
                           cl::sycl::queue& queue,
                           const std::vector<cl::sycl::event>& events) {
  if (n_items > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
#ifdef SNN_USE_INT64
    return launch_vector_pointwise<T, int64_t, PointwiseType, Direction>(
        input_forward, input_backprop, output_backprop, n_items, queue, events);
#else
    return StatusCode::IndexExceeded;
#endif  // SNN_USE_INT64
  } else {
    return launch_vector_pointwise<T, int32_t, PointwiseType, Direction>(
        input_forward, input_backprop, output_backprop, n_items, queue, events);
  }
}

#define SNN_INSTANTIATE_LAUNCH_POINTWISE_GRADIENT_KERNEL(DTYPE, OP, MEMOBJ) \
  template SNN_EXPORT SNNStatus launch_pointwise<OP, DTYPE, Gradient>(      \
      MEMOBJ<DTYPE const> & inp_fwd_access,                                 \
      MEMOBJ<DTYPE const> & inp_bk_access, MEMOBJ<DTYPE> & outp_access,     \
      size_t const n_items, cl::sycl::queue& queue,                         \
      const std::vector<cl::sycl::event>& events);

#define SNN_INSTANTIATE_ALL_LAUNCH_POINTWISE(DTYPE, MEMOBJ)             \
  SNN_INSTANTIATE_LAUNCH_POINTWISE_GRADIENT_KERNEL(DTYPE, Relu, MEMOBJ) \
  SNN_INSTANTIATE_LAUNCH_POINTWISE_GRADIENT_KERNEL(DTYPE, Tanh, MEMOBJ) \
  SNN_INSTANTIATE_LAUNCH_POINTWISE_GRADIENT_KERNEL(DTYPE, Exp, MEMOBJ)  \
  SNN_INSTANTIATE_LAUNCH_POINTWISE_GRADIENT_KERNEL(DTYPE, Log, MEMOBJ)  \
  SNN_INSTANTIATE_LAUNCH_POINTWISE_GRADIENT_KERNEL(DTYPE, Sqrt, MEMOBJ)

#ifdef SNN_ENABLE_USM
SNN_INSTANTIATE_ALL_LAUNCH_POINTWISE(float, USMMemObject)
#endif
SNN_INSTANTIATE_ALL_LAUNCH_POINTWISE(float, BufferMemObject)

#ifdef SNN_USE_HALF
#ifdef SNN_ENABLE_USM
SNN_INSTANTIATE_ALL_LAUNCH_POINTWISE(cl::sycl::half, USMMemObject)
#endif
SNN_INSTANTIATE_ALL_LAUNCH_POINTWISE(cl::sycl::half, BufferMemObject)
#endif  // SNN_USE_HALF

#ifdef SNN_USE_DOUBLE
#ifdef SNN_ENABLE_USM
SNN_INSTANTIATE_ALL_LAUNCH_POINTWISE(double, USMMemObject)
#endif
SNN_INSTANTIATE_ALL_LAUNCH_POINTWISE(double, BufferMemObject)
#endif  // SNN_USE_DOUBLE

#undef SNN_INSTANTIATE_LAUNCH_POINTWISE_GRADIENT_KERNEL
#undef SNN_INSTANTIATE_ALL_LAUNCH_POINTWISE

}  // namespace internal
}  // namespace pointwise
}  // namespace sycldnn
