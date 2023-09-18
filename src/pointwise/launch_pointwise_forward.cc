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

#include "portdnn/internal/pointwise/launch_internal.h"

#include "portdnn/pointwise/direction.h"
#include "portdnn/pointwise/operators.h"

#include "src/pointwise/queue_pointwise_forward.h"

#include <CL/sycl.hpp>

#include "portdnn/export.h"

namespace sycldnn {
namespace pointwise {
namespace internal {

template <typename T, typename Index, template <typename> class PointwiseType,
          typename Direction, template <typename> class MemObj>
SNNStatus launch_vector_pointwise(MemObj<T const>& input, MemObj<T>& output,
                                  Index const n_items, cl::sycl::queue& queue,
                                  const std::vector<cl::sycl::event>& events) {
  if (n_items % 4 == 0) {
    return queue_pointwise<T, Index, PointwiseType, Direction, 4>(
        input, output, n_items, queue, events);
  } else if (n_items % 2 == 0) {
    return queue_pointwise<T, Index, PointwiseType, Direction, 2>(
        input, output, n_items, queue, events);
  } else {
    return queue_pointwise<T, Index, PointwiseType, Direction, 1>(
        input, output, n_items, queue, events);
  }
}

/**
 * Queue a pointwise operation with a thread per element if supported,
 * otherwise return an SNNStatus error code.
 */
template <template <typename> class PointwiseType, typename T,
          typename Direction, template <typename> class MemObj,
          typename EnableIf>
SNNStatus launch_pointwise(MemObj<T const>& input, MemObj<T>& output,
                           size_t const n_items, cl::sycl::queue& queue,
                           const std::vector<cl::sycl::event>& events) {
  if (n_items > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
#ifdef SNN_USE_INT64
    return launch_vector_pointwise<T, int64_t, PointwiseType, Direction>(
        input, output, n_items, queue, events);
#else
    return StatusCode::IndexExceeded;
#endif  // SNN_USE_INT64
  } else {
    return launch_vector_pointwise<T, int32_t, PointwiseType, Direction>(
        input, output, n_items, queue, events);
  }
}

#define SNN_INSTANTIATE_LAUNCH_POINTWISE_KERNEL(DTYPE, OP, MEMOBJ)    \
  template SNN_EXPORT SNNStatus launch_pointwise<OP, DTYPE, Forward>( \
      MEMOBJ<DTYPE const> & inp_access, MEMOBJ<DTYPE> & outp_access,  \
      size_t const n_items, cl::sycl::queue& queue,                   \
      const std::vector<cl::sycl::event>& events);

#define SNN_INSTANTIATE_ALL_LAUNCH_POINTWISE(DTYPE, MEMOBJ)     \
  SNN_INSTANTIATE_LAUNCH_POINTWISE_KERNEL(DTYPE, Relu, MEMOBJ)  \
  SNN_INSTANTIATE_LAUNCH_POINTWISE_KERNEL(DTYPE, Tanh, MEMOBJ)  \
  SNN_INSTANTIATE_LAUNCH_POINTWISE_KERNEL(DTYPE, Exp, MEMOBJ)   \
  SNN_INSTANTIATE_LAUNCH_POINTWISE_KERNEL(DTYPE, Log, MEMOBJ)   \
  SNN_INSTANTIATE_LAUNCH_POINTWISE_KERNEL(DTYPE, Floor, MEMOBJ) \
  SNN_INSTANTIATE_LAUNCH_POINTWISE_KERNEL(DTYPE, Sqrt, MEMOBJ)

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

#undef SNN_INSTANTIATE_LAUNCH_POINTWISE_KERNEL
#undef SNN_INSTANTIATE_ALL_LAUNCH_POINTWISE

}  // namespace internal
}  // namespace pointwise
}  // namespace sycldnn
