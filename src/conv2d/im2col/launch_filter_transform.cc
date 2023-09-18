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
#include "portdnn/mem_object.h"
#include "portdnn/status.h"

#include "portdnn/conv2d/params.h"

#include "src/conv2d/im2col/queue_filter_transform.h"

#include <stddef.h>
#include <cstdint>
#include <limits>

#include <CL/sycl.hpp>

#include "portdnn/export.h"

namespace sycldnn {
namespace conv2d {
namespace internal {
namespace im2col {
namespace {

template <typename T, typename Index, template <typename> class MemObj>
SNNStatus launch_with_index(MemObj<T const>& input, MemObj<T>& output,
                            Conv2DParams const& params, size_t thread_size,
                            cl::sycl::queue& queue,
                            const std::vector<cl::sycl::event>& events) {
  return queue_filter_transform<T, Index>(input, output, params, thread_size,
                                          queue, events);
}
}  // namespace

template <typename T, template <typename> class MemObj>
SNNStatus launch_filter_transform(MemObj<T const>& input, MemObj<T>& output,
                                  Conv2DParams const& params,
                                  cl::sycl::queue& queue,
                                  const std::vector<cl::sycl::event>& events) {
  size_t thread_size = params.window_rows * params.window_cols *
                       params.channels * params.features;
  if (thread_size > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
#ifdef SNN_USE_INT64
    return launch_with_index<T, int64_t>(input, output, params, thread_size,
                                         queue, events);
#else
    return StatusCode::IndexExceeded;
#endif  // SNN_USE_INT64
  } else {
    return launch_with_index<T, int32_t>(input, output, params, thread_size,
                                         queue, events);
  }
}

#define INSTANTIATE_LAUNCHER(DTYPE, MEMOBJ)                     \
  template SNN_EXPORT SNNStatus launch_filter_transform<DTYPE>( \
      MEMOBJ<DTYPE const> & input, MEMOBJ<DTYPE> & output,      \
      Conv2DParams const& params, cl::sycl::queue& queue,       \
      const std::vector<cl::sycl::event>& events);

#ifdef SNN_ENABLE_USM
#define INSTANTIATE_FOR_MEMOBJ(DTYPE)       \
  INSTANTIATE_LAUNCHER(DTYPE, USMMemObject) \
  INSTANTIATE_LAUNCHER(DTYPE, BufferMemObject)
#else
#define INSTANTIATE_FOR_MEMOBJ(DTYPE) \
  INSTANTIATE_LAUNCHER(DTYPE, BufferMemObject)
#endif

INSTANTIATE_FOR_MEMOBJ(float)

#ifdef SNN_USE_DOUBLE
INSTANTIATE_FOR_MEMOBJ(double)
#endif  // SNN_USE_DOUBLE

#ifdef SNN_USE_HALF
INSTANTIATE_FOR_MEMOBJ(cl::sycl::half)
#endif  // SNN_USE_HALF

#undef INSTANTIATE_LAUNCHER
#undef INSTANTIATE_FOR_MEMOBJ

}  // namespace im2col
}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn
