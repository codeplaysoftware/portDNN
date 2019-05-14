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
#include "sycldnn/mem_object.h"
#include "sycldnn/status.h"

#include "sycldnn/conv2d/params.h"

#include "src/conv2d/im2col/queue_filter_transform.h"

#include <stddef.h>
#include <cstdint>
#include <limits>

#include <CL/sycl.hpp>

namespace sycldnn {
namespace conv2d {
namespace internal {
namespace im2col {
namespace {

template <typename T, typename Index>
SNNStatus launch_with_index(BaseMemObject<T const>& input,
                            BaseMemObject<T>& output,
                            Conv2DParams const& params, size_t thread_size,
                            cl::sycl::queue& queue) {
  return queue_filter_transform<T, Index>(input, output, params, thread_size,
                                          queue);
}
}  // namespace

template <typename T>
SNNStatus launch_filter_transform(BaseMemObject<T const>& input,
                                  BaseMemObject<T>& output,
                                  Conv2DParams const& params,
                                  cl::sycl::queue& queue) {
  size_t thread_size = params.window_rows * params.window_cols *
                       params.channels * params.features;
  if (thread_size > std::numeric_limits<int32_t>::max()) {
#ifdef SNN_USE_INT64
    return launch_with_index<T, int64_t>(input, output, params, thread_size,
                                         queue);
#else
    return SNNStatus{{}, StatusCode::IndexExceeded};
#endif  // SNN_USE_INT64
  } else {
    return launch_with_index<T, int32_t>(input, output, params, thread_size,
                                         queue);
  }
}

#define INSTANTIATE_LAUNCHER(DTYPE)                                      \
  template SNNStatus launch_filter_transform<DTYPE>(                     \
      BaseMemObject<DTYPE const> & input, BaseMemObject<DTYPE> & output, \
      Conv2DParams const& params, cl::sycl::queue& queue);

INSTANTIATE_LAUNCHER(float)

#ifdef SNN_USE_DOUBLE
INSTANTIATE_LAUNCHER(double)
#endif  // SNN_USE_DOUBLE

#ifdef SNN_USE_HALF
INSTANTIATE_LAUNCHER(cl::sycl::half)
#endif  // SNN_USE_HALF

#undef INSTANTIATE_LAUNCHER

}  // namespace im2col
}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn
