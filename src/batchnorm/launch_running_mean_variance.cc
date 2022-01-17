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
#include "sycldnn/status.h"

#include "sycldnn/batchnorm/params.h"

#include "src/batchnorm/queue_running_mean_variance_kernel.h"
#include "sycldnn/internal/batchnorm/launch_batchnorm.h"

#include <CL/sycl.hpp>

#include "sycldnn/export.h"

namespace sycldnn {
namespace batchnorm {
namespace internal {

inline bool can_use_vector_width(int32_t n_items, int w) {
  return n_items % w == 0;
};

/**
 * The internal launcher to compute running mean and running variance.
 */
template <typename T>
SNNStatus launch_running_mean_variance(
    BaseMemObject<T const>& input_mean, BaseMemObject<T const>& input_variance,
    BaseMemObject<T>& running_mean, BaseMemObject<T>& running_variance,
    int32_t const n_items, float const momentum, cl::sycl::queue& queue) {
  if (can_use_vector_width(n_items, 4)) {
    return queue_running_mean_variance<T, int32_t, 4>(
        input_mean, input_variance, running_mean, running_variance, n_items,
        momentum, queue);
  } else if (can_use_vector_width(n_items, 2)) {
    return queue_running_mean_variance<T, int32_t, 2>(
        input_mean, input_variance, running_mean, running_variance, n_items,
        momentum, queue);
  } else {
    return queue_running_mean_variance<T, int32_t, 1>(
        input_mean, input_variance, running_mean, running_variance, n_items,
        momentum, queue);
  }
}

#define INSTANTIATE_LAUNCH(DTYPE)                                     \
  template SNN_EXPORT SNNStatus launch_running_mean_variance<DTYPE>(  \
      BaseMemObject<DTYPE const> & input_mean,                        \
      BaseMemObject<DTYPE const> & input_variance,                    \
      BaseMemObject<DTYPE> & running_mean,                            \
      BaseMemObject<DTYPE> & running_variance, int32_t const n_items, \
      float const momentum, cl::sycl::queue& queue)

INSTANTIATE_LAUNCH(float);

#ifdef SNN_USE_HALF
INSTANTIATE_LAUNCH(cl::sycl::half);
#endif

#ifdef SNN_USE_DOUBLE
INSTANTIATE_LAUNCH(double);
#endif

}  // namespace internal
}  // namespace batchnorm
}  // namespace sycldnn
