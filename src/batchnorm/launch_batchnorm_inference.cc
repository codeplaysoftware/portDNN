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

#include "src/batchnorm/queue_batchnorm_kernel_inference.h"
#include "sycldnn/internal/batchnorm/launch_batchnorm.h"

#include <CL/sycl.hpp>

#include "sycldnn/export.h"

namespace sycldnn {
namespace batchnorm {
namespace internal {

inline bool can_use_vector_width(BatchNormParams const& params, int w) {
  return params.channels % w == 0;
};

template <typename T, typename Index>
SNNStatus launch_with_index(
    BaseMemObject<T const>& input, BaseMemObject<T const>& beta,
    BaseMemObject<T const>& gamma, BaseMemObject<T const>& moving_mean,
    BaseMemObject<T const>& moving_variance, BaseMemObject<T>& output,
    BatchNormParams const& params, cl::sycl::queue& queue) {
  if (can_use_vector_width(params, 4)) {
    return queue_batchnorm<T, Index, 4>(input, beta, gamma, moving_mean,
                                        moving_variance, output, params, queue);
  } else if (can_use_vector_width(params, 2)) {
    return queue_batchnorm<T, Index, 2>(input, beta, gamma, moving_mean,
                                        moving_variance, output, params, queue);
  } else {
    return queue_batchnorm<T, Index, 1>(input, beta, gamma, moving_mean,
                                        moving_variance, output, params, queue);
  }
}
/**
 * The internal batchnorm launcher.
 */
template <typename T>
SNNStatus launch_batchnorm(
    BaseMemObject<T const>& input, BaseMemObject<T const>& beta,
    BaseMemObject<T const>& gamma, BaseMemObject<T const>& moving_mean,
    BaseMemObject<T const>& moving_variance, BaseMemObject<T>& output,
    BatchNormParams const& params, cl::sycl::queue& queue) {
  auto total_size = params.batch * params.rows * params.cols * params.channels;
  if (total_size > std::numeric_limits<int64_t>::max()) {
    SNNStatus index_too_large;
    index_too_large.status = StatusCode::IndexExceeded;
    return index_too_large;
  } else if (total_size > std::numeric_limits<int32_t>::max()) {
#ifdef SNN_USE_INT64
    return launch_with_index<T, int64_t>(input, beta, gamma, moving_mean,
                                         moving_variance, output, params,
                                         queue);
#else
    SNNStatus index_too_large;
    index_too_large.status = StatusCode::IndexExceeded;
    return index_too_large;
#endif  // SNN_USE_INT64
  } else {
    return launch_with_index<T, int32_t>(input, beta, gamma, moving_mean,
                                         moving_variance, output, params,
                                         queue);
  }
}

#define INSTANTIATE_LAUNCH(DTYPE)                                            \
  template SNN_EXPORT SNNStatus launch_batchnorm<DTYPE>(                     \
      BaseMemObject<DTYPE const> & input, BaseMemObject<DTYPE const> & beta, \
      BaseMemObject<DTYPE const> & gamma,                                    \
      BaseMemObject<DTYPE const> & moving_mean,                              \
      BaseMemObject<DTYPE const> & moving_variance,                          \
      BaseMemObject<DTYPE> & output, BatchNormParams const& params,          \
      cl::sycl::queue& queue)

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
