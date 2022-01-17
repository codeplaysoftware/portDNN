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

#ifndef SYCLDNN_INCLUDE_INTERNAL_BATCHNORM_LAUNCH_VARIANCE_INTERNAL_H_
#define SYCLDNN_INCLUDE_INTERNAL_BATCHNORM_LAUNCH_VARIANCE_INTERNAL_H_

#include "sycldnn/status.h"

#include "sycldnn/export.h"

namespace sycldnn {
namespace batchnorm {
namespace internal {

/**
 * The internal launcher for computing variance.
 */
template <typename T>
SNN_EXPORT SNNStatus launch_variance(BaseMemObject<T const>& input,
                                     BaseMemObject<T const>& current_mean,
                                     BaseMemObject<T>& current_variance,
                                     BatchNormParams const& params,
                                     cl::sycl::queue& queue);

/**
 * The internal launcher for computing batchnorm.
 */
template <typename T>
SNN_EXPORT SNNStatus launch_batchnorm(
    BaseMemObject<T const>& input, BaseMemObject<T const>& beta,
    BaseMemObject<T const>& gamma, BaseMemObject<T const>& current_mean,
    BaseMemObject<T const>& current_variance, BaseMemObject<T>& output,
    BatchNormParams const& params, cl::sycl::queue& queue);

/**
 * The internal launcher for computing running mean and running variance.
 */
template <typename T>
SNN_EXPORT SNNStatus launch_running_mean_variance(
    BaseMemObject<T const>& input_mean, BaseMemObject<T const>& input_variance,
    BaseMemObject<T>& running_mean, BaseMemObject<T>& running_variance,
    int32_t const n_items, float const momentum, cl::sycl::queue& queue);

}  // namespace internal
}  // namespace batchnorm
}  // namespace sycldnn
#endif  // SYCLDNN_INCLUDE_INTERNAL_BATCHNORM_LAUNCH_VARIANCE_INTERNAL_H_
