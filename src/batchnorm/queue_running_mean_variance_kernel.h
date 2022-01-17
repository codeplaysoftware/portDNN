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

#ifndef SYCLDNN_SRC_RUNNING_MEAN_VARIANCE_QUEUE_BATCHNORM_H
#define SYCLDNN_SRC_RUNNING_MEAN_VARIANCE_QUEUE_BATCHNORM_H

#include "sycldnn/mem_object.h"

#include <CL/sycl.hpp>
#include "sycldnn/status.h"

namespace sycldnn {
namespace batchnorm {
namespace internal {

template <typename T, typename Index, int VectorWidth>
SNNStatus queue_running_mean_variance(BaseMemObject<T const>& input_mean,
                                      BaseMemObject<T const>& input_variance,
                                      BaseMemObject<T>& running_mean,
                                      BaseMemObject<T>& running_variance,
                                      Index const n_items, float const momentum,
                                      cl::sycl::queue& queue);

}  // namespace internal
}  // namespace batchnorm
}  // namespace sycldnn

#endif  // SYCLDNN_SRC_RUNNING_MEAN_VARIANCE_QUEUE_BATCHNORM_H
