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

#ifndef SYCLDNN_SRC_RUNNING_MEAN_VARIANCE_QUEUE_IMPL_H
#define SYCLDNN_SRC_RUNNING_MEAN_VARIANCE_QUEUE_IMPL_H

#include "src/batchnorm/kernels.h"
#include "sycldnn/helpers/ratio.h"
#include "sycldnn/status.h"

#include <CL/sycl.hpp>

namespace sycldnn {
namespace batchnorm {
namespace internal {

template <typename T, typename Index, int VectorWidth>
SNNStatus queue_running_mean_variance(BaseMemObject<T const>& input_mean,
                                      BaseMemObject<T const>& input_variance,
                                      BaseMemObject<T>& running_mean,
                                      BaseMemObject<T>& running_variance,
                                      Index const n_items, float const momentum,
                                      cl::sycl::queue& queue) {
  auto event = queue.submit([&](cl::sycl::handler& cgh) {
    auto input_mean_acc = input_mean.read_accessor(cgh);
    auto input_variance_acc = input_variance.read_accessor(cgh);
    auto running_mean_acc = running_mean.read_write_accessor(cgh);
    auto running_variance_acc = running_variance.read_write_accessor(cgh);
    Index const n_vecs = n_items / VectorWidth;
    RunningMeanVarianceOp<T, Index, VectorWidth> op{
        input_mean_acc,   input_variance_acc,
        running_mean_acc, running_variance_acc,
        n_vecs,           momentum};
    size_t const n_threads = helpers::round_up_to_nearest_multiple(n_vecs, 64);

    cgh.parallel_for(cl::sycl::range<1>{n_threads}, op);
  });

  return {event, StatusCode::OK};
}

}  // namespace internal
}  // namespace batchnorm
}  // namespace sycldnn

#endif  // SYCLDNN_SRC_RUNNING_MEAN_VARIANCE_QUEUE_IMPL_H
