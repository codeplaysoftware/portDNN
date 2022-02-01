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

#ifndef SYCLDNN_SRC_GAMMA_GRADIENT_FROZEN_QUEUE_IMPL_H
#define SYCLDNN_SRC_GAMMA_GRADIENT_FROZEN_QUEUE_IMPL_H

#include "src/batchnorm/gradient/frozen/kernels.h"
#include "sycldnn/helpers/ratio.h"
#include "sycldnn/status.h"

#include <CL/sycl.hpp>

namespace sycldnn {
namespace batchnorm {
namespace internal {

template <typename T, typename Index, int VectorWidth>
SNNStatus queue_gamma_gradient(BaseMemObject<T const>& gradient,
                               BaseMemObject<T const>& input,
                               BaseMemObject<T const>& mean,
                               BaseMemObject<T const>& variance,
                               BaseMemObject<T>& output, Index const n_items,
                               float const epsilon, cl::sycl::queue& queue) {
  auto event = queue.submit([&](cl::sycl::handler& cgh) {
    auto gradient_acc = gradient.read_accessor(cgh);
    auto input_acc = input.read_accessor(cgh);
    auto mean_acc = mean.read_accessor(cgh);
    auto variance_acc = variance.read_accessor(cgh);
    auto output_acc = output.write_accessor(cgh);
    Index const n_vecs = n_items / VectorWidth;
    GammaGradientFrozen<T, Index, VectorWidth> op{
        gradient_acc, input_acc, mean_acc, variance_acc,
        output_acc,   n_vecs,    epsilon};
    size_t const n_threads = helpers::round_up_to_nearest_multiple(n_vecs, 64);

    cgh.parallel_for(cl::sycl::range<1>{n_threads}, op);
  });

  return {event, StatusCode::OK};
}

}  // namespace internal
}  // namespace batchnorm
}  // namespace sycldnn

#endif  // SYCLDNN_SRC_GAMMA_GRADIENT_FROZEN_QUEUE_IMPL_H
