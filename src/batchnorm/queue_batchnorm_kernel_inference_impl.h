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

#ifndef SYCLDNN_SRC_BATCHNORM_QUEUE_INFERENCE_IMPL_H
#define SYCLDNN_SRC_BATCHNORM_QUEUE_INFERENCE_IMPL_H

#include "src/batchnorm/kernels.h"
#include "sycldnn/batchnorm/params.h"
#include "sycldnn/mem_object.h"
#include "sycldnn/status.h"

#include <CL/sycl.hpp>

namespace sycldnn {
namespace batchnorm {
namespace internal {

template <typename T, typename Index, int VectorWidth>
SNNStatus queue_batchnorm(
    BaseMemObject<T const>& input, BaseMemObject<T const>& beta,
    BaseMemObject<T const>& gamma, BaseMemObject<T const>& current_mean,
    BaseMemObject<T const>& current_variance, BaseMemObject<T>& output,
    BatchNormParams const& params, cl::sycl::queue& queue) {
  Index n_items = params.batch * params.channels * params.rows * params.cols;

  auto event = queue.submit([&](cl::sycl::handler& cgh) {
    auto input_acc = input.read_accessor(cgh);
    auto beta_acc = beta.read_accessor(cgh);
    auto gamma_acc = gamma.read_accessor(cgh);
    auto mean_acc = current_mean.read_accessor(cgh);
    auto variance_acc = current_variance.read_accessor(cgh);
    auto output_acc = output.write_accessor(cgh);
    BatchNormOp<T, Index, VectorWidth> op{input_acc, beta_acc,     gamma_acc,
                                          mean_acc,  variance_acc, output_acc,
                                          params};

    cgh.parallel_for(
        cl::sycl::range<1>{static_cast<size_t>(n_items / VectorWidth)}, op);
  });
  return {event, StatusCode::OK};
}

}  // namespace internal
}  // namespace batchnorm
}  // namespace sycldnn

#endif  // SYCLDNN_SRC_BATCHNORM_QUEUE_INFERENCE_IMPL_H
