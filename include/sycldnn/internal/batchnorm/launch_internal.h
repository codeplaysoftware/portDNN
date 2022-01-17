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

#ifndef SYCLDNN_INCLUDE_INTERNAL_BATCHNORM_LAUNCH_INTERNAL_H_
#define SYCLDNN_INCLUDE_INTERNAL_BATCHNORM_LAUNCH_INTERNAL_H_

#include "sycldnn/mem_object.h"

#include "sycldnn/status.h"

#include "sycldnn/export.h"

#include "sycldnn/batchnorm/direction.h"
#include "sycldnn/batchnorm/operation.h"
#include "sycldnn/internal/batchnorm/launch_batchnorm.h"

namespace sycldnn {
namespace batchnorm {
namespace internal {

template <typename Direction, typename Operation>
using EnableIf_Forward_Training = typename std::enable_if<
    std::is_same<Direction, sycldnn::batchnorm::Forward>::value &&
        std::is_same<Operation, sycldnn::batchnorm::Training>::value,
    int>::type;

template <typename Direction, typename Operation>
using DisableIf_Forward_Training = typename std::enable_if<
    std::is_same<Direction, sycldnn::batchnorm::Forward>::value &&
        !std::is_same<Operation, sycldnn::batchnorm::Training>::value,
    int>::type;

/**
 * The internal batchnorm launcher for Forward Direction when computing Mean and
 * Variance.
 *
 * Calculates Mean, then Variance and then Batchnorm.
 */

template <typename T, typename Backend, typename Direction, typename Operation,
          typename = EnableIf_Forward_Training<Direction, Operation>>
SNNStatus launch_forward(
    typename Backend::template pointer_type<T const> input,
    typename Backend::template pointer_type<T const> beta,
    typename Backend::template pointer_type<T const> gamma,
    typename Backend::template pointer_type<T const> input_mean,
    typename Backend::template pointer_type<T const> input_variance,
    typename Backend::template pointer_type<T> running_mean,
    typename Backend::template pointer_type<T> running_variance,
    typename Backend::template pointer_type<T> output,
    BatchNormParams const& params, Backend& backend) {
  using Index = int32_t;
  backend.template reduce_outer<T, Index, sycldnn::batchnorm::BatchNormParams,
                                sycldnn::backend::reduction::Mean>(
      input, running_mean, params);

  auto n_items = params.batch * params.channels * params.rows * params.cols;

  auto queue = backend.get_queue();

  using ConstPointer = typename Backend::template pointer_type<T const>;
  auto const_mean = ConstPointer{running_mean};
  auto const_mean_mem = backend.get_mem_object(const_mean, params.channels);

  auto in_mem = backend.get_mem_object(input, n_items);
  auto variance_mem = backend.get_mem_object(running_variance, params.channels);
  auto beta_mem = backend.get_mem_object(beta, params.channels);
  auto gamma_mem = backend.get_mem_object(gamma, params.channels);
  auto out_mem = backend.get_mem_object(output, n_items);

  SNNStatus status =
      launch_variance(in_mem, const_mean_mem, variance_mem, params, queue);

  auto const_variance = ConstPointer{running_variance};
  auto const_variance_mem =
      backend.get_mem_object(const_variance, params.channels);

  if (sycldnn::StatusCode::OK != status.status) return status;

  status = launch_batchnorm(in_mem, beta_mem, gamma_mem, const_mean_mem,
                            const_variance_mem, out_mem, params, queue);

  if (sycldnn::StatusCode::OK != status.status) return status;

  auto input_mean_mem = backend.get_mem_object(input_mean, params.channels);
  auto running_mean_mem = backend.get_mem_object(running_mean, params.channels);
  auto input_variance_mem =
      backend.get_mem_object(input_variance, params.channels);
  auto running_variance_mem =
      backend.get_mem_object(running_variance, params.channels);

  status = launch_running_mean_variance(
      input_mean_mem, input_variance_mem, running_mean_mem,
      running_variance_mem, params.channels, params.momentum, queue);

  return status;
}

/**
 * The internal batchnorm launcher for Forward Direction when using the existing
 * Mean and Variance.
 *
 * Calculates Batchnorm by using the input, beta, gamma, mean and variance.
 * provided by the user.
 */

template <typename T, typename Direction, typename Operation,
          typename = DisableIf_Forward_Training<Direction, Operation>>
SNNStatus launch_forward(
    BaseMemObject<T const>& input, BaseMemObject<T const>& beta,
    BaseMemObject<T const>& gamma, BaseMemObject<T const>& running_mean,
    BaseMemObject<T const>& running_variance, BaseMemObject<T>& output,
    BatchNormParams const& params, cl::sycl::queue& queue) {
  return launch_batchnorm(input, beta, gamma, running_mean, running_variance,
                          output, params, queue);
}

}  // namespace internal
}  // namespace batchnorm
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_INTERNAL_BATCHNORM_LAUNCH_INTERNAL_H_
