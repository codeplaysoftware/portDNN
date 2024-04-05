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

#ifndef SYCLDNN_SRC_BATCHNORM_QUEUE_IMPL_H_
#define SYCLDNN_SRC_BATCHNORM_QUEUE_IMPL_H_

#include "sycldnn/mem_object.h"
#include "sycldnn/status.h"

#include "kernels.h"
#include "src/batchnorm/queue_batchnorm_kernel.h"
#include "sycldnn/helpers/dims.h"

#include <CL/sycl.hpp>

namespace sycldnn {
namespace batchnorm {
namespace internal {

template <typename T, typename Index, int VectorWidth,
          template <typename> class MemObj>
SNNStatus queue_batchnorm_inference(
    MemObj<T const>& input, MemObj<T const>& beta, MemObj<T const>& gamma,
    MemObj<T const>& input_mean, MemObj<T const>& input_variance,
    MemObj<T>& output, BatchNormParams const& params, cl::sycl::queue& queue,
    const std::vector<cl::sycl::event>& events) {
  auto event = queue.submit([&](cl::sycl::handler& cgh) {
    cgh.depends_on(events);
    auto input_mem = input.read_mem(cgh);
    auto beta_mem = beta.read_mem(cgh);
    auto gamma_mem = gamma.read_mem(cgh);
    auto mean_mem = input_mean.read_mem(cgh);
    auto variance_mem = input_variance.read_mem(cgh);
    auto output_mem = output.write_mem(cgh);

    if (params.input_format == sycldnn::DataFormat::NCHW) {
      BatchnormInferenceOp<T, Index, VectorWidth, /*IsNCHW=*/true,
                           is_usm_obj_v<MemObj<T>, T>>
          batchnorm_op(input_mem, beta_mem, gamma_mem, mean_mem, variance_mem,
                       output_mem, params);
      cgh.parallel_for(batchnorm_op.get_range(), batchnorm_op);
    } else if (params.input_format == sycldnn::DataFormat::NHWC) {
      BatchnormInferenceOp<T, Index, VectorWidth, /*IsNCHW=*/false,
                           is_usm_obj_v<MemObj<T>, T>>
          batchnorm_op(input_mem, beta_mem, gamma_mem, mean_mem, variance_mem,
                       output_mem, params);
      cgh.parallel_for(batchnorm_op.get_range(), batchnorm_op);
    }
  });

  return {event, StatusCode::OK};
}

template <typename T, typename Index, template <typename> class MemObj>
SNNStatus queue_batchnorm_training(
    MemObj<T const>& input, MemObj<T const>& beta, MemObj<T const>& gamma,
    MemObj<T>& running_mean, MemObj<T>& running_variance, MemObj<T>& cache_mean,
    MemObj<T>& cache_invvariance, MemObj<T>& workspace_var, MemObj<T>& output,
    BatchNormParams const& params, cl::sycl::queue& queue,
    const std::vector<cl::sycl::event>& events) {
  auto event = queue.submit([&](cl::sycl::handler& cgh) {
    cgh.depends_on(events);
    auto input_mem = input.read_mem(cgh);
    auto beta_mem = beta.read_mem(cgh);
    auto gamma_mem = gamma.read_mem(cgh);
    auto running_mean_mem = running_mean.write_mem(cgh);
    auto running_variance_mem = running_variance.write_mem(cgh);
    auto cache_mean_mem = cache_mean.write_mem(cgh);
    auto cache_invvariance_mem = cache_invvariance.write_mem(cgh);
    auto workspace_var_mem = workspace_var.write_mem(cgh);
    auto output_mem = output.write_mem(cgh);
    constexpr bool IsUSM = is_usm_obj_v<MemObj<T>, T>;

    if (params.input_format == sycldnn::DataFormat::NCHW) {
      BatchnormTrainingOp<T, Index, /*IsNCHW=*/true, IsUSM> batchnorm_op(
          input_mem, beta_mem, gamma_mem, running_mean_mem,
          running_variance_mem, cache_mean_mem, cache_invvariance_mem,
          workspace_var_mem, output_mem, params);
      cgh.parallel_for(batchnorm_op.get_range(), batchnorm_op);
    } else {
      BatchnormTrainingOp<T, Index, /*IsNCHW=*/false, IsUSM> batchnorm_op(
          input_mem, beta_mem, gamma_mem, running_mean_mem,
          running_variance_mem, cache_mean_mem, cache_invvariance_mem,
          workspace_var_mem, output_mem, params);
      cgh.parallel_for(batchnorm_op.get_range(), batchnorm_op);
    }
  });

  return {event, StatusCode::OK};
}

template <typename T, typename Index, int VectorWidth,
          template <typename> class MemObj>
SNNStatus queue_batchnorm_gradient(
    MemObj<T const>& input, MemObj<T const>& out_grad, MemObj<T>& in_grad,
    MemObj<T const>& gamma, MemObj<T const>& cache_mean,
    MemObj<T const>& cache_invvariance, MemObj<T>& workspace,
    BatchNormParams const& params, cl::sycl::queue& queue,
    const std::vector<cl::sycl::event>& events) {
  auto event = queue.submit([&](cl::sycl::handler& cgh) {
    cgh.depends_on(events);
    auto input_mem = input.read_mem(cgh);
    auto out_grad_mem = out_grad.read_mem(cgh);
    auto in_grad_mem = in_grad.write_mem(cgh);
    auto gamma_mem = gamma.read_mem(cgh);
    auto cache_mean_mem = cache_mean.read_mem(cgh);
    auto cache_invvariance_mem = cache_invvariance.read_mem(cgh);
    auto workspace_mem = workspace.write_mem(cgh);
    constexpr bool IsUSM = is_usm_obj_v<MemObj<T>, T>;

    if (params.input_format == sycldnn::DataFormat::NCHW) {
      bool constexpr IsNCHW = true;
      BatchnormGradientOp<T, Index, VectorWidth, IsNCHW, IsUSM>
          batchnorm_gradient_op(input_mem, out_grad_mem, in_grad_mem, gamma_mem,
                                cache_mean_mem, cache_invvariance_mem,
                                workspace_mem, params);
      cgh.parallel_for(batchnorm_gradient_op.get_range(),
                       batchnorm_gradient_op);
    } else {
      bool constexpr IsNCHW = false;
      BatchnormGradientOp<T, Index, VectorWidth, IsNCHW, IsUSM>
          batchnorm_gradient_op(input_mem, out_grad_mem, in_grad_mem, gamma_mem,
                                cache_mean_mem, cache_invvariance_mem,
                                workspace_mem, params);
      cgh.parallel_for(batchnorm_gradient_op.get_range(),
                       batchnorm_gradient_op);
    }
  });

  return {event, sycldnn::StatusCode::OK};
}

}  // namespace internal
}  // namespace batchnorm
}  // namespace sycldnn

#endif  // SYCLDNN_SRC_BATCHNORM_QUEUE_IMPL_H_
