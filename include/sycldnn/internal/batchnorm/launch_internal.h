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

#include "sycldnn/export.h"
#include "sycldnn/helpers/dims.h"
#include "sycldnn/status.h"

#include "sycldnn/batchnorm/direction.h"
#include "sycldnn/batchnorm/params.h"

#include "sycldnn/binaryop/operators.h"
#include "sycldnn/internal/binaryop/launch.h"

#include "sycldnn/internal/pointwise/launch_internal.h"
#include "sycldnn/pointwise/operators.h"

#include "sycldnn/internal/reduce/launch.h"
#include "sycldnn/reduce/operators.h"

#include "sycldnn/helpers/event_handling.h"
#include "sycldnn/helpers/mem_utils.h"
#include "sycldnn/internal/transpose/launch.h"

namespace sycldnn {
namespace batchnorm {
namespace internal {

template <typename Direction>
static constexpr bool IsForwardTraining =
    std::is_same<Direction, ForwardTraining>::value;

template <typename Direction>
static constexpr bool IsForwardInference =
    std::is_same<Direction, ForwardInference>::value;

template <typename Direction>
static constexpr bool IsGradient = std::is_same<Direction, Gradient>::value;

template <typename Direction>
using EnableIfForwardTraining =
    typename std::enable_if<IsForwardTraining<Direction>>::type;

template <typename Direction>
using EnableIfForwardInference =
    typename std::enable_if<IsForwardInference<Direction>>::type;

template <typename Direction>
using EnableIfGradient = typename std::enable_if<IsGradient<Direction>>::type;

template <typename Direction>
using DisableIfGradient = typename std::enable_if<!IsGradient<Direction>>::type;

inline std::vector<int> get_input_dims(BatchNormParams const& params) {
  switch (params.input_format) {
    case DataFormat::NHWC:
      return {params.batch, params.rows, params.cols, params.channels};

    case DataFormat::NCHW:
      return {params.batch, params.channels, params.rows, params.cols};
  }
  SNN_ASSERT(false, "Unsupported Layout");
  return {};
}

inline int get_non_channel_size(BatchNormParams const& params) {
  return params.batch * params.rows * params.cols;
}

inline std::vector<int> get_4d_channel_dims(BatchNormParams const& params) {
  switch (params.input_format) {
    case DataFormat::NHWC:
      return {1, 1, 1, params.channels};

    case DataFormat::NCHW:
      return {1, params.channels, 1, 1};
  }
  SNN_ASSERT(false, "Unsupported Layout");
  return {};
}

inline int get_total_size(BatchNormParams const& params) {
  return params.batch * params.rows * params.cols * params.channels;
}

template <typename T, typename Backend, template <typename> class MemObj>
SNN_EXPORT SNNStatus launch_inference_internal(
    MemObj<T const>& input, MemObj<T const>& beta, MemObj<T const>& gamma,
    MemObj<T const>& input_mean, MemObj<T const>& input_variance,
    MemObj<T>& output, BatchNormParams const& params, Backend& backend,
    const std::vector<cl::sycl::event>& events);

template <typename T, typename Backend, template <typename> class MemObj>
SNN_EXPORT SNNStatus launch_training_internal(
    MemObj<T const>& input, MemObj<T const>& beta, MemObj<T const>& gamma,
    MemObj<T>& running_mean, MemObj<T>& running_variance, MemObj<T>& cache_mean,
    MemObj<T>& cache_invvariance, MemObj<T>& workspace_var, MemObj<T>& output,
    BatchNormParams const& params, Backend& backend,
    const std::vector<cl::sycl::event>& events);

template <typename T, typename Backend, template <typename> class MemObj>
SNNStatus launch_gradient_internal(
    MemObj<T const>& input, MemObj<T const>& out_grad, MemObj<T>& in_grad,
    MemObj<T const>& gamma, MemObj<T const>& cache_mean,
    MemObj<T const>& cache_invvariance, MemObj<T>& workspace,
    BatchNormParams const& params, Backend& backend,
    const std::vector<cl::sycl::event>& events);

inline std::vector<int> get_binaryop_rhs_dims(const BatchNormParams& params) {
  if (params.input_format == sycldnn::DataFormat::NCHW) {
    return {1, params.channels, 1, 1};
  } else {
    return {params.channels};
  }
}

template <typename T, typename MemObj>
struct native_mem;

template <typename T>
struct native_mem<T, USMMemObject<T>> {
  constexpr static auto get(USMMemObject<T>& mem_obj) {
    return mem_obj.get_pointer();
  }
};

template <typename T>
struct native_mem<T, BufferMemObject<T>> {
  constexpr static auto get(BufferMemObject<T>& mem_obj) {
    return mem_obj.get_buffer();
  }
};

/**
 * The internal launch for batchnorm training.
 */
template <typename T, typename Backend, template <typename> class MemObj>
SNNStatus launch_training(MemObj<T const>& input, MemObj<T const>& beta,
                          MemObj<T const>& gamma, MemObj<T>& running_mean,
                          MemObj<T>& running_variance, MemObj<T>& cache_mean,
                          MemObj<T>& cache_invvariance, MemObj<T>& workspace,
                          MemObj<T>& output_mem, BatchNormParams const& params,
                          Backend& backend,
                          const std::vector<cl::sycl::event>& events) {
  auto const is_nchw = params.input_format == sycldnn::DataFormat::NCHW;
  SNNStatus status;
  std::vector<int> const input_dims = get_input_dims(params);
  std::vector<int> const rhs_dims_binaryop = get_binaryop_rhs_dims(params);
  std::vector<int> out_dims_binaryop;
  status = binaryop::internal::compute_out_dims(input_dims, rhs_dims_binaryop,
                                                out_dims_binaryop);
  if (status.status != sycldnn::StatusCode::OK) {
    return status;
  }

  int const workspace_offset = params.required_workspace_size - params.channels;
  auto workspace_input_minus_mean_sqr =
      make_mem_object(native_mem<T, MemObj<T>>::get(workspace),
                      workspace_offset, workspace.get_offset());
  auto workspace_var =
      make_mem_object(native_mem<T, MemObj<T>>::get(workspace), params.channels,
                      workspace_offset + workspace.get_offset());

  // Setup reduction parameters
  sycldnn::reduce::internal::ReduceParams reduce_params;
  reduce_params.max_sg_size = backend.get_max_sg_size();
  reduce_params.max_wg_size = backend.get_max_wg_size();
  if (is_nchw) {
    reduce_params.batch_size = params.batch * params.channels;
    reduce_params.reduce_size = params.rows * params.cols;
    reduce_params.stride_size = 1;
    reduce_params.batch_groups = params.channels;
  } else {
    reduce_params.batch_size = 1;
    reduce_params.reduce_size = params.batch * params.rows * params.cols;
    reduce_params.stride_size = params.channels;
  }
  auto mean_memset_event =
      backend.fill(cache_mean, params.channels, T{0}, events);
  auto var_memset_event =
      backend.fill(workspace_var, params.channels, T{0}, events);
  // Calculate mean
  status = sycldnn::reduce::internal::launch<T, sycldnn::reduce::Mean>(
      input, cache_mean, reduce_params, backend.get_queue(),
      {mean_memset_event});
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  // Calculate input - mean
  auto const_mean = cache_mean.as_const();
  status = binaryop::internal::launch_binaryop<binaryop::SubSqr>(
      input, const_mean, workspace_input_minus_mean_sqr, input_dims,
      rhs_dims_binaryop, backend.get_queue(), {status.event});
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  // Calculate variance: sum(input_mean_sqr) / num_elems_per_channel
  auto input_minus_mean_sqr_const = workspace_input_minus_mean_sqr.as_const();
  status = sycldnn::reduce::internal::launch<T, sycldnn::reduce::Mean>(
      input_minus_mean_sqr_const, workspace_var, reduce_params,
      backend.get_queue(), {var_memset_event, status.event});
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  // Calculate batch norm output and update running mean & running variance
  status = launch_training_internal(
      input, beta, gamma, running_mean, running_variance, cache_mean,
      cache_invvariance, workspace_var, output_mem, params, backend,
      {status.event});

  return status;
}

/**
 * The internal launch for batchnorm gradient.
 */
template <typename T, typename Backend, template <typename> class MemObj>
SNNStatus launch_gradient(MemObj<T const>& input, MemObj<T const>& out_grad,
                          MemObj<T>& in_grad, MemObj<T const>& gamma,
                          MemObj<T>& beta_grad, MemObj<T>& gamma_grad,
                          MemObj<T const>& cache_mean,
                          MemObj<T const>& cache_invvariance,
                          MemObj<T>& workspace, BatchNormParams const& params,
                          Backend& backend,
                          const std::vector<cl::sycl::event>& events) {
  auto const is_nchw = params.input_format == sycldnn::DataFormat::NCHW;
  SNNStatus status;

  // Setup reduction parameters
  sycldnn::reduce::internal::ReduceParams reduce_params;
  reduce_params.max_sg_size = backend.get_max_sg_size();
  reduce_params.max_wg_size = backend.get_max_wg_size();
  if (is_nchw) {
    reduce_params.batch_size = params.batch * params.channels;
    reduce_params.reduce_size = params.rows * params.cols;
    reduce_params.stride_size = 1;
    reduce_params.batch_groups = params.channels;
  } else {
    reduce_params.batch_size = 1;
    reduce_params.reduce_size = params.batch * params.rows * params.cols;
    reduce_params.stride_size = params.channels;
  }
  auto beta_grad_memset_event =
      backend.fill(beta_grad, params.channels, T{0}, events);
  auto gamma_grad_memset_event =
      backend.fill(gamma_grad, params.channels, T{0}, events);
  status = sycldnn::reduce::internal::launch<T, sycldnn::reduce::Add>(
      out_grad, beta_grad, reduce_params, backend.get_queue(),
      {beta_grad_memset_event});
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  status = launch_gradient_internal(input, out_grad, in_grad, gamma, cache_mean,
                                    cache_invvariance, workspace, params,
                                    backend, {status.event});
  if (status.status != sycldnn::StatusCode::OK) {
    return status;
  }

  auto const_workspace = workspace.as_const();
  return sycldnn::reduce::internal::launch<T, sycldnn::reduce::Add>(
      const_workspace, gamma_grad, reduce_params, backend.get_queue(),
      {gamma_grad_memset_event, status.event});
}

}  // namespace internal
}  // namespace batchnorm
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_INTERNAL_BATCHNORM_LAUNCH_INTERNAL_H_
