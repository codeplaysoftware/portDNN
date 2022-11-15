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

#include "sycldnn/binaryop/operators.h"
#include "sycldnn/internal/binaryop/launch.h"

#include "sycldnn/internal/pointwise/launch_internal.h"
#include "sycldnn/pointwise/operators.h"

#include "sycldnn/internal/reduce/launch.h"
#include "sycldnn/reduce/operators.h"

#include "sycldnn/internal/transpose/launch.h"

namespace sycldnn {
namespace batchnorm {
namespace internal {

template <typename Direction>
static constexpr bool IsGradient = std::is_same<Direction, Gradient>::value;

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

inline int get_non_channel_size(BatchNormParams const& params) {
  return params.batch * params.rows * params.cols;
}

/**
 * The internal launcher for computing variance.
 */
template <typename T, typename Backend>
SNNStatus launch_variance(BaseMemObject<T const>& centered_input,
                          BaseMemObject<T>& variance,
                          BaseMemObject<T>& squared_centered_input,
                          BatchNormParams const& params, Backend& backend) {
  auto queue = backend.get_queue();
  SNNStatus status;
  auto _centered_input = mo_to_bo(centered_input);
  auto _squared_centered_input = mo_to_bo(squared_centered_input);
  status = binaryop::internal::launch_binaryop<binaryop::Mul>(
      _centered_input, _centered_input, _squared_centered_input,
      get_total_size(params), queue, {});
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  auto const_squared_centered_input = squared_centered_input.as_const();
  auto _const_squared_centered_input = mo_to_bo(const_squared_centered_input);
  auto _variance = mo_to_bo(variance);
  status = reduce::internal::launch<reduce::Mean>(
      _const_squared_centered_input, _variance, 1, get_non_channel_size(params),
      params.channels, backend, {});
  return status;
}

/**
 * The internal launcher for computing batchnorm.
 */
template <typename T>
inline SNNStatus launch_batchnorm(
    BaseMemObject<T const>& centered_input, BaseMemObject<T const>& beta,
    BaseMemObject<T const>& gamma, BaseMemObject<T const>& current_variance,
    BaseMemObject<T>& output, BaseMemObject<T>& workspace, const float epsilon,
    const std::vector<int>& input_dims, const std::vector<int>& channel_dims,
    cl::sycl::queue& queue) {
  cl::sycl::buffer<T const, 1> epsilon_buf(&epsilon, cl::sycl::range<1>(1));
  auto epsilon_mem = make_mem_object<T const>(epsilon_buf, 1);

  auto _current_variance = mo_to_bo(current_variance);
  auto _epsilon_mem = mo_to_bo(epsilon_mem);
  auto _workspace = mo_to_bo(workspace);
  SNNStatus status = binaryop::internal::launch_binaryop<binaryop::Add>(
      _current_variance, _epsilon_mem, _workspace, channel_dims, {1}, queue,
      {});
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  auto const_workspace = workspace.as_const();
  auto _const_workspace = mo_to_bo(const_workspace);
  status = pointwise::internal::launch_pointwise<pointwise::Sqrt>(
      _const_workspace, _workspace, helpers::get_total_size(channel_dims),
      queue, {});
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  auto _centered_input = mo_to_bo(centered_input);
  auto _output = mo_to_bo(output);
  status = binaryop::internal::launch_binaryop<binaryop::Div>(
      _centered_input, _const_workspace, _output, input_dims, channel_dims,
      queue, {});
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  auto const_output = output.as_const();
  auto _const_output = mo_to_bo(const_output);
  auto _gamma = mo_to_bo(gamma);
  status = binaryop::internal::launch_binaryop<binaryop::Mul>(
      _const_output, _gamma, _output, input_dims, channel_dims, queue, {});
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  auto _beta = mo_to_bo(beta);
  return binaryop::internal::launch_binaryop<binaryop::Add>(
      _const_output, _beta, _output, input_dims, channel_dims, queue, {});
}

/**
 * Compute running mean and running variance:
 * output = input * momentum + output * (1 - momentum)
 */
template <typename T>
inline SNNStatus launch_running_mean_variance(
    BaseMemObject<T const>& input, BaseMemObject<T const>& momentum,
    BaseMemObject<T const>& one_minus_momentum, BaseMemObject<T>& output,
    BaseMemObject<T>& workspace, int size, cl::sycl::queue& queue) {
  auto const_output = output.as_const();
  auto _const_output = mo_to_bo(const_output);
  auto _one_minus_momentum = mo_to_bo(one_minus_momentum);
  auto _output = mo_to_bo(output);
  SNNStatus status = binaryop::internal::launch_binaryop<binaryop::Mul>(
      _const_output, _one_minus_momentum, _output, {size}, {1}, queue, {});
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  auto _input = mo_to_bo(input);
  auto _momentum = mo_to_bo(momentum);
  auto _workspace = mo_to_bo(workspace);
  status = binaryop::internal::launch_binaryop<binaryop::Mul>(
      _input, _momentum, _workspace, {size}, {1}, queue, {});
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  auto const_workspace = workspace.as_const();
  auto _const_workspace = mo_to_bo(const_workspace);
  status = binaryop::internal::launch_binaryop<binaryop::Add>(
      _const_output, _const_workspace, _output, size, queue, {});
  return status;
}

/**
 * The internal batchnorm launcher for Forward Direction when computing Mean and
 * Variance.
 *
 * Calculates Mean, then Variance and then Batchnorm.
 */

template <typename T, typename Backend>
SNNStatus launch_forward(
    BaseMemObject<T const>& input, BaseMemObject<T const>& beta,
    BaseMemObject<T const>& gamma, BaseMemObject<T const>& input_mean,
    BaseMemObject<T const>& input_variance, BaseMemObject<T>& running_mean,
    BaseMemObject<T>& running_variance, BaseMemObject<T>& output,
    BatchNormParams const& params, Backend& backend) {
  auto n_items = get_total_size(params);
  auto queue = backend.get_queue();
  auto nhwc_dims = {params.batch, params.rows, params.cols, params.channels};
  auto channel_dims = {params.channels};
  const bool is_nchw = params.input_format == DataFormat::NCHW;
  SNNStatus status;

  cl::sycl::buffer<T, 1> tr_input_buf((cl::sycl::range<1>(n_items)));
  auto tr_input = make_mem_object(tr_input_buf, n_items);
  // Transpose NCHW input to NHWC to reduce NHW dimensions in one go.
  if (is_nchw) {
    auto input_dims = get_input_dims(params);
    auto in = mo_to_bo(input);
    auto out = mo_to_bo(tr_input);
    status =
        transpose::internal::launch(in, out, input_dims, NCHW_TO_NHWC, queue);
    if (sycldnn::StatusCode::OK != status.status) {
      return status;
    }
  }
  auto const_tr_input = tr_input.as_const();
  auto& nhwc_input = is_nchw ? const_tr_input : input;

  cl::sycl::buffer<T, 1> centered_input_buf((cl::sycl::range<1>(n_items)));
  auto centered_input = make_mem_object(centered_input_buf, n_items);

  auto _nhwc_input = mo_to_bo(nhwc_input);
  auto _input_mean = mo_to_bo(input_mean);
  auto _centered_input = mo_to_bo(centered_input);
  status = binaryop::internal::launch_binaryop<binaryop::Sub>(
      _nhwc_input, _input_mean, _centered_input, nhwc_dims, channel_dims, queue,
      {});
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  cl::sycl::buffer<T, 1> workspace_buf((cl::sycl::range<1>(params.channels)));
  auto workspace = make_mem_object(workspace_buf, params.channels);
  auto const_centered_input = centered_input.as_const();
  auto& tr_output = centered_input;  // Re-use temporary buffer
  status = launch_batchnorm(const_centered_input, beta, gamma, input_variance,
                            is_nchw ? tr_output : output, workspace,
                            params.epsilon, nhwc_dims, channel_dims, queue);
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  // Transpose NHWC output back to NCHW.
  if (is_nchw) {
    auto const_tr_output = tr_output.as_const();
    auto in = mo_to_bo(const_tr_output);
    auto out = mo_to_bo(output);
    status =
        transpose::internal::launch(in, out, nhwc_dims, NHWC_TO_NCHW, queue);
    if (sycldnn::StatusCode::OK != status.status) {
      return status;
    }
  }

  auto _running_mean = mo_to_bo(running_mean);
  status = reduce::internal::launch<reduce::Mean>(_nhwc_input, _running_mean, 1,
                                                  get_non_channel_size(params),
                                                  params.channels, backend, {});
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  auto const_running_mean = running_mean.as_const();
  auto _const_running_mean = mo_to_bo(const_running_mean);
  status = binaryop::internal::launch_binaryop<binaryop::Sub>(
      _nhwc_input, _const_running_mean, _centered_input, nhwc_dims,
      channel_dims, queue, {});
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  cl::sycl::buffer<T const, 1> momentum_buf(&params.momentum,
                                            cl::sycl::range<1>(1));
  auto momentum = make_mem_object<T const>(momentum_buf, 1);
  const T one_minus_momentum_val = 1 - params.momentum;
  cl::sycl::buffer<T const, 1> one_minus_momentum_buf(&one_minus_momentum_val,
                                                      cl::sycl::range<1>(1));
  auto one_minus_momentum = make_mem_object<T const>(one_minus_momentum_buf, 1);

  status = launch_running_mean_variance(input_mean, momentum,
                                        one_minus_momentum, running_mean,
                                        workspace, params.channels, queue);
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  auto& squared_centered_input = tr_input;  // Re-use temporary buffer
  status = launch_variance(const_centered_input, running_variance,
                           squared_centered_input, params, backend);
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  status = launch_running_mean_variance(input_variance, momentum,
                                        one_minus_momentum, running_variance,
                                        workspace, params.channels, queue);
  return status;
}

/**
 * The internal batchnorm launcher for Forward Direction when using the existing
 * Mean and Variance.
 *
 * Calculates Batchnorm by using the input, beta, gamma, mean and variance.
 * provided by the user.
 */

template <typename T, typename Backend>
SNNStatus launch_forward(BaseMemObject<T const>& input,
                         BaseMemObject<T const>& beta,
                         BaseMemObject<T const>& gamma,
                         BaseMemObject<T const>& running_mean,
                         BaseMemObject<T const>& running_variance,
                         BaseMemObject<T>& output,
                         BatchNormParams const& params, Backend& backend) {
  auto n_items = get_total_size(params);
  auto queue = backend.get_queue();
  auto input_dims = get_input_dims(params);
  auto channel_dims = get_4d_channel_dims(params);

  cl::sycl::buffer<T, 1> centered_input_buf((cl::sycl::range<1>(n_items)));
  auto centered_input = make_mem_object(centered_input_buf, n_items);
  SNNStatus status;
  auto _input = mo_to_bo(input);
  auto _running_mean = mo_to_bo(running_mean);
  auto _centered_input = mo_to_bo(centered_input);
  status = sycldnn::binaryop::internal::launch_binaryop<sycldnn::binaryop::Sub>(
      _input, _running_mean, _centered_input, input_dims, channel_dims, queue,
      {});
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  cl::sycl::buffer<T, 1> workspace_buf((cl::sycl::range<1>(params.channels)));
  auto workspace = make_mem_object(workspace_buf, params.channels);
  auto const_centered_input = centered_input.as_const();
  return launch_batchnorm(const_centered_input, beta, gamma, running_variance,
                          output, workspace, params.epsilon, input_dims,
                          channel_dims, queue);
}

/**
 * The internal batchnorm launcher for Gradient Direction when computing Mean
 * and Variance.
 *
 * Calculates Mean, then Variance and then Batchnorm.
 * https://github.com/tensorflow/tensorflow/blob/d916f20e1f1897696a19158ac7f5bd8d83e1b857/tensorflow/python/ops/nn_grad.py#L924
 */

template <typename T, typename Backend>
SNNStatus launch_gradient(BaseMemObject<T const>& input,
                          BaseMemObject<T const>& gradient,
                          BaseMemObject<T const>& gamma,
                          BaseMemObject<T>& beta_grad,
                          BaseMemObject<T>& gamma_grad,
                          BaseMemObject<T>& output,
                          BatchNormParams const& params, Backend& backend) {
  auto nhwc_dims = {params.batch, params.rows, params.cols, params.channels};
  auto n_items = get_total_size(params);
  auto queue = backend.get_queue();
  const bool is_nchw = params.input_format == DataFormat::NCHW;
  SNNStatus status;

  cl::sycl::buffer<T, 1> tr_input_buf((cl::sycl::range<1>(n_items)));
  auto tr_input = make_mem_object(tr_input_buf, n_items);
  cl::sycl::buffer<T, 1> tr_gradient_buf((cl::sycl::range<1>(n_items)));
  auto tr_gradient = make_mem_object(tr_gradient_buf, n_items);
  // Transpose NCHW input and gradient to NHWC to reduce NHW dimensions in one
  // go.
  if (is_nchw) {
    auto input_dims = get_input_dims(params);
    auto in = mo_to_bo(input);
    auto out = mo_to_bo(tr_input);
    status =
        transpose::internal::launch(in, out, input_dims, NCHW_TO_NHWC, queue);
    if (sycldnn::StatusCode::OK != status.status) {
      return status;
    }

    auto in2 = mo_to_bo(gradient);
    auto out2 = mo_to_bo(tr_gradient);
    status =
        transpose::internal::launch(in2, out2, input_dims, NCHW_TO_NHWC, queue);
    if (sycldnn::StatusCode::OK != status.status) {
      return status;
    }
  }
  auto const_tr_input = tr_input.as_const();
  auto const_tr_gradient = tr_gradient.as_const();
  auto& nhwc_input = is_nchw ? const_tr_input : input;
  auto& nhwc_gradient = is_nchw ? const_tr_gradient : gradient;

  cl::sycl::buffer<T, 1> mean_input_buf((cl::sycl::range<1>(params.channels)));
  auto mean_input = make_mem_object(mean_input_buf, params.channels);
  auto _nhwc_input = mo_to_bo(nhwc_input);
  auto _mean_input = mo_to_bo(mean_input);
  status = reduce::internal::launch<reduce::Mean>(_nhwc_input, _mean_input, 1,
                                                  get_non_channel_size(params),
                                                  params.channels, backend, {});
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  cl::sycl::buffer<T, 1> centered_input_buf((cl::sycl::range<1>(n_items)));
  auto centered_input = make_mem_object(centered_input_buf, n_items);
  auto const_mean_input = mean_input.as_const();
  auto _const_mean_input = mo_to_bo(const_mean_input);
  auto _centered_input = mo_to_bo(centered_input);
  status = sycldnn::binaryop::internal::launch_binaryop<sycldnn::binaryop::Sub>(
      _nhwc_input, _const_mean_input, _centered_input, nhwc_dims,
      {params.channels}, queue, {});
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  cl::sycl::buffer<T, 1> scaled_input_buf((cl::sycl::range<1>(n_items)));
  auto scaled_input = make_mem_object(scaled_input_buf, n_items);
  auto const_centered_input = centered_input.as_const();
  cl::sycl::buffer<T, 1> input_variance_buf(
      (cl::sycl::range<1>(params.channels)));
  auto input_variance = make_mem_object(input_variance_buf, params.channels);
  status = launch_variance(const_centered_input, input_variance, scaled_input,
                           params, backend);
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  cl::sycl::buffer<T const, 1> epsilon_buf(&params.epsilon,
                                           cl::sycl::range<1>(1));
  auto epsilon = make_mem_object<T const>(epsilon_buf, 1);
  auto const_input_variance = input_variance.as_const();
  auto _const_input_variance = mo_to_bo(const_input_variance);
  auto _epsilon = mo_to_bo(epsilon);
  auto _input_variance = mo_to_bo(input_variance);
  status = binaryop::internal::launch_binaryop<binaryop::Add>(
      _const_input_variance, _epsilon, _input_variance, {params.channels}, {1},
      queue, {});
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  auto _nhwc_gradient = mo_to_bo(nhwc_gradient);
  auto _beta_grad = mo_to_bo(beta_grad);
  status = reduce::internal::launch<reduce::Add>(_nhwc_gradient, _beta_grad, 1,
                                                 get_non_channel_size(params),
                                                 params.channels, backend, {});
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  cl::sycl::buffer<T, 1> mean_gradient_buf(
      (cl::sycl::range<1>(params.channels)));
  auto mean_gradient = make_mem_object(mean_gradient_buf, params.channels);
  T num_elts_val = get_non_channel_size(params);
  cl::sycl::buffer<T const, 1> num_elts_buf(&num_elts_val,
                                            cl::sycl::range<1>(1));
  auto num_elts = make_mem_object<T const>(num_elts_buf, 1);
  auto const_beta_grad = beta_grad.as_const();
  auto _const_beta_grad = mo_to_bo(const_beta_grad);
  auto _num_elts = mo_to_bo(num_elts);
  auto _mean_gradient = mo_to_bo(mean_gradient);
  status = binaryop::internal::launch_binaryop<binaryop::Div>(
      _const_beta_grad, _num_elts, _mean_gradient, {params.channels}, {1},
      queue, {});
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  auto const_mean_gradient = mean_gradient.as_const();
  auto _const_mean_gradient = mo_to_bo(const_mean_gradient);
  auto _output = mo_to_bo(output);
  status = sycldnn::binaryop::internal::launch_binaryop<sycldnn::binaryop::Sub>(
      _nhwc_gradient, _const_mean_gradient, _output, nhwc_dims,
      {params.channels}, queue, {});
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  auto _const_centered_input = mo_to_bo(const_centered_input);
  auto _scaled_input = mo_to_bo(scaled_input);
  status = sycldnn::binaryop::internal::launch_binaryop<sycldnn::binaryop::Mul>(
      _nhwc_gradient, _const_centered_input, _scaled_input, nhwc_dims, queue,
      {});
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  auto const_scaled_input = scaled_input.as_const();
  auto _const_scaled_input = mo_to_bo(const_scaled_input);
  auto _gamma_grad = mo_to_bo(gamma_grad);
  status = reduce::internal::launch<reduce::Add>(
      _const_scaled_input, _gamma_grad, 1, get_non_channel_size(params),
      params.channels, backend, {});
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  cl::sycl::buffer<T, 1> workspace_buf((cl::sycl::range<1>(params.channels)));
  auto workspace = make_mem_object(workspace_buf, params.channels);
  auto const_gamma_grad = gamma_grad.as_const();
  auto _const_gamma_grad = mo_to_bo(const_gamma_grad);
  auto _workspace = mo_to_bo(workspace);
  status = binaryop::internal::launch_binaryop<binaryop::Div>(
      _const_gamma_grad, _num_elts, _workspace, {params.channels}, {1}, queue,
      {});
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  auto const_workspace = workspace.as_const();
  auto _const_workspace = mo_to_bo(const_workspace);
  status = binaryop::internal::launch_binaryop<binaryop::Div>(
      _const_workspace, _const_input_variance, _workspace, params.channels,
      queue, {});
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  status = binaryop::internal::launch_binaryop<binaryop::Mul>(
      _const_centered_input, _const_workspace, _centered_input, nhwc_dims,
      {params.channels}, queue, {});
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  auto const_output = output.as_const();
  auto _const_output = mo_to_bo(const_output);
  status = binaryop::internal::launch_binaryop<binaryop::Sub>(
      _const_output, _const_centered_input, _output, nhwc_dims, queue, {});
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  status = pointwise::internal::launch_pointwise<pointwise::Sqrt>(
      _const_input_variance, _input_variance, params.channels, queue, {});
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  auto _gamma = mo_to_bo(gamma);
  status = binaryop::internal::launch_binaryop<binaryop::Mul>(
      _const_output, _gamma, _output, nhwc_dims, {params.channels}, queue, {});
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  auto& tr_output = tr_input;  // Re-use temporary buffer
  auto _tr_output = mo_to_bo(tr_output);
  status = binaryop::internal::launch_binaryop<binaryop::Div>(
      _const_output, _const_input_variance, is_nchw ? _tr_output : _output,
      nhwc_dims, {params.channels}, queue, {});
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  // Transpose NHWC output back to NCHW.
  if (is_nchw) {
    auto const_tr_output = tr_output.as_const();
    auto in = mo_to_bo(const_tr_output);
    auto out = mo_to_bo(output);
    status =
        transpose::internal::launch(in, out, nhwc_dims, NHWC_TO_NCHW, queue);
    if (sycldnn::StatusCode::OK != status.status) {
      return status;
    }
  }

  status = binaryop::internal::launch_binaryop<binaryop::Div>(
      _const_gamma_grad, _const_input_variance, _gamma_grad, params.channels,
      queue, {});
  return status;
}

/**
 * The internal batchnorm launcher for Gradient Direction when using the
 * existing Mean and Variance.
 *
 * Calculates Batchnorm by using the input, beta, gamma, mean and variance.
 * provided by the user.
 */

template <typename T, typename Backend>
SNNStatus launch_gradient(
    BaseMemObject<T const>& input, BaseMemObject<T const>& gradient,
    BaseMemObject<T const>& gamma, BaseMemObject<T const>& pop_mean,
    BaseMemObject<T const>& pop_variance, BaseMemObject<T>& beta_grad,
    BaseMemObject<T>& gamma_grad, BaseMemObject<T>& output,
    BatchNormParams const& params, Backend& backend) {
  auto input_dims = get_input_dims(params);
  auto channel_dims = get_4d_channel_dims(params);
  auto n_items = get_total_size(params);
  auto queue = backend.get_queue();
  const bool is_nchw = params.input_format == DataFormat::NCHW;
  SNNStatus status;

  cl::sycl::buffer<T const, 1> epsilon_buf(&params.epsilon,
                                           cl::sycl::range<1>(1));
  auto epsilon = make_mem_object<T const>(epsilon_buf, 1);
  cl::sycl::buffer<T, 1> workspace_buf((cl::sycl::range<1>(params.channels)));
  auto workspace = make_mem_object(workspace_buf, params.channels);

  auto _pop_variance = mo_to_bo(pop_variance);
  auto _epsilon = mo_to_bo(epsilon);
  auto _workspace = mo_to_bo(workspace);
  status = binaryop::internal::launch_binaryop<binaryop::Add>(
      _pop_variance, _epsilon, _workspace, channel_dims, {1}, queue, {});
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  auto const_workspace = workspace.as_const();
  auto _const_workspace = mo_to_bo(const_workspace);
  status = pointwise::internal::launch_pointwise<pointwise::Sqrt>(
      _const_workspace, _workspace, params.channels, queue, {});
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  auto _input = mo_to_bo(input);
  auto _pop_mean = mo_to_bo(pop_mean);
  auto _output = mo_to_bo(output);
  status = sycldnn::binaryop::internal::launch_binaryop<sycldnn::binaryop::Sub>(
      _input, _pop_mean, _output, input_dims, channel_dims, queue, {});
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  auto const_output = output.as_const();
  auto _const_output = mo_to_bo(const_output);
  auto _gradient = mo_to_bo(gradient);
  status = binaryop::internal::launch_binaryop<binaryop::Mul>(
      _const_output, _gradient, _output, input_dims, queue, {});
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  status = binaryop::internal::launch_binaryop<binaryop::Div>(
      _const_output, _const_workspace, _output, input_dims, channel_dims, queue,
      {});
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  cl::sycl::buffer<T, 1> tr_reduce_buf((cl::sycl::range<1>(n_items)));
  auto tr_reduce = make_mem_object(tr_reduce_buf, n_items);
  // Transpose NCHW tensor to NHWC to reduce NHW dimensions in one go.
  if (is_nchw) {
    auto in = mo_to_bo(const_output);
    auto out = mo_to_bo(tr_reduce);
    status = transpose::internal::launch(in, out, input_dims, NCHW_TO_NHWC,
                                         queue, {});
    if (sycldnn::StatusCode::OK != status.status) {
      return status;
    }
  }
  auto const_tr_reduce = tr_reduce.as_const();
  BaseMemObject<T const>& nhwc_reduce_1 =
      is_nchw ? const_tr_reduce : const_output;

  auto _nhwc_reduce_1 = mo_to_bo(nhwc_reduce_1);
  auto _gamma_grad = mo_to_bo(gamma_grad);
  status = reduce::internal::launch<reduce::Add>(_nhwc_reduce_1, _gamma_grad, 1,
                                                 get_non_channel_size(params),
                                                 params.channels, backend, {});
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  auto _gamma = mo_to_bo(gamma);
  status = binaryop::internal::launch_binaryop<binaryop::Div>(
      _gamma, _const_workspace, _workspace, params.channels, queue, {});
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  status = binaryop::internal::launch_binaryop<binaryop::Mul>(
      _gradient, _const_workspace, _output, input_dims, channel_dims, queue,
      {});
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  // Transpose NCHW tensor to NHWC to reduce NHW dimensions in one go.
  if (is_nchw) {
    auto in = mo_to_bo(gradient);
    auto out = mo_to_bo(tr_reduce);
    status =
        transpose::internal::launch(in, out, input_dims, NCHW_TO_NHWC, queue);
    if (sycldnn::StatusCode::OK != status.status) {
      return status;
    }
    const_tr_reduce = tr_reduce.as_const();
  }
  BaseMemObject<T const>& nhwc_reduce_2 = is_nchw ? const_tr_reduce : gradient;
  auto _nhwc_reduce_2 = mo_to_bo(nhwc_reduce_2);
  auto _beta_grad = mo_to_bo(beta_grad);
  status = reduce::internal::launch<reduce::Add>(_nhwc_reduce_2, _beta_grad, 1,
                                                 get_non_channel_size(params),
                                                 params.channels, backend, {});
  return status;
}

}  // namespace internal
}  // namespace batchnorm
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_INTERNAL_BATCHNORM_LAUNCH_INTERNAL_H_
