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
#ifndef PORTDNN_INCLUDE_CONV2D_IMPLEMENTATION_MATMUL_H_
#define PORTDNN_INCLUDE_CONV2D_IMPLEMENTATION_MATMUL_H_

#include "portdnn/conv2d/conv_type.h"
#include "portdnn/conv2d/params.h"

namespace sycldnn {
namespace conv2d {

namespace internal {

template <typename ConvType>
struct MatmulLauncher;

template <>
struct MatmulLauncher<conv_type::Forward> {
  template <typename T, typename Backend>
  static SNNStatus launch(
      typename Backend::template pointer_type<T const> input,
      typename Backend::template pointer_type<T const> filter,
      typename Backend::template pointer_type<T> output,
      Conv2DParams const& params, Backend& backend,
      const std::vector<cl::sycl::event>& events) {
    auto conv_width = params.batch * params.in_rows * params.in_cols;
    auto event = backend.template matmul<false, false>(
        input, filter, output, T{0}, conv_width, params.channels,
        params.features, events);
    return {event, StatusCode::OK};
  }
};

template <>
struct MatmulLauncher<conv_type::InputBackprop> {
  template <typename T, typename Backend>
  static SNNStatus launch(
      typename Backend::template pointer_type<T const> input,
      typename Backend::template pointer_type<T const> filter,
      typename Backend::template pointer_type<T> output,
      Conv2DParams const& params, Backend& backend,
      const std::vector<cl::sycl::event>& events) {
    auto conv_width = params.batch * params.in_rows * params.in_cols;
    auto event = backend.template matmul<false, true>(
        input, filter, output, T{0}, conv_width, params.features,
        params.channels, events);
    return {event, StatusCode::OK};
  }
};

template <>
struct MatmulLauncher<conv_type::FilterBackprop> {
  template <typename T, typename Backend>
  static SNNStatus launch(
      typename Backend::template pointer_type<T const> input,
      typename Backend::template pointer_type<T const> filter,
      typename Backend::template pointer_type<T> output,
      Conv2DParams const& params, Backend& backend,
      const std::vector<cl::sycl::event>& events) {
    auto conv_width = params.batch * params.in_rows * params.in_cols;
    auto event = backend.template matmul<true, false>(
        input, filter, output, T{0}, params.channels, conv_width,
        params.features, events);
    return {event, StatusCode::OK};
  }
};

}  // namespace internal
/**
 * Launch a matmul to compute a 1x1 2D convolution.
 *
 * Will extract the SYCL buffers and SYCL queue from the backend and forward
 * these on to the precompiled kernels.
 *
 * Returns an SNNStatus containing the SYCL event tied to the kernel launch.
 */
template <typename T, typename ConvType, typename Backend>
inline SNNStatus launch_matmul(
    typename Backend::template pointer_type<T const> input,
    typename Backend::template pointer_type<T const> filter,
    typename Backend::template pointer_type<T> output,
    Conv2DParams const& params, Backend& backend,
    const std::vector<cl::sycl::event>& events) {
  SNN_VALIDATE_PARAM(params.window_rows == 1,
                     "Matmul can only be used for 1x1 NHWC convolutions.");
  SNN_VALIDATE_PARAM(params.window_cols == 1,
                     "Matmul can only be used for 1x1 NHWC convolutions.");
  SNN_VALIDATE_PARAM(params.stride_rows == 1,
                     "Matmul can only be used with stride 1.");
  SNN_VALIDATE_PARAM(params.stride_cols == 1,
                     "Matmul can only be used with stride 1.");
  SNN_VALIDATE_PARAM(params.pad_rows == 0,
                     "Matmul can only be used with zero padding.");
  SNN_VALIDATE_PARAM(params.pad_cols == 0,
                     "Matmul can only be used with zero padding.");

  return internal::MatmulLauncher<ConvType>::template launch<T>(
      input, filter, output, params, backend, events);
}

}  // namespace conv2d
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_CONV2D_IMPLEMENTATION_MATMUL_H_
