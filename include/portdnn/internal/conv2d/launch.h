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
#ifndef PORTDNN_INCLUDE_INTERNAL_CONV2D_LAUNCH_H_
#define PORTDNN_INCLUDE_INTERNAL_CONV2D_LAUNCH_H_

/**
 * \file
 * Implements the \ref sycldnn::conv2d::sublaunch() function, which
 * asynchronously dispatches the SYCL kernels required to perform a 2D
 * convolution.
 */

#include "portdnn/status.h"

#include "portdnn/backend/backend_helpers.h"
#include "portdnn/conv2d/algorithm.h"
#include "portdnn/conv2d/params.h"
#include "portdnn/conv2d/selector/selector.h"

#include "portdnn/conv2d/implementation/direct.h"
#include "portdnn/conv2d/implementation/im2col.h"
#include "portdnn/conv2d/implementation/matmul.h"
#include "portdnn/conv2d/implementation/tiled.h"
#include "portdnn/conv2d/implementation/winograd.h"

namespace sycldnn {
namespace conv2d {

SNNStatus validate_params(Conv2DParams const& params) {
  SNN_VALIDATE_PARAM(params.batch > 0,
                     "The number of batches must be positive.");
  SNN_VALIDATE_PARAM(params.channels > 0,
                     "The number of channels must be positive.");
  SNN_VALIDATE_PARAM(params.features > 0,
                     "The number of features must be positive.");
  SNN_VALIDATE_PARAM(params.in_rows > 0,
                     "The number of input rows must be positive.");
  SNN_VALIDATE_PARAM(params.in_cols > 0,
                     "The number of input columns must be positive.");
  SNN_VALIDATE_PARAM(params.out_rows > 0,
                     "The number of output rows must be positive.");
  SNN_VALIDATE_PARAM(params.out_cols > 0,
                     "The number of output columns must be positive.");
  SNN_VALIDATE_PARAM(params.window_rows > 0,
                     "The number of window rows must be positive.");
  SNN_VALIDATE_PARAM(params.window_cols > 0,
                     "The number of window columns must be positive.");
  SNN_VALIDATE_PARAM(params.stride_rows > 0,
                     "The stride in the row direction must be positive.");
  SNN_VALIDATE_PARAM(params.stride_cols > 0,
                     "The stride in the column direction must be positive.");
  SNN_VALIDATE_PARAM(params.pad_rows >= 0,
                     "The padding in the row direction must be non-negative.");
  SNN_VALIDATE_PARAM(
      params.pad_cols >= 0,
      "The padding in the column direction must be non-negative.");
  SNN_VALIDATE_PARAM(params.groups >= 0,
                     "The number of groups must be non-negative.");
  SNN_VALIDATE_PARAM(params.channels % params.groups == 0,
                     "Channels must be divisble by groups.");
  SNN_VALIDATE_PARAM(params.features % params.groups == 0,
                     "Features must be divisble by groups.");
  SNN_VALIDATE_PARAM(params.dilation_rows == 1,
                     "Currently portDNN only supports dilation 1.");
  SNN_VALIDATE_PARAM(params.dilation_cols == 1,
                     "Currently portDNN only supports dilation 1.");

  auto implies = [](bool x, bool y) { return !x || y; };
  SNN_VALIDATE_PARAM(implies(params.input_format == DataFormat::NHWC,
                             params.filter_format == FilterFormat::HWCF ||
                                 params.filter_format == FilterFormat::FHWC),
                     "Unsupported layout combination.");
  SNN_VALIDATE_PARAM(implies(params.input_format == DataFormat::NCHW,
                             params.filter_format == FilterFormat::FCHW),
                     "Unsupported layout combination.");
  SNN_VALIDATE_PARAM(
      implies(params.groups == 1, params.group_format == BatchFormat::STRIDED),
      "Interleaved is unsupported when group size is one.");
  SNN_VALIDATE_PARAM(implies(params.group_format == BatchFormat::INTERLEAVED,
                             params.filter_format == FilterFormat::HWCF),
                     "Unsupported group and filter format combination.");

  return StatusCode::OK;
}

template <typename T, typename ConvType, typename Backend>
SNNStatus select_and_launch(
    typename Backend::template pointer_type<T const> input,
    typename Backend::template pointer_type<T const> filter,
    typename Backend::template pointer_type<T> output,
    Conv2DParams const& params, Algorithm& algo_tag, Backend& backend,
    typename Backend::template pointer_type<T> workspace,
    size_t workspace_size) {
  switch (algo_tag) {
    case Algorithm::Direct:
      return launch_direct<T, ConvType>(input, filter, output, params, backend,
                                        {});
    case Algorithm::Tiled:
      return launch_tiled<T, ConvType>(input, filter, output, params, backend,
                                       {});
    case Algorithm::Im2col:
      return launch_im2col<T, ConvType>(input, filter, output, workspace,
                                        params, workspace_size, backend, {});
    case Algorithm::Winograd:
      return launch_winograd<T, ConvType>(input, filter, output, workspace,
                                          params, workspace_size, backend, {});
    case Algorithm::WinogradLarge:
      return launch_winograd_large<T, ConvType>(input, filter, output,
                                                workspace, params,
                                                workspace_size, backend, {});
    case Algorithm::Matmul:
      return launch_matmul<T, ConvType>(input, filter, output, params, backend,
                                        {});
    case Algorithm::NotSupported:
    default:
      return StatusCode::InvalidAlgorithm;
  }
}

template <typename T, typename ConvType, typename Backend>
SNNStatus select_and_launch_usm(
    typename Backend::template pointer_type<T const> input,
    typename Backend::template pointer_type<T const> filter,
    typename Backend::template pointer_type<T> output,
    Conv2DParams const& params, Algorithm& algo_tag, Backend& backend,
    typename Backend::template pointer_type<T> workspace, size_t workspace_size,
    const std::vector<cl::sycl::event>& events) {
  // TODO Expand switch statement with more supported USM algos
  switch (algo_tag) {
    case Algorithm::Direct:
      return launch_direct<T, ConvType>(input, filter, output, params, backend,
                                        events);
    case Algorithm::Matmul:
      return launch_matmul<T, ConvType>(input, filter, output, params, backend,
                                        events);
    case Algorithm::Im2col:
      return launch_im2col<T, ConvType>(input, filter, output, workspace,
                                        params, workspace_size, backend,
                                        events);
    case Algorithm::Winograd:
      return launch_winograd<T, ConvType>(input, filter, output, workspace,
                                          params, workspace_size, backend,
                                          events);
    case Algorithm::WinogradLarge:
      return launch_winograd_large<T, ConvType>(
          input, filter, output, workspace, params, workspace_size, backend,
          events);
    case Algorithm::Tiled:
      return launch_tiled<T, ConvType>(input, filter, output, params, backend,
                                       events);
    default:
      return StatusCode::InvalidAlgorithm;
  }
}

template <typename T, typename ConvType, typename Backend>
SNNStatus sublaunch(typename Backend::template pointer_type<T const> input,
                    typename Backend::template pointer_type<T const> filter,
                    typename Backend::template pointer_type<T> output,
                    Conv2DParams const& params, Selector& selector,
                    Backend& backend,
                    typename Backend::template pointer_type<T> workspace,
                    size_t workspace_size,
                    const std::vector<cl::sycl::event>& events) {
  auto status = validate_params(params);
  if (status.status != StatusCode::OK) {
    return status;
  }
  SNN_VALIDATE_PARAM(
      (params.groups == 1 || std::is_same<ConvType, conv_type::Forward>::value),
      "Grouped convolution is only supported for the forward pass.");
  SNN_VALIDATE_PARAM((params.group_format != BatchFormat::INTERLEAVED) ||
                         backend::supports_interleaved_matmul<Backend>::value,
                     "The chosen backend does not support interleaved batched "
                     "matmul, used in im2col algorithm.");

  Algorithm algo_tag = selector.select<ConvType>(params);
  if (params.input_format == DataFormat::NCHW &&
      algo_tag != Algorithm::Direct) {
    return StatusCode::InvalidAlgorithm;
  }
  if (params.groups > 1 && algo_tag != Algorithm::Im2col) {
    return StatusCode::InvalidAlgorithm;
  }
  if constexpr (backend::is_usm_backend<Backend>::value) {
    return select_and_launch_usm<T, ConvType, Backend>(
        input, filter, output, params, algo_tag, backend, workspace,
        workspace_size, events);
  } else {
    return select_and_launch<T, ConvType, Backend>(input, filter, output,
                                                   params, algo_tag, backend,
                                                   workspace, workspace_size);
  }
}

}  // namespace conv2d
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_INTERNAL_CONV2D_LAUNCH_H_
