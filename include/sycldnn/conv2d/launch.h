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
#ifndef SYCLDNN_INCLUDE_CONV2D_LAUNCH_H_
#define SYCLDNN_INCLUDE_CONV2D_LAUNCH_H_

/**
 * \file
 * Implements the \ref sycldnn::conv2d::launch() function, which asynchronously
 * dispatches the SYCL kernels required to perform a 2D convolution.
 */
#include "sycldnn/status.h"

#include "sycldnn/conv2d/algorithm.h"
#include "sycldnn/conv2d/params.h"

#include "sycldnn/conv2d/implementation/direct.h"
#include "sycldnn/conv2d/implementation/im2col.h"
#include "sycldnn/conv2d/implementation/matmul.h"
#include "sycldnn/conv2d/implementation/tiled.h"
#include "sycldnn/conv2d/implementation/winograd.h"
#include "sycldnn/conv2d/selector/selector.h"

namespace sycldnn {
namespace conv2d {
/**
 * Launch a 2D convolution, with the implementation chosen by the Selector.
 *
 * The selector will be used to select which implementation to use, and the
 * corresponding kernels will be launched. If any additional temporary memory is
 * required then it will be allocated through the backend.
 *
 * \param input A pointer to the memory representing the input tensor.
 * \param filter A pointer to the memory representing the tensor of filter
 *               coefficients.
 * \param output A pointer to the memory represnting the output tensor.
 * \param params The convolution parameters, which describe the tensor shapes
 *               and convolution strides.
 * \param selector An instance of \ref sycldnn::conv2d::Selector, used to guide
 *                 the selection of the most appropriate convolution algorithm
 *                 for a specific target platform or problem size.
 * \param backend The backend implementation, used to provide optimized matrix
 *                multiplies and to map between pointer represntations.
 * \param workspace Optional pointer to a workspace buffer for use whenever
 *                  temporary memory is required.
 * \param workspace_size The number of elements available in the workspace
 *                       buffer.
 * \return Returns an SNNStatus containing the SYCL event tied to the kernel
 * launches and a StatusCode enum showing if the launch was OK or whether it
 * encountered some problem.
 */
template <typename T, typename ConvType, typename Backend>
SNNStatus launch(typename Backend::template pointer_type<T const> input,
                 typename Backend::template pointer_type<T const> filter,
                 typename Backend::template pointer_type<T> output,
                 Conv2DParams const& params, Selector& selector,
                 Backend& backend,
                 typename Backend::template pointer_type<T> workspace = {},
                 size_t workspace_size = 0) {
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
  SNN_VALIDATE_PARAM(params.dilation_rows == 1,
                     "Currently SYCL-DNN only supports dilation 1.");
  SNN_VALIDATE_PARAM(params.dilation_cols == 1,
                     "Currently SYCL-DNN only supports dilation 1.");
  SNN_VALIDATE_PARAM(params.input_format == sycldnn::DataFormat::NHWC,
                     "Currently SYCL-DNN only supports the NHWC data format.");
  SNN_VALIDATE_PARAM(
      params.filter_format == sycldnn::FilterFormat::HWCF,
      "Currently SYCL-DNN only supports the HWCF filter format.");

  Algorithm algo_tag = selector.select<ConvType>(params);
  switch (algo_tag) {
    case Algorithm::Direct:
      return launch_direct<T, ConvType>(input, filter, output, params, backend);
    case Algorithm::Tiled:
      return launch_tiled<T, ConvType>(input, filter, output, params, backend);
    case Algorithm::Im2col:
      return launch_im2col<T, ConvType>(input, filter, output, workspace,
                                        params, workspace_size, backend);
    case Algorithm::Winograd:
      return launch_winograd<T, ConvType>(input, filter, output, workspace,
                                          params, workspace_size, backend);
    case Algorithm::WinogradLarge:
      return launch_winograd_large<T, ConvType>(
          input, filter, output, workspace, params, workspace_size, backend);
    case Algorithm::Matmul:
      return launch_matmul<T, ConvType>(input, filter, output, params, backend);
    case Algorithm::NotSupported:
    default:
      SNNStatus not_supported_status;
      not_supported_status.status = StatusCode::InvalidAlgorithm;
      return not_supported_status;
  }
}
}  // namespace conv2d
}  // namespace sycldnn
#endif  // SYCLDNN_INCLUDE_CONV2D_LAUNCH_H_
