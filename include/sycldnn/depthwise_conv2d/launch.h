/*
 * Copyright 2018 Codeplay Software Ltd
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
#ifndef SYCLDNN_INCLUDE_DEPTHWISE_CONV2D_LAUNCH_H_
#define SYCLDNN_INCLUDE_DEPTHWISE_CONV2D_LAUNCH_H_

/**
 * \file
 * Implements the \ref sycldnn::depthwise_conv2d::launch() function, which
 * asynchronously dispatches the SYCL kernels required to perform a 2D
 * convolution.
 */
#include "sycldnn/status.h"

#include "sycldnn/depthwise_conv2d/params.h"
#include "sycldnn/depthwise_conv2d/sizes.h"

#include "sycldnn/helpers/macros.h"

#include "sycldnn/internal/depthwise_conv2d/launch.h"

namespace sycldnn {
namespace depthwise_conv2d {

/**
 * Launch a 2D depthwise convolution.
 *
 * \param input A pointer to the memory representing the input tensor.
 * \param filter A pointer to the memory representing the tensor of filter
 *               coefficients.
 * \param output A pointer to the memory represnting the output tensor.
 * \param params The convolution parameters, which describe the tensor shapes
 *               and convolution strides.
 * \param backend The backend implementation, used to provide optimized matrix
 *                multiplies and to map between pointer represntations.
 * \return Returns an SNNStatus containing the SYCL event tied to the kernel
 * launches and a StatusCode enum showing if the launch was OK or whether it
 * encountered some problem.
 */
template <typename T, typename ConvType, typename Backend>
SNNStatus launch(typename Backend::template pointer_type<T const> input,
                 typename Backend::template pointer_type<T const> filter,
                 typename Backend::template pointer_type<T> output,
                 DepthwiseConv2DParams const& params, Backend& backend) {
  SNN_VALIDATE_PARAM(params.batch > 0,
                     "The number of batches must be positive.");
  SNN_VALIDATE_PARAM(params.channels > 0,
                     "The number of channels must be positive.");
  SNN_VALIDATE_PARAM(params.channel_multiplier > 0,
                     "The channel multiplier must be positive.");
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

  auto conv_sizes = get_sizes<ConvType>(params);

  auto inp_buff = backend.get_buffer(input, conv_sizes.input_size);
  auto fil_buff = backend.get_buffer(filter, conv_sizes.filter_size);
  auto out_buff = backend.get_buffer(output, conv_sizes.output_size);

  auto const inp_offset = backend.get_offset(input);
  auto const fil_offset = backend.get_offset(filter);
  auto const out_offset = backend.get_offset(output);

  ReadAccessor<T const> inp_access{inp_buff,
                                   cl::sycl::range<1>{conv_sizes.input_size},
                                   cl::sycl::id<1>{inp_offset}};
  ReadAccessor<T const> fil_access{fil_buff,
                                   cl::sycl::range<1>{conv_sizes.filter_size},
                                   cl::sycl::id<1>{fil_offset}};
  WriteAccessor<T> out_access{out_buff,
                              cl::sycl::range<1>{conv_sizes.output_size},
                              cl::sycl::id<1>{out_offset}};

  cl::sycl::queue queue = backend.get_queue();

  return internal::launch<ConvType>(inp_access, fil_access, out_access, params,
                                    queue);
}

}  // namespace depthwise_conv2d
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_DEPTHWISE_CONV2D_LAUNCH_H_
