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
#ifndef PORTDNN_INCLUDE_DEPTHWISE_CONV2D_LAUNCH_H_
#define PORTDNN_INCLUDE_DEPTHWISE_CONV2D_LAUNCH_H_

/**
 * \file
 * Implements the \ref sycldnn::depthwise_conv2d::launch() function, which
 * asynchronously dispatches the SYCL kernels required to perform a 2D
 * convolution.
 */
#include "portdnn/backend/backend_helpers.h"
#include "portdnn/mem_object.h"
#include "portdnn/status.h"

#include "portdnn/depthwise_conv2d/params.h"
#include "portdnn/depthwise_conv2d/sizes.h"

#include "portdnn/helpers/macros.h"

#include "portdnn/internal/depthwise_conv2d/launch.h"

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
template <typename T, typename ConvType, typename Backend,
          typename = typename std::enable_if<
              sycldnn::backend::is_buffer_backend_v<Backend>>::type>
SNNStatus launch(typename Backend::template pointer_type<T const> input,
                 typename Backend::template pointer_type<T const> filter,
                 typename Backend::template pointer_type<T> output,
                 DepthwiseConv2DParams const& params, Backend& backend) {
  return internal::sublaunch<T, ConvType>(input, filter, output, params,
                                          backend, {});
}

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
 * \param events Events which should be completed before the operation.
 * \return Returns an SNNStatus containing the SYCL event tied to the kernel
 * launches and a StatusCode enum showing if the launch was OK or whether it
 * encountered some problem.
 */
template <typename T, typename ConvType, typename Backend,
          typename = typename std::enable_if<
              sycldnn::backend::is_usm_backend_v<Backend>>::type>
SNNStatus launch(typename Backend::template pointer_type<T const> input,
                 typename Backend::template pointer_type<T const> filter,
                 typename Backend::template pointer_type<T> output,
                 DepthwiseConv2DParams const& params, Backend& backend,
                 const std::vector<cl::sycl::event>& events = {}) {
  return internal::sublaunch<T, ConvType>(input, filter, output, params,
                                          backend, events);
}
}  // namespace depthwise_conv2d
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_DEPTHWISE_CONV2D_LAUNCH_H_
