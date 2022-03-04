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
#ifndef SYCLDNN_INCLUDE_INTERNAL_DEPTHWISE_CONV2D_LAUNCH_H_
#define SYCLDNN_INCLUDE_INTERNAL_DEPTHWISE_CONV2D_LAUNCH_H_

/**
 * \file
 * Implements the \ref sycldnn::depthwise_conv2d::launch() function, which
 * asynchronously dispatches the SYCL kernels required to perform a 2D
 * convolution.
 */
#include "sycldnn/mem_object.h"
#include "sycldnn/status.h"

#include "sycldnn/depthwise_conv2d/params.h"

#include <CL/sycl.hpp>

#include "sycldnn/export.h"

namespace sycldnn {
namespace depthwise_conv2d {
namespace internal {

/**
 * Launch a 2D depthwise convolution.
 *
 * Implemented in the compiled SYCL-DNN library.
 *
 * \param input  An accessor for the input tensor.
 * \param filter An accessor for the filter tensor.
 * \param output An accessor for the output tensor.
 * \param params The convolution parameters, which describe the tensor shapes
 *               and convolution strides.
 * \param queue  The SYCL queue to enqueue the kernels to.
 * \return Returns an SNNStatus containing the SYCL event tied to the kernel
 * launches and a StatusCode enum showing if the launch was OK or whether it
 * encountered some problem.
 */
template <typename ConvType, typename T>
SNN_EXPORT SNNStatus launch(BaseMemObject<T const>& input,
                            BaseMemObject<T const>& filter,
                            BaseMemObject<T>& output,
                            DepthwiseConv2DParams const& params,
                            cl::sycl::queue& queue);

}  // namespace internal
}  // namespace depthwise_conv2d
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_INTERNAL_DEPTHWISE_CONV2D_LAUNCH_H_
