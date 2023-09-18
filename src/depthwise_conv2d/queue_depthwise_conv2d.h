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
#ifndef PORTDNN_SRC_DEPTHWISE_CONV2D_QUEUE_DEPTHWISE_CONV2D_H_
#define PORTDNN_SRC_DEPTHWISE_CONV2D_QUEUE_DEPTHWISE_CONV2D_H_

#include "portdnn/mem_object.h"
#include "portdnn/status.h"

#include "portdnn/depthwise_conv2d/params.h"

#include <CL/sycl.hpp>

namespace sycldnn {
namespace depthwise_conv2d {
namespace internal {

template <typename ConvType, int VectorWidth, typename T, typename Index,
          template <typename> class MemObj>
SNNStatus queue_kernel(MemObj<T const>& input, MemObj<T const>& filter,
                       MemObj<T>& output,
                       DepthwiseConv2DParams const& kernel_params,
                       Index output_size, cl::sycl::queue& queue,
                       const std::vector<cl::sycl::event>& events);

template <int VectorWidth, typename T, typename Index,
          template <typename> class MemObj>
SNNStatus queue_kernel_fil_bk(MemObj<T const>& input, MemObj<T const>& filter,
                              MemObj<T>& output,
                              DepthwiseConv2DParams const& kernel_params,
                              Index output_size, cl::sycl::queue& queue,
                              const std::vector<cl::sycl::event>& events);

}  // namespace internal
}  // namespace depthwise_conv2d
}  // namespace sycldnn

#endif  // PORTDNN_SRC_DEPTHWISE_CONV2D_QUEUE_DEPTHWISE_CONV2D_H_
