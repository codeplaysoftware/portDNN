/*
 * Copyright 2019 Codeplay Software Ltd
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
#ifndef SYCLDNN_SRC_DEPTHWISE_CONV2D_QUEUE_DEPTHWISE_CONV2D_H_
#define SYCLDNN_SRC_DEPTHWISE_CONV2D_QUEUE_DEPTHWISE_CONV2D_H_

#include "sycldnn/mem_object.h"
#include "sycldnn/status.h"

#include "sycldnn/depthwise_conv2d/params.h"

#include <CL/sycl.hpp>

namespace sycldnn {
namespace depthwise_conv2d {
namespace internal {

template <typename ConvType, typename T, typename Index>
SNNStatus queue_kernel(BaseMemObject<T const>& input,
                       BaseMemObject<T const>& filter, BaseMemObject<T>& output,
                       DepthwiseConv2DParams const& kernel_params,
                       Index output_size, cl::sycl::queue& queue);

template <typename T, typename Index>
SNNStatus queue_kernel_fil_bk(BaseMemObject<T const>& input,
                              BaseMemObject<T const>& filter,
                              BaseMemObject<T>& output,
                              DepthwiseConv2DParams const& kernel_params,
                              Index output_size, cl::sycl::queue& queue);

}  // namespace internal
}  // namespace depthwise_conv2d
}  // namespace sycldnn

#endif  // SYCLDNN_SRC_DEPTHWISE_CONV2D_QUEUE_DEPTHWISE_CONV2D_H_
