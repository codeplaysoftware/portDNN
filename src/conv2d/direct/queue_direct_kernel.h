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
#ifndef SYCLDNN_SRC_CONV2D_DIRECT_LAUNCH_DIRECT_IMPL_H_
#define SYCLDNN_SRC_CONV2D_DIRECT_LAUNCH_DIRECT_IMPL_H_

#include "sycldnn/mem_object.h"
#include "sycldnn/status.h"

#include "sycldnn/conv2d/params.h"

namespace sycldnn {
namespace conv2d {
namespace internal {
/**
 * Queue a direct convolution kernel to the provided SYCL queue.
 */
template <typename T, typename Index, typename ConvType, bool UseFastDiv,
          int Window, int Stride, int VectorWidth>
SNNStatus queue_direct_kernel(BaseMemObject<T const>& input,
                              BaseMemObject<T const>& filter,
                              BaseMemObject<T>& output,
                              Conv2DParams const& kernel_params,
                              Index output_size, cl::sycl::queue& queue);
}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn
#endif  // SYCLDNN_SRC_CONV2D_DIRECT_LAUNCH_DIRECT_IMPL_H_
