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
#ifndef SYCLDNN_INCLUDE_INTERNAL_CONV2D_TILED_H_
#define SYCLDNN_INCLUDE_INTERNAL_CONV2D_TILED_H_

#include "sycldnn/accessor_types.h"
#include "sycldnn/conv2d/params.h"
#include "sycldnn/status.h"

namespace sycldnn {
namespace conv2d {
namespace internal {
/**
 * The internal direct convolution launcher.
 *
 * Implemented in the compiled SYCL DNN library.
 */
template <typename T, typename ConvType>
SNNStatus launch_tiled(ReadAccessor<T const> input,
                       ReadAccessor<T const> filter, WriteAccessor<T> output,
                       Conv2DParams const& params, cl::sycl::queue& queue);
}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn
#endif  // SYCLDNN_INCLUDE_INTERNAL_CONV2D_TILED_H_
