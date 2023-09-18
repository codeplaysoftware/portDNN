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
#ifndef PORTDNN_INCLUDE_CONV2D_IM2COL_H_
#define PORTDNN_INCLUDE_CONV2D_IM2COL_H_

#include "portdnn/conv2d/conv_type.h"
#include "portdnn/conv2d/params.h"
#include "portdnn/conv2d/sizes.h"

#include "portdnn/internal/conv2d/im2col.h"

namespace sycldnn {
namespace conv2d {
/**
 * Launch the 2D convolution using im2col.
 *
 * Will extract the SYCL buffers and SYCL queue from the backend and forward
 * these on to the precompiled kernels.
 *
 * Returns an SNNStatus containing the SYCL event tied to the kernel launch.
 */
template <typename T, typename ConvType, typename Backend>
inline SNNStatus launch_im2col(
    typename Backend::template pointer_type<T const> input,
    typename Backend::template pointer_type<T const> filter,
    typename Backend::template pointer_type<T> output,
    typename Backend::template pointer_type<T> workspace,
    Conv2DParams const& params, size_t workspace_size, Backend& backend,
    const std::vector<cl::sycl::event>& events) {
  return internal::launch_im2col<T, ConvType>(input, filter, output, workspace,
                                              params, workspace_size, backend,
                                              events);
}
}  // namespace conv2d
}  // namespace sycldnn
#endif  // PORTDNN_INCLUDE_CONV2D_IM2COL_H_
