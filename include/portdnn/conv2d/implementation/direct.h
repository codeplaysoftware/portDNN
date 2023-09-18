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
#ifndef PORTDNN_INCLUDE_CONV2D_DIRECT_H_
#define PORTDNN_INCLUDE_CONV2D_DIRECT_H_

#include "portdnn/conv2d/conv_type.h"
#include "portdnn/conv2d/params.h"
#include "portdnn/conv2d/sizes.h"

#include "portdnn/internal/conv2d/direct.h"

namespace sycldnn {
namespace conv2d {
/**
 * Launch the direct implementation of a 2D convolution.
 *
 * Will extract the SYCL buffers and SYCL queue from the backend and forward
 * these on to the precompiled kernels.
 *
 * Returns an SNNStatus containing the SYCL event tied to the kernel launch.
 */
template <typename T, typename ConvType, typename Backend>
inline SNNStatus launch_direct(
    typename Backend::template pointer_type<T const> input,
    typename Backend::template pointer_type<T const> filter,
    typename Backend::template pointer_type<T> output,
    Conv2DParams const& params, Backend& backend,
    const std::vector<cl::sycl::event>& events) {
  auto conv_sizes = get_sizes<ConvType>(params);

  auto inp_access = backend.get_mem_object(input, conv_sizes.input_size);
  auto fil_access = backend.get_mem_object(filter, conv_sizes.filter_size);
  auto out_access = backend.get_mem_object(output, conv_sizes.output_size);

  cl::sycl::queue queue = backend.get_queue();
  return internal::launch_direct<T, ConvType>(
      inp_access, fil_access, out_access, params, queue, events);
}
}  // namespace conv2d
}  // namespace sycldnn
#endif  // PORTDNN_INCLUDE_CONV2D_DIRECT_H_
