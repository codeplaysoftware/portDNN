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
#ifndef SYCLDNN_SRC_CONV2D_DIRECT_LAUNCHER_H_
#define SYCLDNN_SRC_CONV2D_DIRECT_LAUNCHER_H_

#include "sycldnn/helpers/minmax.h"
#include "sycldnn/helpers/ratio.h"

#include "src/conv2d/direct/kernels.h"
#include "src/conv2d/direct/queue_direct_kernel.h"

namespace sycldnn {
namespace conv2d {
namespace internal {

template <typename ConvType, int VectorWidth, typename Index>
Index calculate_required_threads(Index output_size) {
  if (std::is_same<ConvType, conv_type::InputBackprop>::value) {
    return output_size;
  } else {
    return output_size / VectorWidth;
  }
}

template <typename T, typename Index, typename ConvType, bool UseFastDiv,
          int Window, int Stride, int VectorWidth>
SNNStatus queue_direct_kernel(ReadAccessor<T const> input,
                              ReadAccessor<T const> filter,
                              WriteAccessor<T> output,
                              Conv2DParams const& kernel_params,
                              Index output_size, cl::sycl::queue& queue) {
  using Functor = direct::DirectConv2D<T, Index, ConvType, UseFastDiv, Window,
                                       Stride, VectorWidth>;
  cl::sycl::device device = queue.get_device();
  Index const workgroup_size =
      device.get_info<cl::sycl::info::device::max_work_group_size>();
  Index const required_threads =
      calculate_required_threads<ConvType, VectorWidth>(output_size);
  size_t const n_threads =
      helpers::round_up_to_nearest_multiple(required_threads, workgroup_size);

  auto event = queue.submit([&](cl::sycl::handler& cgh) {
    cgh.require(input);
    cgh.require(filter);
    cgh.require(output);

    Index input_offset = input.get_offset().get(0);
    Index filter_offset = filter.get_offset().get(0);
    Index output_offset = output.get_offset().get(0);
    Functor conv(kernel_params, input, input_offset, filter, filter_offset,
                 output, output_offset);

    cgh.parallel_for(cl::sycl::range<1>{n_threads}, conv);
  });
  return {event, StatusCode::OK};
}

}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn

#endif  // SYCLDNN_SRC_CONV2D_DIRECT_LAUNCHER_H_
