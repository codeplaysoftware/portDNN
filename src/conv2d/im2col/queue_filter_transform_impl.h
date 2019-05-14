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
#ifndef SYCLDNN_SRC_CONV2D_IM2COL_QUEUE_FILTER_TRANSFORM_IMPL_H_
#define SYCLDNN_SRC_CONV2D_IM2COL_QUEUE_FILTER_TRANSFORM_IMPL_H_

#include "sycldnn/mem_object.h"
#include "sycldnn/status.h"

#include "sycldnn/conv2d/params.h"

#include "sycldnn/helpers/ratio.h"

#include "src/conv2d/im2col/kernels/extract_filter_tiles.h"
#include "src/conv2d/im2col/queue_filter_transform.h"

#include <CL/sycl.hpp>

namespace sycldnn {
namespace conv2d {
namespace internal {
namespace im2col {

template <typename T, typename Index>
SNNStatus queue_filter_transform(BaseMemObject<T const>& input_mem,
                                 BaseMemObject<T>& output_mem,
                                 Conv2DParams const& params, Index thread_size,
                                 cl::sycl::queue& queue) {
  using Functor = ExtractFilterTiles<T, Index>;
  cl::sycl::device device = queue.get_device();
  Index const workgroup_size =
      device.get_info<cl::sycl::info::device::max_work_group_size>();
  size_t const n_threads =
      helpers::round_up_to_nearest_multiple(thread_size, workgroup_size);

  auto event = queue.submit([&](cl::sycl::handler& cgh) {
    auto input = input_mem.read_accessor(cgh);
    auto output = output_mem.write_accessor(cgh);
    auto in_offset = input.get_offset().get(0);
    auto out_offset = output.get_offset().get(0);
    Functor conv(in_offset, out_offset, params, input, output);

    cgh.parallel_for(cl::sycl::range<1>{n_threads}, conv);
  });

  return SNNStatus{event, StatusCode::OK};
}

}  // namespace im2col
}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn

#endif  // SYCLDNN_SRC_CONV2D_IM2COL_QUEUE_FILTER_TRANSFORM_IMPL_H_
