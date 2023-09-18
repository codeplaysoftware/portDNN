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
#ifndef PORTDNN_SRC_CONV2D_IM2COL_QUEUE_FILTER_TRANSFORM_IMPL_H_
#define PORTDNN_SRC_CONV2D_IM2COL_QUEUE_FILTER_TRANSFORM_IMPL_H_

#include "portdnn/mem_object.h"
#include "portdnn/status.h"

#include "portdnn/conv2d/params.h"

#include "portdnn/helpers/ratio.h"

#include "src/conv2d/im2col/kernels/extract_filter_tiles.h"
#include "src/conv2d/im2col/queue_filter_transform.h"

#include <CL/sycl.hpp>

namespace sycldnn {
namespace conv2d {
namespace internal {
namespace im2col {

template <typename T, typename Index, template <typename> class MemObj>
SNNStatus queue_filter_transform(MemObj<T const>& input_mem,
                                 MemObj<T>& output_mem,
                                 Conv2DParams const& params, Index thread_size,
                                 cl::sycl::queue& queue,
                                 const std::vector<cl::sycl::event>& events) {
  constexpr bool is_usm = is_usm_obj_v<MemObj<T>, T>;

  using Functor = ExtractFilterTiles<T, Index, is_usm>;
  cl::sycl::device device = queue.get_device();
  Index const workgroup_size =
      device.get_info<cl::sycl::info::device::max_work_group_size>();
  size_t const n_threads =
      helpers::round_up_to_nearest_multiple(thread_size, workgroup_size);

  auto event = queue.submit([&](cl::sycl::handler& cgh) {
    cgh.depends_on(events);
    auto input = input_mem.read_mem(cgh);
    auto output = output_mem.write_mem(cgh);
    Functor conv{params, input, output};

    cgh.parallel_for(cl::sycl::range<1>{n_threads}, conv);
  });

  return SNNStatus{event, StatusCode::OK};
}

}  // namespace im2col
}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn

#endif  // PORTDNN_SRC_CONV2D_IM2COL_QUEUE_FILTER_TRANSFORM_IMPL_H_
