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
#ifndef PORTDNN_SRC_DEPTHWISE_CONV2D_QUEUE_DEPTHWISE_CONV2D_IMPL_H_
#define PORTDNN_SRC_DEPTHWISE_CONV2D_QUEUE_DEPTHWISE_CONV2D_IMPL_H_

#include "portdnn/accessor_types.h"

#include "portdnn/depthwise_conv2d/params.h"

#include "portdnn/helpers/minmax.h"
#include "portdnn/helpers/ratio.h"

#include "src/depthwise_conv2d/kernels.h"
#include "src/depthwise_conv2d/queue_depthwise_conv2d.h"

#include <CL/sycl.hpp>

#include <cmath>

namespace sycldnn {
namespace depthwise_conv2d {
namespace internal {

namespace {

template <typename Index>
inline Index pow2_less_than(Index const val) {
  return std::exp2(static_cast<Index>(std::log2(val)));
}

}  // namespace

template <typename ConvType, int VectorWidth, typename T, typename Index,
          template <typename> class MemObj>
SNNStatus queue_kernel(MemObj<T const>& input_mem, MemObj<T const>& filter_mem,
                       MemObj<T>& output_mem,
                       DepthwiseConv2DParams const& kernel_params,
                       Index output_size, cl::sycl::queue& queue,
                       const std::vector<cl::sycl::event>& events) {
  using Functor = DepthwiseConv2D<T, Index, ConvType, VectorWidth,
                                  is_usm_obj_v<MemObj<T>, T>>;

  cl::sycl::device device = queue.get_device();
  size_t const workgroup_size =
      device.get_info<cl::sycl::info::device::max_work_group_size>();
  size_t const n_threads = helpers::round_up_to_nearest_multiple(
      static_cast<size_t>(output_size / VectorWidth), workgroup_size);

  auto event = queue.submit([&](cl::sycl::handler& cgh) {
    cgh.depends_on(events);
    auto input = input_mem.read_mem(cgh);
    auto filter = filter_mem.read_mem(cgh);
    auto output = output_mem.write_mem(cgh);
    Functor conv(output_size, kernel_params, input, filter, output);

    cgh.parallel_for(cl::sycl::range<1>{n_threads}, conv);
  });
  return {event, StatusCode::OK};
}

template <int VectorWidth, typename T, typename Index,
          template <typename> class MemObj>
SNNStatus queue_kernel_fil_bk(MemObj<T const>& input_mem,
                              MemObj<T const>& filter_mem,
                              MemObj<T>& output_mem,
                              DepthwiseConv2DParams const& kernel_params,
                              Index output_size, cl::sycl::queue& queue,
                              const std::vector<cl::sycl::event>& events) {
  using ConvType = conv2d::conv_type::FilterBackprop;
  using Functor = DepthwiseConv2D<T, Index, ConvType, VectorWidth,
                                  is_usm_obj_v<MemObj<T>, T>>;

  cl::sycl::device device = queue.get_device();
  size_t const max_wg_size =
      device.get_info<cl::sycl::info::device::max_work_group_size>();
  size_t const pow2_max_wg_size = pow2_less_than(max_wg_size);

  size_t pow2_batch = pow2_less_than(kernel_params.batch);
  size_t pow2_out_cols = pow2_less_than(kernel_params.out_cols);

  bool div_batch = pow2_batch > 1;
  while (pow2_batch * pow2_out_cols > pow2_max_wg_size) {
    if (div_batch) {
      pow2_batch /= 2;
      div_batch = pow2_out_cols < 2;
    } else {
      pow2_out_cols /= 2;
      div_batch = pow2_batch > 2;
    }
  }
  size_t const workgroup_size = pow2_batch * pow2_out_cols;
  size_t const workspace_size = pow2_batch * pow2_out_cols * VectorWidth;
  size_t const n_outputs = output_size / VectorWidth;

  auto event = queue.submit([&](cl::sycl::handler& cgh) {
    cgh.depends_on(events);
    auto input = input_mem.read_mem(cgh);
    auto filter = filter_mem.read_mem(cgh);
    auto output = output_mem.write_mem(cgh);

    LocalAccessor<T> local_access{cl::sycl::range<1>{workspace_size}, cgh};

    Functor conv(output_size, pow2_batch, pow2_out_cols, kernel_params, input,
                 filter, local_access, output);

    cgh.parallel_for(
        cl::sycl::nd_range<2>{cl::sycl::range<2>{workgroup_size, n_outputs},
                              cl::sycl::range<2>{workgroup_size, 1}},
        conv);
  });
  return {event, StatusCode::OK};
}

}  // namespace internal
}  // namespace depthwise_conv2d
}  // namespace sycldnn

#endif  // PORTDNN_SRC_DEPTHWISE_CONV2D_QUEUE_DEPTHWISE_CONV2D_IMPL_H_
