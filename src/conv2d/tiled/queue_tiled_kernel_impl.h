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
#ifndef SYCLDNN_SRC_CONV2D_TILED_QUEUE_TILED_KERNEL_IMPL_H_
#define SYCLDNN_SRC_CONV2D_TILED_QUEUE_TILED_KERNEL_IMPL_H_

#include "sycldnn/mem_object.h"
#include "sycldnn/status.h"

#include "sycldnn/helpers/ratio.h"

#include "sycldnn/conv2d/params.h"

#include "src/conv2d/tiled/kernels.h"
#include "src/conv2d/tiled/tile_info.h"

#include <CL/sycl.hpp>

namespace sycldnn {
namespace conv2d {
namespace internal {

namespace {

size_t round_up_to_size(int val, cl::sycl::device const& device) {
  int const workgroup_size =
      device.get_info<cl::sycl::info::device::max_work_group_size>();

  int rounded = helpers::round_up_to_nearest_multiple(val, workgroup_size);
  return static_cast<size_t>(rounded);
}

cl::sycl::range<1> get_thread_range(Conv2DParams const& params,
                                    tiled::TileInfo const& tile_info,
                                    cl::sycl::queue const& queue) {
  cl::sycl::device device = queue.get_device();
  auto size = round_up_to_size(params.batch * tile_info.n_rows *
                                   tile_info.n_cols * tile_info.output_vectors,
                               device);
  return {size};
}

}  // namespace

template <typename T, typename Index, typename ConvType, int TileRows,
          int TileCols, int ChannelVectorWidth, int FeatureVectorWidth,
          bool UseFastDiv, int WindowRows, int WindowCols, int Stride>
SNNStatus queue_tiled_kernel(BaseMemObject<T const>& in_mem,
                             BaseMemObject<T const>& fil_mem,
                             BaseMemObject<T>& out_mem,
                             Conv2DParams const& kernel_params,
                             tiled::TileInfo const& tile_info,
                             cl::sycl::queue& queue) {
  using Functor =
      tiled::TiledConv2D<T, Index, ConvType, TileRows, TileCols,
                         ChannelVectorWidth, FeatureVectorWidth, UseFastDiv,
                         WindowRows, WindowCols, Stride>;

  auto event = queue.submit([&](cl::sycl::handler& cgh) {
    auto input = in_mem.read_accessor(cgh);
    auto filter = fil_mem.read_accessor(cgh);
    auto output = out_mem.write_accessor(cgh);

    Functor conv{input, filter, output, kernel_params, tile_info};
    auto threads = get_thread_range(kernel_params, tile_info, queue);

    cgh.parallel_for(threads, conv);
  });
  SNNStatus ok_status{event, StatusCode::OK};
  return ok_status;
}

}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn

#endif  // SYCLDNN_SRC_CONV2D_TILED_QUEUE_TILED_KERNEL_IMPL_H_
