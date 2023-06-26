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
#ifndef SYCLDNN_SRC_CONV2D_TILED_QUEUE_TILED_KERNEL_H_
#define SYCLDNN_SRC_CONV2D_TILED_QUEUE_TILED_KERNEL_H_

#include "sycldnn/mem_object.h"
#include "sycldnn/status.h"

#include "sycldnn/conv2d/params.h"

#include "src/conv2d/tiled/tile_info.h"

#include <CL/sycl.hpp>

namespace sycldnn {
namespace conv2d {
namespace internal {

template <typename T, typename Index, typename ConvType, int TileRows,
          int TileCols, int ChannelVectorWidth, int FeatureVectorWidth,
          bool UseFastDiv, int WindowRows, int WindowCols, int Stride,
          typename Layout, template <typename> class MemObj>
SNNStatus queue_tiled_kernel(MemObj<T const>& input, MemObj<T const>& filter,
                             MemObj<T>& output,
                             Conv2DParams const& kernel_params,
                             tiled::TileInfo const& tile_info,
                             cl::sycl::queue& queue,
                             const std::vector<cl::sycl::event>& events);

}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn

#endif  // SYCLDNN_SRC_CONV2D_TILED_QUEUE_TILED_KERNEL_H_
