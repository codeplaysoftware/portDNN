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
#ifndef PORTDNN_SRC_CONV2D_WINOGRAD_QUEUE_OUTPUT_TRANSFORM_H_
#define PORTDNN_SRC_CONV2D_WINOGRAD_QUEUE_OUTPUT_TRANSFORM_H_

#include "portdnn/mem_object.h"
#include "portdnn/status.h"

#include "portdnn/conv2d/params.h"
#include "portdnn/internal/conv2d/winograd/tile_info.h"

#include <CL/sycl.hpp>

namespace sycldnn {
namespace conv2d {
namespace internal {
namespace winograd {

template <typename T, typename Index, typename ConvType, int M, int N, int R,
          int S, bool Accumulate, template <typename> class MemObj>
SNNStatus queue_output_transform(MemObj<T const>& intermediate,
                                 MemObj<T>& output,
                                 Conv2DParams const& kernel_params,
                                 TileInfo const& tile_info,
                                 cl::sycl::queue& queue,
                                 const std::vector<cl::sycl::event>& events);

}  // namespace winograd
}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn

#endif  // PORTDNN_SRC_CONV2D_WINOGRAD_QUEUE_OUTPUT_TRANSFORM_H_
