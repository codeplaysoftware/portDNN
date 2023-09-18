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
#ifndef PORTDNN_SRC_CONV2D_DIRECT_KERNELS_H_
#define PORTDNN_SRC_CONV2D_DIRECT_KERNELS_H_

#include "src/helpers/math.h"
#include "src/helpers/tensor_index.h"
#include "src/helpers/vector_io.h"
#include "src/helpers/vector_type.h"
#include "src/helpers/window_index.h"

#include "portdnn/accessor_types.h"
#include "portdnn/conv2d/conv_type.h"
#include "portdnn/conv2d/params.h"
#include "portdnn/format_type.h"
#include "portdnn/helpers/macros.h"

namespace sycldnn {
namespace conv2d {
namespace internal {
namespace direct {

/**
 * SYCL kernel for direct convolution computation.
 */
template <typename T, typename Index, typename ConvType, bool UseFastDiv,
          int StaticWindow, int StaticStride, int VectorWidth, typename Layout,
          bool isUSM>
struct DirectConv2D;

}  // namespace direct
}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn

#endif  // PORTDNN_SRC_CONV2D_DIRECT_KERNELS_H_
