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
#include "sycldnn/internal/conv2d/winograd/launch_output_transform.h"

#include "sycldnn/conv2d/conv_type.h"

#include "src/conv2d/winograd/queue_output_transform.h"

namespace sycldnn {
namespace conv2d {
namespace internal {
namespace winograd {

template <typename T, typename ConvType, int M, int N, int R, int S,
          bool Accumulate>
SNNStatus launch_output_transform(ReadAccessor<T const> intermediate,
                                  WriteAccessor<T> output,
                                  Conv2DParams const& params,
                                  TileInfo const& tile_info,
                                  cl::sycl::queue& queue) {
  return queue_output_transform<T, int, ConvType, M, N, R, S, Accumulate>(
      intermediate, output, params, tile_info, queue);
}

#define INSTANTIATE_LAUNCHER(DTYPE, CTYPE, M, N, R, S, ACC)                  \
  template SNNStatus launch_output_transform<DTYPE, CTYPE, M, N, R, S, ACC>( \
      ReadAccessor<DTYPE const> intermediate, WriteAccessor<DTYPE> output,   \
      Conv2DParams const& params, TileInfo const& tile_info,                 \
      cl::sycl::queue& queue);

#define INSTANTIATE_FOR_TYPE(DTYPE)                                         \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::Forward, 4, 4, 3, 3, false)        \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::Forward, 3, 3, 3, 3, false)        \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::Forward, 2, 2, 3, 3, false)        \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::Forward, 1, 2, 1, 3, false)        \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::Forward, 2, 1, 3, 1, false)        \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::InputBackprop, 4, 4, 3, 3, false)  \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::InputBackprop, 3, 3, 3, 3, false)  \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::InputBackprop, 2, 2, 3, 3, false)  \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::InputBackprop, 1, 2, 1, 3, false)  \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::InputBackprop, 2, 1, 3, 1, false)  \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::FilterBackprop, 3, 3, 3, 3, true)  \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::FilterBackprop, 3, 3, 3, 3, false) \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::FilterBackprop, 3, 3, 2, 2, true)  \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::FilterBackprop, 3, 3, 2, 2, false) \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::FilterBackprop, 3, 1, 2, 1, true)  \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::FilterBackprop, 3, 1, 2, 1, false) \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::FilterBackprop, 1, 3, 1, 2, true)  \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::FilterBackprop, 1, 3, 1, 2, false)

INSTANTIATE_FOR_TYPE(float)

#ifdef SNN_USE_DOUBLE
INSTANTIATE_FOR_TYPE(double)
#endif  // SNN_USE_DOUBLE

#ifdef SNN_USE_HALF
INSTANTIATE_FOR_TYPE(cl::sycl::half)
#endif  // SNN_USE_HALF

#undef INSTANTIATE_FOR_TYPE
#undef INSTANTIATE_LAUNCHER

}  // namespace winograd
}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn
