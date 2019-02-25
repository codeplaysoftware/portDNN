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
#include "sycldnn/internal/conv2d/winograd/launch_input_transform.h"

#include "sycldnn/conv2d/conv_type.h"

#include "src/conv2d/winograd/queue_input_transform.h"

namespace sycldnn {
namespace conv2d {
namespace internal {
namespace winograd {

inline bool can_use_vector(Conv2DParams const& params, int vector) {
  return params.channels % vector == 0;
}

template <typename T, typename ConvType, int M, int N, int R, int S>
SNNStatus launch_input_transform(ReadAccessor<T const> input,
                                 WriteAccessor<T> transform,
                                 Conv2DParams const& params,
                                 TileInfo const& tile_info,
                                 cl::sycl::queue& queue) {
  if (can_use_vector(params, 4)) {
    return queue_input_transform<T, int, ConvType, M, N, R, S, 4>(
        input, transform, params, tile_info, queue);
  } else if (can_use_vector(params, 2)) {
    return queue_input_transform<T, int, ConvType, M, N, R, S, 2>(
        input, transform, params, tile_info, queue);
  } else {
    return queue_input_transform<T, int, ConvType, M, N, R, S, 1>(
        input, transform, params, tile_info, queue);
  }
}

#define INSTANTIATE_LAUNCHER(DTYPE, CTYPE, M, N, R, S)                 \
  template SNNStatus launch_input_transform<DTYPE, CTYPE, M, N, R, S>( \
      ReadAccessor<DTYPE const> input, WriteAccessor<DTYPE> transform, \
      Conv2DParams const& params, TileInfo const& tile_info,           \
      cl::sycl::queue& queue);

#define INSTANTIATE_FOR_TYPE(DTYPE)                                  \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::Forward, 4, 4, 3, 3)        \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::Forward, 3, 3, 3, 3)        \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::Forward, 2, 2, 3, 3)        \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::Forward, 2, 1, 3, 1)        \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::Forward, 1, 2, 1, 3)        \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::InputBackprop, 4, 4, 3, 3)  \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::InputBackprop, 3, 3, 3, 3)  \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::InputBackprop, 2, 2, 3, 3)  \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::InputBackprop, 2, 1, 3, 1)  \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::InputBackprop, 1, 2, 1, 3)  \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::FilterBackprop, 3, 3, 3, 3) \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::FilterBackprop, 3, 3, 2, 2) \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::FilterBackprop, 3, 1, 2, 1) \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::FilterBackprop, 1, 3, 1, 2)

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
