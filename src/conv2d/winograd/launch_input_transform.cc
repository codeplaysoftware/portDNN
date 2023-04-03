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
#include "sycldnn/internal/conv2d/winograd/launch_input_transform.h"

#include "sycldnn/conv2d/conv_type.h"

#include "src/conv2d/winograd/queue_input_transform.h"

#include "sycldnn/export.h"

namespace sycldnn {
namespace conv2d {
namespace internal {
namespace winograd {

inline bool can_use_vector(Conv2DParams const& params, int vector) {
  return params.channels % vector == 0;
}

template <typename T, typename ConvType, int M, int N, int R, int S,
          template <typename> class MemObj>
SNNStatus launch_input_transform(MemObj<T const>& input, MemObj<T>& transform,
                                 Conv2DParams const& params,
                                 TileInfo const& tile_info,
                                 cl::sycl::queue& queue,
                                 const std::vector<cl::sycl::event>& events) {
  // The larger input tiles when M is 4 use too many registers if vectorisation
  // is used, which causes performance of the transform kernel to be around half
  // what it is without vectorisation. As we don't currently have a better way
  // of choosing vector sizes for different tile sizes, we just skip
  // vectorisation in this case.
  // TODO(jwlawson): Provide better vector size customisation
  if (M != 4 && can_use_vector(params, 4)) {
    return queue_input_transform<T, int, ConvType, M, N, R, S, 4>(
        input, transform, params, tile_info, queue, events);
  } else if (M != 4 && can_use_vector(params, 2)) {
    return queue_input_transform<T, int, ConvType, M, N, R, S, 2>(
        input, transform, params, tile_info, queue, events);
  } else
    return queue_input_transform<T, int, ConvType, M, N, R, S, 1>(
        input, transform, params, tile_info, queue, events);
}

#define INSTANTIATE_LAUNCHER(DTYPE, CTYPE, M, N, R, S, MEM_OBJ) \
  template SNN_EXPORT SNNStatus                                 \
  launch_input_transform<DTYPE, CTYPE, M, N, R, S>(             \
      MEM_OBJ<DTYPE const> & input, MEM_OBJ<DTYPE> & transform, \
      Conv2DParams const& params, TileInfo const& tile_info,    \
      cl::sycl::queue& queue, const std::vector<cl::sycl::event>& events);

#define INSTANTIATE_FOR_TYPE(DTYPE, MEM_OBJ)                                  \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::Forward, 4, 4, 3, 3, MEM_OBJ)        \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::Forward, 3, 3, 3, 3, MEM_OBJ)        \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::Forward, 2, 2, 3, 3, MEM_OBJ)        \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::Forward, 2, 1, 3, 1, MEM_OBJ)        \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::Forward, 1, 2, 1, 3, MEM_OBJ)        \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::InputBackprop, 4, 4, 3, 3, MEM_OBJ)  \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::InputBackprop, 3, 3, 3, 3, MEM_OBJ)  \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::InputBackprop, 2, 2, 3, 3, MEM_OBJ)  \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::InputBackprop, 2, 1, 3, 1, MEM_OBJ)  \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::InputBackprop, 1, 2, 1, 3, MEM_OBJ)  \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::FilterBackprop, 3, 3, 3, 3, MEM_OBJ) \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::FilterBackprop, 3, 3, 2, 2, MEM_OBJ) \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::FilterBackprop, 3, 1, 2, 1, MEM_OBJ) \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::FilterBackprop, 1, 3, 1, 2, MEM_OBJ)

#ifdef SNN_ENABLE_USM
INSTANTIATE_FOR_TYPE(float, USMMemObject)
#endif  // SNN_ENABLE_USM
INSTANTIATE_FOR_TYPE(float, BufferMemObject)

#ifdef SNN_USE_DOUBLE
#ifdef SNN_ENABLE_USM
INSTANTIATE_FOR_TYPE(double, USMMemObject)
#endif  // SNN_ENABLE_USM
INSTANTIATE_FOR_TYPE(double, BufferMemObject)
#endif  // SNN_USE_DOUBLE

#ifdef SNN_USE_HALF
#ifdef SNN_ENABLE_USM
INSTANTIATE_FOR_TYPE(cl::sycl::half, USMMemObject)
#endif  // SNN_ENABLE_USM
INSTANTIATE_FOR_TYPE(cl::sycl::half, BufferMemObject)
#endif  // SNN_USE_HALF

#undef INSTANTIATE_FOR_TYPE
#undef INSTANTIATE_LAUNCHER

}  // namespace winograd
}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn
