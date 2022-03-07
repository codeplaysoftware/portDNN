
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
#include "sycldnn/mem_object.h"
#include "sycldnn/status.h"

#include "sycldnn/conv2d/conv_type.h"
#include "sycldnn/conv2d/params.h"

#include "sycldnn/internal/conv2d/im2col/launch_input_transform.h"

#include "src/conv2d/im2col/queue_input_transform.h"
#include "src/conv2d/im2col/queue_zero_out_transform.h"

#include <stddef.h>
#include <cstdint>
#include <limits>

#include <CL/sycl.hpp>

#include "sycldnn/export.h"

namespace sycldnn {
namespace conv2d {
namespace internal {
namespace im2col {
namespace {

/** Get the required number of threads for the input transform. */
template <typename ConvType>
size_t get_thread_size(Conv2DParams const& params, int vector_width) {
  return params.batch * params.in_rows * params.in_cols * params.channels /
         vector_width;
}
template <>
size_t get_thread_size<conv_type::InputBackprop>(Conv2DParams const& params,
                                                 int vector_width) {
  return params.batch * params.out_rows * params.out_cols * params.features /
         vector_width;
}

/** Check whether a certain vector size can be used for the given parameters. */
template <typename ConvType>
bool can_use_vector(Conv2DParams const& params, int vector_width) {
  return params.channels % vector_width == 0;
}
template <>
bool can_use_vector<conv_type::InputBackprop>(Conv2DParams const& params,
                                              int vector_width) {
  return params.features % vector_width == 0;
}
template <>
bool can_use_vector<conv_type::FilterBackprop>(Conv2DParams const& /*params*/,
                                               int /*vector_width*/) {
  return false;
}

template <typename T, typename Index, int VectorWidth, typename ConvType>
SNNStatus launch_with_index(BaseMemObject<T const>& input,
                            BaseMemObject<T>& output,
                            Conv2DParams const& params, int n_tiles,
                            int tile_size, cl::sycl::queue& queue) {
  auto status = queue_zero_out_transform<T, VectorWidth>(output, n_tiles,
                                                         tile_size, queue);
  if (status.status != StatusCode::OK) {
    return status;
  } else {
    return queue_input_transform<T, Index, VectorWidth, ConvType>(
        input, output, params, tile_size, queue);
  }
}

template <typename T, int VectorWidth, typename ConvType>
SNNStatus launch_with_vector(BaseMemObject<T const>& input,
                             BaseMemObject<T>& output,
                             Conv2DParams const& params, int n_tiles,
                             int tile_size, cl::sycl::queue& queue) {
  size_t thread_size = get_thread_size<ConvType>(params, VectorWidth);
  if (thread_size > std::numeric_limits<int32_t>::max()) {
#ifdef SNN_USE_INT64
    return launch_with_index<T, int64_t, VectorWidth, ConvType>(
        input, output, params, n_tiles, tile_size, queue);
#else
    return StatusCode::IndexExceeded;
#endif  // SNN_USE_INT64
  } else {
    return launch_with_index<T, int32_t, VectorWidth, ConvType>(
        input, output, params, n_tiles, tile_size, queue);
  }
}
}  // namespace

template <typename T, typename ConvType>
SNNStatus launch_input_transform(BaseMemObject<T const>& input,
                                 BaseMemObject<T>& output,
                                 Conv2DParams const& params, int n_tiles,
                                 int tile_size, cl::sycl::queue& queue) {
  if (can_use_vector<ConvType>(params, 4)) {
    return launch_with_vector<T, 4, ConvType>(input, output, params, n_tiles,
                                              tile_size, queue);
  } else if (can_use_vector<ConvType>(params, 2)) {
    return launch_with_vector<T, 2, ConvType>(input, output, params, n_tiles,
                                              tile_size, queue);
  } else {
    return launch_with_vector<T, 1, ConvType>(input, output, params, n_tiles,
                                              tile_size, queue);
  }
}

#define INSTANTIATE_LAUNCHER(DTYPE, CTYPE)                               \
  template SNN_EXPORT SNNStatus launch_input_transform<DTYPE, CTYPE>(    \
      BaseMemObject<DTYPE const> & input, BaseMemObject<DTYPE> & output, \
      Conv2DParams const& params, int n_tiles, int tile_size,            \
      cl::sycl::queue& queue);

#define INSTANTIATE_FOR_TYPE(DTYPE)                     \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::Forward)       \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::InputBackprop) \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::FilterBackprop)

INSTANTIATE_FOR_TYPE(float)

#ifdef SNN_USE_DOUBLE
INSTANTIATE_FOR_TYPE(double)
#endif  // SNN_USE_DOUBLE

#ifdef SNN_USE_HALF
INSTANTIATE_FOR_TYPE(cl::sycl::half)
#endif  // SNN_USE_HALF

#undef INSTANTIATE_FOR_TYPE
#undef INSTANTIATE_LAUNCHER

}  // namespace im2col
}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn
