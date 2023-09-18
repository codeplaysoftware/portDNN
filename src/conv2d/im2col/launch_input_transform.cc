
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
#include "portdnn/mem_object.h"
#include "portdnn/status.h"

#include "portdnn/conv2d/conv_type.h"
#include "portdnn/conv2d/params.h"

#include "portdnn/internal/conv2d/im2col/launch_input_transform.h"

#include "src/conv2d/im2col/queue_input_transform.h"
#include "src/conv2d/im2col/queue_zero_out_transform.h"

#include <stddef.h>
#include <cstdint>
#include <limits>

#include <CL/sycl.hpp>

#include "portdnn/export.h"

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
  if (params.group_format == sycldnn::BatchFormat::STRIDED) {
    return (params.channels / params.groups) % vector_width == 0;
  } else {
    return params.channels % vector_width == 0;
  }
}
template <>
bool can_use_vector<conv_type::InputBackprop>(Conv2DParams const& params,
                                              int vector_width) {
  return (params.features / params.groups) % vector_width == 0;
}
template <>
bool can_use_vector<conv_type::FilterBackprop>(Conv2DParams const& /*params*/,
                                               int /*vector_width*/) {
  return false;
}

template <typename T, typename Index, int VectorWidth, typename ConvType,
          template <typename> class MemObj>
SNNStatus launch_with_index(MemObj<T const>& input, MemObj<T>& output,
                            Conv2DParams const& params, int n_tiles,
                            int tile_size, cl::sycl::queue& queue,
                            const std::vector<cl::sycl::event>& events) {
  auto status = queue_zero_out_transform<T, VectorWidth>(
      output, n_tiles, params.groups * tile_size, queue, events);
  if (status.status != StatusCode::OK) {
    return status;
  } else {
    std::vector<cl::sycl::event> dependencies{status.event};
    return queue_input_transform<T, Index, VectorWidth, ConvType>(
        input, output, params, tile_size, queue, dependencies);
  }
}

template <typename T, int VectorWidth, typename ConvType,
          template <typename> class MemObj>
SNNStatus launch_with_vector(MemObj<T const>& input, MemObj<T>& output,
                             Conv2DParams const& params, int n_tiles,
                             int tile_size, cl::sycl::queue& queue,
                             const std::vector<cl::sycl::event>& events) {
  size_t thread_size = get_thread_size<ConvType>(params, VectorWidth);
  if (thread_size > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
#ifdef SNN_USE_INT64
    return launch_with_index<T, int64_t, VectorWidth, ConvType>(
        input, output, params, n_tiles, tile_size, queue, events);
#else
    return StatusCode::IndexExceeded;
#endif  // SNN_USE_INT64
  } else {
    return launch_with_index<T, int32_t, VectorWidth, ConvType>(
        input, output, params, n_tiles, tile_size, queue, events);
  }
}
}  // namespace

template <typename T, typename ConvType, template <typename> class MemObj>
SNNStatus launch_input_transform(MemObj<T const>& input, MemObj<T>& output,
                                 Conv2DParams const& params, int n_tiles,
                                 int tile_size, cl::sycl::queue& queue,
                                 const std::vector<cl::sycl::event>& events) {
  if (can_use_vector<ConvType>(params, 4)) {
    return launch_with_vector<T, 4, ConvType>(input, output, params, n_tiles,
                                              tile_size, queue, events);
  } else if (can_use_vector<ConvType>(params, 2)) {
    return launch_with_vector<T, 2, ConvType>(input, output, params, n_tiles,
                                              tile_size, queue, events);
  } else {
    return launch_with_vector<T, 1, ConvType>(input, output, params, n_tiles,
                                              tile_size, queue, events);
  }
}

#define INSTANTIATE_LAUNCHER(DTYPE, CTYPE, MEMOBJ)                            \
  template SNN_EXPORT SNNStatus launch_input_transform<DTYPE, CTYPE, MEMOBJ>( \
      MEMOBJ<DTYPE const> & input, MEMOBJ<DTYPE> & output,                    \
      Conv2DParams const& params, int n_tiles, int tile_size,                 \
      cl::sycl::queue& queue, const std::vector<cl::sycl::event>& events);

#define INSTANTIATE_FOR_TYPE(DTYPE, MEMOBJ)                     \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::Forward, MEMOBJ)       \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::InputBackprop, MEMOBJ) \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::FilterBackprop, MEMOBJ)

#ifdef SNN_ENABLE_USM
#define INSTANTIATE_FOR_MEMOBJ(DTYPE)       \
  INSTANTIATE_FOR_TYPE(DTYPE, USMMemObject) \
  INSTANTIATE_FOR_TYPE(DTYPE, BufferMemObject)
#else
#define INSTANTIATE_FOR_MEMOBJ(DTYPE) \
  INSTANTIATE_FOR_TYPE(DTYPE, BufferMemObject)
#endif
INSTANTIATE_FOR_MEMOBJ(float)

#ifdef SNN_USE_DOUBLE
INSTANTIATE_FOR_MEMOBJ(double)
#endif  // SNN_USE_DOUBLE

#ifdef SNN_USE_HALF
INSTANTIATE_FOR_MEMOBJ(cl::sycl::half)
#endif  // SNN_USE_HALF

#undef INSTANTIATE_FOR_TYPE
#undef INSTANTIATE_FOR_MEMOBJ
#undef INSTANTIATE_LAUNCHER

}  // namespace im2col
}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn
