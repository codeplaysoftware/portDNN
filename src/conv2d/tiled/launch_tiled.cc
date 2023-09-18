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
#include "portdnn/internal/conv2d/tiled.h"

#include "portdnn/mem_object.h"
#include "portdnn/status.h"

#include "portdnn/conv2d/conv_type.h"
#include "portdnn/conv2d/params.h"

#include "portdnn/helpers/ratio.h"

#include "src/conv2d/tiled/kernel_params.h"
#include "src/conv2d/tiled/queue_tiled_kernel.h"
#include "src/conv2d/tiled/tile_info.h"

#include <stddef.h>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <CL/sycl.hpp>

#include "portdnn/export.h"

namespace sycldnn {
namespace conv2d {
namespace internal {
namespace {

template <typename ConvType>
inline bool can_use_fast_div(Conv2DParams const& params,
                             int channel_vector_width, int feature_vector_width,
                             int tile_rows, int tile_cols);
template <>
inline bool can_use_fast_div<conv_type::Forward>(Conv2DParams const& params,
                                                 int /*channel_vector_width*/,
                                                 int feature_vector_width,
                                                 int tile_rows, int tile_cols) {
  return params.features / feature_vector_width != 1 &&
         helpers::round_ratio_up_above_zero(params.out_rows, tile_rows) != 1 &&
         helpers::round_ratio_up_above_zero(params.out_cols, tile_cols) != 1;
}
template <>
inline bool can_use_fast_div<conv_type::InputBackprop>(
    Conv2DParams const& params, int channel_vector_width,
    int /*feature_vector_width*/, int tile_rows, int tile_cols) {
  return params.channels / channel_vector_width != 1 &&
         helpers::round_ratio_up_above_zero(params.in_rows, tile_rows) != 1 &&
         helpers::round_ratio_up_above_zero(params.in_cols, tile_cols) != 1;
}
template <typename ConvType>
inline bool can_use_sizes(Conv2DParams const& params, int channel_vector,
                          int feature_vector, int window, int stride);
template <>
inline bool can_use_sizes<conv_type::Forward>(Conv2DParams const& params,
                                              int const channel_vector,
                                              int const feature_vector,
                                              int const window,
                                              int const stride) {
  return (params.window_rows == window && params.window_cols == window &&
          params.stride_rows == stride && params.stride_cols == stride &&
          params.features % feature_vector == 0 &&
          params.channels % channel_vector == 0);
}
template <>
inline bool can_use_sizes<conv_type::InputBackprop>(Conv2DParams const& params,
                                                    int const channel_vector,
                                                    int const feature_vector,
                                                    int const window,
                                                    int const stride) {
  return (params.window_rows == window && params.window_cols == window &&
          params.stride_rows == stride && params.stride_cols == stride &&
          params.features % feature_vector == 0 &&
          params.channels % channel_vector == 0);
}

/**
 * Check whether fast divisions can be used for the convolution, and launch
 * whichever kernel is required.
 */
template <typename T, typename Index, typename ConvType, int TileRows,
          int TileCols, int ChannelVectorWidth, int FeatureVectorWidth,
          int Window, int Stride, template <typename> class MemObj>
SNNStatus launch_with_index_type(MemObj<T const>& input,
                                 MemObj<T const>& filter, MemObj<T>& output,
                                 Conv2DParams const& params,
                                 tiled::TileInfo const& tile_info,
                                 cl::sycl::queue& queue,
                                 const std::vector<cl::sycl::event>& events) {
  auto kernel_params = get_kernel_params<ConvType>(params);
  if (can_use_fast_div<ConvType>(kernel_params, ChannelVectorWidth,
                                 FeatureVectorWidth, TileRows, TileCols)) {
    return queue_tiled_kernel<T, Index, ConvType, TileRows, TileCols,
                              ChannelVectorWidth, FeatureVectorWidth, true,
                              Window, Window, Stride>(
        input, filter, output, kernel_params, tile_info, queue, events);
  } else {
    return queue_tiled_kernel<T, Index, ConvType, TileRows, TileCols,
                              ChannelVectorWidth, FeatureVectorWidth, false,
                              Window, Window, Stride>(
        input, filter, output, kernel_params, tile_info, queue, events);
  }
}
/**
 * Check what data type is required to fit the index sizes, and launch the
 * required kernel.
 */
template <typename T, typename ConvType, int TileRows, int TileCols,
          int ChannelVectorWidth, int FeatureVectorWidth, int Window,
          int Stride, template <typename> class MemObj>
SNNStatus launch_with_sizes(MemObj<T const>& input, MemObj<T const>& filter,
                            MemObj<T>& output, Conv2DParams const& params,
                            cl::sycl::queue& queue,
                            const std::vector<cl::sycl::event>& events) {
  auto const tile_info = tiled::get_tile_info<ConvType>(
      params, TileRows, TileCols, ChannelVectorWidth, FeatureVectorWidth);
  size_t const output_size = params.batch * tile_info.n_rows *
                             tile_info.n_cols * tile_info.output_vectors;
  if (output_size > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
#ifdef SNN_USE_INT64
    return launch_with_index_type<T, int64_t, ConvType, TileRows, TileCols,
                                  ChannelVectorWidth, FeatureVectorWidth,
                                  Window, Stride>(input, filter, output, params,
                                                  tile_info, queue, events);
#else
    return StatusCode::IndexExceeded;
#endif  // SNN_USE_INT64
  } else {
    return launch_with_index_type<T, int32_t, ConvType, TileRows, TileCols,
                                  ChannelVectorWidth, FeatureVectorWidth,
                                  Window, Stride>(input, filter, output, params,
                                                  tile_info, queue, events);
  }
}

/** Internal tile size launcher for Forward.  */
template <typename T, typename ConvType, template <typename> class MemObj,
          typename std::enable_if<
              std::is_same<ConvType, conv_type::Forward>::value, int>::type = 0>
inline SNNStatus launch_tiled_impl(MemObj<T const>& input,
                                   MemObj<T const>& filter, MemObj<T>& output,
                                   Conv2DParams const& params,
                                   cl::sycl::queue& queue,
                                   const std::vector<cl::sycl::event>& events) {
#define LAUNCH_IF_MATCH(params, window, stride, tile_row, tile_col,           \
                        channel_vector, feature_vector)                       \
  if (can_use_sizes<ConvType>(params, channel_vector, feature_vector, window, \
                              stride)) {                                      \
    return launch_with_sizes<T, ConvType, tile_row, tile_col, channel_vector, \
                             feature_vector, window, stride>(                 \
        input, filter, output, params, queue, events);                        \
  }

// clang-format off
#ifdef POWER_VR
  LAUNCH_IF_MATCH(params, 3, 1, 5, 4, 2, 4)
  LAUNCH_IF_MATCH(params, 3, 1, 3, 4, 1, 8)
  LAUNCH_IF_MATCH(params, 3, 1, 4, 3, 8, 2)
  LAUNCH_IF_MATCH(params, 3, 1, 5, 4, 1, 4)
  LAUNCH_IF_MATCH(params, 3, 1, 5, 4, 8, 1)
  LAUNCH_IF_MATCH(params, 3, 1, 5, 5, 1, 1)
  LAUNCH_IF_MATCH(params, 1, 1, 5, 2, 8, 8)
  LAUNCH_IF_MATCH(params, 1, 1, 4, 4, 1, 8)
  LAUNCH_IF_MATCH(params, 1, 1, 5, 5, 8, 1)
  LAUNCH_IF_MATCH(params, 1, 1, 5, 4, 1, 1)
#endif
#ifdef ARM_GPU
  LAUNCH_IF_MATCH(params, 3, 1, 5, 4, 1, 1)
  LAUNCH_IF_MATCH(params, 1, 1, 2, 4, 4, 2)
  LAUNCH_IF_MATCH(params, 1, 1, 2, 3, 1, 4)
  LAUNCH_IF_MATCH(params, 1, 1, 3, 4, 4, 1)
  LAUNCH_IF_MATCH(params, 1, 1, 2, 4, 1, 1)
#endif
#ifdef AMD_GPU
  LAUNCH_IF_MATCH(params, 3, 1, 4, 5, 4, 2)
  LAUNCH_IF_MATCH(params, 3, 1, 4, 5, 2, 2)
  LAUNCH_IF_MATCH(params, 3, 1, 5, 5, 4, 1)
  LAUNCH_IF_MATCH(params, 3, 1, 4, 3, 1, 4)
  LAUNCH_IF_MATCH(params, 3, 1, 5, 4, 1, 1)
  LAUNCH_IF_MATCH(params, 1, 1, 1, 5, 4, 8)
  LAUNCH_IF_MATCH(params, 1, 1, 2, 3, 4, 4)
  LAUNCH_IF_MATCH(params, 1, 1, 2, 5, 1, 8)
  LAUNCH_IF_MATCH(params, 1, 1, 3, 4, 8, 1)
  LAUNCH_IF_MATCH(params, 1, 1, 2, 3, 1, 1)
#endif
#ifdef INTEL_GPU
  LAUNCH_IF_MATCH(params, 3, 1, 3, 3, 1, 4)
  LAUNCH_IF_MATCH(params, 3, 1, 5, 4, 1, 1)
  LAUNCH_IF_MATCH(params, 1, 1, 4, 2, 4, 8)
  LAUNCH_IF_MATCH(params, 1, 1, 3, 4, 1, 8)
  LAUNCH_IF_MATCH(params, 1, 1, 3, 4, 1, 1)
#endif
#ifdef INTEL_CPU
  LAUNCH_IF_MATCH(params, 3, 1, 5, 4, 1, 16)
  LAUNCH_IF_MATCH(params, 3, 1, 4, 4, 1, 8)
  LAUNCH_IF_MATCH(params, 3, 1, 4, 5, 1, 1)
  LAUNCH_IF_MATCH(params, 1, 1, 1, 4, 1, 16)
  LAUNCH_IF_MATCH(params, 1, 1, 1, 4, 1, 8)
  LAUNCH_IF_MATCH(params, 1, 1, 1, 4, 1, 1)
#endif
  LAUNCH_IF_MATCH(params, 1, 2, 1, 2, 1, 4)
  LAUNCH_IF_MATCH(params, 1, 2, 1, 2, 1, 1)
  LAUNCH_IF_MATCH(params, 3, 2, 2, 2, 1, 4)
  LAUNCH_IF_MATCH(params, 3, 2, 2, 2, 1, 1)
  LAUNCH_IF_MATCH(params, 3, 1, 2, 2, 1, 4)
  LAUNCH_IF_MATCH(params, 3, 1, 3, 4, 1, 1)
  LAUNCH_IF_MATCH(params, 3, 2, 2, 2, 1, 1)
  LAUNCH_IF_MATCH(params, 5, 1, 2, 2, 1, 2)
  LAUNCH_IF_MATCH(params, 5, 1, 2, 4, 1, 1)
  LAUNCH_IF_MATCH(params, 1, 1, 2, 2, 1, 4)
  LAUNCH_IF_MATCH(params, 1, 1, 2, 2, 1, 1)
  LAUNCH_IF_MATCH(params, 1, 2, 2, 2, 1, 1)
  // clang-format on

  return StatusCode::InvalidAlgorithm;
}

/** Internal tile size launcher for InputBackprop.  */
template <
    typename T, typename ConvType, template <typename> class MemObj,
    typename std::enable_if<
        std::is_same<ConvType, conv_type::InputBackprop>::value, int>::type = 0>
inline SNNStatus launch_tiled_impl(MemObj<T const>& input,
                                   MemObj<T const>& filter, MemObj<T>& output,
                                   Conv2DParams const& params,
                                   cl::sycl::queue& queue,
                                   const std::vector<cl::sycl::event>& events) {
  // clang-format off
  LAUNCH_IF_MATCH(params, 1, 2, 2, 2, 1, 4)
  LAUNCH_IF_MATCH(params, 1, 2, 2, 2, 1, 1)
  LAUNCH_IF_MATCH(params, 3, 2, 2, 4, 1, 2)
  LAUNCH_IF_MATCH(params, 3, 1, 3, 4, 1, 4)
  LAUNCH_IF_MATCH(params, 3, 1, 2, 2, 1, 4)
  LAUNCH_IF_MATCH(params, 3, 1, 3, 4, 1, 1)
  LAUNCH_IF_MATCH(params, 3, 2, 2, 2, 1, 1)
  LAUNCH_IF_MATCH(params, 5, 1, 2, 2, 1, 2)
  LAUNCH_IF_MATCH(params, 5, 1, 2, 4, 1, 1)
  LAUNCH_IF_MATCH(params, 1, 1, 2, 2, 1, 4)
  LAUNCH_IF_MATCH(params, 1, 1, 2, 2, 1, 1)
  LAUNCH_IF_MATCH(params, 1, 2, 2, 2, 1, 1)
  // clang-format on

  return StatusCode::InvalidAlgorithm;
}

#undef LAUNCH_IF_MATCH

/** Internal tile size launcher for FilterBackprop.  */
template <typename T, typename ConvType, template <typename> class MemObj,
          typename std::enable_if<
              std::is_same<ConvType, conv_type::FilterBackprop>::value,
              int>::type = 0>
inline SNNStatus launch_tiled_impl(
    MemObj<T const>& /*input*/, MemObj<T const>& /*filter*/,
    MemObj<T>& /*output*/, Conv2DParams const& /*params*/,
    cl::sycl::queue& /*queue*/,
    const std::vector<cl::sycl::event>& /*events*/) {
  // Tiled algorithm is not supported for filter backprop.
  return StatusCode::InvalidAlgorithm;
}
}  // namespace

template <typename T, typename ConvType, template <typename> class MemObj>
inline SNNStatus launch_tiled(MemObj<T const>& input, MemObj<T const>& filter,
                              MemObj<T>& output, Conv2DParams const& params,
                              cl::sycl::queue& queue,
                              const std::vector<cl::sycl::event>& events) {
  return launch_tiled_impl<T, ConvType>(input, filter, output, params, queue,
                                        events);
}

#define INSTANTIATE_LAUNCHER(DTYPE, DIR, MEM_OBJ)                  \
  template SNN_EXPORT SNNStatus launch_tiled<DTYPE, DIR>(          \
      MEM_OBJ<DTYPE const> & input, MEM_OBJ<DTYPE const> & filter, \
      MEM_OBJ<DTYPE> & output, Conv2DParams const& params,         \
      cl::sycl::queue& queue, const std::vector<cl::sycl::event>& events)

#define INSTANTIATE_FOR_TYPE(DTYPE, MEM_OBJ)                      \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::Forward, MEM_OBJ);       \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::InputBackprop, MEM_OBJ); \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::FilterBackprop, MEM_OBJ)

#ifdef SNN_ENABLE_USM
INSTANTIATE_FOR_TYPE(float, USMMemObject);
#endif
INSTANTIATE_FOR_TYPE(float, BufferMemObject);

#ifdef SNN_USE_DOUBLE
#ifdef SNN_ENABLE_USM
INSTANTIATE_FOR_TYPE(double, USMMemObject);
#endif
INSTANTIATE_FOR_TYPE(double, BufferMemObject);
#endif  // SNN_USE_DOUBLE

#ifdef SNN_USE_HALF
#ifdef SNN_ENABLE_USM
INSTANTIATE_FOR_TYPE(cl::sycl::half, USMMemObject);
#endif
INSTANTIATE_FOR_TYPE(cl::sycl::half, BufferMemObject);
#endif  // SNN_USE_HALF

#undef INSTANTIATE_FOR_TYPE
#undef INSTANTIATE_LAUNCHER

}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn
