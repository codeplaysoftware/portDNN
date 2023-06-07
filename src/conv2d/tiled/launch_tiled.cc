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
#include "sycldnn/internal/conv2d/tiled.h"

#include "sycldnn/mem_object.h"
#include "sycldnn/status.h"

#include "sycldnn/conv2d/conv_type.h"
#include "sycldnn/conv2d/params.h"

#include "sycldnn/helpers/ratio.h"
#include "sycldnn/format_type.h"

#include "src/conv2d/tiled/kernel_params.h"
#include "src/conv2d/tiled/queue_tiled_kernel.h"
#include "src/conv2d/tiled/tile_info.h"

#include <stddef.h>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <CL/sycl.hpp>

#include "sycldnn/export.h"

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
 * Check the Data Layout, and launch whichever kernel is required.
 */
template <typename T, typename Index, typename ConvType, DataFormat Format,
          int TileRows, int TileCols, int ChannelVectorWidth,
          int FeatureVectorWidth, bool UseFastDiv, int Window, int Stride,
          template <typename> class MemObj>
SNNStatus launch_with_fast_div(MemObj<T const>& input, MemObj<T const>& filter,
                               MemObj<T>& output, Conv2DParams const& params,
                               tiled::TileInfo const& tile_info,
                               cl::sycl::queue& queue,
                               const std::vector<cl::sycl::event>& events) {
  if constexpr (Format == DataFormat::NHWC) {
    return queue_tiled_kernel<T, Index, ConvType, TileRows, TileCols,
                              ChannelVectorWidth, FeatureVectorWidth,
                              UseFastDiv, Window, Window, Stride, layout::NHWC>(
        input, filter, output, params, tile_info, queue, events);
  } else {
    return queue_tiled_kernel<T, Index, ConvType, TileRows, TileCols,
                              ChannelVectorWidth, FeatureVectorWidth,
                              UseFastDiv, Window, Window, Stride, layout::NCHW>(
        input, filter, output, params, tile_info, queue, events);
  }
}

/**
 * Check whether fast divisions can be used for the convolution, and launch
 * whichever kernel is required.
 */
template <typename T, typename Index, typename ConvType, DataFormat Format, int TileRows,
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
    return launch_with_fast_div<T, Index, ConvType, Format, TileRows, TileCols,
                              ChannelVectorWidth, FeatureVectorWidth, true,
                              Window, Stride>(
        input, filter, output, kernel_params, tile_info, queue, events);
  } else {
    return launch_with_fast_div<T, Index, ConvType, Format, TileRows, TileCols,
                              ChannelVectorWidth, FeatureVectorWidth, false,
                              Window, Stride>(
        input, filter, output, kernel_params, tile_info, queue, events);
  }
}
/**
 * Check what data type is required to fit the index sizes, and launch the
 * required kernel.
 */
template <typename T, typename ConvType, DataFormat Format, int TileRows, int TileCols,
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
    return launch_with_index_type<T, int64_t, ConvType, Format, TileRows, TileCols,
                                  ChannelVectorWidth, FeatureVectorWidth,
                                  Window, Stride>(input, filter, output, params,
                                                  tile_info, queue, events);
#else
    return StatusCode::IndexExceeded;
#endif  // SNN_USE_INT64
  } else {
    return launch_with_index_type<T, int32_t, ConvType, Format, TileRows, TileCols,
                                  ChannelVectorWidth, FeatureVectorWidth,
                                  Window, Stride>(input, filter, output, params,
                                                  tile_info, queue, events);
  }
}

// The launch_tiled_impl functions use X Macros to automatically
// include available tiled convolution kernels as defined in
// conv2d/CMakeLists.txt. Four macros are defined for every combination of
// NCHW/NHWC and Forward/InputBackprop, and the generated file
// tiled_variants.def defines the various window sizes, tile sizes &
// vectorizations available. For each of the four cases, it's necessary
// to define the other 3 macros as empty ('empty def') so that only
// the relevant parameter sets are exposed.

/** Internal tile size launcher for Forward.  */
template <
    typename T, typename ConvType, DataFormat Format,
    template <typename> class MemObj,
    typename std::enable_if<std::is_same<ConvType, conv_type::Forward>::value, int>::type = 0>
inline SNNStatus launch_tiled_impl(MemObj<T const>& input,
                                   MemObj<T const>& filter, MemObj<T>& output,
                                   Conv2DParams const& params,
                                   cl::sycl::queue& queue,
                                   const std::vector<cl::sycl::event>& events) {
#define LAUNCH_IF_MATCH(params, window, stride, tile_row, tile_col,           \
                        channel_vector, feature_vector)                       \
  if (can_use_sizes<ConvType>(params, channel_vector, feature_vector, window, \
                              stride)) {                                      \
    return launch_with_sizes<T, ConvType, Format, tile_row, tile_col, channel_vector, \
                             feature_vector, window, stride>(                 \
        input, filter, output, params, queue, events);                        \
  }

#define X_INPUTBACKPROP_NHWC(window, stride, tile_row, tile_col, \
                             channel_vector, feature_vector)  // empty def
#define X_INPUTBACKPROP_NCHW(window, stride, tile_row, tile_col, \
                             channel_vector, feature_vector)  // empty def

  if constexpr (Format == DataFormat::NHWC) {
#define X_FORWARD_NHWC(window, stride, tile_row, tile_col, channel_vector,    \
                       feature_vector)                                        \
  LAUNCH_IF_MATCH(params, window, stride, tile_row, tile_col, channel_vector, \
                  feature_vector)
#define X_FORWARD_NCHW(window, stride, tile_row, tile_col, channel_vector, \
                       feature_vector)  // empty def

#include "conv2d/tiled/tiled_variants.def"

#undef X_FORWARD_NHWC
#undef X_FORWARD_NCHW

    return StatusCode::InvalidAlgorithm;
  } else {  // NCHW

#define X_FORWARD_NCHW(window, stride, tile_row, tile_col, channel_vector,    \
                       feature_vector)                                        \
  LAUNCH_IF_MATCH(params, window, stride, tile_row, tile_col, channel_vector, \
                  feature_vector)
#define X_FORWARD_NHWC(window, stride, tile_row, tile_col, channel_vector, \
                       feature_vector)  // empty def

#include "conv2d/tiled/tiled_variants.def"

#undef X_FORWARD_NHWC
#undef X_FORWARD_NCHW
    return StatusCode::InvalidAlgorithm;
  }
#undef X_INPUTBACKPROP_NHWC
#undef X_INPUTBACKPROP_NCHW
}

/** Internal tile size launcher for InputBackprop.  */
template <
    typename T, typename ConvType, DataFormat Format,
    template <typename> class MemObj,
    typename std::enable_if<
        std::is_same<ConvType, conv_type::InputBackprop>::value, int>::type = 0>
inline SNNStatus launch_tiled_impl(MemObj<T const>& input,
                                   MemObj<T const>& filter, MemObj<T>& output,
                                   Conv2DParams const& params,
                                   cl::sycl::queue& queue,
                                   const std::vector<cl::sycl::event>& events) {
#define X_FORWARD_NHWC(window, stride, tile_row, tile_col, channel_vector, \
                       feature_vector)  // empty def
#define X_FORWARD_NCHW(window, stride, tile_row, tile_col, channel_vector, \
                       feature_vector)  // empty def

  if constexpr (Format == DataFormat::NHWC) {
#define X_INPUTBACKPROP_NHWC(window, stride, tile_row, tile_col,              \
                             channel_vector, feature_vector)                  \
  LAUNCH_IF_MATCH(params, window, stride, tile_row, tile_col, channel_vector, \
                  feature_vector)
#define X_INPUTBACKPROP_NCHW(window, stride, tile_row, tile_col, \
                             channel_vector, feature_vector)  // empty def

#include "conv2d/tiled/tiled_variants.def"

#undef X_INPUTBACKPROP_NHWC
#undef X_INPUTBACKPROP_NCHW

    return StatusCode::InvalidAlgorithm;

  } else {  // NCHW

#define X_INPUTBACKPROP_NCHW(window, stride, tile_row, tile_col,              \
                             channel_vector, feature_vector)                  \
  LAUNCH_IF_MATCH(params, window, stride, tile_row, tile_col, channel_vector, \
                  feature_vector)
#define X_INPUTBACKPROP_NHWC(window, stride, tile_row, tile_col, \
                             channel_vector, feature_vector)  // empty def

#include "conv2d/tiled/tiled_variants.def"

#undef X_INPUTBACKPROP_NHWC
#undef X_INPUTBACKPROP_NCHW

    return StatusCode::InvalidAlgorithm;
  }
#undef X_FORWARD_NHWC
#undef X_FORWARD_NCHW
}

#undef LAUNCH_IF_MATCH

/** Internal tile size launcher for FilterBackprop.  */
template <typename T, typename ConvType, DataFormat Format, template <typename> class MemObj,
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
  if (params.input_format == DataFormat::NHWC) {
    return launch_tiled_impl<T, ConvType, DataFormat::NHWC>(input, filter, output,
                                                        params, queue, events);
  } else { //NCHW
    return launch_tiled_impl<T, ConvType, DataFormat::NCHW>(input, filter, output,
                                                        params, queue, events);
  }
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
