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
#include "portdnn/internal/conv2d/direct.h"

#include "portdnn/format_type.h"
#include "portdnn/mem_object.h"
#include "portdnn/status.h"

#include "portdnn/conv2d/conv_type.h"

#include "portdnn/conv2d/params.h"
#include "portdnn/conv2d/sizes.h"

#include "src/conv2d/direct/kernel_params.h"
#include "src/conv2d/direct/queue_direct_kernel.h"

#include <CL/sycl.hpp>

#include <stddef.h>
#include <cstdint>
#include <limits>

#include "portdnn/export.h"

namespace sycldnn {
namespace conv2d {
namespace internal {
namespace {
/** Check whether fast divisions can be used for the given parameters. */
template <typename ConvType>
static inline bool can_use_fast_div(Conv2DParams const& params, int vec_width);

template <>
inline bool can_use_fast_div<conv_type::Forward>(Conv2DParams const& params,
                                                 int vec_width) {
  return (params.features / vec_width) != 1 && params.out_rows != 1 &&
         params.out_cols != 1;
}
template <>
inline bool can_use_fast_div<conv_type::InputBackprop>(
    Conv2DParams const& params, int /*vec_width*/) {
  return params.features != 1 && params.in_rows != 1 && params.in_cols != 1;
}
template <>
inline bool can_use_fast_div<conv_type::FilterBackprop>(
    Conv2DParams const& params, int vec_width) {
  return (params.features / vec_width) != 1 && params.channels != 1 &&
         params.out_cols != 1;
}
/**
 * Check whether the provided window and stride can be used with the given
 * convolution parameters.
 */
template <typename ConvType>
inline bool can_use_static_conv(Conv2DParams const& params, int const window,
                                int const stride) {
  return (params.window_cols == window && params.window_rows == window &&
          params.stride_rows == stride && params.stride_cols == stride);
}

/**
 * Check whether a given vector width can be used for the given convolution.
 *
 * Expects the convolution parameters to be the original parameters, not the
 * kernel parameters.
 * */
template <typename ConvType>
inline bool can_use_vector_width(Conv2DParams const& params, int const width) {
  return params.input_format == DataFormat::NHWC &&
         params.filter_format == FilterFormat::HWCF &&
         params.features % width == 0;
}

/**
 * \brief The helper ensures that only the instantiated symbols are used.
 */
template <typename T, typename Index, typename ConvType, bool UseFastDiv,
          int Window, int Stride, int VectorWidth, typename Layout,
          template <typename> class MemObj>
struct queue_kernel_helper {
  SNNStatus operator()(MemObj<T const>&, MemObj<T const>&, MemObj<T>&,
                       Conv2DParams const&, Index, cl::sycl::queue&,
                       const std::vector<cl::sycl::event>& events) {
    SNN_UNUSED_VAR(events)
    return StatusCode::InvalidAlgorithm;
  }
};

template <typename T, typename Index, typename ConvType, bool UseFastDiv,
          int Window, int Stride, int VectorWidth,
          template <typename> class MemObj>
struct queue_kernel_helper<T, Index, ConvType, UseFastDiv, Window, Stride,
                           VectorWidth, layout::NHWC, MemObj> {
  SNNStatus operator()(MemObj<T const>& input, MemObj<T const>& filter,
                       MemObj<T>& output, Conv2DParams const& params,
                       Index output_size, cl::sycl::queue& queue,
                       const std::vector<cl::sycl::event>& events) {
    return queue_direct_kernel<T, Index, ConvType, UseFastDiv, Window, Stride,
                               VectorWidth, layout::NHWC, MemObj>(
        input, filter, output, params, output_size, queue, events);
  }
};

#ifdef SNN_ENABLE_NCHW
template <typename T, typename Index, typename ConvType, bool UseFastDiv,
          int Window, int Stride, template <typename> class MemObj>
struct queue_kernel_helper<T, Index, ConvType, UseFastDiv, Window, Stride, 1,
                           layout::NCHW, MemObj> {
  SNNStatus operator()(MemObj<T const>& input, MemObj<T const>& filter,
                       MemObj<T>& output, Conv2DParams const& params,
                       Index output_size, cl::sycl::queue& queue,
                       const std::vector<cl::sycl::event>& events) {
    return queue_direct_kernel<T, Index, ConvType, UseFastDiv, Window, Stride,
                               /*VectorWidth=*/1, layout::NCHW, MemObj>(
        input, filter, output, params, output_size, queue, events);
  }
};
#endif

template <typename T, typename Index, typename ConvType, bool UseFastDiv,
          int Window, int Stride, int VectorWidth,
          template <typename> class MemObj>
SNNStatus launch_with_fast_div(MemObj<T const>& input, MemObj<T const>& filter,
                               MemObj<T>& output, Conv2DParams const& params,
                               Index output_size, cl::sycl::queue& queue,
                               const std::vector<cl::sycl::event>& events) {
  if (params.input_format == DataFormat::NCHW &&
      params.filter_format == FilterFormat::FCHW) {
    return queue_kernel_helper<T, Index, ConvType, UseFastDiv, Window, Stride,
                               VectorWidth, layout::NCHW, MemObj>()(
        input, filter, output, params, output_size, queue, events);
  } else if (params.input_format == DataFormat::NHWC &&
             params.filter_format == FilterFormat::HWCF) {
    return queue_kernel_helper<T, Index, ConvType, UseFastDiv, Window, Stride,
                               VectorWidth, layout::NHWC, MemObj>()(
        input, filter, output, params, output_size, queue, events);
  }
  return StatusCode::InvalidAlgorithm;
}

/**
 * Check whether fast divisions can be used for the convolution, and launch
 * the convolution kernel to do the computation.
 */
template <typename T, typename Index, typename ConvType, int Window, int Stride,
          int VectorWidth, template <typename> class MemObj>
SNNStatus launch_with_vector(MemObj<T const>& input, MemObj<T const>& filter,
                             MemObj<T>& output, Conv2DParams const& params,
                             Index output_size, cl::sycl::queue& queue,
                             const std::vector<cl::sycl::event>& events) {
  auto kernel_params = direct::get_kernel_params<ConvType>(params);
  if (can_use_fast_div<ConvType>(kernel_params, VectorWidth)) {
    return launch_with_fast_div<T, Index, ConvType, true, Window, Stride,
                                VectorWidth, MemObj>(
        input, filter, output, kernel_params, output_size, queue, events);
  } else {
    return launch_with_fast_div<T, Index, ConvType, false, Window, Stride,
                                VectorWidth, MemObj>(
        input, filter, output, kernel_params, output_size, queue, events);
  }
}

/**
 * Check which vector widths can be used for the convolution, and launch
 * the convolution kernel to do the computation.
 */
template <typename T, typename Index, typename ConvType, int Window, int Stride,
          template <typename> class MemObj>
SNNStatus launch_with_index(MemObj<T const>& input, MemObj<T const>& filter,
                            MemObj<T>& output, Conv2DParams const& params,
                            Index output_size, cl::sycl::queue& queue,
                            const std::vector<cl::sycl::event>& events) {
  if (can_use_vector_width<ConvType>(params, 4)) {
    return launch_with_vector<T, Index, ConvType, Window, Stride, 4, MemObj>(
        input, filter, output, params, output_size, queue, events);
  } else if (can_use_vector_width<ConvType>(params, 2)) {
    return launch_with_vector<T, Index, ConvType, Window, Stride, 2, MemObj>(
        input, filter, output, params, output_size, queue, events);
  } else {
    return launch_with_vector<T, Index, ConvType, Window, Stride, 1, MemObj>(
        input, filter, output, params, output_size, queue, events);
  }
}

/**
 * Check what data type is required to fit the index sizes, and launch the
 * required kernel.
 */
template <typename T, typename ConvType, int Window, int Stride,
          template <typename> class MemObj>
SNNStatus launch_with_static_sizes(MemObj<T const>& input,
                                   MemObj<T const>& filter, MemObj<T>& output,
                                   Conv2DParams const& params,
                                   cl::sycl::queue& queue,
                                   const std::vector<cl::sycl::event>& events) {
  auto conv_sizes = get_sizes<ConvType>(params);
  size_t output_size = conv_sizes.output_size;
  if (output_size > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
#ifdef SNN_USE_INT64
    return launch_with_index<T, int64_t, ConvType, Window, Stride, MemObj>(
        input, filter, output, params, static_cast<int64_t>(output_size), queue,
        events);
#else
    return StatusCode::IndexExceeded;
#endif  // SNN_USE_INT64
  } else {
    return launch_with_index<T, int32_t, ConvType, Window, Stride, MemObj>(
        input, filter, output, params, static_cast<int32_t>(output_size), queue,
        events);
  }
}
}  // namespace
/**
 * Use static window and stride sizes for the most common cases, or fall back
 * to using dynamic window and strides. This allows the compiler to make use of
 * the static window and stride sizes to better optimise when possible.
 */
template <typename T, typename ConvType, template <typename> class MemObj>
SNNStatus launch_direct(MemObj<T const>& input, MemObj<T const>& filter,
                        MemObj<T>& output, Conv2DParams const& params,
                        cl::sycl::queue& queue,
                        const std::vector<cl::sycl::event>& events) {
#ifdef SNN_CONV2D_STATIC_DIRECT
  if (can_use_static_conv<ConvType>(params, 1, 1)) {
    return launch_with_static_sizes<T, ConvType, 1, 1, MemObj>(
        input, filter, output, params, queue, events);
  } else if (can_use_static_conv<ConvType>(params, 3, 1)) {
    return launch_with_static_sizes<T, ConvType, 3, 1, MemObj>(
        input, filter, output, params, queue, events);
  } else if (can_use_static_conv<ConvType>(params, 3, 2)) {
    return launch_with_static_sizes<T, ConvType, 3, 2, MemObj>(
        input, filter, output, params, queue, events);
  } else if (can_use_static_conv<ConvType>(params, 5, 1)) {
    return launch_with_static_sizes<T, ConvType, 5, 1, MemObj>(
        input, filter, output, params, queue, events);
  } else if (can_use_static_conv<ConvType>(params, 5, 2)) {
    return launch_with_static_sizes<T, ConvType, 5, 2, MemObj>(
        input, filter, output, params, queue, events);
  } else
#endif  // SNN_CONV2D_STATIC_DIRECT
  {
    return launch_with_static_sizes<T, ConvType, 0, 0, MemObj>(
        input, filter, output, params, queue, events);
  }
}

#define INSTANTIATE_LAUNCHER(DTYPE, DIR, MEMOBJ)                   \
  template SNN_EXPORT SNNStatus launch_direct<DTYPE, DIR, MEMOBJ>( \
      MEMOBJ<DTYPE const> & input, MEMOBJ<DTYPE const> & filter,   \
      MEMOBJ<DTYPE> & output, Conv2DParams const& params,          \
      cl::sycl::queue& queue, const std::vector<cl::sycl::event>& events)

#ifdef SNN_ENABLE_USM
#define INSTANTIATE_FOR_MEMOBJ(DTYPE, DIR)        \
  INSTANTIATE_LAUNCHER(DTYPE, DIR, USMMemObject); \
  INSTANTIATE_LAUNCHER(DTYPE, DIR, BufferMemObject);
#else
#define INSTANTIATE_FOR_MEMOBJ(DTYPE, DIR) \
  INSTANTIATE_LAUNCHER(DTYPE, DIR, BufferMemObject);

#endif  // SNN_ENABLE_USM

#define INSTANTIATE_FOR_TYPE(DTYPE)                        \
  INSTANTIATE_FOR_MEMOBJ(DTYPE, conv_type::Forward);       \
  INSTANTIATE_FOR_MEMOBJ(DTYPE, conv_type::InputBackprop); \
  INSTANTIATE_FOR_MEMOBJ(DTYPE, conv_type::FilterBackprop);

INSTANTIATE_FOR_TYPE(float);

#ifdef SNN_USE_DOUBLE
INSTANTIATE_FOR_TYPE(double);
#endif  // SNN_USE_DOUBLE

#ifdef SNN_USE_HALF
INSTANTIATE_FOR_TYPE(cl::sycl::half);
#endif  // SNN_USE_HALF

#undef INSTANTIATE_FOR_TYPE
#undef INSTANTIATE_LAUNCHER
#undef INSTANTIATE_FOR_MEMOBJ

}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn
