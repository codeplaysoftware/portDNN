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
#include "sycldnn/internal/conv2d/direct.h"

#include "sycldnn/mem_object.h"
#include "sycldnn/status.h"

#include "sycldnn/conv2d/conv_type.h"

#include "sycldnn/conv2d/params.h"
#include "sycldnn/conv2d/sizes.h"

#include "src/conv2d/direct/kernel_params.h"
#include "src/conv2d/direct/queue_direct_kernel.h"

#include <CL/sycl.hpp>

#include <stddef.h>
#include <cstdint>
#include <limits>

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
  return params.features % width == 0;
}

/**
 * Check whether fast divisions can be used for the convolution, and launch
 * the convolution kernel to do the computation.
 */
template <typename T, typename Index, typename ConvType, int Window, int Stride,
          int VectorWidth>
SNNStatus launch_with_vector(BaseMemObject<T const>& input,
                             BaseMemObject<T const>& filter,
                             BaseMemObject<T>& output,
                             Conv2DParams const& params, Index output_size,
                             cl::sycl::queue& queue) {
  auto kernel_params = direct::get_kernel_params<ConvType>(params);
  if (can_use_fast_div<ConvType>(kernel_params, VectorWidth)) {
    return queue_direct_kernel<T, Index, ConvType, true, Window, Stride,
                               VectorWidth>(input, filter, output,
                                            kernel_params, output_size, queue);
  } else {
    return queue_direct_kernel<T, Index, ConvType, false, Window, Stride,
                               VectorWidth>(input, filter, output,
                                            kernel_params, output_size, queue);
  }
}

/**
 * Check which vector widths can be used for the convolution, and launch
 * the convolution kernel to do the computation.
 */
template <typename T, typename Index, typename ConvType, int Window, int Stride>
SNNStatus launch_with_index(BaseMemObject<T const>& input,
                            BaseMemObject<T const>& filter,
                            BaseMemObject<T>& output,
                            Conv2DParams const& params, Index output_size,
                            cl::sycl::queue& queue) {
  if (can_use_vector_width<ConvType>(params, 4)) {
    return launch_with_vector<T, Index, ConvType, Window, Stride, 4>(
        input, filter, output, params, output_size, queue);
  } else if (can_use_vector_width<ConvType>(params, 2)) {
    return launch_with_vector<T, Index, ConvType, Window, Stride, 2>(
        input, filter, output, params, output_size, queue);
  } else {
    return launch_with_vector<T, Index, ConvType, Window, Stride, 1>(
        input, filter, output, params, output_size, queue);
  }
}

/**
 * Check what data type is required to fit the index sizes, and launch the
 * required kernel.
 */
template <typename T, typename ConvType, int Window, int Stride>
SNNStatus launch_with_static_sizes(BaseMemObject<T const>& input,
                                   BaseMemObject<T const>& filter,
                                   BaseMemObject<T>& output,
                                   Conv2DParams const& params,
                                   cl::sycl::queue& queue) {
  auto conv_sizes = get_sizes<ConvType>(params);
  size_t output_size = conv_sizes.output_size;
  if (output_size > std::numeric_limits<int32_t>::max()) {
#ifdef SNN_USE_INT64
    return launch_with_index<T, int64_t, ConvType, Window, Stride>(
        input, filter, output, params, static_cast<int64_t>(output_size),
        queue);
#else
    SNNStatus tensor_too_large;
    tensor_too_large.status = StatusCode::IndexExceeded;
    return tensor_too_large;
#endif  // SNN_USE_INT64
  } else {
    return launch_with_index<T, int32_t, ConvType, Window, Stride>(
        input, filter, output, params, static_cast<int32_t>(output_size),
        queue);
  }
}
}  // namespace
/**
 * Use static window and stride sizes for the most common cases, or fall back
 * to using dynamic window and strides. This allows the compiler to make use of
 * the static window and stride sizes to better optimise when possible.
 */
template <typename T, typename ConvType>
SNNStatus launch_direct(BaseMemObject<T const>& input,
                        BaseMemObject<T const>& filter,
                        BaseMemObject<T>& output, Conv2DParams const& params,
                        cl::sycl::queue& queue) {
#ifdef SNN_CONV2D_STATIC_DIRECT
  if (can_use_static_conv<ConvType>(params, 1, 1)) {
    return launch_with_static_sizes<T, ConvType, 1, 1>(input, filter, output,
                                                       params, queue);
  } else if (can_use_static_conv<ConvType>(params, 3, 1)) {
    return launch_with_static_sizes<T, ConvType, 3, 1>(input, filter, output,
                                                       params, queue);
  } else if (can_use_static_conv<ConvType>(params, 3, 2)) {
    return launch_with_static_sizes<T, ConvType, 3, 2>(input, filter, output,
                                                       params, queue);
  } else if (can_use_static_conv<ConvType>(params, 5, 1)) {
    return launch_with_static_sizes<T, ConvType, 5, 1>(input, filter, output,
                                                       params, queue);
  } else if (can_use_static_conv<ConvType>(params, 5, 2)) {
    return launch_with_static_sizes<T, ConvType, 5, 2>(input, filter, output,
                                                       params, queue);
  } else
#endif  // SNN_CONV2D_STATIC_DIRECT
  {
    return launch_with_static_sizes<T, ConvType, 0, 0>(input, filter, output,
                                                       params, queue);
  }
}

#define INSTANTIATE_LAUNCHER(DTYPE, DIR)                                       \
  template SNNStatus launch_direct<DTYPE, DIR>(                                \
      BaseMemObject<DTYPE const> & input, BaseMemObject<DTYPE const> & filter, \
      BaseMemObject<DTYPE> & output, Conv2DParams const& params,               \
      cl::sycl::queue& queue)

#define INSTANTIATE_FOR_TYPE(DTYPE)                      \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::Forward);       \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::InputBackprop); \
  INSTANTIATE_LAUNCHER(DTYPE, conv_type::FilterBackprop)

INSTANTIATE_FOR_TYPE(float);

#ifdef SNN_USE_DOUBLE
INSTANTIATE_FOR_TYPE(double);
#endif  // SNN_USE_DOUBLE

#ifdef SNN_USE_HALF
INSTANTIATE_FOR_TYPE(cl::sycl::half);
#endif  // SNN_USE_HALF

#undef INSTANTIATE_FOR_TYPE
#undef INSTANTIATE_LAUNCHER

}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn
