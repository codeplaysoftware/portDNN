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
#include "sycldnn/accessor_types.h"
#include "sycldnn/status.h"

#include "sycldnn/depthwise_conv2d/params.h"

#include "src/depthwise_conv2d/kernel_params.h"
#include "src/depthwise_conv2d/output_size.h"
#include "src/depthwise_conv2d/queue_depthwise_conv2d_impl.h"

#include <CL/sycl.hpp>

namespace sycldnn {
namespace depthwise_conv2d {
namespace internal {

template <typename ConvType, typename T, typename Index>
struct Launcher {
  static SNNStatus launch(ReadAccessor<T const> input,
                          ReadAccessor<T const> filter, WriteAccessor<T> output,
                          DepthwiseConv2DParams const& params,
                          Index output_size, cl::sycl::queue& queue) {
    return queue_kernel<ConvType>(input, filter, output, params, output_size,
                                  queue);
  }
};

template <typename T, typename Index>
struct Launcher<conv2d::conv_type::FilterBackprop, T, Index> {
  static SNNStatus launch(ReadAccessor<T const> input,
                          ReadAccessor<T const> filter, WriteAccessor<T> output,
                          DepthwiseConv2DParams const& params,
                          Index output_size, cl::sycl::queue& queue) {
    return queue_kernel_fil_bk(input, filter, output, params, output_size,
                               queue);
  }
};

template <typename ConvType, typename T>
SNNStatus launch(ReadAccessor<T const> input, ReadAccessor<T const> filter,
                 WriteAccessor<T> output, DepthwiseConv2DParams const& params,
                 cl::sycl::queue& queue) {
  size_t output_size = get_output_size<ConvType>(params);
  auto kernel_params = get_kernel_params<ConvType>(params);
  if (output_size > std::numeric_limits<int32_t>::max()) {
#ifdef SNN_USE_INT64
    return Launcher<ConvType, T, int64_t>::launch(
        input, filter, output, kernel_params, static_cast<int64_t>(output_size),
        queue);
#else
    SNNStatus tensor_too_large;
    tensor_too_large.status = StatusCode::IndexExceeded;
    return tensor_too_large;
#endif  // SNN_USE_INT64
  } else {
    return Launcher<ConvType, T, int32_t>::launch(
        input, filter, output, kernel_params, static_cast<int32_t>(output_size),
        queue);
  }
}

#define INSTANTIATE_LAUNCHER(DTYPE, DIRECTION)                           \
  template SNNStatus launch<DIRECTION, DTYPE>(                           \
      ReadAccessor<DTYPE const> input, ReadAccessor<DTYPE const> filter, \
      WriteAccessor<DTYPE> output, DepthwiseConv2DParams const& params,  \
      cl::sycl::queue& queue)

#define INSTANTIATE_FOR_TYPE(DTYPE)                              \
  INSTANTIATE_LAUNCHER(DTYPE, conv2d::conv_type::Forward);       \
  INSTANTIATE_LAUNCHER(DTYPE, conv2d::conv_type::InputBackprop); \
  INSTANTIATE_LAUNCHER(DTYPE, conv2d::conv_type::FilterBackprop)

INSTANTIATE_FOR_TYPE(float);

#ifdef SNN_USE_DOUBLE
INSTANTIATE_FOR_TYPE(double);
#endif

#ifdef SNN_USE_HALF
INSTANTIATE_FOR_TYPE(cl::sycl::half);
#endif

#undef INSTANTIATE_FOR_TYPE
#undef INSTANTIATE_LAUNCHER

}  // namespace internal
}  // namespace depthwise_conv2d
}  // namespace sycldnn
