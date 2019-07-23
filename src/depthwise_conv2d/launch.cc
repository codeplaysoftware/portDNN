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
#include "sycldnn/mem_object.h"
#include "sycldnn/status.h"

#include "sycldnn/conv2d/conv_type.h"

#include "sycldnn/depthwise_conv2d/params.h"

#include "src/depthwise_conv2d/kernel_params.h"
#include "src/depthwise_conv2d/output_size.h"
#include "src/depthwise_conv2d/queue_depthwise_conv2d.h"

#include <stddef.h>
#include <cstdint>
#include <limits>

#include <CL/sycl.hpp>

#include "sycldnn/export.h"

namespace sycldnn {
namespace depthwise_conv2d {
namespace internal {

namespace {

template <typename ConvType>
bool can_vectorize(DepthwiseConv2DParams const& p, int vector_width) {
  // TODO(dmcbain): depthwise convolutions do not support vectorisation
  // for channel multipliers that are not 1
  if (p.channel_multiplier != 1) {
    return false;
  }
  return (p.channels * p.channel_multiplier) % vector_width == 0;
}

template <typename ConvType, typename T, typename Index, int VectorWidth>
struct Launcher {
  static SNNStatus launch(BaseMemObject<T const>& input,
                          BaseMemObject<T const>& filter,
                          BaseMemObject<T>& output,
                          DepthwiseConv2DParams const& params,
                          Index output_size, cl::sycl::queue& queue) {
    return queue_kernel<ConvType, VectorWidth>(input, filter, output, params,
                                               output_size, queue);
  }
};

template <typename T, typename Index, int VectorWidth>
struct Launcher<conv2d::conv_type::FilterBackprop, T, Index, VectorWidth> {
  static SNNStatus launch(BaseMemObject<T const>& input,
                          BaseMemObject<T const>& filter,
                          BaseMemObject<T>& output,
                          DepthwiseConv2DParams const& params,
                          Index output_size, cl::sycl::queue& queue) {
    return queue_kernel_fil_bk<VectorWidth>(input, filter, output, params,
                                            output_size, queue);
  }
};

template <typename ConvType, typename T, typename IndexType>
SNNStatus launch_vectorised(BaseMemObject<T const>& input,
                            BaseMemObject<T const>& filter,
                            BaseMemObject<T>& output,
                            DepthwiseConv2DParams const& params,
                            IndexType output_size, cl::sycl::queue& queue) {
  if (can_vectorize<ConvType>(params, 4)) {
    return Launcher<ConvType, T, IndexType, 4>::launch(
        input, filter, output, params, output_size, queue);
  } else if (can_vectorize<ConvType>(params, 2)) {
    return Launcher<ConvType, T, IndexType, 2>::launch(
        input, filter, output, params, output_size, queue);
  } else {
    return Launcher<ConvType, T, IndexType, 1>::launch(
        input, filter, output, params, output_size, queue);
  }
}

}  // namespace

template <typename ConvType, typename T>
SNNStatus launch(BaseMemObject<T const>& input, BaseMemObject<T const>& filter,
                 BaseMemObject<T>& output, DepthwiseConv2DParams const& params,
                 cl::sycl::queue& queue) {
  size_t output_size = get_output_size<ConvType>(params);
  auto kernel_params = get_kernel_params<ConvType>(params);
  if (output_size > std::numeric_limits<int32_t>::max()) {
#ifdef SNN_USE_INT64
    return launch_vectorised<ConvType, T, int64_t>(
        input, filter, output, kernel_params, static_cast<int64_t>(output_size),
        queue);
#else
    SNNStatus tensor_too_large;
    tensor_too_large.status = StatusCode::IndexExceeded;
    return tensor_too_large;
#endif  // SNN_USE_INT64
  } else {
    return launch_vectorised<ConvType, T, int32_t>(
        input, filter, output, kernel_params, static_cast<int32_t>(output_size),
        queue);
  }
}

#define INSTANTIATE_LAUNCHER(DTYPE, DIRECTION)                                 \
  template SNN_EXPORT SNNStatus launch<DIRECTION, DTYPE>(                      \
      BaseMemObject<DTYPE const> & input, BaseMemObject<DTYPE const> & filter, \
      BaseMemObject<DTYPE> & output, DepthwiseConv2DParams const& params,      \
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
