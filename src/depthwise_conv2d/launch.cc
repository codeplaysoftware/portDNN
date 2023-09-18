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

#include "portdnn/depthwise_conv2d/params.h"

#include "src/depthwise_conv2d/kernel_params.h"
#include "src/depthwise_conv2d/output_size.h"
#include "src/depthwise_conv2d/queue_depthwise_conv2d.h"

#include <stddef.h>
#include <cstdint>
#include <limits>

#include <CL/sycl.hpp>

#include "portdnn/export.h"

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

template <typename ConvType, typename T, typename Index, int VectorWidth,
          template <typename> class MemObj>
struct Launcher {
  static SNNStatus launch(MemObj<T const>& input, MemObj<T const>& filter,
                          MemObj<T>& output,
                          DepthwiseConv2DParams const& params,
                          Index output_size, cl::sycl::queue& queue,
                          const std::vector<cl::sycl::event>& events) {
    return queue_kernel<ConvType, VectorWidth>(input, filter, output, params,
                                               output_size, queue, events);
  }
};

template <typename T, typename Index, int VectorWidth,
          template <typename> class MemObj>
struct Launcher<conv2d::conv_type::FilterBackprop, T, Index, VectorWidth,
                MemObj> {
  static SNNStatus launch(MemObj<T const>& input, MemObj<T const>& filter,
                          MemObj<T>& output,
                          DepthwiseConv2DParams const& params,
                          Index output_size, cl::sycl::queue& queue,
                          const std::vector<cl::sycl::event>& events) {
    return queue_kernel_fil_bk<VectorWidth>(input, filter, output, params,
                                            output_size, queue, events);
  }
};

template <typename ConvType, typename T, typename IndexType,
          template <typename> class MemObj,
          typename = std::enable_if<is_mem_obj_v<MemObj<T>, T>>>
SNNStatus launch_vectorised(MemObj<T const>& input, MemObj<T const>& filter,
                            MemObj<T>& output,
                            DepthwiseConv2DParams const& params,
                            IndexType output_size, cl::sycl::queue& queue,
                            const std::vector<cl::sycl::event>& events) {
  if (can_vectorize<ConvType>(params, 4)) {
    return Launcher<ConvType, T, IndexType, 4, MemObj>::launch(
        input, filter, output, params, output_size, queue, events);
  } else if (can_vectorize<ConvType>(params, 2)) {
    return Launcher<ConvType, T, IndexType, 2, MemObj>::launch(
        input, filter, output, params, output_size, queue, events);
  } else {
    return Launcher<ConvType, T, IndexType, 1, MemObj>::launch(
        input, filter, output, params, output_size, queue, events);
  }
}

}  // namespace

template <typename ConvType, typename T, template <typename> class MemObj,
          typename = std::enable_if<is_mem_obj_v<MemObj<T>, T>>>
SNNStatus launch(MemObj<T const>& input, MemObj<T const>& filter,
                 MemObj<T>& output, DepthwiseConv2DParams const& params,
                 cl::sycl::queue& queue,
                 const std::vector<cl::sycl::event>& events) {
  size_t output_size = get_output_size<ConvType>(params);
  auto kernel_params = get_kernel_params<ConvType>(params);
  if (output_size > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
#ifdef SNN_USE_INT64
    return launch_vectorised<ConvType, T, int64_t>(
        input, filter, output, kernel_params, static_cast<int64_t>(output_size),
        queue, events);
#else
    return StatusCode::IndexExceeded;
#endif  // SNN_USE_INT64
  } else {
    return launch_vectorised<ConvType, T, int32_t>(
        input, filter, output, kernel_params, static_cast<int32_t>(output_size),
        queue, events);
  }
}

#define INSTANTIATE_LAUNCHER(DTYPE, DIRECTION, MEM_OBJ)             \
  template SNN_EXPORT SNNStatus launch<DIRECTION, DTYPE>(           \
      MEM_OBJ<DTYPE const> & input, MEM_OBJ<DTYPE const> & filter,  \
      MEM_OBJ<DTYPE> & output, DepthwiseConv2DParams const& params, \
      cl::sycl::queue& queue, const std::vector<cl::sycl::event>& events)

#define INSTANTIATE_FOR_TYPE(DTYPE, MEM_OBJ)                              \
  INSTANTIATE_LAUNCHER(DTYPE, conv2d::conv_type::Forward, MEM_OBJ);       \
  INSTANTIATE_LAUNCHER(DTYPE, conv2d::conv_type::InputBackprop, MEM_OBJ); \
  INSTANTIATE_LAUNCHER(DTYPE, conv2d::conv_type::FilterBackprop, MEM_OBJ)

#ifdef SNN_ENABLE_USM
INSTANTIATE_FOR_TYPE(float, USMMemObject);
#endif
INSTANTIATE_FOR_TYPE(float, BufferMemObject);

#ifdef SNN_USE_DOUBLE
#ifdef SNN_ENABLE_USM
INSTANTIATE_FOR_TYPE(double, USMMemObject);
#endif
INSTANTIATE_FOR_TYPE(double, BufferMemObject);
#endif

#ifdef SNN_USE_HALF
#ifdef SNN_ENABLE_USM
INSTANTIATE_FOR_TYPE(cl::sycl::half, USMMemObject);
#endif
INSTANTIATE_FOR_TYPE(cl::sycl::half, BufferMemObject);
#endif

#undef INSTANTIATE_FOR_TYPE
#undef INSTANTIATE_LAUNCHER

}  // namespace internal
}  // namespace depthwise_conv2d
}  // namespace sycldnn
