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

#include "portdnn/pooling/params.h"
#include "portdnn/pooling/sizes.h"

#include "portdnn/helpers/macros.h"

#include "portdnn/internal/pooling/launch_internal.h"

#include "src/pooling/can_fastdiv.h"
#include "src/pooling/can_vectorize.h"
#include "src/pooling/kernels.h"
#include "src/pooling/queue_max_grad_kernel.h"

#include <CL/sycl.hpp>

#include <type_traits>

#include "portdnn/export.h"

namespace sycldnn {
namespace pooling {
namespace internal {

template <typename T, typename Index, template <typename> class PoolType,
          typename Direction, int VectorWidth, bool UseFastDiv,
          template <typename> class MemObj>
SNNStatus launch_with_fastdiv(MemObj<T const>& inp_data,
                              MemObj<T const>& outp_data,
                              MemObj<T const>& inp_backprop,
                              MemObj<T>& outp_backprop, const PoolingParams& pp,
                              size_t threads, cl::sycl::queue& queue,
                              const std::vector<cl::sycl::event>& events) {
  if (DataFormat::NHWC == pp.input_format) {
    return queue_max_grad_pooling<T, Index, PoolType, Direction, VectorWidth,
                                  UseFastDiv, layout::NHWC>(
        inp_data, outp_data, inp_backprop, outp_backprop, pp, threads, queue,
        events);
  } else if (DataFormat::NCHW == pp.input_format) {
    return StatusCode::InvalidAlgorithm;
  } else {
    return StatusCode::InvalidAlgorithm;
  }
}

template <typename T, typename Index, template <typename> class PoolType,
          typename Direction, int VectorWidth, template <typename> class MemObj>
SNNStatus launch_with_vector_size(MemObj<T const>& inp_data,
                                  MemObj<T const>& outp_data,
                                  MemObj<T const>& inp_backprop,
                                  MemObj<T>& outp_backprop,
                                  const PoolingParams& pp, size_t threads,
                                  cl::sycl::queue& queue,
                                  const std::vector<cl::sycl::event>& events) {
  threads /= VectorWidth;
  if (can_use_fastdiv<Direction>(pp, VectorWidth)) {
    return launch_with_fastdiv<T, Index, PoolType, Direction, VectorWidth,
                               true>(inp_data, outp_data, inp_backprop,
                                     outp_backprop, pp, threads, queue, events);
  } else {
    return launch_with_fastdiv<T, Index, PoolType, Direction, VectorWidth,
                               false>(inp_data, outp_data, inp_backprop,
                                      outp_backprop, pp, threads, queue,
                                      events);
  }
}

template <typename T, typename Index, template <typename> class PoolType,
          typename Direction, template <typename> class MemObj>
SNNStatus launch_with_index(MemObj<T const>& inp_data,
                            MemObj<T const>& outp_data,
                            MemObj<T const>& inp_backprop,
                            MemObj<T>& outp_backprop, const PoolingParams& pp,
                            size_t threads, cl::sycl::queue& queue,
                            const std::vector<cl::sycl::event>& events) {
  if (can_vectorize<Direction, PoolType>(pp, 4)) {
    return launch_with_vector_size<T, Index, PoolType, Direction, 4>(
        inp_data, outp_data, inp_backprop, outp_backprop, pp, threads, queue,
        events);
  } else if (can_vectorize<Direction, PoolType>(pp, 2)) {
    return launch_with_vector_size<T, Index, PoolType, Direction, 2>(
        inp_data, outp_data, inp_backprop, outp_backprop, pp, threads, queue,
        events);
  } else {
    return launch_with_vector_size<T, Index, PoolType, Direction, 1>(
        inp_data, outp_data, inp_backprop, outp_backprop, pp, threads, queue,
        events);
  }
}

template <typename T, template <typename> class PoolType, typename Direction,
          template <typename> class MemObj,
          EnableIfMaxGradient<T, PoolType, Direction>>
SNNStatus launch_pooling(MemObj<T const>& inp_data, MemObj<T const>& outp_data,
                         MemObj<T const>& inp_backprop,
                         MemObj<T>& outp_backprop, const PoolingParams& pp,
                         cl::sycl::queue& queue,
                         const std::vector<cl::sycl::event>& events) {
  auto sizes = get_sizes<Direction>(pp);
  size_t threads = sizes.output_size;
  if (threads > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
#ifdef SNN_USE_INT64
    return launch_with_index<T, int64_t, PoolType, Direction>(
        inp_data, outp_data, inp_backprop, outp_backprop, pp, threads, queue,
        events);
#else
    return StatusCode::IndexExceeded;
#endif  // SNN_USE_INT64
  } else {
    return launch_with_index<T, int32_t, PoolType, Direction>(
        inp_data, outp_data, inp_backprop, outp_backprop, pp, threads, queue,
        events);
  }
}

#define INSTANTIATE_LAUNCH(DTYPE, OP, DIRECTION, MEM_OBJ)                      \
  template SNN_EXPORT SNNStatus launch_pooling<DTYPE, OP, DIRECTION, MEM_OBJ>( \
      MEM_OBJ<DTYPE const> & input_data, MEM_OBJ<DTYPE const> & output_data,   \
      MEM_OBJ<DTYPE const> & input_backprop, MEM_OBJ<DTYPE> & outp_backprop,   \
      const PoolingParams& pp, cl::sycl::queue& queue,                         \
      const std::vector<cl::sycl::event>& events)

#define INSTANTIATE_FOR_TYPE(DTYPE, MEM_OBJ)              \
  INSTANTIATE_LAUNCH(DTYPE, Max, Backpropagate, MEM_OBJ); \
  INSTANTIATE_LAUNCH(DTYPE, MaxWithNan, Backpropagate, MEM_OBJ)

INSTANTIATE_FOR_TYPE(float, BufferMemObject);
#ifdef SNN_ENABLE_USM
INSTANTIATE_FOR_TYPE(float, USMMemObject);
#endif

#ifdef SNN_USE_HALF
INSTANTIATE_FOR_TYPE(cl::sycl::half, BufferMemObject);
#ifdef SNN_ENABLE_USM
INSTANTIATE_FOR_TYPE(cl::sycl::half, USMMemObject);
#endif  // SNN_ENABLE_USM
#endif  // SNN_USE_HALF

#ifdef SNN_USE_DOUBLE
INSTANTIATE_FOR_TYPE(double, BufferMemObject);
#ifdef SNN_ENABLE_USM
INSTANTIATE_FOR_TYPE(double, USMMemObject);
#endif  // SNN_ENABLE_USM
#endif  // SNN_USE_DOUBLE

#undef INSTANTIATE_FOR_TYPE
#undef INSTANTIATE_LAUNCH

}  // namespace internal
}  // namespace pooling
}  // namespace sycldnn
