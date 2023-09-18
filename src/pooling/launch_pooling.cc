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

#include "portdnn/data_format.h"
#include "portdnn/mem_object.h"
#include "portdnn/status.h"

#include "portdnn/pooling/operators.h"
#include "portdnn/pooling/params.h"
#include "portdnn/pooling/sizes.h"

#include "portdnn/helpers/macros.h"

#include "portdnn/internal/pooling/launch_internal.h"

#include "src/pooling/can_fastdiv.h"
#include "src/pooling/can_vectorize.h"
#include "src/pooling/kernels.h"
#include "src/pooling/queue_pooling_kernel.h"

#include <CL/sycl.hpp>

#include <type_traits>

#include "portdnn/export.h"

namespace sycldnn {
namespace pooling {
namespace internal {

/**
 * \brief The helper ensures that only the instantiated symbols are used.
 */
template <typename T, typename Index, template <typename> class PoolType,
          typename Direction, int VectorWidth, bool UseFastDiv, typename Format,
          template <typename> class MemObj>
struct queue_pooling_helper {
  SNNStatus operator()(MemObj<T const>&, MemObj<T>&, const PoolingParams&,
                       size_t, cl::sycl::queue&,
                       const std::vector<cl::sycl::event>&) {
    return StatusCode::InvalidAlgorithm;
  }
};

template <typename T, typename Index, template <typename> class PoolType,
          typename Direction, int VectorWidth, bool UseFastDiv,
          template <typename> class MemObj>
struct queue_pooling_helper<T, Index, PoolType, Direction, VectorWidth,
                            UseFastDiv, layout::NHWC, MemObj> {
  SNNStatus operator()(MemObj<T const>& input, MemObj<T>& output,
                       const PoolingParams& pp, size_t threads,
                       cl::sycl::queue& queue,
                       const std::vector<cl::sycl::event>& events) {
    return queue_pooling<T, Index, PoolType, Direction, VectorWidth, UseFastDiv,
                         layout::NHWC, MemObj>(input, output, pp, threads,
                                               queue, events);
  }
};

#ifdef SNN_ENABLE_NCHW
template <typename T, typename Index, template <typename> class PoolType,
          bool UseFastDiv, template <typename> class MemObj>
struct queue_pooling_helper<T, Index, PoolType, Forward, 1, UseFastDiv,
                            layout::NCHW, MemObj> {
  SNNStatus operator()(MemObj<T const>& input, MemObj<T>& output,
                       const PoolingParams& pp, size_t threads,
                       cl::sycl::queue& queue,
                       const std::vector<cl::sycl::event>& events) {
    return queue_pooling<T, Index, PoolType, Forward, /*VectorWidth=*/1,
                         UseFastDiv, layout::NCHW, MemObj>(
        input, output, pp, threads, queue, events);
  }
};
#endif

template <typename T, typename Index, template <typename> class PoolType,
          typename Direction, int VectorWidth, bool UseFastDiv,
          template <typename> class MemObj>
SNNStatus launch_with_fastdiv(MemObj<T const>& input, MemObj<T>& output,
                              const PoolingParams& pp, size_t threads,
                              cl::sycl::queue& queue,
                              const std::vector<cl::sycl::event>& events) {
  if (DataFormat::NHWC == pp.input_format) {
    return queue_pooling_helper<T, Index, PoolType, Direction, VectorWidth,
                                UseFastDiv, layout::NHWC, MemObj>{}(
        input, output, pp, threads, queue, events);
  } else if (DataFormat::NCHW == pp.input_format) {
    SNN_ASSERT((std::is_same<Direction, Forward>::value),
               "Must have forward-only NCHW pooling");
    return queue_pooling_helper<T, Index, PoolType, Direction, VectorWidth,
                                UseFastDiv, layout::NCHW, MemObj>{}(
        input, output, pp, threads, queue, events);
  } else {
    return StatusCode::InvalidAlgorithm;
  }
}

template <typename T, typename Index, template <typename> class PoolType,
          typename Direction, int VectorWidth, template <typename> class MemObj>
SNNStatus launch_with_vector_size(MemObj<T const>& input, MemObj<T>& output,
                                  const PoolingParams& pp, size_t threads,
                                  cl::sycl::queue& queue,
                                  const std::vector<cl::sycl::event>& events) {
  threads /= VectorWidth;
  if (can_use_fastdiv<Direction>(pp, VectorWidth)) {
    return launch_with_fastdiv<T, Index, PoolType, Direction, VectorWidth,
                               true>(input, output, pp, threads, queue, events);
  } else {
    return launch_with_fastdiv<T, Index, PoolType, Direction, VectorWidth,
                               false>(input, output, pp, threads, queue,
                                      events);
  }
}

template <typename T, typename Index, template <typename> class PoolType,
          typename Direction, template <typename> class MemObj>
SNNStatus launch_with_index(MemObj<T const>& input, MemObj<T>& output,
                            const PoolingParams& pp, size_t threads,
                            cl::sycl::queue& queue,
                            const std::vector<cl::sycl::event>& events) {
  if (can_vectorize<Direction, PoolType>(pp, 4)) {
    return launch_with_vector_size<T, Index, PoolType, Direction, 4>(
        input, output, pp, threads, queue, events);
  } else if (can_vectorize<Direction, PoolType>(pp, 2)) {
    return launch_with_vector_size<T, Index, PoolType, Direction, 2>(
        input, output, pp, threads, queue, events);
  } else {
    return launch_with_vector_size<T, Index, PoolType, Direction, 1>(
        input, output, pp, threads, queue, events);
  }
}

template <typename T, template <typename> class PoolType, typename Direction,
          template <typename> class MemObj,
          DisableIfMaxGradient<T, PoolType, Direction>>
SNNStatus launch_pooling(MemObj<T const>& input, MemObj<T>& output,
                         const PoolingParams& pp, cl::sycl::queue& queue,
                         const std::vector<cl::sycl::event>& events) {
  auto sizes = get_sizes<Direction>(pp);
  size_t threads = sizes.output_size;
  if (threads > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
#ifdef SNN_USE_INT64
    return launch_with_index<T, int64_t, PoolType, Direction>(
        input, output, pp, threads, queue, events);
#else
    return StatusCode::IndexExceeded;
#endif  // SNN_USE_INT64
  } else {
    return launch_with_index<T, int32_t, PoolType, Direction>(
        input, output, pp, threads, queue, events);
  }
}

#define INSTANTIATE_LAUNCH(DTYPE, OP, DIRECTION, MEM_OBJ)                      \
  template SNN_EXPORT SNNStatus launch_pooling<DTYPE, OP, DIRECTION, MEM_OBJ>( \
      MEM_OBJ<DTYPE const> & inp_access, MEM_OBJ<DTYPE> & outp_access,         \
      const PoolingParams& pp, cl::sycl::queue& queue,                         \
      const std::vector<cl::sycl::event>& events)

#define INSTANTIATE_FOR_TYPE(DTYPE, MEM_OBJ)               \
  INSTANTIATE_LAUNCH(DTYPE, Max, Forward, MEM_OBJ);        \
  INSTANTIATE_LAUNCH(DTYPE, MaxWithNan, Forward, MEM_OBJ); \
  INSTANTIATE_LAUNCH(DTYPE, Average, Forward, MEM_OBJ);    \
  INSTANTIATE_LAUNCH(DTYPE, Average, Backpropagate, MEM_OBJ)

INSTANTIATE_FOR_TYPE(float, BufferMemObject);
#ifdef SNN_ENABLE_USM
INSTANTIATE_FOR_TYPE(float, USMMemObject);
#endif  // SNN_ENABLE_USM

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
