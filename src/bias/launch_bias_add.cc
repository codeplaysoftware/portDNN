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
#include "sycldnn/mem_object.h"

#include "sycldnn/bias/params.h"
#include "sycldnn/bias/sizes.h"

#include "sycldnn/internal/bias/launch.h"

#include "src/bias/queue_bias_kernel.h"

#include <CL/sycl.hpp>

#include "sycldnn/export.h"

namespace sycldnn {
namespace bias {
namespace internal {

template <typename T>
SNNStatus launch_bias_add(BaseMemObject<T const>& input,
                          BaseMemObject<T const>& bias,
                          BaseMemObject<T>& output, const BiasParams& pp,
                          cl::sycl::queue& queue) {
  auto sizes = get_sizes(pp);
  size_t threads = sizes.output_size;
  if (threads > std::numeric_limits<int32_t>::max()) {
    SNNStatus index_too_large;
    index_too_large.status = StatusCode::IndexExceeded;
    return index_too_large;
  } else {
    if (sizes.bias_size % 8 == 0)
      return queue_bias_add<T, int32_t, 8>(input, bias, output, pp, threads,
                                           queue);
    else if (sizes.bias_size % 4 == 0)
      return queue_bias_add<T, int32_t, 4>(input, bias, output, pp, threads,
                                           queue);
    else if (sizes.bias_size % 2 == 0)
      return queue_bias_add<T, int32_t, 2>(input, bias, output, pp, threads,
                                           queue);
    else
      return queue_bias_add<T, int32_t, 1>(input, bias, output, pp, threads,
                                           queue);
  }
}

#define INSTANTIATE_LAUNCH(DTYPE)                               \
  template SNN_EXPORT SNNStatus launch_bias_add<DTYPE>(         \
      BaseMemObject<DTYPE const> & inp_access,                  \
      BaseMemObject<DTYPE const> & biasp_access,                \
      BaseMemObject<DTYPE> & outp_access, const BiasParams& pp, \
      cl::sycl::queue& queue)

#define INSTANTIATE_FOR_TYPE(DTYPE) INSTANTIATE_LAUNCH(DTYPE);

INSTANTIATE_FOR_TYPE(float);

#ifdef SNN_USE_HALF
INSTANTIATE_FOR_TYPE(cl::sycl::half);
#endif  // SNN_USE_HALF

#ifdef SNN_USE_DOUBLE
INSTANTIATE_FOR_TYPE(double);
#endif  // SNN_USE_DOUBLE

#undef INSTANTIATE_LAUNCH

}  // namespace internal
}  // namespace bias
}  // namespace sycldnn
