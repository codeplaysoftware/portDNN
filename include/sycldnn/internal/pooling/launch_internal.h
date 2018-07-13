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

#ifndef SYCLDNN_INCLUDE_POOLING_LAUNCH_INTERNAL_H_
#define SYCLDNN_INCLUDE_POOLING_LAUNCH_INTERNAL_H_

#include "sycldnn/accessor_types.h"
#include "sycldnn/pooling/params.h"
#include "sycldnn/status.h"

namespace sycldnn {
namespace pooling {
namespace internal {

template <typename T, template <typename U> class PoolingType,
          typename Direction>
SNNStatus launch_pooling(ReadAccessor<T const> inp_access,
                         WriteAccessor<T> outp_access, const PoolingParams& pp,
                         cl::sycl::queue& queue);

}  // namespace internal
}  // namespace pooling
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_POOLING_LAUNCH_INTERNAL_H_
