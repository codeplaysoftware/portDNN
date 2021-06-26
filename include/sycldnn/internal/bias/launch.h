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

#ifndef SYCLDNN_INCLUDE_BIAS_LAUNCH_INTERNAL_H_
#define SYCLDNN_INCLUDE_BIAS_LAUNCH_INTERNAL_H_

#include "sycldnn/mem_object.h"
#include "sycldnn/status.h"

#include "sycldnn/bias/params.h"

#include "sycldnn/export.h"

namespace sycldnn {
namespace bias {
namespace internal {

template <typename T>
SNN_EXPORT SNNStatus launch_bias_add(BaseMemObject<T const>& inp_data,
                                     BaseMemObject<T const>& bias_data,
                                     BaseMemObject<T>& outp_data,
                                     const BiasParams& pp,
                                     cl::sycl::queue& queue);

}  // namespace internal
}  // namespace bias
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_BIAS_LAUNCH_INTERNAL_H_
