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

#include <CL/sycl.hpp>

#include "src/pointwise/launch_forward_pointwise.h"
#include "src/pointwise/launch_grad_pointwise.h"
#include "sycldnn/accessor_types.h"
#include "sycldnn/internal/pointwise/launch_internal.h"
#include "sycldnn/status.h"

namespace sycldnn {
namespace pointwise {
namespace internal {

SNN_INSTANTIATE_LAUNCH_POINTWISE_KERNEL(float, Relu, Forward)
#ifdef SNN_USE_HALF
SNN_INSTANTIATE_LAUNCH_POINTWISE_KERNEL(cl::sycl::half, Relu, Forward)
#endif  // SNN_USE_HALF
#ifdef SNN_USE_DOUBLE
SNN_INSTANTIATE_LAUNCH_POINTWISE_KERNEL(double, Relu, Forward)
#endif  // SNN_USE_DOUBLE

SNN_INSTANTIATE_LAUNCH_POINTWISE_GRADIENT_KERNEL(float, Relu)
#ifdef SNN_USE_HALF
SNN_INSTANTIATE_LAUNCH_POINTWISE_GRADIENT_KERNEL(cl::sycl::half, Relu)
#endif  // SNN_USE_HALF
#ifdef SNN_USE_DOUBLE
SNN_INSTANTIATE_LAUNCH_POINTWISE_GRADIENT_KERNEL(double, Relu)
#endif  // SNN_USE_DOUBLE

}  // namespace internal
}  // namespace pointwise
}  // namespace sycldnn
