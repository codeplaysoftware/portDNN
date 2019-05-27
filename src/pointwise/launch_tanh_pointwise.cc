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
#include "src/pointwise/launch_forward_pointwise.h"
#include "src/pointwise/launch_grad_pointwise.h"

#include <CL/sycl.hpp>

namespace sycldnn {
namespace pointwise {
namespace internal {

SNN_INSTANTIATE_LAUNCH_POINTWISE_KERNEL(float, Tanh, Forward)
#ifdef SNN_USE_HALF
SNN_INSTANTIATE_LAUNCH_POINTWISE_KERNEL(cl::sycl::half, Tanh, Forward)
#endif  // SNN_USE_HALF
#ifdef SNN_USE_DOUBLE
SNN_INSTANTIATE_LAUNCH_POINTWISE_KERNEL(double, Tanh, Forward)
#endif  // SNN_USE_DOUBLE

SNN_INSTANTIATE_LAUNCH_POINTWISE_GRADIENT_KERNEL(float, Tanh)
#ifdef SNN_USE_HALF
SNN_INSTANTIATE_LAUNCH_POINTWISE_GRADIENT_KERNEL(cl::sycl::half, Tanh)
#endif  // SNN_USE_HALF
#ifdef SNN_USE_DOUBLE
SNN_INSTANTIATE_LAUNCH_POINTWISE_GRADIENT_KERNEL(double, Tanh)
#endif  // SNN_USE_DOUBLE

}  // namespace internal
}  // namespace pointwise
}  // namespace sycldnn
