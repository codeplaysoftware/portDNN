/*
 * Copyright 2019 Codeplay Software Ltd
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
#ifndef SYCLDNN_INCLUDE_INTERNAL_TRANSPOSE_LAUNCH_H_
#define SYCLDNN_INCLUDE_INTERNAL_TRANSPOSE_LAUNCH_H_

#include <CL/sycl.hpp>

#include "sycldnn/mem_object.h"
#include "sycldnn/status.h"

#include <vector>

namespace sycldnn {
namespace transpose {
namespace internal {

/**
 * The internal tensor transpose launcher.
 *
 * Implemented in the compiled SYCL DNN library.
 */
template <typename T>
SNNStatus launch(BaseMemObject<T const>& input, BaseMemObject<T>& output,
                 std::vector<int> dimensions, std::vector<int> permutation,
                 cl::sycl::queue& queue);

}  // namespace internal
}  // namespace transpose
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_INTERNAL_TRANSPOSE_LAUNCH_H_
