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

#ifndef SYCLDNN_SRC_SOFTMAX_QUEUE_SOFTMAX_H
#define SYCLDNN_SRC_SOFTMAX_QUEUE_SOFTMAX_H

#include "sycldnn/softmax/params.h"

#include <CL/sycl.hpp>
#include "sycldnn/status.h"

namespace sycldnn {
namespace softmax {
namespace internal {

template <typename T, typename Index, typename SoftmaxType, typename Backend,
          int VectorWidth>
SNNStatus queue_softmax(typename Backend::template pointer_type<T const>& input,
                        typename Backend::template pointer_type<T>& workspace,
                        typename Backend::template pointer_type<T>& output,
                        SoftmaxParams const& params, Backend& backend);
}  // namespace internal
}  // namespace softmax
}  // namespace sycldnn

#endif  // SYCLDNN_SRC_SOFTMAX_QUEUE_SOFTMAX_H