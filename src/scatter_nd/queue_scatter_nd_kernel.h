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

#ifndef SYCLDNN_SRC_SCATTER_ND_QUEUE_SCATTER_ND_H
#define SYCLDNN_SRC_SCATTER_ND_QUEUE_SCATTER_ND_H

#include "sycldnn/scatter_nd/params.h"

#include <CL/sycl.hpp>
#include "sycldnn/status.h"

namespace sycldnn {
namespace scatter_nd {
namespace internal {

template <typename T, typename Index, typename ScatterNDType, int IndexDepth,
          int VectorWidth>
SNNStatus queue_scatter_nd(BaseMemObject<Index const>& ind_mem,
                           BaseMemObject<T const>& upd_mem,
                           BaseMemObject<T>& out_mem,
                           ScatterNDSizes const& sizes, cl::sycl::queue& queue);
}  // namespace internal
}  // namespace scatter_nd
}  // namespace sycldnn

#endif  // SYCLDNN_SRC_SCATTER_ND_QUEUE_SCATTER_ND_H
