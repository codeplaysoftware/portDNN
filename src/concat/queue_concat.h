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
#ifndef SYCLDNN_SRC_CONCAT_QUEUE_CONCAT_H_
#define SYCLDNN_SRC_CONCAT_QUEUE_CONCAT_H_

#include <CL/sycl.hpp>

#include "sycldnn/status.h"

#include "sycldnn/concat/params.h"
#include "sycldnn/concat/tarray.h"

namespace sycldnn {
namespace concat {
namespace internal {

/**
 * Submit a concat operation to a SYCL queue.
 */
template <typename T, int VectorWidth, typename Backend, int TArrayMaxSize,
          template <typename> class MemObj>
SNNStatus queue_concat(TArray<T, Backend, TArrayMaxSize>& input_tarray,
                       std::array<int, TArrayMaxSize> chunk_sizes,
                       std::array<int, TArrayMaxSize> input_offsets,
                       std::array<int, TArrayMaxSize> flat_sizes,
                       const int chunk_offset, MemObj<T>& out_mem,
                       cl::sycl::queue& queue,
                       const std::vector<cl::sycl::event>& events);

}  // namespace internal
}  // namespace concat
}  // namespace sycldnn

#endif