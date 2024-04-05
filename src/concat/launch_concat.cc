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
#include <CL/sycl.hpp>

#include "sycldnn/backend/snn_backend.h"
#include "sycldnn/backend/snn_usm_backend.h"
#include "sycldnn/mem_object.h"

#include "sycldnn/export.h"

#include "src/concat/queue_concat.h"
#include "sycldnn/internal/concat/launch_internal.h"

#include "sycldnn/concat/params.h"
#include "sycldnn/concat/tarray.h"

#include <algorithm>

namespace sycldnn {
namespace concat {
namespace internal {

template <int TArrayMaxSize>
bool can_vectorize(std::array<int, TArrayMaxSize>& sizes, const int num_tensors,
                   const int vec_size) {
  return std::all_of(sizes.begin(), sizes.begin() + num_tensors,
                     [vec_size](int val) { return val % vec_size == 0; });
}

template <typename T, typename Backend, int TArrayMaxSize,
          template <typename> class MemObj>
SNNStatus launch(TArray<T, Backend, TArrayMaxSize>& input_tarray,
                 std::array<int, TArrayMaxSize> chunk_sizes,
                 std::array<int, TArrayMaxSize> input_offsets,
                 std::array<int, TArrayMaxSize> flat_sizes,
                 const int chunk_offset, MemObj<T>& output,
                 cl::sycl::queue& queue,
                 const std::vector<cl::sycl::event>& event) {
  if (can_vectorize<TArrayMaxSize>(flat_sizes, input_tarray.get_size(), 4)) {
    return queue_concat<T, 4, Backend, TArrayMaxSize>(
        input_tarray, chunk_sizes, input_offsets, flat_sizes, chunk_offset,
        output, queue, event);
  } else if (can_vectorize<TArrayMaxSize>(flat_sizes, input_tarray.get_size(),
                                          2)) {
    return queue_concat<T, 2, Backend, TArrayMaxSize>(
        input_tarray, chunk_sizes, input_offsets, flat_sizes, chunk_offset,
        output, queue, event);
  } else {
    return queue_concat<T, 1, Backend, TArrayMaxSize>(
        input_tarray, chunk_sizes, input_offsets, flat_sizes, chunk_offset,
        output, queue, event);
  }
}

#define SNN_INSTANTIATE_LAUNCH_CONCAT_KERNEL(DTYPE, MEMOBJ, BACKEND,     \
                                             TARRAYMAXSIZE)              \
  template SNN_EXPORT SNNStatus launch<DTYPE, BACKEND, TARRAYMAXSIZE>(   \
      TArray<DTYPE, BACKEND, TARRAYMAXSIZE> & input_tarray,              \
      std::array<int, TARRAYMAXSIZE> chunk_sizes,                        \
      std::array<int, TARRAYMAXSIZE> input_offsets,                      \
      std::array<int, TARRAYMAXSIZE> flat_sizes, const int chunk_offset, \
      MEMOBJ<DTYPE>& outp_access, cl::sycl::queue& queue,                \
      const std::vector<cl::sycl::event>& events);

#ifdef SNN_ENABLE_USM
SNN_INSTANTIATE_LAUNCH_CONCAT_KERNEL(float, USMMemObject,
                                     sycldnn::backend::SNNUSMBackend, 6)
#endif
SNN_INSTANTIATE_LAUNCH_CONCAT_KERNEL(float, BufferMemObject,
                                     sycldnn::backend::SNNBackend, 6)

#ifdef SNN_USE_HALF
#ifdef SNN_ENABLE_USM
SNN_INSTANTIATE_LAUNCH_CONCAT_KERNEL(cl::sycl::half, USMMemObject,
                                     sycldnn::backend::SNNUSMBackend, 6)
#endif
SNN_INSTANTIATE_LAUNCH_CONCAT_KERNEL(cl::sycl::half, BufferMemObject,
                                     sycldnn::backend::SNNBackend, 6)
#endif

#ifdef SNN_USE_DOUBLE
#ifdef SNN_ENABLE_USM
SNN_INSTANTIATE_LAUNCH_CONCAT_KERNEL(double, USMMemObject,
                                     sycldnn::backend::SNNUSMBackend, 6)
#endif
SNN_INSTANTIATE_LAUNCH_CONCAT_KERNEL(double, BufferMemObject,
                                     sycldnn::backend::SNNBackend, 6)
#endif

#undef SNN_INSTANTIATE_LAUNCH_CONCAT_KERNEL
#undef SNN_INSTANTIATE_ALL_LAUNCH_CONCAT
}  // namespace internal
}  // namespace concat
}  // namespace sycldnn
