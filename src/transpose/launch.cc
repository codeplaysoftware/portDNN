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
#include "sycldnn/internal/transpose/launch.h"

#include "sycldnn/mem_object.h"

#include "src/transpose/queue_kernel.h"

#include <iterator>
#include <vector>

#include "sycldnn/export.h"

namespace sycldnn {
namespace transpose {
namespace internal {

namespace {

template <typename T, typename Index, int N, template <typename> class MemObj,
          typename = std::enable_if<is_mem_obj_v<MemObj<T>, T>>>
struct Transposer {
  static SNNStatus transpose(MemObj<T const>& input, MemObj<T>& output,
                             std::vector<int> const& dimensions,
                             std::vector<int> const& permutation,
                             cl::sycl::queue& queue,
                             const std::vector<cl::sycl::event>& events) {
    return queue_kernel<T, Index, N>(input, output, dimensions, permutation,
                                     queue, events);
  }
};

template <typename T, typename Index>
struct Transposer<T, Index, 1, BufferMemObject> {
  // A 1D transpose can only possibly be an identity operation, so just copy
  // from the input to the output.
  static SNNStatus transpose(BufferMemObject<T const>& input_mem,
                             BufferMemObject<T>& output_mem,
                             std::vector<int> const& /*dimensions*/,
                             std::vector<int> const& /*permutation*/,
                             cl::sycl::queue& queue,
                             const std::vector<cl::sycl::event>& /*events*/) {
    auto event = queue.submit([&](cl::sycl::handler& cgh) {
      auto input = input_mem.read_accessor(cgh).get_accessor();
      auto output = output_mem.write_accessor(cgh).get_accessor();
      cgh.copy(input, output);
    });
    return {event, StatusCode::OK};
  }
};

#ifdef SNN_ENABLE_USM
template <typename T, typename Index>
struct Transposer<T, Index, 1, USMMemObject> {
  // A 1D transpose can only possibly be an identity operation, so just copy
  // from the input to the output.
  static SNNStatus transpose(USMMemObject<T const>& input_mem,
                             USMMemObject<T>& output_mem,
                             std::vector<int> const& dimensions,
                             std::vector<int> const& /*permutation*/,
                             cl::sycl::queue& queue,
                             const std::vector<cl::sycl::event>& events) {
    auto event = queue.submit([&](cl::sycl::handler& cgh) {
      cgh.depends_on(events);
      auto input = input_mem.read_mem(cgh).get_pointer();
      auto output = output_mem.write_mem(cgh).get_pointer();

      size_t count = 0;
      for (unsigned int i = 0; i < dimensions.size(); ++i)
        count += dimensions[i];

      // TODO: make copy when ComputeCpp implements it
      cgh.memcpy(static_cast<void*>(output), static_cast<const void*>(input),
                 count * sizeof(T));
    });
    return {event, StatusCode::OK};
  }
};
#endif  // SNN_ENABLE_USM

void merge_consecutive_indices(int index, std::vector<int>& dimensions,
                               std::vector<int>& permutation) {
  int permuted_index = permutation[index];
  int permuted_index_to_remove = permutation[index + 1];
  int removed_size = dimensions[permuted_index_to_remove];
  dimensions.erase(begin(dimensions) + permuted_index_to_remove);
  dimensions[permuted_index] *= removed_size;

  permutation.erase(begin(permutation) + index + 1);
  for (int& perm : permutation) {
    if (perm > permuted_index_to_remove) {
      perm -= 1;
    }
  }
}

// Two consecutive indices can be merged into one, as they will not be split up
// in the transpose.
//
// e.g. The two following transposes are equivalent:
// dim: [a, b, c, d]  perm: [3, 1, 2, 0]
// dim: [a, b * c, d] perm: [2, 1, 0]
void simplify_transpose(std::vector<int>& dimensions,
                        std::vector<int>& permutation) {
  bool changed = false;
  do {
    changed = false;
    for (int previous = -2, idx = 0, size = permutation.size(); idx < size;
         ++idx) {
      int this_perm = permutation[idx];
      if (previous + 1 == this_perm) {
        merge_consecutive_indices(idx - 1, dimensions, permutation);
        changed = true;
        break;
      }
      previous = this_perm;
    }
  } while (changed);
}

}  // namespace

template <typename T, template <typename> class MemObj>
SNNStatus launch_impl(MemObj<T const>& input, MemObj<T>& output,
                      std::vector<int> dimensions, std::vector<int> permutation,
                      cl::sycl::queue& queue,
                      const std::vector<cl::sycl::event>& events) {
  simplify_transpose(dimensions, permutation);
  switch (dimensions.size()) {
    case 6:
      return Transposer<T, int, 6, MemObj>::transpose(
          input, output, dimensions, permutation, queue, events);
    case 5:
      return Transposer<T, int, 5, MemObj>::transpose(
          input, output, dimensions, permutation, queue, events);
    case 4:
      return Transposer<T, int, 4, MemObj>::transpose(
          input, output, dimensions, permutation, queue, events);
    case 3:
      return Transposer<T, int, 3, MemObj>::transpose(
          input, output, dimensions, permutation, queue, events);
    case 2:
      return Transposer<T, int, 2, MemObj>::transpose(
          input, output, dimensions, permutation, queue, events);
    case 1:
      return Transposer<T, int, 1, MemObj>::transpose(
          input, output, dimensions, permutation, queue, events);
  }
  return StatusCode::InvalidAlgorithm;
}

#define INSTANTIATE_FOR_TYPE(DTYPE, MEM_OBJ)                     \
  template SNN_EXPORT SNNStatus launch_impl(                     \
      MEM_OBJ<DTYPE const>& input, MEM_OBJ<DTYPE>& output,       \
      std::vector<int> dimensions, std::vector<int> permutation, \
      cl::sycl::queue& backend, const std::vector<cl::sycl::event>& events)

#ifdef SNN_ENABLE_USM
INSTANTIATE_FOR_TYPE(uint8_t, USMMemObject);
INSTANTIATE_FOR_TYPE(uint16_t, USMMemObject);
INSTANTIATE_FOR_TYPE(uint32_t, USMMemObject);
INSTANTIATE_FOR_TYPE(uint64_t, USMMemObject);
#endif  // SNN_ENABLE_USM

INSTANTIATE_FOR_TYPE(uint8_t, BufferMemObject);
INSTANTIATE_FOR_TYPE(uint16_t, BufferMemObject);
INSTANTIATE_FOR_TYPE(uint32_t, BufferMemObject);
INSTANTIATE_FOR_TYPE(uint64_t, BufferMemObject);
#undef INSTANTIATE_FOR_TYPE

}  // namespace internal
}  // namespace transpose
}  // namespace sycldnn
