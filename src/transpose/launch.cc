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

template <typename T, typename Index, int N>
struct Transposer {
  static SNNStatus transpose(BaseMemObject<T const>& input,
                             BaseMemObject<T>& output,
                             std::vector<int> const& dimensions,
                             std::vector<int> const& permutation,
                             cl::sycl::queue& queue) {
    return queue_kernel<T, Index, N>(input, output, dimensions, permutation,
                                     queue);
  }
};

template <typename T, typename Index>
struct Transposer<T, Index, 1> {
  // A 1D transpose can only possibly be an identity operation, so just copy
  // from the input to the output.
  static SNNStatus transpose(BaseMemObject<T const>& input_mem,
                             BaseMemObject<T>& output_mem,
                             std::vector<int> const& /*dimensions*/,
                             std::vector<int> const& /*permutation*/,
                             cl::sycl::queue& queue) {
    auto event = queue.submit([&](cl::sycl::handler& cgh) {
      auto input = input_mem.read_accessor(cgh).get_accessor();
      auto output = output_mem.write_accessor(cgh).get_accessor();
      cgh.copy(input, output);
    });
    return {event, StatusCode::OK};
  }
};

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

// Two consectutive indices can be merged into one, as they will not be split up
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

template <typename T>
SNNStatus launch(BaseMemObject<T const>& input, BaseMemObject<T>& output,
                 std::vector<int> dimensions, std::vector<int> permutation,
                 cl::sycl::queue& queue) {
  simplify_transpose(dimensions, permutation);
  switch (dimensions.size()) {
    case 6:
      return Transposer<T, int, 6>::transpose(input, output, dimensions,
                                              permutation, queue);
    case 5:
      return Transposer<T, int, 5>::transpose(input, output, dimensions,
                                              permutation, queue);
    case 4:
      return Transposer<T, int, 4>::transpose(input, output, dimensions,
                                              permutation, queue);
    case 3:
      return Transposer<T, int, 3>::transpose(input, output, dimensions,
                                              permutation, queue);
    case 2:
      return Transposer<T, int, 2>::transpose(input, output, dimensions,
                                              permutation, queue);
    case 1:
      return Transposer<T, int, 1>::transpose(input, output, dimensions,
                                              permutation, queue);
  }
  return {{}, StatusCode::InvalidAlgorithm};
}

#define INSTANTIATE_FOR_TYPE(DTYPE)                                    \
  template SNN_EXPORT SNNStatus launch(                                \
      BaseMemObject<DTYPE const>& input, BaseMemObject<DTYPE>& output, \
      std::vector<int> dimensions, std::vector<int> permutation,       \
      cl::sycl::queue& backend)

INSTANTIATE_FOR_TYPE(float);

#ifdef SNN_USE_DOUBLE
INSTANTIATE_FOR_TYPE(double);
#endif  // SNN_USE_DOUBLE

#ifdef SNN_USE_HALF
INSTANTIATE_FOR_TYPE(cl::sycl::half);
#endif  // SNN_USE_HALF

#undef INSTANTIATE_FOR_TYPE

}  // namespace internal
}  // namespace transpose
}  // namespace sycldnn
