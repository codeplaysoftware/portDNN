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
#ifndef PORTDNN_INCLUDE_INTERNAL_TRANSPOSE_LAUNCH_H_
#define PORTDNN_INCLUDE_INTERNAL_TRANSPOSE_LAUNCH_H_

#include <CL/sycl.hpp>

#include "portdnn/helpers/sycl_language_helpers.h"
#include "portdnn/mem_object.h"
#include "portdnn/status.h"

#include <algorithm>
#include <numeric>
#include <vector>

#include "portdnn/export.h"

namespace sycldnn {

const std::vector<int> NHWC_TO_NCHW{0, 3, 1, 2};
const std::vector<int> NCHW_TO_NHWC{0, 2, 3, 1};

namespace transpose {
namespace internal {

/**
 * The internal tensor transpose launcher.
 *
 * Implemented in the compiled SYCL DNN library.
 */
template <typename T, template <typename> class MemObj>
SNN_EXPORT SNNStatus launch_impl(MemObj<T const>& input, MemObj<T>& output,
                                 std::vector<int> dimensions,
                                 std::vector<int> permutation,
                                 cl::sycl::queue& queue,
                                 const std::vector<cl::sycl::event>& events);

/**
 * Internal tensor transpose launcher that is able to cast tensor types to the
 * implemented types.
 */
template <typename SrcT, typename DstT, template <typename> class MemObj,
          typename = std::enable_if<is_mem_obj_v<MemObj<SrcT>, SrcT>>>
SNNStatus launch_cast(MemObj<SrcT const>& input, MemObj<SrcT>& output,
                      std::vector<int> dimensions, std::vector<int> permutation,
                      cl::sycl::queue& queue,
                      const std::vector<cl::sycl::event>& events) {
  if (std::is_same<SrcT, DstT>::value) {
    return launch_impl(input, output, dimensions, permutation, queue, events);
  }
  auto input_int_mem = input.template cast<DstT const>();
  auto output_int_mem = output.template cast<DstT>();

  return launch_impl(input_int_mem, output_int_mem, dimensions, permutation,
                     queue, events);
}

#define SNN_LAUNCH_CAST(DST_T, MEM_OBJ)                                       \
  template <typename T, typename std::enable_if<sizeof(T) == sizeof(DST_T),   \
                                                int>::type = 0>               \
  SNNStatus launch(MEM_OBJ<T const>& input, MEM_OBJ<T>& output,               \
                   std::vector<int> dimensions, std::vector<int> permutation, \
                   cl::sycl::queue& queue,                                    \
                   const std::vector<cl::sycl::event>& events) {              \
    return launch_cast<T, DST_T>(input, output, dimensions, permutation,      \
                                 queue, events);                              \
  }

SNN_LAUNCH_CAST(uint8_t, USMMemObject);
SNN_LAUNCH_CAST(uint16_t, USMMemObject);
SNN_LAUNCH_CAST(uint32_t, USMMemObject);
SNN_LAUNCH_CAST(uint64_t, USMMemObject);

SNN_LAUNCH_CAST(uint8_t, BufferMemObject);
SNN_LAUNCH_CAST(uint16_t, BufferMemObject);
SNN_LAUNCH_CAST(uint32_t, BufferMemObject);
SNN_LAUNCH_CAST(uint64_t, BufferMemObject);

#undef SNN_LAUNCH_CAST

/**
 * The internal tensor transpose sublauncher.
 * Performs checks, and creates memory objects.
 *
 * Implemented in the compiled SYCL DNN library.
 */
template <typename T, typename Backend>
SNNStatus sublaunch(typename Backend::template pointer_type<T const> input,
                    typename Backend::template pointer_type<T> output,
                    std::vector<int> const& dimensions,
                    std::vector<int> const& permutation, Backend& backend,
                    const std::vector<cl::sycl::event>& events) {
  auto n_dimensions = dimensions.size();
  SNN_VALIDATE_PARAM(n_dimensions > 0,
                     "The number of dimensions must be positive.");
  SNN_VALIDATE_PARAM(n_dimensions < 7,
                     "Only dimensions 6 and fewer are supported.");
  SNN_VALIDATE_PARAM(
      permutation.size() == static_cast<unsigned int>(n_dimensions),
      "The number of permutations entries must match the number "
      "of dimensions.");

  std::vector<bool> not_seen(n_dimensions, true);
  for (int value : permutation) {
    SNN_VALIDATE_PARAM(value >= 0 && value < static_cast<int>(n_dimensions),
                       "Each permutation value must index a dimension.");
    SNN_VALIDATE_PARAM(not_seen[value],
                       "Each permutation value must be distinct.");
    not_seen[value] = false;
  }

  size_t tensor_size =
      std::accumulate(begin(dimensions), end(dimensions),
                      static_cast<size_t>(1), std::multiplies<int>());
  SNN_VALIDATE_PARAM(tensor_size > 0, "Tensor size must be positive.");

  auto in_acc = backend.get_mem_object(input, tensor_size);
  auto out_acc = backend.get_mem_object(output, tensor_size);

  auto sycl_queue = backend.get_queue();

  return internal::launch<T>(in_acc, out_acc, dimensions, permutation,
                             sycl_queue, events);
}

}  // namespace internal
}  // namespace transpose
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_INTERNAL_TRANSPOSE_LAUNCH_H_
