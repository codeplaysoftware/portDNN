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
#ifndef SYCLDNN_SRC_CONCAT_QUEUE_CONCAT_IMPL_H_
#define SYCLDNN_SRC_CONCAT_QUEUE_CONCAT_IMPL_H_

#include <CL/sycl.hpp>

#include "sycldnn/accessor_types.h"

#include "sycldnn/concat/params.h"
#include "sycldnn/concat/tarray.h"

#include "sycldnn/status.h"

#include "sycldnn/helpers/mem_utils.h"
#include "sycldnn/helpers/ratio.h"

#include "src/concat/kernels.h"

#include <numeric>

namespace sycldnn {
namespace concat {
namespace internal {

template <typename T, int VectorWidth, typename Backend, int TArrayMaxSize,
          template <typename> class MemObj>
SNNStatus queue_concat(TArray<T, Backend, TArrayMaxSize>& input_tarray,
                       std::array<int, TArrayMaxSize> chunk_sizes,
                       std::array<int, TArrayMaxSize> input_offsets,
                       std::array<int, TArrayMaxSize> flat_sizes,
                       const int chunk_offset, MemObj<T>& out_mem,
                       cl::sycl::queue& queue,
                       const std::vector<cl::sycl::event>& events) {
  constexpr bool is_usm = is_usm_obj_v<MemObj<T>, T>;

  const size_t num_tensors = input_tarray.get_size();

  // Copy meta data onto device
  // Each block of 3 values maps to a tensor:
  //   {chunk_size, input_offset, flat_size}
  constexpr auto size_of_metadata = 3;
  std::array<int, size_of_metadata * TArrayMaxSize> concat_meta_data;
  for (int i = 0; i < size_of_metadata * TArrayMaxSize; i += size_of_metadata) {
    concat_meta_data[i] = chunk_sizes[i / size_of_metadata];
    concat_meta_data[i + 1] = input_offsets[i / size_of_metadata];
    concat_meta_data[i + 2] = flat_sizes[i / size_of_metadata];
  }
  auto concat_meta_data_gpu = sycldnn::helpers::alloc_and_assign<int, is_usm>(
      size_of_metadata * TArrayMaxSize, concat_meta_data.data(), queue);
  auto concat_meta_data_mem = make_mem_object<int const>(
      concat_meta_data_gpu, size_of_metadata * TArrayMaxSize);

  auto event = queue.submit([&](cl::sycl::handler& cgh) {
    cgh.depends_on(events);

    std::array<sycldnn::ReadMem<T const, is_usm>, TArrayMaxSize>
        input_tensors_host;
    for (size_t i = 0; i < num_tensors; i++) {
      input_tensors_host[i] = input_tarray.get_data()[i].read_mem(cgh);
    }
    // Fill superfluous indexes in generic_data to prevent placeholder_access
    // runtime error
    for (int i = num_tensors; i < TArrayMaxSize; i++) {
      input_tensors_host[i] = input_tensors_host[num_tensors - 1];
    }

    auto concat_meta_data_read = concat_meta_data_mem.read_mem(cgh);

    auto largest_tensor_size =
        *std::max_element(flat_sizes.begin(), flat_sizes.begin() + num_tensors);

    auto output = out_mem.write_mem(cgh);

    cl::sycl::device device = queue.get_device();
    size_t const wg_size = std::sqrt(
        device.get_info<cl::sycl::info::device::max_work_group_size>());
    const size_t num_threads_x = num_tensors;
    const size_t num_threads_y = helpers::round_up_to_nearest_multiple(
        largest_tensor_size / VectorWidth, wg_size);

    ConcatOp<T, VectorWidth, TArrayMaxSize, is_usm> concat_op{
        input_tensors_host, output, chunk_offset, concat_meta_data_read};
    cgh.parallel_for(
        cl::sycl::nd_range<2>{
            cl::sycl::range<2>{num_threads_x, num_threads_y},
            cl::sycl::range<2>{num_threads_x,
                               std::min(num_threads_y, wg_size)}},
        concat_op);
  });

  return {event, StatusCode::OK};
}

}  // namespace internal
}  // namespace concat
}  // namespace sycldnn

#endif
