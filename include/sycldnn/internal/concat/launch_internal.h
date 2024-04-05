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
#ifndef SYCLDNN_INCLUDE_CONCAT_LAUNCH_INTERNAL_H_
#define SYCLDNN_INCLUDE_CONCAT_LAUNCH_INTERNAL_H_

#include <array>
#include <numeric>
#include <vector>

#include "sycldnn/export.h"
#include "sycldnn/status.h"

#include "sycldnn/concat/params.h"
#include "sycldnn/concat/tarray.h"

namespace sycldnn {
namespace concat {

/**
 * Calculate the output dimensions of the concat operation.
 *
 * \param [in]  params  The parameters of the concat operation.
 *
 * \return A vector containing the output dimensions of the concat operation.
 */
std::vector<int> calculate_concat_output_dims(ConcatParams& params) {
  if (params.axis < 0) {
    params.axis += params.rank;
  }

  std::vector<int> result_dim = params.input_descriptors[0];
  result_dim[params.axis] = 0;
  // Sum the dimension along the concatenation axis for each input tensor
  std::for_each(
      params.input_descriptors.begin(), params.input_descriptors.end(),
      [&](auto desc) { result_dim[params.axis] += desc[params.axis]; });
  return result_dim;
}

namespace internal {

/**
 * Check that the dimensions of each input tensor are valid for a concat op.
 *
 * \param params The parameters of the concat operation.
 */
void check_dimensions(
    ConcatParams& params) {  // Make sure each vector of descriptors in params
                             // has the sam enumber of dims as rank
  SNN_ASSERT(params.rank > 0 && params.rank <= 4, "Rank is not in valid range");
  SNN_ASSERT(params.axis >= -params.rank && params.axis <= params.rank - 1,
             "Axis is not in valid range");

  for (size_t i = 0; i < params.input_descriptors.size(); i++) {
    SNN_ASSERT(params.rank == params.input_descriptors[i].size(),
               "The rank of concat operation should be equal to the input "
               "tensors size");
    for (int j = 0; j < params.rank; j++) {
      if (j != params.axis) {
        SNN_ASSERT(
            params.input_descriptors[i][j] == params.input_descriptors[0][j],
            "Input tensors should have the same dimensions on all but the "
            "concat axis");
      }
    }
  }
}

/**
 * Check input size and if there are enough descriptors for each input
 * tensor
 *
 * \tparam T The data type of the input and output tensors.
 * \tparam Backend The type of the backend
 *
 * \param input   A vector of pointers to device memory.
 * \param params  The parameters of the concat operation.
 */
template <typename T, typename Backend>
SNNStatus inline validate_params(
    std::vector<typename Backend::template pointer_type<T const>>& input,
    ConcatParams& params) {
  SNN_VALIDATE_PARAM(input.size() > 0,
                     "The number of input tensors must be positive");
  SNN_VALIDATE_PARAM(input.size() == params.input_descriptors.size(),
                     "Descriptors must be supplied for all input tensors");
  return StatusCode::OK;
}

/**
 * Calculate the size (i.e., number of elements) of the output tensor.
 *
 * \param params The parameters of the concat operation.
 *
 * \return Returns the size of the output tensor.
 */
size_t get_concat_output_size(ConcatParams& params) {
  size_t size = 0;
  for (auto desc : params.input_descriptors) {
    size +=
        std::accumulate(desc.begin(), desc.end(), 1, std::multiplies<int>());
  }
  return size;
}

/**
 * The internal concat launcher
 *
 * Implemented in the compiled SYCL DNN library.
 */
template <typename T, typename Backend, int TArrayMaxSize,
          template <typename> class MemObj>
SNN_EXPORT SNNStatus launch(TArray<T, Backend, TArrayMaxSize>& input_tarray,
                            std::array<int, TArrayMaxSize> chunk_sizes,
                            std::array<int, TArrayMaxSize> input_offsets,
                            std::array<int, TArrayMaxSize> flat_sizes,
                            const int chunk_offset, MemObj<T>& output,
                            cl::sycl::queue& queue,
                            const std::vector<cl::sycl::event>& event);

/**
 * Calculates data needed in the kernel for finding an element's position in the
 * output tensor
 *
 * \tparam T              The data type of the input tensors.
 * \tparam Backend        The type of the Backend.
 * \tparam TArrayMaxSize  The maximum number of input tensors that can be stored
 * in a single TArray. \tparam MemObj         The type of memory object used.
 *
 * \param input_tarray    The input tensor array
 * \param output          The output tensor
 * \param params          The parameters of the concat operation.
 * \param queue           The SYCL queue to submit the kernel to.
 * \param event           The SYCL events to wait on before starting the kernel.
 *
 * \return Returns an SNNStatus containing the SYCL event tied to the kernel
 * launches and a StatusCode enum showing if the launch was OK or whether it
 * encountered some problem.
 */
template <typename T, typename Backend, int TArrayMaxSize,
          template <typename> class MemObj>
SNNStatus launch_with_meta_data(TArray<T, Backend, TArrayMaxSize>& input_tarray,
                                MemObj<T>& output, ConcatParams& params,
                                cl::sycl::queue& queue,
                                const std::vector<cl::sycl::event>& event) {
  // chunk_sizes stores the size of a chunk across the concatenation axis for
  // each input tensor
  std::array<int, TArrayMaxSize> chunk_sizes;
  // input_offsets stores the offset of the first element of each input tensor
  // in the output tensor
  std::array<int, TArrayMaxSize> input_offsets;
  // flat_sizes stores the total number of elements in each input tensor
  std::array<int, TArrayMaxSize> flat_sizes;
  for (int i = 0; i < input_tarray.get_size(); i++) {
    auto& desc = params.input_descriptors[i];
    chunk_sizes[i] = std::accumulate(desc.begin() + params.axis, desc.end(), 1,
                                     std::multiplies<int>());
    input_offsets[i] = std::accumulate(
        chunk_sizes.begin(), chunk_sizes.begin() + i, 0, std::plus<int>());
    flat_sizes[i] =
        std::accumulate(desc.begin(), desc.end(), 1, std::multiplies<int>());
  }
  const int chunk_offset = std::accumulate(
      chunk_sizes.begin(), chunk_sizes.begin() + input_tarray.get_size(), 0,
      std::plus<int>());

  return internal::launch<T, Backend, TArrayMaxSize>(
      input_tarray, chunk_sizes, input_offsets, flat_sizes, chunk_offset,
      output, queue, event);
}

/**
 * Load input tensors into a TArray before launching concat kernel.
 * If the number of input tensors is greater than TArrayMaxSize, the first n
 * tensors will be concatenated, then appended to the remainder, which are then
 * recursively concatenated.
 *
 * \tparam T              The data type of the input tensors.
 * \tparam Backend        The type of the Backend.
 * \tparam TArrayMaxSize  The maximum number of input tensors that can be stored
 * in a single TArray.
 *
 * \param input           A vector of pointers to device memory.
 * \param output          A pointer to the memory representing the output
 * tensor. \param params          The parameters of the concat operation. \param
 * backend         The backend implementation, used to map between pointer
 * representations. \param events          A vector of events to wait on before
 * starting the concat operation.
 *
 * \return Returns an SNNStatus containing the SYCL event tied to the kernel
 * launches and a StatusCode enum showing if the launch was OK or whether it
 * encountered some problem.
 */
template <typename T, typename Backend, int TArrayMaxSize>
SNNStatus launch_with_tarray(
    std::vector<typename Backend::template pointer_type<T const>>& input,
    typename Backend::template pointer_type<T> output, ConcatParams& params,
    Backend& backend, const std::vector<cl::sycl::event>& events) {
  TArray<T, Backend, TArrayMaxSize> tarray_inst;
  auto queue = backend.get_queue();
  if (input.size() > TArrayMaxSize) {
    std::vector<typename Backend::template pointer_type<T const>> first_chunk(
        input.begin(), input.begin() + TArrayMaxSize);
    input.erase(input.begin(), input.begin() + TArrayMaxSize);

    ConcatParams first_chunk_params;
    first_chunk_params.axis = params.axis;
    first_chunk_params.rank = params.rank;
    std::for_each(params.input_descriptors.begin(),
                  params.input_descriptors.begin() + TArrayMaxSize,
                  [&](std::vector<int> x) {
                    first_chunk_params.input_descriptors.push_back(x);
                  });

    size_t first_chunk_output_size = get_concat_output_size(first_chunk_params);
    auto first_chunk_output =
        backend.template allocate<T>(first_chunk_output_size);

    auto queue = backend.get_queue();
    tarray_inst.set_data(first_chunk, first_chunk_params, backend);

    auto first_chunk_output_access =
        backend.get_mem_object(first_chunk_output, first_chunk_output_size);

    auto first_chunk_event = launch_with_meta_data<T, Backend, TArrayMaxSize>(
        tarray_inst, first_chunk_output_access, first_chunk_params, queue,
        events);
    first_chunk_event.event.wait_and_throw();

    if (first_chunk_event.status == StatusCode::OK) {
      input.insert(input.begin(), first_chunk_output);

      params.input_descriptors.erase(
          params.input_descriptors.begin(),
          params.input_descriptors.begin() + TArrayMaxSize);
      params.input_descriptors.insert(
          params.input_descriptors.begin(),
          calculate_concat_output_dims(first_chunk_params));

      return launch_with_tarray<T, Backend, TArrayMaxSize>(
          input, output, params, backend, events);
    } else {
      return first_chunk_event;
    }
  } else {
    tarray_inst.set_data(input, params, backend);

    auto outp_access =
        backend.get_mem_object(output, get_concat_output_size(params));

    return launch_with_meta_data<T, Backend, TArrayMaxSize>(
        tarray_inst, outp_access, params, queue, events);
  }
}

/**
 * Launch a concat operation.
 * Negative axes will be concatenated on axis + rank(input)
 *
 * \tparam T The data type of the input and output tensors.
 * \tparam Backend The type of the backend
 * \tparam TArrayMaxSize The maximum number of input tensors that can be stored
 * in a single TArray
 *
 * \param input   A vector of pointers to device memory.
 * \param output  A pointer to the memory representing the output tensor.
 * \param params  The parameters of the concat operation.
 * \param backend The backend implementation, used to map between pointer
 * representations. \param events  A vector of events to wait on before starting
 * the concat operation.
 *
 * \return Returns an SNNStatus containing the SYCL event tied to the kernel
 * launches and a StatusCode enum showing if the launch was OK or whether it
 * encountered some problem.
 */
template <typename T, typename Backend, int TArrayMaxSize>
SNNStatus sublaunch(
    std::vector<typename Backend::template pointer_type<T const>>& input,
    typename Backend::template pointer_type<T> output, ConcatParams& params,
    Backend& backend, const std::vector<cl::sycl::event>& events) {
  auto validation_status = internal::validate_params<T, Backend>(input, params);
  if (validation_status.status != StatusCode::OK) {
    return validation_status;
  }

  if (params.axis < 0) {
    params.axis += params.rank;
  }

  check_dimensions(params);

  return launch_with_tarray<T, Backend, TArrayMaxSize>(input, output, params,
                                                       backend, events);
}

}  // namespace internal
}  // namespace concat
}  // namespace sycldnn

#endif