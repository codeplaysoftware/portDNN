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
#ifndef SYCLDNN_INCLUDE_CONCAT_TARRAY_H
#define SYCLDNN_INCLUDE_CONCAT_TARRAY_H

#include <array>
#include <vector>

#include "src/backend/backend_provider.h"
#include "sycldnn/helpers/macros.h"

#include "sycldnn/backend/snn_backend.h"
#include "sycldnn/backend/snn_usm_backend.h"

#include "sycldnn/concat/params.h"

/**
 * \file
 * Contains the definition of the \ref sycldnn::concat::TArray struct, which is
 * used to store the input tensors for the concat operation.
 *
 * TArray exists to solve the problem of concat requiring a dynamic list of
 * input tensors. An array of MemObjects is allocated upfront with a templated
 * size known at compile time. TArray then uses a runtime attribute to track the
 * actual number of tensors in use.
 */
namespace sycldnn {
namespace concat {

/**
 * Struct that contains the array of MemObjects used in a concat op.
 * Template specializes on the backend type.
 */
template <typename T, typename Backend, int MaxCapacity>
struct TArrayData;

/**
 * Struct that contains the array of BufferMemObjects used in a concat op with
 * the SNNBackend.
 */
template <typename T, int MaxCapacity>
struct TArrayData<T, sycldnn::backend::SNNBackend, MaxCapacity> {
 private:
  std::array<sycldnn::BufferMemObject<T const>, MaxCapacity> data;

 public:
  TArrayData() : data() {}

  /**
   * \return a pointer to the array of MemObjects.
   */
  sycldnn::BufferMemObject<T const>* get_data() { return data.data(); }
};

/**
 * Struct that contains the array of USMMemObjects used in a concat op with the
 * SNNUSMBackend.
 */
template <typename T, int MaxCapacity>
struct TArrayData<T, sycldnn::backend::SNNUSMBackend, MaxCapacity> {
 private:
  std::array<sycldnn::USMMemObject<T const>, MaxCapacity> data;

 public:
  TArrayData() : data() {}

  /**
   * \return a pointer to the array of MemObjects.
   */
  sycldnn::USMMemObject<T const>* get_data() { return data.data(); }
};

/**
 * TArray struct
 *
 * \tparam T              The data type of the input tensors.
 * \tparam Backend        The type of the Backend.
 * \tparam MaxCapacity    The maximum number of input tensors that can be stored
 * in a single TArray
 */
template <typename T, typename Backend, int MaxCapacity = 6>
struct TArray {
 private:
  using DeviceMem = typename Backend::template pointer_type<T const>;
  size_t size = 0;
  TArrayData<T, Backend, MaxCapacity> data;

 public:
  /**
   * Default constructor.
   */
  TArray() : data() {}

  /**
   * Construct a TArray from a vector of pointers to device memory.
   *
   * \param input   A vector of pointers to device memory.
   * \param params  The parameters of the concat operation.
   * \param backend The SYCL backend to use.
   */
  TArray(const std::vector<DeviceMem>& input, ConcatParams& params,
         Backend& backend) {
    set_data(input, params, backend);
  }

  /**
   * Load a vector of pointers to device memory into the TArray.
   *
   * \param input   A vector of pointers to device memory.
   * \param params  The parameters of the concat operation.
   * \param backend The SYCL backend to use.
   */
  void set_data(std::vector<DeviceMem>& input, ConcatParams& params,
                Backend& backend) {
    size = input.size();
    for (size_t i = 0; i < input.size(); i++) {
      data.get_data()[i] = backend.get_mem_object(
          input[i], std::accumulate(params.input_descriptors[i].begin(),
                                    params.input_descriptors[i].end(), 1,
                                    std::multiplies<int>()));
    }
  }

  /**
   * \return the number of tensors stored in the TArray.
   */
  int get_size() { return size; }

  /**
   * \return the maximum number of tensors that can be stored in the TArray.
   */
  int get_max_size() { return MaxCapacity; }

  /**
   * \return a pointer to the array of device memory
   */
  auto get_data() -> decltype(data.get_data()) { return data.get_data(); }
};

}  // namespace concat
}  // namespace sycldnn

#endif
