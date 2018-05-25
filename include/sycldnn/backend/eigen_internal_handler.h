/*
 * Copyright 2018 Codeplay Software Ltd
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
#ifndef SYCLDNN_INCLUDE_BACKEND_EIGEN_INTERNAL_HANDLER_H_
#define SYCLDNN_INCLUDE_BACKEND_EIGEN_INTERNAL_HANDLER_H_

/**
 * \file
 * Contains the implementation of \ref sycldnn::backend::EigenInternalHandler,
 * which provides single and batch matrix multiply implementations using Eigen,
 * as well as internal tensor allocation and buffer fetching methods.
 */
#include <utility>

#include "sycldnn/helpers/macros.h"

namespace sycldnn {
namespace backend {
/**
 * Handler struct to provide matmul and batch_matmul implementations using
 * Eigen, as well as internal tensor allocations and buffer fetching methods.
 *
 * This expects the Eigen Tensor module to have already been included. We don't
 * explicitly include it in this file so that the user has control of how Eigen
 * is included and which files are actually needed.
 */
struct EigenInternalHandler {
  /** The pointer representation required by the internal handler. */
  template <typename T>
  using internal_pointer_type = T*;

  /**
   * Constructs an instance of \ref sycldnn::backend::EigenInternalHandler from
   * an instance of Eigen's SyclDevice.
   * \param device The Eigen::SyclDevice to construct the handler from.
   */
  EigenInternalHandler(Eigen::SyclDevice const& device) : device_(device) {}

  /**
   * Allocate a tensor to be used internally.
   * \param n_bytes The size of the allocation in bytes.
   * \return Returns a pointer to allocation, using the internal pointer
   *         representation.
   * */
  template <typename T>
  internal_pointer_type<T> allocate(size_t n_bytes) {
    return static_cast<internal_pointer_type<T>>(device_.allocate(n_bytes));
  }

  /**
   * Deallocate an internal tensor.
   * \param ptr A pointer to the allocation to deallocate.
   * \return void
   */
  template <typename T>
  void deallocate(internal_pointer_type<T> ptr) {
    device_.deallocate(ptr);
  }

  /**
   * Get a SYCL buffer from an internal pointer.
   * \param ptr The pointer for which to retrieve the corresponding SYCL buffer.
   * \return Returns a SYCL buffer corresponding to ptr.
   */
  template <typename T>
  auto get_buffer_internal(internal_pointer_type<T> ptr, size_t /*n_elems*/)
      -> decltype(
          std::declval<Eigen::SyclDevice>()
              .get_sycl_buffer(ptr)
              .template reinterpret<T>(std::declval<cl::sycl::range<1>>())) {
    // The deduced return type is required to ensure that the buffer type
    // matches the allocator used in the Eigen device. We cannot assume that
    // std::allocator is used.
    auto raw_buffer = device_.get_sycl_buffer(ptr);
    auto buffer_size = raw_buffer.get_size();
    SNN_ASSERT(buffer_size % sizeof(T) == 0,
               "Buffer size must be a multiple of sizeof(T)");
    auto cast_size = cl::sycl::range<1>{buffer_size / sizeof(T)};
    return raw_buffer.template reinterpret<T>(cast_size);
  }

  /**
   * Get the offset from an internal pointer. An internal pointer may be an
   * offset from some base address, where the base address corresponds to a
   * SYCL buffer, and the offset refers to some address internal to the SYCL
   * buffer. This function enables querying such an offset.
   * \param ptr The internal pointer to query the offset for.
   * \return Returns the offset from the buffer base address, in elements.
   */
  template <typename T>
  size_t get_offset_internal(internal_pointer_type<T> ptr) {
    return device_.get_offset(ptr) / sizeof(T);
  }

  /**
   * Make TensorMap objects out of the provided pointers and dimensions, then
   * use Tensor Contraction to compute the matrix multiply.
   */
  template <bool TransposeLHS, bool TransposeRHS, typename T, typename Index>
  cl::sycl::event matmul(T const* const lhs, T const* const rhs,
                         T* const output, T const alpha, Index const m,
                         Index const k, Index const n) {
    static constexpr auto lhs_dim = TransposeLHS ? 0 : 1;
    static constexpr auto rhs_dim = TransposeRHS ? 1 : 0;
    using ConstTensorType = Eigen::Tensor<T const, 2, Eigen::RowMajor, Index>;
    using ConstTensor = Eigen::TensorMap<ConstTensorType, Eigen::Aligned>;
    using TensorType = Eigen::Tensor<T, 2, Eigen::RowMajor, Index>;
    using Tensor = Eigen::TensorMap<TensorType, Eigen::Aligned>;
    using TensorShape = Eigen::DSizes<Index, 2>;
    using ContractDims =
        Eigen::IndexPairList<Eigen::type2indexpair<lhs_dim, rhs_dim>>;

    TensorShape const lhs_shape{TransposeLHS ? k : m, TransposeLHS ? m : k};
    TensorShape const rhs_shape{TransposeRHS ? n : k, TransposeRHS ? k : n};
    TensorShape const out_shape{m, n};

    ConstTensor lhs_tensor{lhs, lhs_shape};
    ConstTensor rhs_tensor{rhs, rhs_shape};
    Tensor out_tensor{output, out_shape};

    if (alpha == static_cast<T>(0)) {
      out_tensor.device(device_) =
          lhs_tensor.contract(rhs_tensor, ContractDims{});
    } else {
      out_tensor.device(device_) =
          alpha * out_tensor + lhs_tensor.contract(rhs_tensor, ContractDims{});
    }
    // Eigen does not provide a way to access the SYCL event from kernels.
    return cl::sycl::event{};
  }

  /**
   * As Eigen Tensor does not have a batch matrix multiply, just fall back to
   * multiple calls to the standard matrix multiply.
   */
  template <bool TransposeLHS, bool TransposeRHS, typename T, typename Index>
  cl::sycl::event batch_matmul(T const* const lhs, T const* const rhs,
                               T* const output, Index const n_batches,
                               Index const m, Index const k, Index const n) {
    Index const lhs_size = m * k;
    Index const rhs_size = k * n;
    Index const out_size = m * n;

    cl::sycl::event event;
    for (int i = 0; i < n_batches; ++i) {
      Index const lhs_offset = lhs_size * i;
      Index const rhs_offset = rhs_size * i;
      Index const out_offset = out_size * i;
      event = matmul<TransposeLHS, TransposeRHS>(
          lhs + lhs_offset, rhs + rhs_offset, output + out_offset,
          static_cast<T>(0), m, k, n);
    }
    return event;
  }

 private:
  Eigen::SyclDevice const& device_;
};
}  // namespace backend
}  // namespace sycldnn
#endif  // SYCLDNN_INCLUDE_BACKEND_EIGEN_MATMUL_HANDLER_H_
