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
#ifndef PORTDNN_INCLUDE_BACKEND_EIGEN_MATMUL_PROVIDER_H_
#define PORTDNN_INCLUDE_BACKEND_EIGEN_MATMUL_PROVIDER_H_

/**
 * \file
 * Contains the implementation of \ref sycldnn::backend::EigenMatmulProvider,
 * which provides single and batch matrix multiply implementations using Eigen.
 */

#include "portdnn/backend/backend_traits.h"
#include "portdnn/backend/crtp_backend.h"
#include "portdnn/batch_format.h"

namespace sycldnn {
namespace backend {

/**
 * Handler struct to provide matmul and batch_matmul implementations using
 * Eigen.
 *
 * This expects the Eigen Tensor module to have already been included. We don't
 * explicitly include it in this file so that the user has control of how Eigen
 * is included and which files are actually needed.
 */
template <typename EigenBackend>
struct EigenMatmulProvider
    : public CRTPBackend<EigenBackend, EigenMatmulProvider> {
  /**
   * Compute a single matrix multiply using Eigen.
   *
   * Perform the matrix multiply operation:
   * \code
   *   output = lhs * rhs + alpha * output
   * \endcode
   * where lhs is a [m x k] matrix, rhs is a [k x n] matrix. The `bool`
   * template parameters determine whether or not to transpose the matrices.
   * The matrices provided here are assumed to be in row-major ordering.
   *
   * \param [in]     lhs    Pointer to a buffer containing the LHS matrix.
   * \param [in]     rhs    Pointer to a buffer containing the RHS matrix.
   * \param [in,out] output Pointer to a buffer containing the output matrix.
   * \param [in]     alpha  Scale multiplier for the output matrix.
   * \param [in]     m      Number of rows in the LHS matrix.
   * \param [in]     k      Number of columns in the LHS matrix and rows in the
   *                        RHS matrix.
   * \param [in]     n      Number of columns in the RHS matrix.
   *
   * \return A SYCL event corresponding to the matmul kernel launch.
   */
  template <bool TransposeLHS, bool TransposeRHS, typename T, typename Index>
  cl::sycl::event matmul(T const* const lhs, T const* const rhs,
                         T* const output, T const alpha, Index const m,
                         Index const k, Index const n,
                         const std::vector<cl::sycl::event>& = {}) {
    static constexpr auto lhs_dim = TransposeLHS ? 0 : 1;
    static constexpr auto rhs_dim = TransposeRHS ? 1 : 0;
    using ConstTensorType = Eigen::Tensor<T const, 2, Eigen::RowMajor, Index>;
    using ConstTensor = Eigen::TensorMap<ConstTensorType, Eigen::Aligned>;
    using TensorType = Eigen::Tensor<T, 2, Eigen::RowMajor, Index>;
    using Tensor = Eigen::TensorMap<TensorType, Eigen::Aligned>;
    using TensorShape = Eigen::DSizes<Index, 2>;
    using ContractDims =
        Eigen::IndexPairList<Eigen::type2indexpair<lhs_dim, rhs_dim>>;

    auto eigen_device = this->underlying_backend().get_eigen_device();

    TensorShape const lhs_shape{TransposeLHS ? k : m, TransposeLHS ? m : k};
    TensorShape const rhs_shape{TransposeRHS ? n : k, TransposeRHS ? k : n};
    TensorShape const out_shape{m, n};

    ConstTensor lhs_tensor{lhs, lhs_shape};
    ConstTensor rhs_tensor{rhs, rhs_shape};
    Tensor out_tensor{output, out_shape};

    if (alpha == static_cast<T>(0)) {
      out_tensor.device(eigen_device) =
          lhs_tensor.contract(rhs_tensor, ContractDims{});
    } else {
      out_tensor.device(eigen_device) =
          alpha * out_tensor + lhs_tensor.contract(rhs_tensor, ContractDims{});
    }
    // Eigen does not provide a way to access the SYCL event from kernels.
    return cl::sycl::event{};
  }

  /**
   * Compute a batch of matrix multiplies.
   *
   * Perform the batched matrix multiply operation:
   * \code
   *   output[i] = lhs[i] * rhs[i]
   * \endcode
   * for 0 <= i < batch, where lhs is a [batch x m x k] tensor and rhs is a
   * [batch x k x n] tensor. Each matrix is assumed to be contiguous in memory
   * and in row-major format. The `bool` template parameters determine whether
   * or not to transpose the matrices.
   * As Eigen Tensor does not have a batch matrix multiply, just fall back to
   * multiple calls to the standard matrix multiply.
   *
   * \param [in]     lhs        Pointer to a buffer containing the LHS matrix.
   * \param [in]     rhs        Pointer to a buffer containing the RHS matrix.
   * \param [in,out] output     Pointer to a buffer containing the output
   *                            matrix.
   * \param [in]     n_batches  Scale multiplier for the output matrix.
   * \param [in]     m          Number of rows in the LHS matrix.
   * \param [in]     k          Number of columns in the LHS matrix and rows in
   *                            the RHS matrix.
   * \param [in]     n          Number of columns in the RHS matrix.
   *                            will wait on before launching the kernels.
   * \param [in]     batch_type Format indicating how the batches are layed out.
   * \return A SYCL event corresponding to the matmul kernel launch.
   */
  template <bool TransposeLHS, bool TransposeRHS, typename T, typename Index>
  cl::sycl::event batch_matmul(
      T const* const lhs, T const* const rhs, T* const output,
      Index const n_batches, Index const m, Index const k, Index const n,
      sycldnn::BatchFormat const batch_type = sycldnn::BatchFormat::STRIDED,
      const std::vector<cl::sycl::event>& = {}) {
    if (batch_type != sycldnn::BatchFormat::STRIDED) {
      throw std::runtime_error(
          "Eigen batch matmul only supports strided batch format.");
    }
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
};

}  // namespace backend
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_BACKEND_EIGEN_MATMUL_HANDLER_H_
