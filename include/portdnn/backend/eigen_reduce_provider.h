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
#ifndef PORTDNN_INCLUDE_BACKEND_EIGEN_REDUCE_PROVIDER_H_
#define PORTDNN_INCLUDE_BACKEND_EIGEN_REDUCE_PROVIDER_H_

/**
 * \file
 * Contains the implementation of \ref sycldnn::backend::EigenReduceProvider,
 * which provides single and batch matrix multiply implementations using Eigen.
 */

#include "portdnn/backend/backend_traits.h"
#include "portdnn/backend/crtp_backend.h"
#include "portdnn/reduce/operators.h"

namespace sycldnn {
namespace backend {
namespace internal {

/**
 * \brief Functor to implement Eigen operators for each reduction operations.
 *
 * \tparam Op Reduction operation.
 */
template <typename Op>
struct reduce_helper;

/** Implement Eigen Add reduction. */
template <>
struct reduce_helper<reduce::Add> {
  /**
   * \brief Add reduction functor.
   *
   * \tparam InputTensor
   * \tparam OutputTensor
   * \param input_tensor
   * \param output_tensor
   * \param eigen_device
   */
  template <typename InputTensor, typename OutputTensor>
  void operator()(InputTensor input_tensor, OutputTensor output_tensor,
                  Eigen::SyclDevice eigen_device) {
    output_tensor.device(eigen_device) =
        input_tensor.sum(Eigen::type2index<1>());
  }
};

/** Implement Eigen Mean reduction. */
template <>
struct reduce_helper<reduce::Mean> {
  /**
   * \brief Mean reduction functor.
   *
   * \tparam InputTensor
   * \tparam OutputTensor
   * \param input_tensor
   * \param output_tensor
   * \param eigen_device
   */
  template <typename InputTensor, typename OutputTensor>
  void operator()(InputTensor input_tensor, OutputTensor output_tensor,
                  Eigen::SyclDevice eigen_device) {
    output_tensor.device(eigen_device) =
        input_tensor.mean(Eigen::type2index<1>());
  }
};
}  // namespace internal

/**
 * Handler struct to provide a reduce implementation using Eigen.
 *
 * This expects the Eigen Tensor module to have already been included. We don't
 * explicitly include it in this file so that the user has control of how Eigen
 * is included and which files are actually needed.
 */
template <typename EigenBackend>
struct EigenReduceProvider
    : public CRTPBackend<EigenBackend, EigenReduceProvider> {
  /**
   * Compute a reduction.
   *
   * Perform a reduction using Op on the outer axis from an input:
   * [batch, outer, inner].
   * \param [in]  input  Pointer to a buffer containing the input tensor.
   * \param [out] output Pointer to a buffer containing the output tensor.
   * \param [in]  batch  Batch size.
   * \param [in]  outer  Outer size.
   * \param [in]  inner  Inner size.
   * \return A SYCL event corresponding to the reduce kernel launch.
   */
  template <typename Op, typename T, typename Index>
  cl::sycl::event reduce(T const* const input, T* const output, Index batch,
                         Index outer, Index inner) {
    using InputTensorType = Eigen::Tensor<T const, 3, Eigen::RowMajor, Index>;
    using InputTensor = Eigen::TensorMap<InputTensorType, Eigen::Aligned>;
    using OutputTensorType = Eigen::Tensor<T, 2, Eigen::RowMajor, Index>;
    using OutputTensor = Eigen::TensorMap<OutputTensorType, Eigen::Aligned>;

    auto eigen_device = this->underlying_backend().get_eigen_device();

    const Eigen::DSizes<Index, 3> input_shape{batch, outer, inner};
    const Eigen::DSizes<Index, 2> output_shape{batch, inner};

    InputTensor input_tensor{input, input_shape};
    OutputTensor output_tensor{output, output_shape};
    internal::reduce_helper<Op>()(input_tensor, output_tensor, eigen_device);

    // Eigen does not provide a way to access the SYCL event from kernels.
    return cl::sycl::event{};
  }
};

}  // namespace backend
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_BACKEND_EIGEN_REDUCE_PROVIDER_H_
