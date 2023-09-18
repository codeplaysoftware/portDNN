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
#ifndef PORTDNN_INCLUDE_BACKEND_EIGEN_BACKEND_WITH_SNN_MATMUL_H_
#define PORTDNN_INCLUDE_BACKEND_EIGEN_BACKEND_WITH_SNN_MATMUL_H_

/**
 * \file
 * Contains the implementation of \ref sycldnn::backend::EigenBackendSNNMatmul,
 * which provides pointer handling through Eigen and matrix multiplies via
 * portDNN's internal matmul kernels.
 */
#include "portdnn/backend/common_backend.h"
#include "portdnn/backend/eigen_external_handler.h"
#include "portdnn/backend/eigen_internal_handler.h"
#include "portdnn/backend/eigen_pointer_to_eigen_pointer.h"
#include "portdnn/backend/snn_matmul_provider.h"
#include "portdnn/backend/snn_reduce_provider.h"

namespace sycldnn {
namespace backend {

// Forward declaration to allow the BackendTraits specialisation.
struct EigenBackendSNNMatmul;

/**
 * The template specialisation of \ref
 * sycldnn::backend::BackendTraits<EigenBackendSNNMatmul>.
 *
 * Provides the pointer types for the EigenBackendSNNMatmul.
 */
template <>
struct BackendTraits<EigenBackendSNNMatmul> {
  /**
   * The external pointer type for EigenBackendSNNMatmul.
   */
  template <typename T>
  using pointer_type = T*;

  /**
   * The internal pointer type for EigenBackendSNNMatmul.
   */
  template <typename T>
  using internal_pointer_type = T*;
};

/**
 * Eigen backend for portDNN.
 *
 * Provides pointer handling and matrix multiplies using Eigen.
 */
struct EigenBackendSNNMatmul final
    : public CommonBackend,
      public EigenExternalHandler<EigenBackendSNNMatmul>,
      public EigenToEigenPointer,
      public EigenInternalHandler<EigenBackendSNNMatmul>,
      public SNNReduceProvider<EigenBackendSNNMatmul>,
      public SNNMatmulProvider<EigenBackendSNNMatmul> {
  /** The pointer type used in interface of the EigenBackend. */
  template <typename T>
  using pointer_type =
      typename BackendTraits<EigenBackendSNNMatmul>::template pointer_type<T>;
  /** The internal pointer type used internally by the EigenBackend. */
  template <typename T>
  using internal_pointer_type = typename BackendTraits<
      EigenBackendSNNMatmul>::template internal_pointer_type<T>;

  /**
   * Constructs an instance of \ref sycldnn::backend::EigenBackendSNNMatmul from
   * an
   * instance of Eigen's SyclDevice.
   * \param device The Eigen::SyclDevice to construct the backend from.
   */
  explicit EigenBackendSNNMatmul(Eigen::SyclDevice const& device)
      : CommonBackend(device.sycl_queue()), device_{device} {}

  /**
   * Gets a descriptive name for this backend.
   * \return a descriptive name for this backend.
   */
  static char const* name() { return "EigenBackendSNNMatmul"; }

  /**
   * Gets the SYCL queue that the backend is bound to.
   * \return Returns the SYCL queue that the backend is bound to.
   */
  cl::sycl::queue get_queue() { return device_.sycl_queue(); }

  /**
   * Get a const reference to the Eigen SyclDevice used in this backend.
   * \return A const reference to the Eigen SyclDevice.
   */
  Eigen::SyclDevice const& get_eigen_device() const { return device_; }
  /**
   * Get a reference to the Eigen SyclDevice used in this backend.
   * \return A reference to the Eigen SyclDevice.
   */
  Eigen::SyclDevice& get_eigen_device() { return device_; }

 private:
  Eigen::SyclDevice device_;
};

}  // namespace backend
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_BACKEND_EIGEN_BACKEND_WITH_SNN_MATMUL_H_
