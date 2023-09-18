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
#ifndef PORTDNN_INCLUDE_BACKEND_EIGEN_POINTER_TO_EIGEN_POINTER_H_
#define PORTDNN_INCLUDE_BACKEND_EIGEN_POINTER_TO_EIGEN_POINTER_H_

/**
 * \file
 * Contains the implementation of \ref sycldnn::backend::EigenToEigenPointer,
 * which provides mapping from externally to internal pointer representations
 * for the Eigen backend. For the Eigen backend this is a no-op.
 */
namespace sycldnn {
namespace backend {
/**
 * Handler to convert external Eigen pointers to internal Eigen pointers.
 *
 * In this case the conversion doesn't need to do anything, as the same pointers
 * are used externally and internally.
 */
struct EigenToEigenPointer {
  /**
   * Maps from external to internal pointer representations. This is a no-op for
   * the Eigen backend.
   * \param ptr The external pointer to transform to the corresponding internal
   *            pointer representation.
   * \return Returns a pointer representation compatible with \ref
   *         sycldnn::backend::EigenInternalHandler.
   */
  template <typename T>
  T* to_internal_pointer(T* ptr) {
    return ptr;
  }

  /**
   * Release the internal pointer, which has previously been returned from \ref
   * sycldnn::backend::EigenToEigenPointer::to_internal_pointer.
   *
   * In this case it is a no-op.
   *
   * \param ptr The internal pointer to release.
   */
  template <typename T>
  void release_internal_pointer(T* ptr) {
    SNN_UNUSED_VAR(ptr);
  }
};
}  // namespace backend
}  // namespace sycldnn
#endif  // PORTDNN_INCLUDE_BACKEND_EIGEN_POINTER_TO_EIGEN_POINTER_H_
