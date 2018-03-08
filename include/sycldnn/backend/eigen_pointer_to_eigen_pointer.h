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
#ifndef SYCLDNN_INCLUDE_BACKEND_EIGEN_POINTER_TO_EIGEN_POINTER_H_
#define SYCLDNN_INCLUDE_BACKEND_EIGEN_POINTER_TO_EIGEN_POINTER_H_
namespace sycldnn {
namespace backend {
/**
 * Handler to convert external Eigen pointers to internal Eigen pointers.
 *
 * In this case the conversion doesn't need to do anything, as the same pointers
 * are used externally and internally.
 */
struct EigenToEigenPointer {
  template <typename T>
  T* to_internal_pointer(T* ptr) {
    return ptr;
  }
};
}  // namespace backend
}  // namespace sycldnn
#endif  // SYCLDNN_INCLUDE_BACKEND_EIGEN_POINTER_TO_EIGEN_POINTER_H_
