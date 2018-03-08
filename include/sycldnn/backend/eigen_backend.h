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
#ifndef SYCLDNN_INCLUDE_BACKEND_EIGEN_BACKEND_H_
#define SYCLDNN_INCLUDE_BACKEND_EIGEN_BACKEND_H_

#include "sycldnn/backend/eigen_external_handler.h"
#include "sycldnn/backend/eigen_internal_handler.h"
#include "sycldnn/backend/eigen_pointer_to_eigen_pointer.h"

namespace sycldnn {
namespace backend {
/**
 * Eigen backend for SYCL-DNN.
 *
 * Provides pointer handling and matrix multiplies using Eigen.
 */
struct EigenBackend final : public EigenExternalHandler,
                            public EigenToEigenPointer,
                            public EigenInternalHandler {
  EigenBackend(Eigen::SyclDevice const& device)
      : EigenExternalHandler{device},
        EigenToEigenPointer{},
        EigenInternalHandler{device} {}
};
}  // namespace backend
}  // namespace sycldnn
#endif  // SYCLDNN_INCLUDE_BACKEND_EIGEN_MATMUL_HANDLER_H_
