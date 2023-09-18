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
#ifndef PORTDNN_TEST_TYPES_TEST_BACKEND_TYPES_H_
#define PORTDNN_TEST_TYPES_TEST_BACKEND_TYPES_H_

#include "test/types/to_gtest_types.h"
#include "test/types/type_list.h"

#include "portdnn/backend/snn_backend.h"
#include "portdnn/backend/snn_usm_backend.h"

#if defined(SNN_TEST_EIGEN_MATMULS) || defined(SNN_TEST_EIGEN)
#include <unsupported/Eigen/CXX11/Tensor>

#include "portdnn/backend/eigen_backend.h"
#endif

#if defined(SNN_TEST_SYCLBLAS_MATMULS) || defined(SNN_TEST_SYCLBLAS)
#include "portdnn/backend/sycl_blas_backend.h"
#endif

#if defined(SNN_TEST_CLBLAST_MATMULS) || defined(SNN_TEST_CLBLAST)
#include "portdnn/backend/clblast_backend.h"
#endif

namespace sycldnn {
namespace types {

/** List of backend types to use by default in tests.  */
using DefaultBackendTypes = sycldnn::types::TypeList<
#ifdef SNN_ENABLE_USM
    sycldnn::backend::SNNUSMBackend,
#endif
    sycldnn::backend::SNNBackend>;
/** The same as DefaultBackends but in the googletest Types format. */
using GTestDefaultBackendTypes = ToGTestTypes<DefaultBackendTypes>::type;

/**
 * Expanded list of all supported backend types to use in tests using matmuls
 * with USM backend added.
 */
using AllMatmulBackendTypes = sycldnn::types::TypeList<
#ifdef SNN_TEST_EIGEN_MATMULS
    sycldnn::backend::EigenBackend,
#endif
#ifdef SNN_TEST_SYCLBLAS_MATMULS
    sycldnn::backend::SyclBLASBackend,
#endif
#ifdef SNN_TEST_CLBLAST_MATMULS
    sycldnn::backend::CLBlastBackend,
#endif
#ifdef SNN_ENABLE_USM
    sycldnn::backend::SNNUSMBackend,
#endif
    sycldnn::backend::SNNBackend>;

/** Expanded list of all supported backend types to use in tests. */
// TODO: Remove _AllBackendTypes once all testing supports USM
using AllBackendTypes = sycldnn::types::TypeList<
#ifdef SNN_TEST_EIGEN
    sycldnn::backend::EigenBackend,
#endif
#ifdef SNN_TEST_SYCLBLAS
    sycldnn::backend::SyclBLASBackend,
#endif
#ifdef SNN_TEST_CLBLAST
    sycldnn::backend::CLBlastBackend,
#endif
#ifdef SNN_ENABLE_USM
    sycldnn::backend::SNNUSMBackend,
#endif
    sycldnn::backend::SNNBackend>;

}  // namespace types
}  // namespace sycldnn

#endif  // PORTDNN_TEST_TYPES_TEST_BACKEND_TYPES_H_
