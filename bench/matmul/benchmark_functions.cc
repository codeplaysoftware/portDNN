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
#include "snn_fixture.h"

#ifdef SNN_BENCH_EIGEN
#include "portdnn/backend/eigen_backend.h"
#include "src/backend/eigen_backend_provider.h"
#endif  // SNN_BENCH_EIGEN

#ifdef SNN_BENCH_SYCLBLAS
#include "portdnn/backend/sycl_blas_backend.h"
#include "src/backend/syclblas_backend_provider.h"
#endif  // SNN_BENCH_SYCLBLAS

#ifdef SNN_BENCH_CLBLAST
#include "portdnn/backend/clblast_backend.h"
#include "src/backend/clblast_backend_provider.h"
#endif  // SNN_BENCH_SYCLBLAS

#ifdef SNN_BENCH_SNNBACKEND
#include "portdnn/backend/snn_backend.h"
#include "src/backend/snn_backend_provider.h"
#endif  // SNN_BENCH_SNNBACKEND

#define BM_INSTANTIATE(BACK, DTYPE) \
  MATMUL_BENCHMARK(BACK, sycldnn::backend::BACK, DTYPE)

#define BM_WITH_BACKEND(BACK) BM_INSTANTIATE(BACK, float)

#ifdef SNN_BENCH_EIGEN
BM_WITH_BACKEND(EigenBackend);
#endif

#ifdef SNN_BENCH_SYCLBLAS
BM_WITH_BACKEND(SyclBLASBackend);
#endif

#ifdef SNN_BENCH_CLBLAST
BM_WITH_BACKEND(CLBlastBackend);
#endif

#ifdef SNN_BENCH_SNNBACKEND
BM_WITH_BACKEND(SNNBackend);
#endif
