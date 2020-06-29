/*
 * Copyright 2019 Codeplay Software Ltd.
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
#include "src/backend/eigen_backend_provider.h"

#include "sycldnn/backend/eigen_backend.h"
#endif  // SNN_BENCH_EIGEN

#ifdef SNN_BENCH_SYCLBLAS
#include "src/backend/syclblas_backend_provider.h"
#include "sycldnn/backend/sycl_blas_backend.h"
#endif  // SNN_BENCH_SYCLBLAS

#ifdef SNN_BENCH_CLBLAST
#include "src/backend/clblast_backend_provider.h"
#include "sycldnn/backend/clblast_backend.h"
#endif  // SNN_BENCH_SYCLBLAS

#include "sycldnn/conv2d/conv_type.h"

#include "sycldnn/conv2d/selector/direct_selector.h"
#include "sycldnn/conv2d/selector/im2col_selector.h"
#include "sycldnn/conv2d/selector/matmul_selector.h"
#include "sycldnn/conv2d/selector/tiled_selector.h"
#include "sycldnn/conv2d/selector/winograd_selector.h"

#if !defined(SNN_BENCH_EIGEN) && !defined(SNN_BENCH_SYCLBLAS)
#error At least one of SNN_BENCH_EIGEN or SNN_BENCH_SYCLBLAS must be set.
#endif

#define BM_WITH_ALGO_DIR_BACK_DTYPE(ALGO, DIR, BACK, DTYPE)                   \
  CONVOLUTION_BENCHMARK(ALGO##_##DIR##_##BACK, sycldnn::backend::BACK, DTYPE, \
                        sycldnn::conv2d::conv_type::DIR,                      \
                        sycldnn::conv2d::ALGO##Selector)

#define BM_WITH_ALGO_DIR_BACK(ALGO, DIR, BACK) \
  BM_WITH_ALGO_DIR_BACK_DTYPE(ALGO, DIR, BACK, float)

#ifdef SNN_BENCH_EIGEN
#define BM_WITH_EIGEN(ALGO, DIR) BM_WITH_ALGO_DIR_BACK(ALGO, DIR, EigenBackend)
#else
#define BM_WITH_EIGEN(ALGO, DIR)
#endif

#ifdef SNN_BENCH_SYCLBLAS
#define BM_WITH_SYCLBLAS(ALGO, DIR) \
  BM_WITH_ALGO_DIR_BACK(ALGO, DIR, SyclBLASBackend)
#else
#define BM_WITH_SYCLBLAS(ALGO, DIR)
#endif

#ifdef SNN_BENCH_CLBLAST
#define BM_WITH_CLBLAST(ALGO, DIR) \
  BM_WITH_ALGO_DIR_BACK(ALGO, DIR, CLBlastBackend)
#else
#define BM_WITH_CLBLAST(ALGO, DIR)
#endif

#define BM_WITH_ALGO_AND_DIR(ALGO, DIR) \
  BM_WITH_EIGEN(ALGO, DIR)              \
  BM_WITH_SYCLBLAS(ALGO, DIR)           \
  BM_WITH_CLBLAST(ALGO, DIR)

#define BM_WITH_ALGO(ALGO)                  \
  BM_WITH_ALGO_AND_DIR(ALGO, Forward)       \
  BM_WITH_ALGO_AND_DIR(ALGO, InputBackprop) \
  BM_WITH_ALGO_AND_DIR(ALGO, FilterBackprop)

BM_WITH_ALGO(Direct);
BM_WITH_ALGO(Tiled);
BM_WITH_ALGO(Im2col);
BM_WITH_ALGO(Winograd);
BM_WITH_ALGO(WinogradLarge);
BM_WITH_ALGO(Matmul);
