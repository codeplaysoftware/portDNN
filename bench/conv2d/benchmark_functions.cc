/*
 * Copyright Codeplay Software Ltd.
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

#ifdef SNN_BENCH_PORTBLAS
#include "src/backend/syclblas_backend_provider.h"
#include "sycldnn/backend/sycl_blas_backend.h"
#endif  // SNN_BENCH_PORTBLAS

#ifdef SNN_BENCH_CLBLAST
#include "src/backend/clblast_backend_provider.h"
#include "sycldnn/backend/clblast_backend.h"
#endif  // SNN_BENCH_PORTBLAS

#ifdef SNN_ENABLE_USM
#include "src/backend/snn_usm_backend_provider.h"
#include "sycldnn/backend/snn_usm_backend.h"
#endif  // SNN_ENABLE_USM

#include "src/backend/snn_backend_provider.h"
#include "sycldnn/backend/snn_backend.h"

#include "sycldnn/conv2d/conv_type.h"

#include "sycldnn/format_type.h"

#include "sycldnn/conv2d/selector/direct_selector.h"
#include "sycldnn/conv2d/selector/im2col_selector.h"
#include "sycldnn/conv2d/selector/matmul_selector.h"
#include "sycldnn/conv2d/selector/tiled_selector.h"
#include "sycldnn/conv2d/selector/winograd_selector.h"

#define TensorOp true
#define NonTensorOp false
#define InOrder true
#define OutOfOrder false

#define BM_WITH_ALGO_DIR_BACK_DTYPE_LAYOUT(ALGO, DIR, BACK, TENSORMATH, DTYPE, \
                                           LAYOUT)                             \
  CONVOLUTION_BENCHMARK(                                                       \
      ALGO##_##DIR##_##BACK##_##LAYOUT##_##DTYPE##_##TENSORMATH##_OutOfOrder,  \
      sycldnn::backend::BACK, DTYPE, sycldnn::layout::LAYOUT,                  \
      sycldnn::conv2d::conv_type::DIR, sycldnn::conv2d::ALGO##Selector,        \
      TENSORMATH, OutOfOrder)                                                  \
  CONVOLUTION_BENCHMARK(                                                       \
      ALGO##_##DIR##_##BACK##_##LAYOUT##_##DTYPE##_##TENSORMATH##_InOrder,     \
      sycldnn::backend::BACK, DTYPE, sycldnn::layout::LAYOUT,                  \
      sycldnn::conv2d::conv_type::DIR, sycldnn::conv2d::ALGO##Selector,        \
      TENSORMATH, InOrder)

#ifdef SNN_ENABLE_NCHW
#define BM_WITH_ALGO_DIR_BACK_DTYPE(ALGO, DIR, BACK, TENSORMATH, DTYPE)        \
  BM_WITH_ALGO_DIR_BACK_DTYPE_LAYOUT(ALGO, DIR, BACK, TENSORMATH, DTYPE, NHWC) \
  BM_WITH_ALGO_DIR_BACK_DTYPE_LAYOUT(ALGO, DIR, BACK, TENSORMATH, DTYPE, NCHW)
#else
#define BM_WITH_ALGO_DIR_BACK_DTYPE(ALGO, DIR, BACK, TENSORMATH, DTYPE) \
  BM_WITH_ALGO_DIR_BACK_DTYPE_LAYOUT(ALGO, DIR, BACK, TENSORMATH, DTYPE, NHWC)
#endif

#define BM_WITH_ALGO_DIR_BACK(ALGO, DIR, BACK, TENSORMATH) \
  BM_WITH_ALGO_DIR_BACK_DTYPE(ALGO, DIR, BACK, TENSORMATH, float)

#ifdef SNN_BENCH_EIGEN
#define BM_WITH_EIGEN(ALGO, DIR) \
  BM_WITH_ALGO_DIR_BACK(ALGO, DIR, EigenBackend, NonTensorOp)
#else
#define BM_WITH_EIGEN(ALGO, DIR)
#endif

#ifdef SNN_BENCH_PORTBLAS
#define BM_WITH_PORTBLAS(ALGO, DIR)                              \
  BM_WITH_ALGO_DIR_BACK(ALGO, DIR, SyclBLASBackend, NonTensorOp) \
  BM_WITH_ALGO_DIR_BACK(ALGO, DIR, SyclBLASBackend, TensorOp)
#else
#define BM_WITH_PORTBLAS(ALGO, DIR)
#endif

#ifdef SNN_BENCH_CLBLAST
#define BM_WITH_CLBLAST(ALGO, DIR) \
  BM_WITH_ALGO_DIR_BACK(ALGO, DIR, CLBlastBackend, NonTensorOp)
#else
#define BM_WITH_CLBLAST(ALGO, DIR)
#endif

#ifdef SNN_BENCH_SNNBACKEND
#ifdef SNN_ENABLE_USM
#define BM_WITH_SNNBACKEND(ALGO, DIR)                       \
  BM_WITH_ALGO_DIR_BACK(ALGO, DIR, SNNBackend, NonTensorOp) \
  BM_WITH_ALGO_DIR_BACK(ALGO, DIR, SNNUSMBackend, NonTensorOp)
#else
#define BM_WITH_SNNBACKEND(ALGO, DIR) \
  BM_WITH_ALGO_DIR_BACK(ALGO, DIR, SNNBackend, NonTensorOp)
#endif
#else
#define BM_WITH_SNNBACKEND(ALGO, DIR)
#endif

#define BM_WITH_ALGO_AND_DIR(ALGO, DIR) \
  BM_WITH_EIGEN(ALGO, DIR)              \
  BM_WITH_PORTBLAS(ALGO, DIR)           \
  BM_WITH_CLBLAST(ALGO, DIR)            \
  BM_WITH_SNNBACKEND(ALGO, DIR)

#define BM_WITH_ALGO(ALGO)                  \
  BM_WITH_ALGO_AND_DIR(ALGO, Forward)       \
  BM_WITH_ALGO_AND_DIR(ALGO, InputBackprop) \
  BM_WITH_ALGO_AND_DIR(ALGO, FilterBackprop)

#ifdef SNN_ENABLE_USM
#define BM_ALGO_WITH_SNNBACKEND(ALGO)                                    \
  BM_WITH_ALGO_DIR_BACK(ALGO, Forward, SNNBackend, NonTensorOp)          \
  BM_WITH_ALGO_DIR_BACK(ALGO, InputBackprop, SNNBackend, NonTensorOp)    \
  BM_WITH_ALGO_DIR_BACK(ALGO, FilterBackprop, SNNBackend, NonTensorOp)   \
  BM_WITH_ALGO_DIR_BACK(ALGO, Forward, SNNUSMBackend, NonTensorOp)       \
  BM_WITH_ALGO_DIR_BACK(ALGO, InputBackprop, SNNUSMBackend, NonTensorOp) \
  BM_WITH_ALGO_DIR_BACK(ALGO, FilterBackprop, SNNUSMBackend, NonTensorOp)
#else
#define BM_ALGO_WITH_SNNBACKEND(ALGO)                                 \
  BM_WITH_ALGO_DIR_BACK(ALGO, Forward, SNNBackend, NonTensorOp)       \
  BM_WITH_ALGO_DIR_BACK(ALGO, InputBackprop, SNNBackend, NonTensorOp) \
  BM_WITH_ALGO_DIR_BACK(ALGO, FilterBackprop, SNNBackend, NonTensorOp)
#endif

BM_ALGO_WITH_SNNBACKEND(Direct)
BM_ALGO_WITH_SNNBACKEND(Tiled)

BM_WITH_ALGO(Im2col);
BM_WITH_ALGO(Winograd);
BM_WITH_ALGO(WinogradLarge);
BM_WITH_ALGO(Matmul);
