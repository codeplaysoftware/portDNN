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
#include "cudnn_conv2d_bench_executor.h"
#include "sycldnn/format_type.h"

#define BM_WITH_DIR_ALGO_DTYPE_LAYOUT(DIR, ALGO, DTYPE, LAYOUT)         \
  CONVOLUTION_BENCHMARK(                                                \
      CUDNN##_##DIR##_##LAYOUT##_##DTYPE##_##ALGO##_TensorOp, DTYPE,    \
      sycldnn::layout::LAYOUT, sycldnn::conv2d::conv_type::DIR,         \
      sycldnn::bench::ALGO, true)                                       \
  CONVOLUTION_BENCHMARK(                                                \
      CUDNN##_##DIR##_##LAYOUT##_##DTYPE##_##ALGO##_NonTensorOp, DTYPE, \
      sycldnn::layout::LAYOUT, sycldnn::conv2d::conv_type::DIR,         \
      sycldnn::bench::ALGO, false)

#ifdef SNN_ENABLE_NCHW
#define BM_WITH_DIR_ALGO_DTYPE(DIR, ALGO, DTYPE)              \
  BM_WITH_DIR_ALGO_DTYPE_LAYOUT(DIR, ALGO, DTYPE, NHWC_CUDNN) \
  BM_WITH_DIR_ALGO_DTYPE_LAYOUT(DIR, ALGO, DTYPE, NCHW)
#else
#define BM_WITH_DIR_ALGO_DTYPE(DIR, ALGO, DTYPE) \
  BM_WITH_DIR_ALGO_DTYPE_LAYOUT(DIR, ALGO, DTYPE, NHWC_CUDNN)
#endif

#define BM_WITH_DIR_ALGO(DIR, ALGO) BM_WITH_DIR_ALGO_DTYPE(DIR, ALGO, float)

BM_WITH_DIR_ALGO(Forward, IMPLICIT_GEMM_DETERMINISTIC)
BM_WITH_DIR_ALGO(Forward, GEMM)
BM_WITH_DIR_ALGO(Forward, WINOGRAD)
BM_WITH_DIR_ALGO(Forward, FFT)

BM_WITH_DIR_ALGO(InputBackprop, IMPLICIT_GEMM_DETERMINISTIC)
BM_WITH_DIR_ALGO(InputBackprop, IMPLICIT_GEMM_NONDETERMINISTIC)
BM_WITH_DIR_ALGO(InputBackprop, WINOGRAD)
BM_WITH_DIR_ALGO(InputBackprop, FFT)

BM_WITH_DIR_ALGO(FilterBackprop, IMPLICIT_GEMM_DETERMINISTIC)
BM_WITH_DIR_ALGO(FilterBackprop, IMPLICIT_GEMM_NONDETERMINISTIC)
BM_WITH_DIR_ALGO(FilterBackprop, IMPLICIT_GEMM_PRECOMP)
BM_WITH_DIR_ALGO(FilterBackprop, WINOGRAD)
BM_WITH_DIR_ALGO(FilterBackprop, FFT)
