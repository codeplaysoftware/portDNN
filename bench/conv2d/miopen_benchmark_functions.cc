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
#include "miopen_conv2d_bench_executor.h"
#include "sycldnn/format_type.h"

#define BM_WITH_ALGO_DTYPE_LAYOUT(ALGO, DTYPE, LAYOUT)               \
  CONVOLUTION_BENCHMARK(MIOPEN##_##LAYOUT##_##DTYPE##_##ALGO, DTYPE, \
                        sycldnn::layout::LAYOUT,                     \
                        miopenConvolutionFwdAlgo##ALGO)

// MIOpen only supports NCHW so it is enabled by default
#define BM_WITH_ALGO_DTYPE(ALGO, DTYPE) \
  BM_WITH_ALGO_DTYPE_LAYOUT(ALGO, DTYPE, NCHW)

#define BM_WITH_ALGO(ALGO) BM_WITH_ALGO_DTYPE(ALGO, float)

#ifdef SNN_ENABLE_NCHW
BM_WITH_ALGO(ImplicitGEMM)
BM_WITH_ALGO(GEMM)
BM_WITH_ALGO(Winograd)
BM_WITH_ALGO(FFT)
#else
#error To compile the MIOpen benchmarks please enable -DSNN_ENABLE_NCHW=ON
#endif
