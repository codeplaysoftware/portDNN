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
#include "cudnn_conv2d_executor.h"

CONVOLUTION_BENCHMARK(cuDNN_implicit_gemm, float,
                      CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM);
CONVOLUTION_BENCHMARK(cuDNN_gemm, float, CUDNN_CONVOLUTION_FWD_ALGO_GEMM);
CONVOLUTION_BENCHMARK(cuDNN_fft, float, CUDNN_CONVOLUTION_FWD_ALGO_FFT);
CONVOLUTION_BENCHMARK(cuDNN_winograd, float,
                      CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD);
