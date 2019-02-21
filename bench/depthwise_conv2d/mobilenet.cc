/*
 * Copyright 2019 Codeplay Software Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use these files except in compliance with the License.
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
#include "mobilenet_param_set.h"
#include "snn_fixture.h"

#include "src/backend/eigen_backend_provider.h"

#include "sycldnn/backend/eigen_backend.h"

#include "sycldnn/conv2d/conv_type.h"

#define MOBILENET_BENCHMARK_WITH_DIR(N, Win, Str, Row, Col, Ch, Dir)           \
  DEPTHWISE_CONVOLUTION_BENCHMARK(                                             \
      "MobileNet", Dir##_##N##_##Win##_##Str##_##Row##_##Col##_##Ch,           \
      sycldnn::backend::EigenBackend, ParameterSet<N, Win, Str, Row, Col, Ch>, \
      sycldnn::conv2d::conv_type::Dir)

#define MOBILENET_BENCHMARK(N, Win, Str, Row, Col, Ch)                   \
  MOBILENET_BENCHMARK_WITH_DIR(N, Win, Str, Row, Col, Ch, Forward)       \
  MOBILENET_BENCHMARK_WITH_DIR(N, Win, Str, Row, Col, Ch, InputBackprop) \
  MOBILENET_BENCHMARK_WITH_DIR(N, Win, Str, Row, Col, Ch, FilterBackprop)

// Standard benchmark sizes (batch size: 1, 4, optionally 32
#define MOBILENET_PARAMS(Win, Str, Row, Col, Ch) \
  MOBILENET_BENCHMARK(1, Win, Str, Row, Col, Ch);
#include "bench/depthwise_conv2d/mobilenet_params.def"
#undef MOBILENET_PARAMS

#define MOBILENET_PARAMS(Win, Str, Row, Col, Ch) \
  MOBILENET_BENCHMARK(4, Win, Str, Row, Col, Ch);
#include "bench/depthwise_conv2d/mobilenet_params.def"
#undef MOBILENET_PARAMS

#ifdef SNN_LARGE_BATCH_BENCHMARKS
#define MOBILENET_PARAMS(Win, Str, Row, Col, Ch) \
  MOBILENET_BENCHMARK(32, Win, Str, Row, Col, Ch);
#include "bench/depthwise_conv2d/mobilenet_params.def"
#undef MOBILENET_PARAMS
#endif  // SNN_LARGE_BATCH_BENCHMARKS

// Extended benchmarks (batch size: 2, optionally 8, 16, 64)
#ifdef SNN_EXTENDED_BENCHMARKS
#define MOBILENET_PARAMS(Win, Str, Row, Col, Ch) \
  MOBILENET_BENCHMARK(2, Win, Str, Row, Col, Ch);
#include "bench/depthwise_conv2d/mobilenet_params.def"
#undef MOBILENET_PARAMS

#ifdef SNN_LARGE_BATCH_BENCHMARKS
#define MOBILENET_PARAMS(Win, Str, Row, Col, Ch) \
  MOBILENET_BENCHMARK(8, Win, Str, Row, Col, Ch);
#include "bench/depthwise_conv2d/mobilenet_params.def"
#undef MOBILENET_PARAMS

#define MOBILENET_PARAMS(Win, Str, Row, Col, Ch) \
  MOBILENET_BENCHMARK(16, Win, Str, Row, Col, Ch);
#include "bench/depthwise_conv2d/mobilenet_params.def"
#undef MOBILENET_PARAMS

#define MOBILENET_PARAMS(Win, Str, Row, Col, Ch) \
  MOBILENET_BENCHMARK(64, Win, Str, Row, Col, Ch);
#include "bench/depthwise_conv2d/mobilenet_params.def"
#undef MOBILENET_PARAMS
#endif  // SNN_LARGE_BATCH_BENCHMARKS
#endif  // SNN_EXTENDED_BENCHMARKS
