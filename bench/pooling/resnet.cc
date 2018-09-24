/*
 * Copyright 2018 Codeplay Software Ltd.
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
#include "resnet_param_set.h"
#include "snn_fixture.h"

#define RESNET_BENCHMARK(N, C, W, H, K, S)                                     \
  POOLING_BENCHMARK(MaxPool_Forward_##N##_##C##_##W##_##H##_##K##_##S,         \
                    ParameterSet<N, C, W, H, K, S>, sycldnn::pooling::Forward, \
                    sycldnn::pooling::Max)                                     \
  POOLING_BENCHMARK(AveragePool_Forward_##N##_##C##_##W##_##H##_##K##_##S,     \
                    ParameterSet<N, C, W, H, K, S>, sycldnn::pooling::Forward, \
                    sycldnn::pooling::Average)

#define RESNET_PARAMS(channels, width, height, window, stride) \
  RESNET_BENCHMARK(1, channels, width, height, window, stride);
#include "bench/pooling/resnet_params.def"
#undef RESNET_PARAMS

#define RESNET_PARAMS(channels, width, height, window, stride) \
  RESNET_BENCHMARK(32, channels, width, height, window, stride);
#include "bench/pooling/resnet_params.def"
#undef RESNET_PARAMS

#define RESNET_PARAMS(channels, width, height, window, stride) \
  RESNET_BENCHMARK(64, channels, width, height, window, stride);
#include "bench/pooling/resnet_params.def"
#undef RESNET_PARAMS
