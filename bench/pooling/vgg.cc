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
#include "snn_fixture.h"
#include "vgg_param_set.h"

#define VGG_BENCHMARK(N, C, W, H)                                           \
  POOLING_BENCHMARK(MaxPool_Forward_##N##_##C##_##W##_##H##_2,              \
                    ParameterSet<N, C, W, H, 2>, sycldnn::pooling::Forward, \
                    sycldnn::pooling::Max)

#define VGG_PARAMS(channels, width, height) \
  VGG_BENCHMARK(1, channels, width, height);
#include "bench/pooling/vgg_params.def"
#undef VGG_PARAMS

#define VGG_PARAMS(channels, width, height) \
  VGG_BENCHMARK(32, channels, width, height);
#include "bench/pooling/vgg_params.def"
#undef VGG_PARAMS

#define VGG_PARAMS(channels, width, height) \
  VGG_BENCHMARK(64, channels, width, height);
#include "bench/pooling/vgg_params.def"
#undef VGG_PARAMS
