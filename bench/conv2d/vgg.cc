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

#define VGG_BENCHMARK(N, C, W, H, F)                                           \
  CONVOLUTION_BENCHMARK(                                                       \
      Direct_Forward_##N##_##C##_##W##_##H##_##F, ParameterSet<N, C, W, H, F>, \
      sycldnn::conv2d::conv_type::Forward, sycldnn::conv2d::DirectSelector)    \
  CONVOLUTION_BENCHMARK(                                                       \
      Tiled_Forward_##N##_##C##_##W##_##H##_##F, ParameterSet<N, C, W, H, F>,  \
      sycldnn::conv2d::conv_type::Forward, sycldnn::conv2d::TiledSelector)     \
  CONVOLUTION_BENCHMARK(Direct_InputBackprop_##N##_##C##_##W##_##H##_##F,      \
                        ParameterSet<N, C, W, H, F>,                           \
                        sycldnn::conv2d::conv_type::InputBackprop,             \
                        sycldnn::conv2d::DirectSelector)                       \
  CONVOLUTION_BENCHMARK(Tiled_InputBackprop_##N##_##C##_##W##_##H##_##F,       \
                        ParameterSet<N, C, W, H, F>,                           \
                        sycldnn::conv2d::conv_type::InputBackprop,             \
                        sycldnn::conv2d::TiledSelector)                        \
  CONVOLUTION_BENCHMARK(Direct_FilterBackprop_##N##_##C##_##W##_##H##_##F,     \
                        ParameterSet<N, C, W, H, F>,                           \
                        sycldnn::conv2d::conv_type::FilterBackprop,            \
                        sycldnn::conv2d::DirectSelector)                       \
  CONVOLUTION_BENCHMARK(Tiled_FilterBackprop_##N##_##C##_##W##_##H##_##F,      \
                        ParameterSet<N, C, W, H, F>,                           \
                        sycldnn::conv2d::conv_type::FilterBackprop,            \
                        sycldnn::conv2d::TiledSelector)

#define VGG_PARAMS(channels, width, height, features) \
  VGG_BENCHMARK(1, channels, width, height, features);
#include "bench/conv2d/vgg_params.def"
#undef VGG_PARAMS

#define VGG_PARAMS(channels, width, height, features) \
  VGG_BENCHMARK(2, channels, width, height, features);
#include "bench/conv2d/vgg_params.def"
#undef VGG_PARAMS

#define VGG_PARAMS(channels, width, height, features) \
  VGG_BENCHMARK(4, channels, width, height, features);
#include "bench/conv2d/vgg_params.def"
#undef VGG_PARAMS
