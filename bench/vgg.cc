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

#ifdef ARM_COMPUTE
#include "arm_fixture.h"

#define VGG_BENCHMARK(N, C, W, H, F)                           \
  CONVOLUTION_BENCHMARK(                                       \
      ARM_Forward_##N##_##C##_##W##_##H##_##Flt##_##S##_##Ftr, \
      ParameterSet<N, C, W, H, F>, sycldnn::conv2d::conv_type::Forward)
#else
#include "snn_fixture.h"

#define VGG_BENCHMARK(N, C, W, H, F)                                           \
  CONVOLUTION_BENCHMARK(                                                       \
      Direct_Forward_##N##_##C##_##W##_##H##_##Flt##_##S##_##Ftr,              \
      ParameterSet<N, C, W, H, F>, sycldnn::conv2d::conv_type::Forward,        \
      sycldnn::conv2d::DirectSelector)                                         \
  CONVOLUTION_BENCHMARK(                                                       \
      Tiled_Forward_##N##_##C##_##W##_##H##_##Flt##_##S##_##Ftr,               \
      ParameterSet<N, C, W, H, F>, sycldnn::conv2d::conv_type::Forward,        \
      sycldnn::conv2d::TiledSelector)                                          \
  CONVOLUTION_BENCHMARK(                                                       \
      Direct_InputBackprop_##N##_##C##_##W##_##H##_##Flt##_##S##_##Ftr,        \
      ParameterSet<N, C, W, H, F>, sycldnn::conv2d::conv_type::InputBackprop,  \
      sycldnn::conv2d::DirectSelector)                                         \
  CONVOLUTION_BENCHMARK(                                                       \
      Tiled_InputBackprop_##N##_##C##_##W##_##H##_##Flt##_##S##_##Ftr,         \
      ParameterSet<N, C, W, H, F>, sycldnn::conv2d::conv_type::InputBackprop,  \
      sycldnn::conv2d::TiledSelector)                                          \
  CONVOLUTION_BENCHMARK(                                                       \
      Direct_FilterBackprop_##N##_##C##_##W##_##H##_##Flt##_##S##_##Ftr,       \
      ParameterSet<N, C, W, H, F>, sycldnn::conv2d::conv_type::FilterBackprop, \
      sycldnn::conv2d::DirectSelector)                                         \
  CONVOLUTION_BENCHMARK(                                                       \
      Tiled_FilterBackprop_##N##_##C##_##W##_##H##_##Flt##_##S##_##Ftr,        \
      ParameterSet<N, C, W, H, F>, sycldnn::conv2d::conv_type::FilterBackprop, \
      sycldnn::conv2d::TiledSelector)

#endif
/*
 * Channels | Width | Height | Filter | Stride | Features
 * ---------|-------|--------|--------|--------|---------
 *        3 |   224 |    224 |      3 |      1 |       64
 *       64 |   224 |    224 |      3 |      1 |       64
 *       64 |   112 |    112 |      3 |      1 |      128
 *      128 |   112 |    112 |      3 |      1 |      128
 *      128 |    56 |     56 |      3 |      1 |      256
 *      256 |    56 |     56 |      3 |      1 |      256
 *      256 |    28 |     28 |      3 |      1 |      512
 *      512 |    28 |     28 |      3 |      1 |      512
 *      512 |    14 |     14 |      3 |      1 |      512
 */
namespace {

template <int N, int C, int W, int H, int F>
struct ParameterSet {
  sycldnn::conv2d::Conv2DParams operator()() {
    sycldnn::conv2d::Conv2DParams params;
    params.channels = C;
    params.features = F;
    params.batch = N;
    params.in_rows = H;
    params.in_cols = W;
    params.window_rows = 3;
    params.window_cols = 3;
    params.stride_rows = 1;
    params.stride_cols = 1;
    params.out_rows = H;
    params.out_cols = W;
    params.pad_rows = 1;
    params.pad_cols = 1;
    params.dilation_rows = 1;
    params.dilation_cols = 1;
    return params;
  }
};
}

VGG_BENCHMARK(1, 3, 224, 224, 64);
VGG_BENCHMARK(1, 64, 224, 224, 64);
VGG_BENCHMARK(1, 64, 112, 112, 128);
VGG_BENCHMARK(1, 128, 112, 112, 128);
VGG_BENCHMARK(1, 128, 56, 56, 256);
VGG_BENCHMARK(1, 256, 56, 56, 256);
VGG_BENCHMARK(1, 256, 28, 28, 512);
VGG_BENCHMARK(1, 512, 28, 28, 512);
VGG_BENCHMARK(1, 512, 14, 14, 512);

VGG_BENCHMARK(32, 3, 224, 224, 64);
VGG_BENCHMARK(32, 64, 224, 224, 64);
VGG_BENCHMARK(32, 64, 112, 112, 128);
VGG_BENCHMARK(32, 128, 112, 112, 128);
VGG_BENCHMARK(32, 128, 56, 56, 256);
VGG_BENCHMARK(32, 256, 56, 56, 256);
VGG_BENCHMARK(32, 256, 28, 28, 512);
VGG_BENCHMARK(32, 512, 28, 28, 512);
VGG_BENCHMARK(32, 512, 14, 14, 512);

VGG_BENCHMARK(64, 3, 224, 224, 64);
VGG_BENCHMARK(64, 64, 224, 224, 64);
VGG_BENCHMARK(64, 64, 112, 112, 128);
VGG_BENCHMARK(64, 128, 112, 112, 128);
VGG_BENCHMARK(64, 128, 56, 56, 256);
VGG_BENCHMARK(64, 256, 56, 56, 256);
VGG_BENCHMARK(64, 256, 28, 28, 512);
VGG_BENCHMARK(64, 512, 28, 28, 512);
VGG_BENCHMARK(64, 512, 14, 14, 512);
