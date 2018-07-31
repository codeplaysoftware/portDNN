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

#define RESNET_BENCHMARK(N, C, W, H, Flt, S, Ftr)              \
  CONVOLUTION_BENCHMARK(                                       \
      ARM_Forward_##N##_##C##_##W##_##H##_##Flt##_##S##_##Ftr, \
      ParameterSet<N, C, W, H, Flt, S, Ftr>,                   \
      sycldnn::conv2d::conv_type::Forward)
#else
#include "snn_fixture.h"

#define RESNET_BENCHMARK(N, C, W, H, Flt, S, Ftr)                           \
  CONVOLUTION_BENCHMARK(                                                    \
      Direct_Forward_##N##_##C##_##W##_##H##_##Flt##_##S##_##Ftr,           \
      ParameterSet<N, C, W, H, Flt, S, Ftr>,                                \
      sycldnn::conv2d::conv_type::Forward, sycldnn::conv2d::DirectSelector) \
  CONVOLUTION_BENCHMARK(                                                    \
      Tiled_Forward_##N##_##C##_##W##_##H##_##Flt##_##S##_##Ftr,            \
      ParameterSet<N, C, W, H, Flt, S, Ftr>,                                \
      sycldnn::conv2d::conv_type::Forward, sycldnn::conv2d::TiledSelector)  \
  CONVOLUTION_BENCHMARK(                                                    \
      Direct_InputBackprop_##N##_##C##_##W##_##H##_##Flt##_##S##_##Ftr,     \
      ParameterSet<N, C, W, H, Flt, S, Ftr>,                                \
      sycldnn::conv2d::conv_type::InputBackprop,                            \
      sycldnn::conv2d::DirectSelector)                                      \
  CONVOLUTION_BENCHMARK(                                                    \
      Tiled_InputBackprop_##N##_##C##_##W##_##H##_##Flt##_##S##_##Ftr,      \
      ParameterSet<N, C, W, H, Flt, S, Ftr>,                                \
      sycldnn::conv2d::conv_type::InputBackprop,                            \
      sycldnn::conv2d::TiledSelector)                                       \
  CONVOLUTION_BENCHMARK(                                                    \
      Direct_FilterBackprop_##N##_##C##_##W##_##H##_##Flt##_##S##_##Ftr,    \
      ParameterSet<N, C, W, H, Flt, S, Ftr>,                                \
      sycldnn::conv2d::conv_type::FilterBackprop,                           \
      sycldnn::conv2d::DirectSelector)                                      \
  CONVOLUTION_BENCHMARK(                                                    \
      Tiled_FilterBackprop_##N##_##C##_##W##_##H##_##Flt##_##S##_##Ftr,     \
      ParameterSet<N, C, W, H, Flt, S, Ftr>,                                \
      sycldnn::conv2d::conv_type::FilterBackprop,                           \
      sycldnn::conv2d::TiledSelector)

#endif

/*
 * Channels | Width | Height | Filter | Stride | Features
 * ---------|-------|--------|--------|--------|---------
 *        3 |   230 |    230 |      7 |      2 |       64
 *       64 |    56 |     56 |      1 |      1 |      256
 *       64 |    56 |     56 |      1 |      1 |       64
 *       64 |    56 |     56 |      3 |      1 |       64
 *      256 |    56 |     56 |      1 |      1 |       64
 *      256 |    56 |     56 |      1 |      2 |      512
 *      256 |    56 |     56 |      1 |      2 |      128
 *      128 |    28 |     28 |      3 |      1 |      128
 *      128 |    28 |     28 |      1 |      1 |      512
 *      512 |    28 |     28 |      1 |      1 |      128
 *      512 |    28 |     28 |      1 |      1 |      128
 *      512 |    28 |     28 |      1 |      2 |     1024
 *      512 |    28 |     28 |      1 |      2 |      256
 *      256 |    14 |     14 |      3 |      1 |      256
 *      256 |    14 |     14 |      1 |      1 |     1024
 *     1024 |    14 |     14 |      1 |      1 |      256
 *     1024 |    14 |     14 |      1 |      2 |     2048
 *     1024 |    14 |     14 |      1 |      2 |      512
 *      512 |     7 |      7 |      3 |      1 |      512
 *      512 |     7 |      7 |      1 |      1 |     2048
 *     2048 |     7 |      7 |      1 |      1 |      512
 */
namespace {

template <int N, int C, int W, int H, int Flt, int S, int Ftr>
struct ParameterSet {
  sycldnn::conv2d::Conv2DParams operator()() {
    sycldnn::conv2d::Conv2DParams params;
    params.channels = C;
    params.features = Ftr;
    params.batch = N;
    params.in_rows = H;
    params.in_cols = W;
    params.window_rows = Flt;
    params.window_cols = Flt;
    params.stride_rows = S;
    params.stride_cols = S;
    params.out_rows = H / S;
    params.out_cols = W / S;
    params.pad_rows = Flt / 2;
    params.pad_cols = Flt / 2;
    params.dilation_rows = 1;
    params.dilation_cols = 1;
    return params;
  }
};
}

RESNET_BENCHMARK(1, 3, 230, 230, 7, 2, 64);
RESNET_BENCHMARK(1, 64, 56, 56, 1, 1, 256);
RESNET_BENCHMARK(1, 64, 56, 56, 1, 1, 64);
RESNET_BENCHMARK(1, 64, 56, 56, 3, 1, 64);
RESNET_BENCHMARK(1, 256, 56, 56, 1, 1, 64);
RESNET_BENCHMARK(1, 256, 56, 56, 1, 2, 512);
RESNET_BENCHMARK(1, 256, 56, 56, 1, 2, 128);
RESNET_BENCHMARK(1, 128, 28, 28, 3, 1, 128);
RESNET_BENCHMARK(1, 128, 28, 28, 1, 1, 512);
RESNET_BENCHMARK(1, 512, 28, 28, 1, 1, 128);
RESNET_BENCHMARK(1, 512, 28, 28, 1, 2, 1024);
RESNET_BENCHMARK(1, 512, 28, 28, 1, 2, 256);
RESNET_BENCHMARK(1, 256, 14, 14, 3, 1, 256);
RESNET_BENCHMARK(1, 256, 14, 14, 1, 1, 1024);
RESNET_BENCHMARK(1, 1024, 14, 14, 1, 1, 256);
RESNET_BENCHMARK(1, 1024, 14, 14, 1, 2, 2048);
RESNET_BENCHMARK(1, 1024, 14, 14, 1, 2, 512);
RESNET_BENCHMARK(1, 512, 7, 7, 3, 1, 512);
RESNET_BENCHMARK(1, 512, 7, 7, 1, 1, 2048);
RESNET_BENCHMARK(1, 2048, 7, 7, 1, 1, 512);

RESNET_BENCHMARK(32, 3, 230, 230, 7, 2, 64);
RESNET_BENCHMARK(32, 64, 56, 56, 1, 1, 256);
RESNET_BENCHMARK(32, 64, 56, 56, 1, 1, 64);
RESNET_BENCHMARK(32, 64, 56, 56, 3, 1, 64);
RESNET_BENCHMARK(32, 256, 56, 56, 1, 1, 64);
RESNET_BENCHMARK(32, 256, 56, 56, 1, 2, 512);
RESNET_BENCHMARK(32, 256, 56, 56, 1, 2, 128);
RESNET_BENCHMARK(32, 128, 28, 28, 3, 1, 128);
RESNET_BENCHMARK(32, 128, 28, 28, 1, 1, 512);
RESNET_BENCHMARK(32, 512, 28, 28, 1, 1, 128);
RESNET_BENCHMARK(32, 512, 28, 28, 1, 2, 1024);
RESNET_BENCHMARK(32, 512, 28, 28, 1, 2, 256);
RESNET_BENCHMARK(32, 256, 14, 14, 3, 1, 256);
RESNET_BENCHMARK(32, 256, 14, 14, 1, 1, 1024);
RESNET_BENCHMARK(32, 1024, 14, 14, 1, 1, 256);
RESNET_BENCHMARK(32, 1024, 14, 14, 1, 2, 2048);
RESNET_BENCHMARK(32, 1024, 14, 14, 1, 2, 512);
RESNET_BENCHMARK(32, 512, 7, 7, 3, 1, 512);
RESNET_BENCHMARK(32, 512, 7, 7, 1, 1, 2048);
RESNET_BENCHMARK(32, 2048, 7, 7, 1, 1, 512);

RESNET_BENCHMARK(64, 3, 230, 230, 7, 2, 64);
RESNET_BENCHMARK(64, 64, 56, 56, 1, 1, 256);
RESNET_BENCHMARK(64, 64, 56, 56, 1, 1, 64);
RESNET_BENCHMARK(64, 64, 56, 56, 3, 1, 64);
RESNET_BENCHMARK(64, 256, 56, 56, 1, 1, 64);
RESNET_BENCHMARK(64, 256, 56, 56, 1, 2, 512);
RESNET_BENCHMARK(64, 256, 56, 56, 1, 2, 128);
RESNET_BENCHMARK(64, 128, 28, 28, 3, 1, 128);
RESNET_BENCHMARK(64, 128, 28, 28, 1, 1, 512);
RESNET_BENCHMARK(64, 512, 28, 28, 1, 1, 128);
RESNET_BENCHMARK(64, 512, 28, 28, 1, 2, 1024);
RESNET_BENCHMARK(64, 512, 28, 28, 1, 2, 256);
RESNET_BENCHMARK(64, 256, 14, 14, 3, 1, 256);
RESNET_BENCHMARK(64, 256, 14, 14, 1, 1, 1024);
RESNET_BENCHMARK(64, 1024, 14, 14, 1, 1, 256);
RESNET_BENCHMARK(64, 1024, 14, 14, 1, 2, 2048);
RESNET_BENCHMARK(64, 1024, 14, 14, 1, 2, 512);
RESNET_BENCHMARK(64, 512, 7, 7, 3, 1, 512);
RESNET_BENCHMARK(64, 512, 7, 7, 1, 1, 2048);
RESNET_BENCHMARK(64, 2048, 7, 7, 1, 1, 512);
