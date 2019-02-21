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
#ifndef SYCLDNN_BENCH_DEPTHWISE_CONV2D_MOBILENET_PARAM_SET_H_
#define SYCLDNN_BENCH_DEPTHWISE_CONV2D_MOBILENET_PARAM_SET_H_

#include "sycldnn/padding_mode.h"

#include "sycldnn/depthwise_conv2d/params.h"

#include "sycldnn/helpers/padding.h"

/**
 * Function object which returns a depthwise conv2d parameter struct required
 * for the MobileNet model.
 *
 * \tparam Batches Number of batches
 * \tparam Window Size of convolution window
 * \tparam Stride Stride of the convolution
 * \tparam Rows Number of rows in the input
 * \tparam Cols Number of columns in the input
 * \tparam Channels Number of channels
 */
template <int Batches, int Window, int Stride, int Rows, int Cols, int Channels>
struct ParameterSet {
  sycldnn::depthwise_conv2d::DepthwiseConv2DParams operator()() {
    sycldnn::depthwise_conv2d::DepthwiseConv2DParams params;
    params.channels = Channels;
    params.channel_multiplier = 1;
    params.batch = Batches;
    params.in_rows = Rows;
    params.in_cols = Cols;
    params.window_rows = Window;
    params.window_cols = Window;
    params.stride_rows = Stride;
    params.stride_cols = Stride;
    return sycldnn::helpers::add_padding_to(params, sycldnn::PaddingMode::SAME);
  }
};

#endif  // SYCLDNN_BENCH_DEPTHWISE_CONV2D_MOBILENET_PARAM_SET_H_
