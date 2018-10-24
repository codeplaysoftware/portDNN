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
#ifndef SYCLDNN_BENCH_CONV2D_MOBILENET_PARAM_SET_H_
#define SYCLDNN_BENCH_CONV2D_MOBILENET_PARAM_SET_H_

#include "sycldnn/conv2d/params.h"

/**
 * Function object which returns a conv2d parameter struct required for the
 * MobileNet model.
 *
 * \tparam Batches Number of batches
 * \tparam Window Size of convolution window
 * \tparam Stride Stride of the convolution
 * \tparam Rows Number of rows in the input
 * \tparam Cols Number of columns in the input
 * \tparam Channels Number of channels
 * \tparam Features Number of features
 */
template <int Batches, int Window, int Stride, int Rows, int Cols, int Channels,
          int Features>
struct ParameterSet {
  sycldnn::conv2d::Conv2DParams operator()() {
    sycldnn::conv2d::Conv2DParams params;
    params.channels = Channels;
    params.features = Features;
    params.batch = Batches;
    params.in_rows = Rows;
    params.in_cols = Cols;
    params.window_rows = Window;
    params.window_cols = Window;
    params.stride_rows = Stride;
    params.stride_cols = Stride;
    params.out_rows = Rows / Stride;
    params.out_cols = Cols / Stride;
    params.pad_rows = Window / Stride;
    params.pad_cols = Window / Stride;
    params.dilation_rows = 1;
    params.dilation_cols = 1;
    return params;
  }
};

#endif  // SYCLDNN_BENCH_CONV2D_MOBILENET_PARAM_SET_H_
