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
#ifndef SYCLDNN_BENCH_CONV2D_RESNET_PARAM_SET_H_
#define SYCLDNN_BENCH_CONV2D_RESNET_PARAM_SET_H_

#include "sycldnn/conv2d/params.h"

/**
 * Function object which returns a conv2d parameter struct required for the
 * Resnet model.
 *
 * \tparam N Number of batches
 * \tparam C Number of channels
 * \tparam W Width of the input
 * \tparam H Height of the input
 * \tparam Flt Size of the filter
 * \tparam S Size of the stride
 * \tparam F Number of features
 */
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

#endif  // SYCLDNN_BENCH_CONV2D_RESNET_PARAM_SET_H_
