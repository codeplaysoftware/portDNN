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
#ifndef SYCLDNN_BENCH_POOLING_VGG_PARAM_SET_H_
#define SYCLDNN_BENCH_POOLING_VGG_PARAM_SET_H_

#include "sycldnn/pooling/params.h"

/**
 * Function object which returns a pooling parameter struct required for the VGG
 * model. All strides are 1. The output size is always
 * input size / neighbourhood size, and is guaranteed to be evenly divisible.
 *
 * \tparam N Number of batches
 * \tparam C Number of channels
 * \tparam W Width of the input
 * \tparam H Height of the input
 * \tparam K Pooling neighbourhood
 */
template <int N, int C, int W, int H, int K>
struct ParameterSet {
  sycldnn::pooling::PoolingParams operator()() {
    sycldnn::pooling::PoolingParams params;
    params.channels = C;
    params.batch = N;
    params.in_rows = H;
    params.in_cols = W;
    params.window_rows = K;
    params.window_cols = K;
    params.stride_rows = K;
    params.stride_cols = K;
    params.out_rows = H / K;
    params.out_cols = W / K;
    params.pad_rows = 0;
    params.pad_cols = 0;
    return params;
  }
};

#endif  // SYCLDNN_BENCH_POOLING_VGG_PARAM_SET_H_
