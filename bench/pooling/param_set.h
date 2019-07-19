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
#ifndef SYCLDNN_BENCH_POOLING_PARAM_SET_H_
#define SYCLDNN_BENCH_POOLING_PARAM_SET_H_

#include "sycldnn/padding_mode.h"

#include "sycldnn/pooling/params.h"

#include "sycldnn/helpers/padding.h"

/**
 * Function object which returns a pooling parameter struct with the given
 * parameters.
 *
 * \tparam N Number of batches
 * \tparam C Number of channels
 * \tparam H Height of the input
 * \tparam W Width of the input
 * \tparam K Size of the pooling window
 * \tparam S Stride of the pooling
 * \tparam Mode Padding mode to apply
 */
template <int N, int C, int W, int H, int K, int S, sycldnn::PaddingMode Mode>
struct ParameterSet {
  sycldnn::pooling::PoolingParams operator()() {
    sycldnn::pooling::PoolingParams params;
    params.batch = N;
    params.window_rows = K;
    params.window_cols = K;
    params.stride_rows = S;
    params.stride_cols = S;
    params.in_rows = H;
    params.in_cols = W;
    params.channels = C;
    return sycldnn::helpers::add_padding_to(params, Mode);
  }
};

#endif  // SYCLDNN_BENCH_POOLING_PARAM_SET_H_
