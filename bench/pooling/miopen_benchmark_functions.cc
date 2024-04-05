/*
 * Copyright Codeplay Software Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
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

#include "miopen_pooling_bench_executor.h"
#include "sycldnn/format_type.h"

#define BM_WITH_DIR_OP_DTYPE_LAYOUT(DIR, OP, DTYPE, LAYOUT)         \
  POOLING_BENCHMARK(MIOpen##_##LAYOUT##_##OP##_##DIR, DTYPE,        \
                    sycldnn::layout::LAYOUT, sycldnn::pooling::DIR, \
                    sycldnn::pooling::OP)

#ifdef SNN_ENABLE_NCHW
#define BM_WITH_DIR_OP_DTYPE(DIR, OP, DTYPE) \
  BM_WITH_DIR_OP_DTYPE_LAYOUT(DIR, OP, DTYPE, NCHW)
#else
#error To compile the MIOpen benchmarks please enable -DSNN_ENABLE_NCHW=ON
#endif

#define BM_WITH_DIR_OP(DIR, OP) BM_WITH_DIR_OP_DTYPE(DIR, OP, float)

#define BM_WITH_DIRECTION(DIR) \
  BM_WITH_DIR_OP(DIR, Max)     \
  BM_WITH_DIR_OP(DIR, Average)

BM_WITH_DIRECTION(Forward);
