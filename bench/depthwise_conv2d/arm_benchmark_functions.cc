/*
 * Copyright Codeplay Software Ltd.
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
#include "arm_fixture.h"

// For ARM Compute Library, need to provide the executor to specify whether
// running on NEON or OpenCL.
#ifdef ACL_NEON
#define EXEC sycldnn::bench::ACLNeonExecutor
#else
#define EXEC sycldnn::bench::ACLOpenCLExecutor
#endif  // ACL_NEON

#define BM_WITH_DTYPE(DTYPE) \
  DEPTHWISE_CONVOLUTION_BENCHMARK(Forward, DTYPE, EXEC)

BM_WITH_DTYPE(float);
