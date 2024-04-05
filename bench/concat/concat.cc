/*
 * Copyright Codeplay Software Ltd.
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

#include "src/backend/snn_backend_provider.h"
#include "src/backend/snn_usm_backend_provider.h"

#include "sycldnn/backend/snn_backend.h"
#include "sycldnn/backend/snn_usm_backend.h"

std::vector<std::vector<int>> large_rank4_descriptors(16,
                                                      std::vector<int>(4, 56));
std::vector<std::vector<int>> small_rank4_descriptors(4,
                                                      std::vector<int>(4, 56));

std::vector<std::vector<int>> large_rank3_descriptors(16,
                                                      std::vector<int>(3, 208));
std::vector<std::vector<int>> small_rank3_descriptors(4,
                                                      std::vector<int>(3, 208));

std::vector<std::vector<int>> large_rank2_descriptors(16,
                                                      std::vector<int>(2,
                                                                       3004));
std::vector<std::vector<int>> small_rank2_descriptors(4,
                                                      std::vector<int>(2,
                                                                       3004));

std::vector<std::vector<int>> large_rank1_descriptors(
    16, std::vector<int>(1, 9000000));
std::vector<std::vector<int>> small_rank1_descriptors(
    4, std::vector<int>(1, 9000000));

#define CONCAT_BENCHMARK_WITH_AXIS(TENSOR_DESC, RANK, BACKEND, BACKEND_STR, \
                                   AXIS)                                    \
  CONCAT_BENCHMARK(Rank##_##RANK##_##Axis##_##AXIS##_##BACKEND_STR,         \
                   TENSOR_DESC, RANK, AXIS, BACKEND, float)

#define CREATE_CONCAT_BENCHMARK_RANK_1(TENSOR_DESC, BACKEND, BACKEND_STR) \
  CONCAT_BENCHMARK_WITH_AXIS(TENSOR_DESC, 1, BACKEND, BACKEND_STR, 0)

#define CREATE_CONCAT_BENCHMARK_RANK_2(TENSOR_DESC, BACKEND, BACKEND_STR) \
  CONCAT_BENCHMARK_WITH_AXIS(TENSOR_DESC, 2, BACKEND, BACKEND_STR, 0)     \
  CONCAT_BENCHMARK_WITH_AXIS(TENSOR_DESC, 2, BACKEND, BACKEND_STR, 1)

#define CREATE_CONCAT_BENCHMARK_RANK_3(TENSOR_DESC, BACKEND, BACKEND_STR) \
  CONCAT_BENCHMARK_WITH_AXIS(TENSOR_DESC, 3, BACKEND, BACKEND_STR, 0)     \
  CONCAT_BENCHMARK_WITH_AXIS(TENSOR_DESC, 3, BACKEND, BACKEND_STR, 1)     \
  CONCAT_BENCHMARK_WITH_AXIS(TENSOR_DESC, 3, BACKEND, BACKEND_STR, 2)

#define CREATE_CONCAT_BENCHMARK_RANK_4(TENSOR_DESC, BACKEND, BACKEND_STR) \
  CONCAT_BENCHMARK_WITH_AXIS(TENSOR_DESC, 4, BACKEND, BACKEND_STR, 0)     \
  CONCAT_BENCHMARK_WITH_AXIS(TENSOR_DESC, 4, BACKEND, BACKEND_STR, 1)     \
  CONCAT_BENCHMARK_WITH_AXIS(TENSOR_DESC, 4, BACKEND, BACKEND_STR, 2)     \
  CONCAT_BENCHMARK_WITH_AXIS(TENSOR_DESC, 4, BACKEND, BACKEND_STR, 3)

#define CREATE_CONCAT_BENCHMARK_SPECIFIC_AXIS(TENSOR_DESC, AXIS, RANK, \
                                              BACKEND, BACKEND_STR)    \
  CONCAT_BENCHMARK_WITH_AXIS(TENSOR_DESC, RANK, BACKEND, BACKEND_STR, AXIS)

#ifdef SNN_ENABLE_USM
CREATE_CONCAT_BENCHMARK_RANK_4(large_rank4_descriptors,
                               sycldnn::backend::SNNUSMBackend,
                               SNNUSMBackend_Recurse)
CREATE_CONCAT_BENCHMARK_RANK_3(large_rank3_descriptors,
                               sycldnn::backend::SNNUSMBackend,
                               SNNUSMBackend_Recurse)
CREATE_CONCAT_BENCHMARK_RANK_2(large_rank2_descriptors,
                               sycldnn::backend::SNNUSMBackend,
                               SNNUSMBackend_Recurse)
CREATE_CONCAT_BENCHMARK_RANK_1(large_rank1_descriptors,
                               sycldnn::backend::SNNUSMBackend,
                               SNNUSMBackend_Recurse)

CREATE_CONCAT_BENCHMARK_RANK_4(small_rank4_descriptors,
                               sycldnn::backend::SNNUSMBackend,
                               SNNUSMBackend_NoRecurse)
CREATE_CONCAT_BENCHMARK_RANK_3(small_rank3_descriptors,
                               sycldnn::backend::SNNUSMBackend,
                               SNNUSMBackend_NoRecurse)
CREATE_CONCAT_BENCHMARK_RANK_2(small_rank2_descriptors,
                               sycldnn::backend::SNNUSMBackend,
                               SNNUSMBackend_NoRecurse)
CREATE_CONCAT_BENCHMARK_RANK_1(small_rank1_descriptors,
                               sycldnn::backend::SNNUSMBackend,
                               SNNUSMBackend_NoRecurse)
#endif

CREATE_CONCAT_BENCHMARK_RANK_4(large_rank4_descriptors,
                               sycldnn::backend::SNNBackend, SNNBackend_Recurse)
CREATE_CONCAT_BENCHMARK_RANK_3(large_rank3_descriptors,
                               sycldnn::backend::SNNBackend, SNNBackend_Recurse)
CREATE_CONCAT_BENCHMARK_RANK_2(large_rank2_descriptors,
                               sycldnn::backend::SNNBackend, SNNBackend_Recurse)
CREATE_CONCAT_BENCHMARK_RANK_1(large_rank1_descriptors,
                               sycldnn::backend::SNNBackend, SNNBackend_Recurse)
/**
 * Benchmarks marked "Recurse" are ones where the number of tensors is larger
 * than the maximum size of the tarray, thus triggering the recursive
 * concatenation case. Favour the "NoRecurse" benchmarks for measuring kernel
 * performance as "Recurse" benchmarks will launch multiple kernels
 */
CREATE_CONCAT_BENCHMARK_RANK_4(small_rank4_descriptors,
                               sycldnn::backend::SNNBackend,
                               SNNBackend_NoRecurse)
CREATE_CONCAT_BENCHMARK_RANK_3(small_rank3_descriptors,
                               sycldnn::backend::SNNBackend,
                               SNNBackend_NoRecurse)
CREATE_CONCAT_BENCHMARK_RANK_2(small_rank2_descriptors,
                               sycldnn::backend::SNNBackend,
                               SNNBackend_NoRecurse)
CREATE_CONCAT_BENCHMARK_RANK_1(small_rank1_descriptors,
                               sycldnn::backend::SNNBackend,
                               SNNBackend_NoRecurse)

// ######################################
// ##### Shapes from popular models #####
// ######################################

// Pointnet
std::vector<std::vector<int>> pointnet_desc = {{32, 4096, 1, 1024},
                                               {32, 4096, 1, 128}};
CREATE_CONCAT_BENCHMARK_SPECIFIC_AXIS(pointnet_desc, 3, 4,
                                      sycldnn::backend::SNNUSMBackend, Pointnet)

// DLRM
std::vector<std::vector<int>> dlrm_desc1 = {{80000, 2}, {80000, 3}};
std::vector<std::vector<int>> dlrm_desc2 = {{80000, 2}, {80000, 6}};
CREATE_CONCAT_BENCHMARK_SPECIFIC_AXIS(dlrm_desc1, 1, 2,
                                      sycldnn::backend::SNNUSMBackend, DLRM1)
CREATE_CONCAT_BENCHMARK_SPECIFIC_AXIS(dlrm_desc2, 1, 2,
                                      sycldnn::backend::SNNUSMBackend, DLRM2)

// Mask-RCNN
std::vector<std::vector<int>> maskrcnn_desc1 = {
    {4096, 1}, {4096, 1}, {4096, 1}, {4096, 1}};
std::vector<std::vector<int>> maskrcnn_desc2 = {
    {7, 7, 256}, {7, 7, 256}, {7, 7, 256}, {7, 7, 256}};
std::vector<std::vector<int>> maskrcnn_desc3 = {
    {14, 14, 256}, {14, 14, 256}, {14, 14, 256}, {14, 14, 256}};
CREATE_CONCAT_BENCHMARK_SPECIFIC_AXIS(maskrcnn_desc1, 1, 2,
                                      sycldnn::backend::SNNUSMBackend,
                                      MaskRCNN1)
CREATE_CONCAT_BENCHMARK_SPECIFIC_AXIS(maskrcnn_desc2, 0, 3,
                                      sycldnn::backend::SNNUSMBackend,
                                      MaskRCNN2)
CREATE_CONCAT_BENCHMARK_SPECIFIC_AXIS(maskrcnn_desc3, 0, 3,
                                      sycldnn::backend::SNNUSMBackend,
                                      MaskRCNN3)
