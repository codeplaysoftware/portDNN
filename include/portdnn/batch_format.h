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
#ifndef PORTDNN_INCLUDE_BATCH_FORMAT_H_
#define PORTDNN_INCLUDE_BATCH_FORMAT_H_

/**
 * \file
 * Contains the declaration of the \ref sycldnn::BatchFormat enumerated
 * type. This type is used to specify the batch memory layout used for a given
 * Batchd Matmul operation.
 */
namespace sycldnn {

/**
 * For a given batched matmul with left matrices BxMxK and right matrices BxKxN
 * to get result shape BxMxK. The formats used to describe how the batches are
 * strided in memory are:
 */
enum class BatchFormat {
  /**
   * BatchFormat::STRIDED where the batches are the slowest moving dimension.
   * I.e. elements 0 to MxK-1 will be in the same batch.
   */
  STRIDED,
  /**
   * BatchFormat::INTERLEAVED where the batches are the fastest moving
   * dimension. I.e. for a matrix with B>1 then elements 0 and 1 will belong to
   * different batches.
   */
  INTERLEAVED
};
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_BATCH_FORMAT_H_
