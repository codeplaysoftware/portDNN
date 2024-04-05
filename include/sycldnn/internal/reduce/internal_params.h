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
#ifndef SYCLDNN_INCLUDE_INTERNAL_REDUCE_INTERNAL_PARAMS_H_
#define SYCLDNN_INCLUDE_INTERNAL_REDUCE_INTERNAL_PARAMS_H_

/**
 * \file
 * Defines the \ref sycldnn::reduce::internal::ReduceParams struct,
 * which contains the values used in a reduce operation.
 */
namespace sycldnn {
namespace reduce {
namespace internal {
/** Struct that contains values used in a Pointwise op. */
struct ReduceParams {
  /** The type of the params is int, providing a decent
   * upper bound on the tensor sizes.*/
  using Index = int;

  /** Batch size of reduce operation */
  Index batch_size = 1;
  /** Size of the dimension being reduced */
  Index reduce_size = 1;
  /** Size of inner stride between elements to be reduce */
  Index stride_size = 1;
  /** Specifies the number of batch groups to also reduce for. E.g.
   * batch_size=N*C and batch_groups=C then output batch will equal C. */
  Index batch_groups = -1;
  /** Maximum size of a singular subgroup */
  Index max_sg_size = 32;
  /** Maximum size of a singular workgroup */
  Index max_wg_size = 256;
};
}  // namespace internal
}  // namespace reduce
}  // namespace sycldnn
#endif  // SYCLDNN_INCLUDE_REDUCE_PARAMS_H_
