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
 * distributed under the License is distributed on an "AS IS" BASIS
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef PORTDNN_INCLUDE_POINTWISE_PARAMS_H_
#define PORTDNN_INCLUDE_POINTWISE_PARAMS_H_

/**
 * \file
 * Defines the \ref sycldnn::pointwise::PointwiseParams struct,
 * which contains the values used in a pointwise operation.
 */
namespace sycldnn {
namespace pointwise {

/** Struct that contains values used in a Pointwise op. */
struct PointwiseParams {
  /** The type of the params is int, providing a decent
   * upper bound on the tensor sizes.*/
  using Index = int;

  /** The total number of input/output values. */
  Index size;
};

}  // namespace pointwise
}  // namespace sycldnn
#endif  // PORTDNN_INCLUDE_POINTWISE_PARAMS_H_
