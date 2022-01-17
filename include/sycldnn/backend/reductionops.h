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

#ifndef SYCLDNN_INCLUDE_BACKEND_REDUCTION_H_
#define SYCLDNN_INCLUDE_BACKEND_REDUCTION_H_

/**
 * \file
 * Contains the declarations of the Add, Mean tag types.
 */

namespace sycldnn {
namespace backend {
namespace reduction {

struct Add;

struct Mean;

/**
 * Function to check if the provided reduction is valid or not.
 * \tparam Op   Type of reduction.
 * \return
 */
template <typename Op>
inline SNN_ALWAYS_INLINE void is_valid_reduction() {
  static_assert(std::is_same<Op, reduction::Add>::value ||
                    std::is_same<Op, reduction::Mean>::value,
                "Invalid Reduction Type");
}

}  // namespace reduction
}  // namespace backend
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_BACKEND_REDUCTION_H_
