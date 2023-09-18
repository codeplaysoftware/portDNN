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
#ifndef PORTDNN_TEST_TYPES_TYPE_TUPLE4_H_
#define PORTDNN_TEST_TYPES_TYPE_TUPLE4_H_

namespace sycldnn {
namespace types {
template <typename T0_, typename T1_, typename T2_, typename T3_>
struct TypeTuple4 {
  using T0 = T0_;
  using T1 = T1_;
  using T2 = T2_;
  using T3 = T3_;
};
}  // namespace types
}  // namespace sycldnn
#endif  // PORTDNN_TEST_TYPES_TYPE_TUPLE4_H_
