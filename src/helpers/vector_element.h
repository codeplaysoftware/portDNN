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
#ifndef PORTDNN_SRC_HELPERS_VECTOR_ELEMENT_H_
#define PORTDNN_SRC_HELPERS_VECTOR_ELEMENT_H_

#include "portdnn/helpers/macros.h"

#include <CL/sycl.hpp>

namespace sycldnn {
namespace helpers {
/**
 * Vector helper functions to get and set elements within SYCL vectors.
 *
 * As SYCL vectors don't provide a subscript operator, it is hard to use them
 * inside loops. These helper functions provide a programmatic way of accessing
 * and setting values in vectors without using the hardcoded swizzle functions.
 *
 * In practice these function calls should always be inlined, and if the loops
 * are unrolled then the index values will always be statically known so there
 * will be no overhead from using these functions. However be aware that this
 * relies on the loops being unrolled and the functions inlined.
 */
namespace vector_element {
/** If the type is not a vector, then fall back to returning that value. */
template <typename T>
static inline SNN_ALWAYS_INLINE T get(T const& val, int /*index*/) {
  return val;
}
/** If the type is not a vector, then fall back to setting that value. */
template <typename T>
static inline SNN_ALWAYS_INLINE void set(T& ref, int /*index*/, T val) {
  ref = val;
}
template <typename T>
static inline SNN_ALWAYS_INLINE T get(cl::sycl::vec<T, 1> const& vec,
                                      int index) {
  switch (index) {
    case 0:
      return vec.s0();
    default:
      SNN_UNREACHABLE;
      return 0;
  }
}
template <typename T>
static inline SNN_ALWAYS_INLINE void set(cl::sycl::vec<T, 1>& vec, int index,
                                         T val) {
  switch (index) {
    case 0:
      vec.s0() = val;
      break;
  }
}
template <typename T>
static inline SNN_ALWAYS_INLINE T get(cl::sycl::vec<T, 2> const& vec,
                                      int index) {
  switch (index) {
    case 0:
      return vec.s0();
    case 1:
      return vec.s1();
    default:
      SNN_UNREACHABLE;
      return 0;
  }
}
template <typename T>
static inline SNN_ALWAYS_INLINE void set(cl::sycl::vec<T, 2>& vec, int index,
                                         T val) {
  switch (index) {
    case 0:
      vec.s0() = val;
      break;
    case 1:
      vec.s1() = val;
      break;
  }
}
template <typename T>
static inline SNN_ALWAYS_INLINE T get(cl::sycl::vec<T, 3> const& vec,
                                      int index) {
  switch (index) {
    case 0:
      return vec.s0();
    case 1:
      return vec.s1();
    case 2:
      return vec.s2();
    default:
      SNN_UNREACHABLE;
      return 0;
  }
}
template <typename T>
static inline SNN_ALWAYS_INLINE void set(cl::sycl::vec<T, 3>& vec, int index,
                                         T val) {
  switch (index) {
    case 0:
      vec.s0() = val;
      break;
    case 1:
      vec.s1() = val;
      break;
    case 2:
      vec.s2() = val;
      break;
  }
}
template <typename T>
static inline SNN_ALWAYS_INLINE T get(cl::sycl::vec<T, 4> const& vec,
                                      int index) {
  switch (index) {
    case 0:
      return vec.s0();
    case 1:
      return vec.s1();
    case 2:
      return vec.s2();
    case 3:
      return vec.s3();
    default:
      SNN_UNREACHABLE;
      return 0;
  }
}
template <typename T>
static inline SNN_ALWAYS_INLINE void set(cl::sycl::vec<T, 4>& vec, int index,
                                         T val) {
  switch (index) {
    case 0:
      vec.s0() = val;
      break;
    case 1:
      vec.s1() = val;
      break;
    case 2:
      vec.s2() = val;
      break;
    case 3:
      vec.s3() = val;
      break;
  }
}
template <typename T>
static inline SNN_ALWAYS_INLINE T get(cl::sycl::vec<T, 8> const& vec,
                                      int index) {
  switch (index) {
    case 0:
      return vec.s0();
    case 1:
      return vec.s1();
    case 2:
      return vec.s2();
    case 3:
      return vec.s3();
    case 4:
      return vec.s4();
    case 5:
      return vec.s5();
    case 6:
      return vec.s6();
    case 7:
      return vec.s7();
    default:
      SNN_UNREACHABLE;
      return 0;
  }
}
template <typename T>
static inline SNN_ALWAYS_INLINE void set(cl::sycl::vec<T, 8>& vec, int index,
                                         T val) {
  switch (index) {
    case 0:
      vec.s0() = val;
      break;
    case 1:
      vec.s1() = val;
      break;
    case 2:
      vec.s2() = val;
      break;
    case 3:
      vec.s3() = val;
      break;
    case 4:
      vec.s4() = val;
      break;
    case 5:
      vec.s5() = val;
      break;
    case 6:
      vec.s6() = val;
      break;
    case 7:
      vec.s7() = val;
      break;
  }
}
template <typename T>
static inline SNN_ALWAYS_INLINE T get(cl::sycl::vec<T, 16> const& vec,
                                      int index) {
  switch (index) {
    case 0:
      return vec.s0();
    case 1:
      return vec.s1();
    case 2:
      return vec.s2();
    case 3:
      return vec.s3();
    case 4:
      return vec.s4();
    case 5:
      return vec.s5();
    case 6:
      return vec.s6();
    case 7:
      return vec.s7();
    case 8:
      return vec.s8();
    case 9:
      return vec.s9();
    case 10:
      return vec.sA();
    case 11:
      return vec.sB();
    case 12:
      return vec.sC();
    case 13:
      return vec.sD();
    case 14:
      return vec.sE();
    case 15:
      return vec.sF();
    default:
      SNN_UNREACHABLE;
      return 0;
  }
}
template <typename T>
static inline SNN_ALWAYS_INLINE void set(cl::sycl::vec<T, 16>& vec, int index,
                                         T val) {
  switch (index) {
    case 0:
      vec.s0() = val;
      break;
    case 1:
      vec.s1() = val;
      break;
    case 2:
      vec.s2() = val;
      break;
    case 3:
      vec.s3() = val;
      break;
    case 4:
      vec.s4() = val;
      break;
    case 5:
      vec.s5() = val;
      break;
    case 6:
      vec.s6() = val;
      break;
    case 7:
      vec.s7() = val;
      break;
    case 8:
      vec.s8() = val;
      break;
    case 9:
      vec.s9() = val;
      break;
    case 10:
      vec.sA() = val;
      break;
    case 11:
      vec.sB() = val;
      break;
    case 12:
      vec.sC() = val;
      break;
    case 13:
      vec.sD() = val;
      break;
    case 14:
      vec.sE() = val;
      break;
    case 15:
      vec.sF() = val;
      break;
  }
}
}  // namespace vector_element
}  // namespace helpers
}  // namespace sycldnn
#endif  // PORTDNN_SRC_HELPERS_VECTOR_ELEMENT_H_
