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
#ifndef PORTDNN_SRC_HELPERS_MATH_H_
#define PORTDNN_SRC_HELPERS_MATH_H_

#include "portdnn/helpers/macros.h"

#include <CL/sycl.hpp>

namespace sycldnn {
namespace helpers {
namespace math {

template <typename T>
inline SNN_ALWAYS_INLINE T mad(T a, T b, T c) {
  return cl::sycl::mad(a, b, c);
}

/** Overload for 1-element vectors to workaround missing mad() function. */
template <typename T>
inline SNN_ALWAYS_INLINE cl::sycl::vec<T, 1> mad(cl::sycl::vec<T, 1> a,
                                                 cl::sycl::vec<T, 1> b,
                                                 cl::sycl::vec<T, 1> c) {
  return cl::sycl::vec<T, 1>{cl::sycl::mad(a.s0(), b.s0(), c.s0())};
}

/**
 * ComputeCpp's SYCL vectors have stringent alignment requirements, that can
 * conflict with some calling conventions preventing passing vectors by value.
 * To get around this we can pass by reference. The function will always be
 * inlined on the device so this should make no functional difference.
 */
template <typename T, int Dim>
inline SNN_ALWAYS_INLINE cl::sycl::vec<T, Dim> mad(
    cl::sycl::vec<T, Dim> const& a, cl::sycl::vec<T, Dim> const& b,
    cl::sycl::vec<T, Dim> const& c) {
  return cl::sycl::mad(a, b, c);
}

template <typename T>
inline SNN_ALWAYS_INLINE T dot(T a, T b) {
  return a * b;
}

/** For 1, 2, 3 and 4 element vectors just use cl::sycl::dot. */
template <typename T, int Width,
          typename std::enable_if<(Width <= 4), int>::type = 0>
inline SNN_ALWAYS_INLINE T dot(cl::sycl::vec<T, Width> a,
                               cl::sycl::vec<T, Width> b) {
  return cl::sycl::dot(a, b);
}

/** For larger vectors, compute dot on the upper half and lower half then sum
 * the results. */
template <typename T, int Width,
          typename std::enable_if<(Width > 4), int>::type = 0>
inline SNN_ALWAYS_INLINE T dot(cl::sycl::vec<T, Width> a,
                               cl::sycl::vec<T, Width> b) {
  return dot(a.hi(), b.hi()) + dot(a.lo(), b.lo());
}

template <
    typename T, typename U,
    typename std::enable_if<!std::is_floating_point<T>::value, int>::type = 0>
inline SNN_ALWAYS_INLINE T ratio(T a, U b) {
  return a / b;
}

template <
    typename T, int Width, typename U,
    typename std::enable_if<!std::is_floating_point<T>::value, int>::type = 0>
inline SNN_ALWAYS_INLINE cl::sycl::vec<T, Width> ratio(
    cl::sycl::vec<T, Width> a, U b) {
  return a / b;
}

template <
    typename T, typename U,
    typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
inline SNN_ALWAYS_INLINE T ratio(T a, U b) {
  return a * (T{1} / b);
}

template <
    typename T, int Width, typename U,
    typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
inline SNN_ALWAYS_INLINE cl::sycl::vec<T, Width> ratio(
    cl::sycl::vec<T, Width> a, U b) {
  return a * (T{1} / b);
}

/**
 * \brief Return ceil of \p x divided by \p y
 */
template <typename T>
inline SNN_ALWAYS_INLINE T divide_ceil(T x, T y) {
  return (x + y - 1) / y;
}

/**
 * \brief Round up \p x to the next multiple of \p alignment
 */
template <typename T>
inline SNN_ALWAYS_INLINE T align(T x, T alignment) {
  T r = x % alignment;
  return x + (r == 0 ? 0 : alignment - r);
}

}  // namespace math
}  // namespace helpers
}  // namespace sycldnn
#endif  // PORTDNN_SRC_HELPERS_MATH_H_
