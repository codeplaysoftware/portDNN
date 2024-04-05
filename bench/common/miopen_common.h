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
#ifndef SYCLDNN_BENCH_COMMON_MIOPEN_COMMON_H_
#define SYCLDNN_BENCH_COMMON_MIOPEN_COMMON_H_

#include <hip/hip_runtime.h>
#include <miopen/miopen.h>
#include <CL/sycl.hpp>

#include <stdexcept>

template <typename F, typename... Args>
inline void hip_check(F f, Args... args) {
  auto st = f(args...);
  if (hipSuccess != st) {
    throw std::runtime_error("Unexpected HIP function failure: " +
                             std::to_string(st));
  }
}

template <template <typename T> class F, typename U, typename... Args>
inline void hip_check(F<U> f, U u, size_t s) {
  auto st = f(u, s);
  if (hipSuccess != st) {
    throw std::runtime_error("Unexpected MIOpen function failure: " +
                             std::to_string(st));
  }
}

template <typename F, typename... Args>
inline void miopen_check(F f, Args... args) {
  auto st = f(args...);
  if (miopenStatusSuccess != st) {
    throw std::runtime_error("Unexpected MIOpen function failure: " +
                             std::to_string(st));
  }
}

template <typename T>
miopenDataType_t get_miopen_dtype() {
  constexpr auto is_float = std::is_same<T, float>::value;
  constexpr auto is_half = std::is_same<T, cl::sycl::half>::value;

  if (is_float) {
    return miopenDataType_t::miopenFloat;
  } else if (is_half) {
    return miopenDataType_t::miopenHalf;
  } else {
    throw std::runtime_error(
        "SNN MIOpen benchmarks do not support specified data type");
  }
}

#endif  // SYCLDNN_BENCH_MIOPEN_COMMON_H_
