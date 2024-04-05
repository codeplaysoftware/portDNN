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
#ifndef SYCLDNN_BENCH_COMMON_CUDNN_COMMON_H_
#define SYCLDNN_BENCH_COMMON_CUDNN_COMMON_H_

#include <stdexcept>

#include <cuda.h>
#include <cudnn.h>
#include <CL/sycl.hpp>

#include "cudnn_exceptions.h"

template <typename F, typename... Args>
inline void cuda_check(F f, Args... args) {
  auto st = f(args...);
  if (cudaSuccess != st) {
    throw std::runtime_error("Unexpected CUDA function failure: " +
                             std::to_string(st));
  }
}

template <template <typename T> class F, typename U, typename... Args>
inline void cuda_check(F<U> f, U u, size_t s) {
  auto st = f(u, s);
  if (cudaSuccess != st) {
    throw std::runtime_error("Unexpected CUDA function failure: " +
                             std::to_string(st));
  }
}

template <typename F, typename... Args>
inline void cudnn_check(F f, Args... args) {
  auto st = f(args...);
  if (CUDNN_STATUS_SUCCESS != st) {
    throw CUDNNUnexpectedFailure(st);
  }
}

template <typename T>
cudnnDataType_t get_cudnn_dtype() {
  constexpr auto is_float = std::is_same<T, float>::value;
  constexpr auto is_double = std::is_same<T, double>::value;
  constexpr auto is_half = std::is_same<T, cl::sycl::half>::value;
  static_assert(is_float || is_double || is_half,
                "SNN CUDNN benchmarks do not support specified datatype.");

  if (is_float) {
    return cudnnDataType_t::CUDNN_DATA_FLOAT;
  } else if (is_double) {
    return cudnnDataType_t::CUDNN_DATA_DOUBLE;
  } else {
    return cudnnDataType_t::CUDNN_DATA_HALF;
  }
}

cudnnTensorFormat_t get_cudnn_tensor_format(sycldnn::DataFormat format) {
  if (format == sycldnn::DataFormat::NCHW) {
    return cudnnTensorFormat_t::CUDNN_TENSOR_NCHW;
  } else if (format == sycldnn::DataFormat::NHWC) {
    return cudnnTensorFormat_t::CUDNN_TENSOR_NHWC;
  } else {
    throw std::runtime_error(
        "SNN CUDNN benchmarks do not support specified tensor format.");
  }
}

template <bool UseTensorOp>
struct get_cudnn_math_type;

template <>
struct get_cudnn_math_type<false> {
  static constexpr cudnnMathType_t value = cudnnMathType_t::CUDNN_FMA_MATH;
};

template <>
struct get_cudnn_math_type<true> {
  static constexpr cudnnMathType_t value =
      cudnnMathType_t::CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION;
};

#endif  // SYCLDNN_BENCH_CUDNN_COMMON_H_
