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
#ifndef SYCLDNN_BENCH_COMMON_CUDNN_EXCEPTIONS_H_
#define SYCLDNN_BENCH_COMMON_CUDNN_EXCEPTIONS_H_

#include <cudnn.h>
#include <exception>
#include <string>

inline std::string get_cudnn_status_str(cudnnStatus_t status) {
  switch (status) {
    case CUDNN_STATUS_SUCCESS:
      return "CUDNN_STATUS_SUCCESS";
    case CUDNN_STATUS_NOT_INITIALIZED:
      return "CUDNN_STATUS_NOT_INITIALIZED";
    case CUDNN_STATUS_ALLOC_FAILED:
      return "CUDNN_STATUS_ALLOC_FAILED";
    case CUDNN_STATUS_BAD_PARAM:
      return "CUDNN_STATUS_BAD_PARAM";
    case CUDNN_STATUS_ARCH_MISMATCH:
      return "CUDNN_STATUS_ARCH_MISMATCH";
    case CUDNN_STATUS_MAPPING_ERROR:
      return "CUDNN_STATUS_MAPPING_ERROR";
    case CUDNN_STATUS_EXECUTION_FAILED:
      return "CUDNN_STATUS_EXECUTION_FAILED";
    case CUDNN_STATUS_INTERNAL_ERROR:
      return "CUDNN_STATUS_INTERNAL_ERROR";
    case CUDNN_STATUS_NOT_SUPPORTED:
      return "CUDNN_STATUS_NOT_SUPPORTED";
    case CUDNN_STATUS_LICENSE_ERROR:
      return "CUDNN_STATUS_LICENSE_ERROR";
    case CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING:
      return "CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING";
    case CUDNN_STATUS_RUNTIME_IN_PROGRESS:
      return "CUDNN_STATUS_RUNTIME_IN_PROGRESS";
    case CUDNN_STATUS_RUNTIME_FP_OVERFLOW:
      return "CUDNN_STATUS_RUNTIME_FP_OVERFLOW";
    default:
      throw std::runtime_error("Unknown CuDNN status");
  }
}

class CUDNNFailure : public std::exception {
 public:
  CUDNNFailure() = default;
  explicit CUDNNFailure(cudnnStatus_t status)
      : message_("CuDNN failed with error status " +
                 get_cudnn_status_str(status) + ".") {}
  explicit CUDNNFailure(std::string const& msg) : message_(msg) {}

  virtual const char* what() const noexcept override {
    return message_.c_str();
  }

 protected:
  std::string message_;
};

class CUDNNExpectedFailure : public CUDNNFailure {
 public:
  CUDNNExpectedFailure(cudnnStatus_t status) : CUDNNFailure(status) {}
  CUDNNExpectedFailure(std::string const& msg) : CUDNNFailure(msg) {}
};

class CUDNNUnexpectedFailure : public CUDNNFailure {
 public:
  CUDNNUnexpectedFailure(cudnnStatus_t status) : CUDNNFailure(status) {}
  CUDNNUnexpectedFailure(std::string const& msg) : CUDNNFailure(msg) {}
};

#endif  // SYCLDNN_BENCH_COMMON_CUDNN_EXCEPTIONS_H_
