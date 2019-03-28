/*
 * Copyright 2019 Codeplay Software Ltd
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
#include <string>

#include "sycldnn/conv2d/algorithm.h"
#include "sycldnn/conv2d/params.h"

#include "sycldnn/conv2d/selector/default_selector.h"
#include "sycldnn/helpers/macros.h"

#include "sycldnn/conv2d/selector/selector.h"

/**
 * Implemented selectors for convolution algorithms on various devices,
 * selecting the fastest algorithm for the given device and convolution
 * parameters.
 */

namespace {

/**
 * A selector which makes no assumption about the underlying device.
 * This is chosen as a fall-back when the available device is not recognised.
 */
class DefaultSelector final : public sycldnn::conv2d::Selector {
 public:
  /**
   * Selects a suitable convolution algorithm for the target platform, given
   * the convolution parameters.
   */
  sycldnn::conv2d::Algorithm select(
      sycldnn::conv2d::Conv2DParams const& params) override {
    SNN_UNUSED_VAR(params)
    return sycldnn::conv2d::Algorithm::Direct;
  }

  char const* name() const override { return "Default"; }
};

/** A selector which assumes the underlying device to run on is an Intel CPU. */
class IntelCPUSelector final : public sycldnn::conv2d::Selector {
 public:
  /**
   * Selects a suitable convolution algorithm given the convolution parameters.
   */
  sycldnn::conv2d::Algorithm select(
      sycldnn::conv2d::Conv2DParams const& params) override {
    if (params.stride_rows != 1 && params.stride_cols != 1 &&
        params.stride_rows == params.stride_cols &&
        params.window_rows == params.window_cols) {
      return sycldnn::conv2d::Algorithm::Tiled;
    } else if ((params.window_rows == 3 && params.window_cols == 3) ||
               (params.window_rows == 1 && params.window_cols == 3) ||
               (params.window_rows == 3 && params.window_cols == 1)) {
      return sycldnn::conv2d::Algorithm::Winograd;
    } else {
      return sycldnn::conv2d::Algorithm::Direct;
    }
  }

  char const* name() const override { return "IntelCPUSelector"; }
};

/** A selector which assumes the underlying device to run on is an Intel GPU. */
class IntelGPUSelector final : public sycldnn::conv2d::Selector {
 public:
  /**
   * Selects a suitable convolution algorithm given the convolution parameters.
   */
  sycldnn::conv2d::Algorithm select(
      sycldnn::conv2d::Conv2DParams const& params) override {
    if (params.stride_rows != 1 && params.stride_cols != 1 &&
        params.stride_rows == params.stride_cols &&
        params.window_rows == params.window_cols) {
      return sycldnn::conv2d::Algorithm::Tiled;
    } else if ((params.window_rows == 3 && params.window_cols == 3) ||
               (params.window_rows == 1 && params.window_cols == 3) ||
               (params.window_rows == 3 && params.window_cols == 1)) {
      return sycldnn::conv2d::Algorithm::Winograd;
    } else {
      return sycldnn::conv2d::Algorithm::Direct;
    }
  }

  char const* name() const override { return "IntelGPUSelector"; }
};

/** A selector which assumes the underlying device to run on is an ARM GPU. */
class ARMGPUSelector final : public sycldnn::conv2d::Selector {
 public:
  /**
   * Selects a suitable convolution algorithm given the convolution parameters.
   */
  sycldnn::conv2d::Algorithm select(
      sycldnn::conv2d::Conv2DParams const& params) override {
    if (params.stride_rows != 1 && params.stride_cols != 1 &&
        params.stride_rows == params.stride_cols &&
        params.window_rows == params.window_cols) {
      return sycldnn::conv2d::Algorithm::Tiled;
    } else if ((params.window_rows == 3 && params.window_cols == 3) ||
               (params.window_rows == 1 && params.window_cols == 3) ||
               (params.window_rows == 3 && params.window_cols == 1)) {
      return sycldnn::conv2d::Algorithm::Winograd;
    } else {
      return sycldnn::conv2d::Algorithm::Direct;
    }
  }

  char const* name() const override { return "ARMGPUSelector"; }
};

/** A selector which assumes the underlying device to run on is an AMD GPU. */
class AMDGPUSelector final : public sycldnn::conv2d::Selector {
 public:
  /**
   * Selects a suitable convolution algorithm given the convolution parameters.
   */
  sycldnn::conv2d::Algorithm select(
      sycldnn::conv2d::Conv2DParams const& params) override {
    if (params.stride_rows != 1 && params.stride_cols != 1 &&
        params.stride_rows == params.stride_cols &&
        params.window_rows == params.window_cols) {
      return sycldnn::conv2d::Algorithm::Tiled;
    } else if ((params.window_rows == 3 && params.window_cols == 3) ||
               (params.window_rows == 1 && params.window_cols == 3) ||
               (params.window_rows == 3 && params.window_cols == 1)) {
      return sycldnn::conv2d::Algorithm::Winograd;
    } else {
      return sycldnn::conv2d::Algorithm::Direct;
    }
  }

  char const* name() const override { return "AMDGPUSelector"; }
};

}  // namespace

namespace sycldnn {
namespace conv2d {

std::unique_ptr<Selector> get_default_selector(const cl::sycl::device& device) {
  auto vendor = device.get_info<cl::sycl::info::device::vendor>();
  bool is_intel = vendor.find("Intel(R) Corporation") != std::string::npos;
  bool is_amd =
      vendor.find("Advanced Micro Devices, Inc.") != std::string::npos;
  bool is_arm = vendor.find("ARM") != std::string::npos;

  bool is_cpu = device.is_cpu();
  bool is_gpu = device.is_gpu();

  std::unique_ptr<Selector> selector{new DefaultSelector{}};

  if (is_intel && is_cpu) {
    selector.reset(new IntelCPUSelector{});
  } else if (is_intel && is_gpu) {
    selector.reset(new IntelGPUSelector{});
  } else if (is_amd && is_gpu) {
    selector.reset(new AMDGPUSelector{});
  } else if (is_arm && is_gpu) {
    selector.reset(new ARMGPUSelector{});
  }

  return selector;
}

}  // namespace conv2d
}  // namespace sycldnn
