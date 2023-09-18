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
#include "portdnn/conv2d/algorithm.h"
#include "portdnn/conv2d/params.h"

#include "portdnn/conv2d/selector/default_selector.h"
#include "portdnn/helpers/macros.h"

#include "portdnn/conv2d/selector/selector.h"

#include <memory>
#include <string>

#include <CL/sycl.hpp>

#include "portdnn/export.h"

/**
 * \file Implemented selectors for convolution algorithms on various devices,
 * selecting the fastest algorithm for the given device and convolution
 * parameters.
 */

namespace {

/**
 * A selector which makes no assumption about the underlying device.
 * This is chosen as a fall-back when the available device is not recognised.
 */
class DefaultSelector : public sycldnn::conv2d::Selector {
 public:
  /**
   * Provide a reasonable default selection for all devices.
   * \copydoc Selector::select_forward
   */
  sycldnn::conv2d::Algorithm select_forward(
      sycldnn::conv2d::Conv2DParams const& params) override {
    // Im2Col is the only algorithm that supports Grouped Convolution.
    if (params.groups > 1) {
      return sycldnn::conv2d::Algorithm::Im2col;
    }
    // For 1x1s1 the convolution is equivalent to a matrix multiply.
    if (params.stride_rows == 1 && params.stride_cols == 1 &&
        params.window_rows == 1 && params.window_cols == 1) {
      return sycldnn::conv2d::Algorithm::Matmul;
    }
    // Winograd is supported for 1x3s1, 3x1s1, 3x3s1.
    if (params.stride_rows == 1 && params.stride_cols == 1) {
      if (params.window_rows == 3 && params.window_cols == 3) {
        return sycldnn::conv2d::Algorithm::WinogradLarge;
      } else if ((params.window_rows == 1 && params.window_cols == 3) ||
                 (params.window_rows == 3 && params.window_cols == 1)) {
        return sycldnn::conv2d::Algorithm::Winograd;
      }
    }
    // Tiled is supported for 1x1s1, 1x1s2, 3x3s1, 3x3s2, 5x5s1.
    if (params.stride_rows == params.stride_cols &&
        params.window_rows == params.window_cols) {
      if ((params.window_rows == 5 && params.stride_rows == 1) ||
          ((params.window_rows == 1 || params.window_rows == 3) &&
           (params.stride_rows == 1 || params.stride_rows == 2))) {
        return sycldnn::conv2d::Algorithm::Tiled;
      }
    }
    // Fallback to use Im2col for anything else.
    return sycldnn::conv2d::Algorithm::Im2col;
  }

  /**
   * Provide a reasonable default selection for all devices.
   * \copydoc Selector::select_input_backprop
   */
  sycldnn::conv2d::Algorithm select_input_backprop(
      sycldnn::conv2d::Conv2DParams const& params) override {
    // For 1x1s1 the convolution is equivalent to a matrix multiply.
    if (params.stride_rows == 1 && params.stride_cols == 1 &&
        params.window_rows == 1 && params.window_cols == 1) {
      return sycldnn::conv2d::Algorithm::Matmul;
    }
    // Winograd is supported for 1x3s1, 3x1s1, 3x3s1.
    if (params.stride_rows == 1 && params.stride_cols == 1) {
      if (params.window_rows == 3 && params.window_cols == 3) {
        return sycldnn::conv2d::Algorithm::WinogradLarge;
      } else if ((params.window_rows == 1 && params.window_cols == 3) ||
                 (params.window_rows == 3 && params.window_cols == 1)) {
        return sycldnn::conv2d::Algorithm::Winograd;
      }
    }
    // Fallback to use Im2col for anything else.
    return sycldnn::conv2d::Algorithm::Im2col;
  }

  /**
   * Provide a reasonable default selection for all devices.
   * \copydoc Selector::select_filter_backprop
   */
  sycldnn::conv2d::Algorithm select_filter_backprop(
      sycldnn::conv2d::Conv2DParams const& params) override {
    // For 1x1s1 the convolution is equivalent to a matrix multiply.
    if (params.stride_rows == 1 && params.stride_cols == 1 &&
        params.window_rows == 1 && params.window_cols == 1) {
      return sycldnn::conv2d::Algorithm::Matmul;
    }
    // Winograd is supported for 1x3s1, 3x1s1, 3x3s1.
    if (params.stride_rows == 1 && params.stride_cols == 1) {
      if (params.window_rows == 3 && params.window_cols == 3) {
        return sycldnn::conv2d::Algorithm::WinogradLarge;
      } else if ((params.window_rows == 1 && params.window_cols == 3) ||
                 (params.window_rows == 3 && params.window_cols == 1)) {
        return sycldnn::conv2d::Algorithm::Winograd;
      }
    }
    // Fallback to use Im2col for anything else.
    return sycldnn::conv2d::Algorithm::Im2col;
  }

  char const* name() const override { return "DefaultSelector"; }
};

/** A selector which assumes the underlying device to run on is an Intel CPU. */
class IntelCPUSelector final : public DefaultSelector {
 public:
  sycldnn::conv2d::Algorithm select_forward(
      sycldnn::conv2d::Conv2DParams const& params) override {
    return this->DefaultSelector::select_forward(params);
  }

  sycldnn::conv2d::Algorithm select_input_backprop(
      sycldnn::conv2d::Conv2DParams const& params) override {
    return this->DefaultSelector::select_input_backprop(params);
  }

  sycldnn::conv2d::Algorithm select_filter_backprop(
      sycldnn::conv2d::Conv2DParams const& params) override {
    return this->DefaultSelector::select_filter_backprop(params);
  }

  char const* name() const override { return "IntelCPUSelector"; }
};

/** A selector which assumes the underlying device to run on is an Intel GPU. */
class IntelGPUSelector final : public DefaultSelector {
 public:
  sycldnn::conv2d::Algorithm select_forward(
      sycldnn::conv2d::Conv2DParams const& params) override {
    return this->DefaultSelector::select_forward(params);
  }

  sycldnn::conv2d::Algorithm select_input_backprop(
      sycldnn::conv2d::Conv2DParams const& params) override {
    return this->DefaultSelector::select_input_backprop(params);
  }

  sycldnn::conv2d::Algorithm select_filter_backprop(
      sycldnn::conv2d::Conv2DParams const& params) override {
    return this->DefaultSelector::select_filter_backprop(params);
  }

  char const* name() const override { return "IntelGPUSelector"; }
};

/** A selector which assumes the underlying device to run on is an ARM GPU. */
class ARMGPUSelector final : public DefaultSelector {
 public:
  sycldnn::conv2d::Algorithm select_forward(
      sycldnn::conv2d::Conv2DParams const& params) override {
    return this->DefaultSelector::select_forward(params);
  }

  sycldnn::conv2d::Algorithm select_input_backprop(
      sycldnn::conv2d::Conv2DParams const& params) override {
    return this->DefaultSelector::select_input_backprop(params);
  }

  sycldnn::conv2d::Algorithm select_filter_backprop(
      sycldnn::conv2d::Conv2DParams const& params) override {
    return this->DefaultSelector::select_filter_backprop(params);
  }

  char const* name() const override { return "ARMGPUSelector"; }
};

/** A selector which assumes the underlying device to run on is an AMD GPU. */
class AMDGPUSelector final : public DefaultSelector {
 public:
  sycldnn::conv2d::Algorithm select_forward(
      sycldnn::conv2d::Conv2DParams const& params) override {
    return this->DefaultSelector::select_forward(params);
  }

  sycldnn::conv2d::Algorithm select_input_backprop(
      sycldnn::conv2d::Conv2DParams const& params) override {
    return this->DefaultSelector::select_input_backprop(params);
  }

  sycldnn::conv2d::Algorithm select_filter_backprop(
      sycldnn::conv2d::Conv2DParams const& params) override {
    return this->DefaultSelector::select_filter_backprop(params);
  }

  char const* name() const override { return "AMDGPUSelector"; }
};

/** A selector specialised for PowerVR GPUs. */
class PowerVRSelector final : public DefaultSelector {
 public:
  sycldnn::conv2d::Algorithm select_forward(
      sycldnn::conv2d::Conv2DParams const& params) override {
    if (params.stride_cols > 1 && params.stride_cols > 1) {
      return sycldnn::conv2d::Algorithm::Im2col;
    }
    if (params.window_rows > 1 && params.out_rows < 10 &&
        params.out_cols < 10) {
      return sycldnn::conv2d::Algorithm::Im2col;
    }
    if (params.channels == 3) {
      if (params.window_rows == 3 && params.window_cols == 3) {
        return sycldnn::conv2d::Algorithm::Tiled;
      }
      return sycldnn::conv2d::Algorithm::Im2col;
    }
    if (params.window_rows == 3 && params.window_cols == 3) {
      if (params.features < 30) {
        return sycldnn::conv2d::Algorithm::Tiled;
      }
      if (params.batch < 4 && params.out_rows < 15 && params.out_cols < 15) {
        return sycldnn::conv2d::Algorithm::Winograd;
      }
      if (params.out_rows < 13 && params.out_cols < 13) {
        return sycldnn::conv2d::Algorithm::Winograd;
      }
    }
    return this->DefaultSelector::select_forward(params);
  }

  char const* name() const override { return "PowerVRSelector"; }
};

}  // namespace

namespace sycldnn {
namespace conv2d {

SNN_EXPORT std::unique_ptr<Selector> get_default_selector(
    const cl::sycl::device& device) {
  auto vendor = device.get_info<cl::sycl::info::device::vendor>();
  bool is_intel = vendor.find("Intel(R) Corporation") != std::string::npos;
  bool is_amd =
      vendor.find("Advanced Micro Devices, Inc.") != std::string::npos;
  bool is_arm = vendor.find("ARM") != std::string::npos;
  bool is_img = vendor.find("Imagination Technologies") != std::string::npos;

  bool is_cpu = device.is_cpu();
  bool is_gpu = device.is_gpu();

  std::unique_ptr<Selector> selector{};

  if (is_intel && is_cpu) {
    selector.reset(new IntelCPUSelector{});
  } else if (is_intel && is_gpu) {
    selector.reset(new IntelGPUSelector{});
  } else if (is_amd && is_gpu) {
    selector.reset(new AMDGPUSelector{});
  } else if (is_arm && is_gpu) {
    selector.reset(new ARMGPUSelector{});
  } else if (is_img && is_gpu) {
    selector.reset(new PowerVRSelector{});
  } else {
    selector.reset(new DefaultSelector{});
  }

  return selector;
}

}  // namespace conv2d
}  // namespace sycldnn
