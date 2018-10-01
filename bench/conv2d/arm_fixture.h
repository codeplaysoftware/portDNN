/*
 * Copyright 2018 Codeplay Software Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use these files except in compliance with the License.
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
#ifndef SYCLDNN_BENCH_CONV2D_ARM_FIXTURE_H_
#define SYCLDNN_BENCH_CONV2D_ARM_FIXTURE_H_

#include "arm_conv2d_executor.h"
#include "base_convolution_fixture.h"

#include "bench/fixture/add_arm_opencl_device_info.h"
#include "bench/fixture/string_reporter.h"

// The OpenCL C++ wrapper, used by ARM Compute Library, generates warnings
// about deprecated functions. This silences those warnings.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <arm_compute/runtime/CL/CLFunctions.h>
#include <arm_compute/runtime/CL/CLScheduler.h>
#include <arm_compute/runtime/CL/CLTensor.h>
#pragma GCC diagnostic pop

extern const char* commit_date;
extern const char* commit_hash;

namespace arm = arm_compute;

template <typename ParamGen, typename ConvType>
class ARMConvolutionBenchmark
    : public sycldnn::bench::ARMConv2DExecutor<
          ARMConvolutionBenchmark<ParamGen, ConvType>, ConvType>,
      public sycldnn::bench::StringReporter,
      public BaseConvolutionBenchmark {
 private:
  using State = benchmark::State;

 protected:
  void run(State& state) {
    auto params = ParamGen()();
    this->execute(state, params);

    // Get the SYCL device, and add device and driver info to the benchmark.
    auto device = cl::Device::getDefault();
    sycldnn::bench::device_info::add_opencl_device_info(device, *this);

    this->add_to_label("selector", "ARMCompute");
    this->add_to_label("git_hash", commit_hash);
    this->set_label(state);
  };
};

#define CONVOLUTION_BENCHMARK(name, ...)                                  \
  BENCHMARK_TEMPLATE_DEFINE_F(ARMConvolutionBenchmark, name, __VA_ARGS__) \
  (benchmark::State & state) { this->run(state); }                        \
  BENCHMARK_REGISTER_F(ARMConvolutionBenchmark, name)                     \
      ->UseManualTime()                                                   \
      ->Unit(benchmark::kNanosecond);

#endif  // define SYCLDNN_BENCH_CONV2D_ARM_FIXTURE_H_
