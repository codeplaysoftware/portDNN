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
#ifndef SYCLDNN_BENCH_ACL_FIXTURE_H_
#define SYCLDNN_BENCH_ACL_FIXTURE_H_

#include "fixture.h"

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
class ARMConvolutionBenchmark : public BaseConvolutionBenchmark {
 private:
  using State = benchmark::State;
  using Conv2DParams = sycldnn::conv2d::Conv2DParams;

 protected:
  void add_opencl_device_info(const cl::Device& device);
  void execute(benchmark::State& state,
               sycldnn::conv2d::Conv2DParams const& params);
  void run(State& state) {
    auto params = ParamGen()();
    this->execute(state, params);
  };
};

template <typename ParamGen, typename ConvType>
void ARMConvolutionBenchmark<ParamGen, ConvType>::execute(
    benchmark::State& state, sycldnn::conv2d::Conv2DParams const& params) {
  auto& scheduler = arm::CLScheduler::get();
  auto context = cl::Context::getDefault();
  auto queue = cl::CommandQueue::getDefault();
  auto device = cl::Device::getDefault();
  scheduler.init(context, queue, device);

  // Allocate tensors.
  arm::CLTensor W, X, Z, B;
  X.allocator()->init(
      arm::TensorInfo(arm::TensorShape(params.in_rows, params.in_cols,
                                       params.channels, params.batch),
                      arm::Format::F32));
  Z.allocator()->init(
      arm::TensorInfo(arm::TensorShape(params.out_rows, params.out_cols,
                                       params.features, params.batch),
                      arm::Format::F32));
  W.allocator()->init(
      arm::TensorInfo(arm::TensorShape(params.window_rows, params.window_cols,
                                       params.channels, params.features),
                      arm::Format::F32));
  B.allocator()->init(
      arm::TensorInfo(arm::TensorShape(params.features), arm::Format::F32));

  // Construct a convolution layer.
  arm::CLConvolutionLayer conv1;
  arm::PadStrideInfo psi(params.stride_cols, params.stride_rows,
                         params.window_cols / 2, params.window_rows / 2);
  conv1.configure(&X, &W, &B, &Z, psi);

  // Validate the configuration.
  auto status = conv1.validate(X.info(), W.info(), B.info(), Z.info(), psi);
  if (!status) {
    state.SkipWithError(status.error_description().c_str());
    return;
  }

  // Allocate the tensors themselves.
  X.allocator()->allocate();
  Z.allocator()->allocate();
  W.allocator()->allocate();
  B.allocator()->allocate();

  // Run the layer once to eliminate lazy behaviour.
  conv1.run();
  scheduler.sync();

  for (auto _ : state) {
    auto start = std::chrono::high_resolution_clock::now();
    conv1.run();
    scheduler.sync();

    auto end = std::chrono::high_resolution_clock::now();

    auto elapsed_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    state.SetIterationTime(elapsed_seconds.count());
  }

  X.allocator()->free();
  Z.allocator()->free();
  W.allocator()->free();
  B.allocator()->free();

  // Get the OpenCL device, and add device and driver info to the key-value map.
  add_opencl_device_info(device);
  set_items_processed<ConvType>(state, params);
  add_param_counters(state, params);
  add_bandwidth_counters<float>(state,
                                sycldnn::conv2d::get_sizes<ConvType>(params));

  key_value_map["selector"] = "ARMCompute";
  key_value_map["git_hash"] = commit_hash;
  set_label(state);
}

template <typename ParamGen, typename ConvType>
void ARMConvolutionBenchmark<ParamGen, ConvType>::add_opencl_device_info(
    const cl::Device& device) {
  // OpenCL is unclear whether strings returned from clGet*Info() should be
  // null terminated, and ComputeCpp currently copies embedded nulls. On some
  // OpenCL implemntations this results in strings that behave unexpectedly
  // when appended to. This lambda trims those strings.
  auto trim = [](std::string s) -> std::string {
    s.resize(strlen(s.c_str()));
    return s;
  };
  std::string device_name;
  std::string device_version;
  std::string vendor_name;
  std::string driver_version;
  device.getInfo(CL_DEVICE_NAME, &device_name);
  device.getInfo(CL_DEVICE_VERSION, &device_version);
  device.getInfo(CL_DEVICE_VENDOR, &vendor_name);
  device.getInfo(CL_DRIVER_VERSION, &driver_version);

  key_value_map["device_name"] = trim(device_name);
  key_value_map["device_version"] = trim(device_version);
  key_value_map["vendor_name"] = trim(vendor_name);
  key_value_map["driver_version"] = trim(driver_version);
}

#define CONVOLUTION_BENCHMARK(name, ...)                                  \
  BENCHMARK_TEMPLATE_DEFINE_F(ARMConvolutionBenchmark, name, __VA_ARGS__) \
  (benchmark::State & state) { this->run(state); }                        \
  BENCHMARK_REGISTER_F(ARMConvolutionBenchmark, name)                     \
      ->UseManualTime()                                                   \
      ->Unit(benchmark::kNanosecond);

#endif  // define SYCLDNN_BENCH_FIXTURE_H_
