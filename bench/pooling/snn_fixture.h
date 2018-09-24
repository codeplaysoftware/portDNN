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
#ifndef SYCLDNN_BENCH_POOLING_SNN_FIXTURE_H_
#define SYCLDNN_BENCH_POOLING_SNN_FIXTURE_H_

#include <cassert>
#include <unsupported/Eigen/CXX11/Tensor>
#include "fixture.h"
// Need to ensure that Eigen is included before the backend.
// The backend itself doesn't include Eigen to allow users of SYCL-DNN to
// include it however they wish.
#include "sycldnn/backend/eigen_backend.h"
#include "sycldnn/pooling/launch.h"

template <typename ParamGen, typename Direction,
          template <typename> class Operator>
class SNNPoolingBenchmark : public BasePoolingBenchmark {
 private:
  using State = benchmark::State;

 protected:
  void add_opencl_device_info(const cl::sycl::device& device);
  void execute(benchmark::State& state, PoolingParams const& params);

  void run(State& state) {
    auto params = ParamGen()();
    this->execute(state, params);
  };

 private:
  Eigen::QueueInterface* get_eigen_queue() {
    static Eigen::QueueInterface queue_interface{cl::sycl::default_selector{}};
    return &queue_interface;
  }
};

template <typename ParamGen, typename Direction,
          template <typename> class Operator>
void SNNPoolingBenchmark<ParamGen, Direction, Operator>::execute(
    benchmark::State& state, PoolingParams const& params) {
  Eigen::SyclDevice device{get_eigen_queue()};
  sycldnn::backend::EigenBackend backend{device};

  auto pool_sizes = sycldnn::pooling::get_sizes<Direction>(params);

  size_t inp_bytes = pool_sizes.input_size * sizeof(float);
  float* inp_gpu = static_cast<float*>(device.allocate(inp_bytes));
  size_t out_bytes = pool_sizes.output_size * sizeof(float);
  float* out_gpu = static_cast<float*>(device.allocate(out_bytes));

  {  // Ensure the kernel is built before benchmarking
    auto status = sycldnn::pooling::launch<float, Operator, Direction>(
        inp_gpu, out_gpu, params, backend);
    status.event.wait();

    if (sycldnn::StatusCode::OK != status.status) {
      state.SkipWithError(
          "Invalid or unsupported benchmark configuration. "
          "This may be expected behaviour and does not indicate a problem.");
      return;
    }
  }

  for (auto _ : state) {
    auto start = std::chrono::high_resolution_clock::now();
    auto status = sycldnn::pooling::launch<float, Operator, Direction>(
        inp_gpu, out_gpu, params, backend);

    status.event.wait();
    auto end = std::chrono::high_resolution_clock::now();

    auto elapsed_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    state.SetIterationTime(elapsed_seconds.count());
  }

  device.deallocate_all();

  // Get the SYCL device, and add device and driver info to the key-value map.
  auto dev = backend.get_queue().get_device();
  add_opencl_device_info(dev);

  set_items_processed<Direction>(state, params);
  add_param_counters(state, params);
  add_bandwidth_counters<float>(state, pool_sizes);

  key_value_map["git_hash"] = commit_hash;
  set_label(state);
}

template <typename ParamGen, typename Direction,
          template <typename> class Operator>
void SNNPoolingBenchmark<ParamGen, Direction, Operator>::add_opencl_device_info(
    const cl::sycl::device& device) {
  // OpenCL is unclear whether strings returned from clGet*Info() should be
  // null terminated, and ComputeCpp currently copies embedded nulls.
  // On some OpenCL implementations this results in strings that behave
  // unexpectedly when appended to. This lambda trims those strings.
  auto trim = [](std::string s) -> std::string {
    s.resize(strlen(s.c_str()));
    return s;
  };
  auto device_name = device.get_info<cl::sycl::info::device::name>();
  auto device_version = device.get_info<cl::sycl::info::device::version>();
  auto vendor_name = device.get_info<cl::sycl::info::device::vendor>();
  auto driver_version =
      device.get_info<cl::sycl::info::device::driver_version>();
  key_value_map["device_name"] = trim(device_name);
  key_value_map["device_version"] = trim(device_version);
  key_value_map["vendor_name"] = trim(vendor_name);
  key_value_map["driver_version"] = trim(driver_version);
}

#define POOLING_BENCHMARK(name, ...)                                  \
  BENCHMARK_TEMPLATE_DEFINE_F(SNNPoolingBenchmark, name, __VA_ARGS__) \
  (benchmark::State & state) { this->run(state); }                    \
  BENCHMARK_REGISTER_F(SNNPoolingBenchmark, name)                     \
      ->UseManualTime()                                               \
      ->Unit(benchmark::kNanosecond);

#endif  // define SYCLDNN_BENCH_POOLING_SNN_FIXTURE_H_
