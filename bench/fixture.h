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
#ifndef SYCLDNN_BENCH_FIXTURE_H_
#define SYCLDNN_BENCH_FIXTURE_H_

#include <benchmark/benchmark.h>

#include <unsupported/Eigen/CXX11/Tensor>
// Need to ensure that Eigen is included before the backend.
// The backend itself doesn't include Eigen to allow users of SYCL-DNN to
// include it however they wish.
#include "sycldnn/backend/eigen_backend.h"
#include "sycldnn/conv2d/launch.h"
#include "sycldnn/conv2d/params.h"
#include "sycldnn/conv2d/selector/direct_selector.h"
#include "sycldnn/conv2d/selector/tiled_selector.h"

extern const char* commit_date;
extern const char* commit_hash;

namespace {

class BaseConvolutionBenchmark : public benchmark::Fixture {
 private:
  using State = benchmark::State;
  using Conv2DParams = sycldnn::conv2d::Conv2DParams;

 protected:
  template <typename ConvType>
  void execute(benchmark::State& state,
               sycldnn::conv2d::Conv2DParams const& params,
               sycldnn::conv2d::Selector& selector);

 private:
  // Adds the convolution parameters to the counter set.
  void add_param_counters(State& state, Conv2DParams const& params);

  // Adds theoretical best-case bandwidth requirements to the counter set.
  template <typename T>
  void add_bandwidth_counters(State& state,
                              sycldnn::conv2d::ConvSizes const& sizes);

  // Adds information about the OpenCL device and driver version to the
  // key-value map.
  void add_opencl_device_info(cl::sycl::device const& dev);

  // Records the number of elements processed to the counter set. How this
  // calculated varies based on the type of convolution.
  template <typename ConvType>
  void set_items_processed(benchmark::State& state,
                           sycldnn::conv2d::Conv2DParams const& params);

  // Serializes the key-value map into a single comma separated string and
  // stores it in the benchmark label.
  void set_label(State& state);

  // A map holding key-value pairs to be emitted along with the counter set.
  std::map<std::string, std::string> key_value_map;
};

template <typename ParamGen, typename ConvType, typename Selector>
class ConvolutionBenchmark : public BaseConvolutionBenchmark {
 private:
  using State = benchmark::State;
  using Conv2DParams = sycldnn::conv2d::Conv2DParams;

 protected:
  void run(State& state) {
    auto params = ParamGen()();
    auto selector = Selector();
    this->execute<ConvType>(state, params, selector);
  };
};

// Add a full set of counters corresponding to the convolution parameters.
void BaseConvolutionBenchmark::add_param_counters(
    benchmark::State& state, sycldnn::conv2d::Conv2DParams const& params) {
  state.counters["batch"] = params.batch;
  state.counters["in_rows"] = params.in_rows;
  state.counters["in_cols"] = params.in_cols;
  state.counters["channels"] = params.channels;
  state.counters["out_rows"] = params.out_rows;
  state.counters["out_cols"] = params.out_cols;
  state.counters["features"] = params.features;
  state.counters["stride_rows"] = params.stride_rows;
  state.counters["stride_cols"] = params.stride_cols;
  state.counters["fil_rows"] = params.window_rows;
  state.counters["fil_cols"] = params.window_cols;
  state.counters["pad_rows"] = params.pad_rows;
  state.counters["pad_cols"] = params.pad_cols;
}

// Calculate the optimal bandwidth requirements, and add corresponding counters.
// This assumes each filter and input element is read exactly once, rather than
// the actual behaviour where multiple threads may re-read the same values.
template <typename ElementType>
void BaseConvolutionBenchmark::add_bandwidth_counters(
    benchmark::State& state, sycldnn::conv2d::ConvSizes const& sizes) {
  // Compute the size of each element in bytes.
  auto element_bytes = sizeof(ElementType);

  state.counters["bytes_read"] =
      (sizes.filter_size + sizes.input_size) * element_bytes;
  state.counters["bytes_written"] = sizes.output_size * element_bytes;
}

// Records the number of elements processed to the counter set. How this
// calculated varies based on the type of convolution.
template <>
void BaseConvolutionBenchmark::set_items_processed<
    sycldnn::conv2d::conv_type::Forward>(
    benchmark::State& state, sycldnn::conv2d::Conv2DParams const& params) {
  state.SetItemsProcessed(state.iterations() * params.batch * params.out_rows *
                          params.out_cols * params.window_rows *
                          params.window_cols * params.channels *
                          params.features * 2);
}

template <>
void BaseConvolutionBenchmark::set_items_processed<
    sycldnn::conv2d::conv_type::InputBackprop>(
    benchmark::State& state, sycldnn::conv2d::Conv2DParams const& params) {
  state.SetItemsProcessed(state.iterations() * params.batch * params.in_rows *
                          params.in_cols * params.window_rows *
                          params.window_cols * params.channels *
                          params.features * 2);
}

template <>
void BaseConvolutionBenchmark::set_items_processed<
    sycldnn::conv2d::conv_type::FilterBackprop>(
    benchmark::State& state, sycldnn::conv2d::Conv2DParams const& params) {
  state.SetItemsProcessed(state.iterations() * params.batch * params.in_rows *
                          params.in_cols * params.window_rows *
                          params.window_cols * params.channels *
                          params.features * 2);
}

void BaseConvolutionBenchmark::add_opencl_device_info(
    const cl::sycl::device& device) {
  // OpenCL is unclear whether strings returned from clGet*Info() should be
  // null terminated, and ComputeCpp currently copies embedded nulls. On some
  // OpenCL implemntations this results in strings that behave unexpectedly
  // when appended to. This lambda trims those strings.
  auto trim = [](std::string s) -> std::string {
    s.resize(strlen(s.c_str()));
    return s;
  };
  auto device_name = device.get_info<cl::sycl::info::device::name>();
  auto vendor_name = device.get_info<cl::sycl::info::device::vendor>();
  auto driver_version = device.get_info<cl::sycl::info::device::version>();
  key_value_map["device_name"] = trim(device_name);
  key_value_map["vendor_name"] = trim(vendor_name);
  key_value_map["driver_version"] = trim(driver_version);
}

void BaseConvolutionBenchmark::set_label(benchmark::State& state) {
  std::string label;
  for (auto& kv : key_value_map) {
    if (label.size()) label += ",";

    label += kv.first + "=" + kv.second;
  }
  state.SetLabel(label);
}

template <typename ConvType>
void BaseConvolutionBenchmark::execute(
    benchmark::State& state, sycldnn::conv2d::Conv2DParams const& params,
    sycldnn::conv2d::Selector& selector) {
  Eigen::QueueInterface queue_interface{cl::sycl::default_selector{}};
  Eigen::SyclDevice device{&queue_interface};
  sycldnn::backend::EigenBackend backend{device};

  auto conv_sizes = sycldnn::conv2d::get_sizes<ConvType>(params);

  size_t inp_bytes = conv_sizes.input_size * sizeof(float);
  float* inp_gpu = static_cast<float*>(device.allocate(inp_bytes));
  size_t fil_bytes = conv_sizes.filter_size * sizeof(float);
  float* fil_gpu = static_cast<float*>(device.allocate(fil_bytes));
  size_t out_bytes = conv_sizes.output_size * sizeof(float);
  float* out_gpu = static_cast<float*>(device.allocate(out_bytes));

  {  // Ensure the kernel is built before benchmarking
    auto status = sycldnn::conv2d::launch<float, ConvType>(
        inp_gpu, fil_gpu, out_gpu, params, selector, backend);
    status.event.wait();
  }

  for (auto _ : state) {
    auto start = std::chrono::high_resolution_clock::now();
    auto status = sycldnn::conv2d::launch<float, ConvType>(
        inp_gpu, fil_gpu, out_gpu, params, selector, backend);

    status.event.wait();
    auto end = std::chrono::high_resolution_clock::now();

    auto elapsed_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    state.SetIterationTime(elapsed_seconds.count());
  }

  device.deallocate_all();

  // Get the SYCL device, and add device and driver info to the key-value map.
  auto dev = queue_interface.sycl_queue().get_device();
  add_opencl_device_info(dev);

  set_items_processed<ConvType>(state, params);
  add_param_counters(state, params);
  add_bandwidth_counters<float>(state, conv_sizes);

  key_value_map["selector"] = selector.name();
  key_value_map["git-hash"] = commit_hash;
  key_value_map["git-date"] = commit_date;
  set_label(state);
}
}

#define CONVOLUTION_BENCHMARK(name, ...)                               \
  BENCHMARK_TEMPLATE_DEFINE_F(ConvolutionBenchmark, name, __VA_ARGS__) \
  (benchmark::State & state) { this->run(state); }                     \
  BENCHMARK_REGISTER_F(ConvolutionBenchmark, name)->UseManualTime()

#endif  // define SYCLDNN_BENCH_FIXTURE_H_
