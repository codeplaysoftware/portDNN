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
#ifndef SYCLDNN_BENCH_MIOPEN_BATCHNORM_EXECUTOR_H_
#define SYCLDNN_BENCH_MIOPEN_BATCHNORM_EXECUTOR_H_

#include "sycldnn/batchnorm/params.h"
#include "sycldnn/format_type.h"
#include "sycldnn/helpers/scope_exit.h"

#include "base_batchnorm_fixture.h"
#include "benchmark_config.h"
#include "benchmark_params.h"

#include "bench/common/miopen_common.h"
#include "bench/common/verification_utils.h"
#include "bench/fixture/add_datatype_info.h"
#include "bench/fixture/base_bench_executor.h"
#include "bench/fixture/statistic.h"
#include "bench/fixture/string_reporter.h"
#include "bench/fixture/typenames.h"
#include "snn_batchnorm_bench_executor.h"

#include <benchmark/benchmark.h>

#include <hip/hip_runtime.h>
#include <miopen/miopen.h>

#include <array>
#include <functional>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <string>

namespace sycldnn {
namespace bench {

template <typename Benchmark, typename DataType>
struct MIOpenBatchnormExecutor : public BaseBenchExecutor {
 private:
  using State = ::benchmark::State;
  using Params = batchnorm::BatchNormParams;

  /** Get a reference to the underlying benchmark fixture. */
  Benchmark& underlying_benchmark() { return static_cast<Benchmark&>(*this); }

  void validate_results(void* miopen_gpu,
                        std::vector<std::vector<DataType>> const& host_inputs,
                        Params const& params, DataType tolerance = 1e-5) {
    using Executor =
        SNNBatchnormExecutor<DataType, batchnorm::Forward, backend::SNNBackend>;

    Executor snn_batchnorm(host_inputs[0], host_inputs[1], host_inputs[2],
                           host_inputs[3], host_inputs[4], params);

    SNN_ON_SCOPE_EXIT { snn_batchnorm.teardown(); };

    // Allocate SNN device memory
    snn_batchnorm.setup();

    // Execute SNN batchnorm
    auto status = snn_batchnorm.execute();
    if (status.status != sycldnn::StatusCode::OK) {
      throw std::runtime_error(
          "Unexpected issue executing SNN batchnorm for MIOpen benchmark "
          "validation.");
    }

    status.event.wait_and_throw();

    const auto output_size =
        params.batch * params.rows * params.cols * params.channels;

    std::vector<DataType> snn_output(output_size);
    std::vector<DataType> miopen_output(output_size);

    // Copy SNN results to host
    snn_batchnorm.copy_output_to_host(snn_output);
    // Copy MIOpen results to host
    hip_check(hipMemcpy, miopen_output.data(), miopen_gpu,
              sizeof(DataType) * output_size, hipMemcpyDeviceToHost);
    hip_check(hipDeviceSynchronize);

    // Assert that results are equal.
    if (!internal::compare_results(snn_output, miopen_output, tolerance)) {
      throw std::runtime_error(
          "MIOpen results do not match SNN results, exiting benchmark.");
    }
  }

 protected:
  miopenHandle_t handle_;

 public:
  /** Execute a batchnorm benchmark with the given parameters and selector. */
  void execute(State& state, Params const& params) {
    using TensorShape = std::array<int32_t, 4>;

    if (params.input_format == sycldnn::DataFormat::NHWC) {
      state.SkipWithError("MIOpen only supports NCHW, skipping.");
      return;
    }

    if (params.is_training) {
      state.SkipWithError(
          "Unexpected parameter in miopen_batchnorm_executor: "
          "params.is_training set to true.");
      return;
    }

    auto alpha = 1.f;
    auto beta = 0.f;

    TensorShape layer_shape = {params.batch, params.channels, params.rows,
                               params.cols};
    miopenBatchNormMode_t mode = miopenBNSpatial;  // 1, C, 1, 1

    miopenTensorDescriptor_t inp_out_desc;
    miopenTensorDescriptor_t scale_bias_desc;
    miopen_check(miopenCreateTensorDescriptor, &inp_out_desc);
    miopen_check(miopenCreateTensorDescriptor, &scale_bias_desc);
    miopen_check(miopenSet4dTensorDescriptor, inp_out_desc, miopenFloat,
                 layer_shape[0], layer_shape[1], layer_shape[2],
                 layer_shape[3]);
    miopen_check(miopenDeriveBNTensorDescriptor, scale_bias_desc, inp_out_desc,
                 mode);

    size_t input_size =
        params.batch * params.rows * params.cols * params.channels;
    size_t scale_bias_size = params.channels;

    const DataType max_val = 2.0;
    const auto inp_host = internal::iota_initialised_data(input_size, max_val);
    const auto scale_host =
        internal::iota_initialised_data(params.channels, max_val);
    const auto bias_host =
        internal::iota_initialised_data(params.channels, max_val);
    const auto estimated_mean_host =
        internal::iota_initialised_data(params.channels, max_val);
    const auto estimated_variance_host =
        internal::iota_initialised_data(params.channels, max_val);

    DataType* inp_gpu;
    DataType* out_gpu;
    DataType* scale_gpu;
    DataType* bias_gpu;
    DataType* estimated_mean_gpu;
    DataType* estimated_variance_gpu;

    const auto input_size_bytes = input_size * sizeof(DataType);
    const auto scale_bias_size_bytes = scale_bias_size * sizeof(DataType);

    hip_check(hipMalloc<DataType>, &inp_gpu, input_size_bytes);
    hip_check(hipMalloc<DataType>, &out_gpu, input_size_bytes);
    hip_check(hipMalloc<DataType>, &scale_gpu, scale_bias_size_bytes);
    hip_check(hipMalloc<DataType>, &bias_gpu, scale_bias_size_bytes);
    hip_check(hipMalloc<DataType>, &estimated_mean_gpu, scale_bias_size_bytes);
    hip_check(hipMalloc<DataType>, &estimated_variance_gpu,
              scale_bias_size_bytes);

    hip_check(hipMemcpy, inp_gpu, inp_host.data(), input_size_bytes,
              hipMemcpyHostToDevice);
    hip_check(hipMemcpy, scale_gpu, scale_host.data(), scale_bias_size_bytes,
              hipMemcpyHostToDevice);
    hip_check(hipMemcpy, bias_gpu, bias_host.data(), scale_bias_size_bytes,
              hipMemcpyHostToDevice);
    hip_check(hipMemcpy, estimated_mean_gpu, estimated_mean_host.data(),
              scale_bias_size_bytes, hipMemcpyHostToDevice);
    hip_check(hipMemcpy, estimated_variance_gpu, estimated_variance_host.data(),
              scale_bias_size_bytes, hipMemcpyHostToDevice);
    hip_check(hipDeviceSynchronize);

    SNN_ON_SCOPE_EXIT {
      miopen_check(miopenDestroyTensorDescriptor, inp_out_desc);
      miopen_check(miopenDestroyTensorDescriptor, scale_bias_desc);
      hip_check(hipFree, inp_gpu);
      hip_check(hipFree, out_gpu);
      hip_check(hipFree, scale_gpu);
      hip_check(hipFree, bias_gpu);
      hip_check(hipFree, estimated_mean_gpu);
      hip_check(hipFree, estimated_variance_gpu);
    };

    miopen_check(miopenBatchNormalizationForwardInference, this->handle_, mode,
                 &alpha, &beta, inp_out_desc, inp_gpu, inp_out_desc, out_gpu,
                 scale_bias_desc, scale_gpu, bias_gpu, estimated_mean_gpu,
                 estimated_variance_gpu, params.epsilon);
    hip_check(hipDeviceSynchronize);

    constexpr auto tolerance = 1e-5;
    std::vector<std::vector<DataType>> snn_inputs = {
        inp_host, scale_host, bias_host, estimated_mean_host,
        estimated_variance_host};
    this->validate_results(out_gpu, snn_inputs, params, tolerance);

    for (auto _ : state) {
      this->start_timing();

      miopen_check(miopenBatchNormalizationForwardInference, this->handle_,
                   mode, &alpha, &beta, inp_out_desc, inp_gpu, inp_out_desc,
                   out_gpu, scale_bias_desc, scale_gpu, bias_gpu,
                   estimated_mean_gpu, estimated_variance_gpu, params.epsilon);
      hip_check(hipDeviceSynchronize);

      this->end_timing();
      this->set_iteration_time(state);
    }

    auto& benchmark = underlying_benchmark();

    benchmark.set_items_processed(state, params);
    benchmark.add_param_counters(state, params);
    benchmark.template add_bandwidth_counters<float>(state, params);

    this->finish_benchmark(state);
  }
};

}  // namespace bench
}  // namespace sycldnn

extern const char* commit_date;
extern const char* commit_hash;

template <typename DataType, typename DataLayout>
class MIOpenBatchnormBenchmark
    : public sycldnn::bench::MIOpenBatchnormExecutor<
          MIOpenBatchnormBenchmark<DataType, DataLayout>, DataType>,
      public sycldnn::bench::StringReporter,
      public BaseBatchnormBenchmark {
  using State = benchmark::State;
  hipDeviceProp_t properties_;
  int version_;

 protected:
  void run(State& state) {
    auto params = benchmark_params::deserialize(state);
    params.input_format = DataLayout::input_layout;
    this->add_statistic(std::unique_ptr<sycldnn::bench::Statistic>{
        new sycldnn::bench::MaxStatistic{}});
    this->add_statistic(std::unique_ptr<sycldnn::bench::Statistic>{
        new sycldnn::bench::MinStatistic{}});
    this->add_statistic(std::unique_ptr<sycldnn::bench::Statistic>{
        new sycldnn::bench::StdDevStatistic{}});
    try {
      this->execute(state, params);
    } catch (const std::runtime_error& e) {
      state.SkipWithError(e.what());
    }

    sycldnn::bench::datatype_info::add_datatype_info<DataType>(*this);

    this->add_to_label("@layout", sycldnn::bench::TypeName<DataLayout>::name);
    this->add_to_label("@library", "SYCL-DNN");
    this->add_to_label("@backend", "MIOpen");
    this->add_to_label("short_name", "Batchnorm");
    this->add_to_label("git_hash", commit_hash);

    this->add_to_label("vendor_name", "AMD");
    this->add_to_label("device_name", properties_.name);
    this->add_to_label("device_version",
                       std::to_string(version_ / 1000) + "." +
                           std::to_string((version_ % 1000) / 10));
    this->add_to_label("driver_version", "n/a");
    this->set_label(state);
  }

  void set_model(const char* model_name) {
    this->add_to_label("@model_name", model_name);
  }

 public:
  void SetUp(const State&) {
    hip_check(hipSetDevice, 0);
    hip_check(hipGetDeviceProperties, &this->properties_, 0);
    hip_check(hipRuntimeGetVersion, &this->version_);
    miopen_check(miopenCreate, &this->handle_);
  }

  void TearDown(const State&) { miopen_check(miopenDestroy, this->handle_); }
};

#define BATCHNORM_BENCHMARK(name, ...)                                     \
  BENCHMARK_TEMPLATE_DEFINE_F(MIOpenBatchnormBenchmark, name, __VA_ARGS__) \
  (benchmark::State & state) {                                             \
    this->set_model(get_benchmark_name());                                 \
    this->run(state);                                                      \
  }                                                                        \
  BENCHMARK_REGISTER_F(MIOpenBatchnormBenchmark, name)                     \
      ->UseManualTime()                                                    \
      ->Unit(benchmark::kNanosecond)                                       \
      ->Apply(RunForAllParamSets);

#endif  // SYCLDNN_BENCH_MIOPEN_BATCHNORM_EXECUTOR_H_
