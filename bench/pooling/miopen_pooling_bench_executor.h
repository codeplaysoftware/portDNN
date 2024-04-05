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
#ifndef SYCLDNN_BENCH_POOL_MIOPEN_POOL_EXECUTOR_H_
#define SYCLDNN_BENCH_POOL_MIOPEN_POOL_EXECUTOR_H_

#include <benchmark/benchmark.h>

#include <hip/hip_runtime.h>
#include <miopen/miopen.h>

#include "base_pooling_fixture.h"
#include "benchmark_config.h"
#include "benchmark_params.h"

#include "sycldnn/helpers/scope_exit.h"

#include "bench/fixture/add_datatype_info.h"
#include "bench/fixture/base_bench_executor.h"
#include "bench/fixture/operator_typenames.h"
#include "bench/fixture/statistic.h"
#include "bench/fixture/string_reporter.h"
#include "bench/fixture/typenames.h"

#include "bench/pooling/snn_pooling_executor.h"

#include "bench/common/miopen_common.h"
#include "bench/common/verification_utils.h"

#include "src/backend/snn_backend_provider.h"

namespace sycldnn {
namespace bench {

template <template <typename> class SNNPoolOperator>
struct get_miopen_pool_mode;

template <>
struct get_miopen_pool_mode<pooling::Max> {
  static constexpr auto pool_mode = miopenPoolingMode_t::miopenPoolingMax;
};

template <>
struct get_miopen_pool_mode<pooling::Average> {
  static constexpr auto pool_mode = miopenPoolingMode_t::miopenPoolingAverage;
};

template <typename Benchmark, typename DataType,
          template <typename> class SNNPoolOperator, typename Direction>
struct MIOpenPoolingBenchExecutor;

template <typename Benchmark, typename DataType,
          template <typename> class SNNPoolOperator>
struct MIOpenPoolingBenchExecutor<Benchmark, DataType, SNNPoolOperator,
                                  pooling::Forward> : public BaseBenchExecutor {
 private:
  using State = ::benchmark::State;
  using PoolingParams = pooling::PoolingParams;
  using Executor = SNNPoolingExecutor<DataType, SNNPoolOperator,
                                      pooling::Forward, backend::SNNBackend>;
  /** Get a reference to the underlying benchmark fixture. */
  Benchmark& underlying_benchmark() { return static_cast<Benchmark&>(*this); }

 protected:
  miopenHandle_t handle_;

 public:
  void validate_results(void* miopen_gpu, std::vector<DataType> const& inp_vec,
                        PoolingParams const& params,
                        DataType tolerance = 1e-5) {
    Executor snn_pool(inp_vec, params);

    SNN_ON_SCOPE_EXIT { snn_pool.teardown(); };

    // Allocate SNN device memory
    snn_pool.setup();

    // Execute SNN pooling
    auto status = snn_pool.execute();
    if (status.status != StatusCode::OK) {
      throw std::runtime_error(
          "Unexpected issue executing SNN pooling for MIOpen benchmark "
          "validation.");
    }

    status.event.wait_and_throw();

    const auto pool_sizes = pooling::get_sizes<pooling::Forward>(params);
    const auto output_size = pool_sizes.output_size;

    std::vector<DataType> snn_output(output_size);
    std::vector<DataType> miopen_output(output_size);

    // Copy SNN results to host
    snn_pool.copy_output_to_host(snn_output);
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

  /** Execute a pool benchmark with the given parameters and selector. */
  void execute(State& state, PoolingParams const& params) {
    auto& benchmark = underlying_benchmark();

    miopenTensorDescriptor_t inp_desc;
    miopenTensorDescriptor_t out_desc;
    miopenPoolingDescriptor_t pool_desc;
    miopen_check(miopenCreateTensorDescriptor, &inp_desc);
    miopen_check(miopenCreateTensorDescriptor, &out_desc);
    miopen_check(miopenCreatePoolingDescriptor, &pool_desc);

    using TensorShape = std::array<int32_t, 4>;
    TensorShape inp_shape = {params.batch, params.channels, params.in_rows,
                             params.in_cols};
    TensorShape out_shape = {params.batch, params.channels, params.out_rows,
                             params.out_cols};

    const auto pool_sizes =
        sycldnn::pooling::get_sizes<pooling::Forward>(params);
    const DataType max_val = 4;
    std::vector<DataType> inp_vec =
        internal::iota_initialised_data(pool_sizes.input_size, max_val);

    DataType* inp_gpu;
    DataType* out_gpu;

    hip_check(hipMalloc<DataType>, &inp_gpu,
              pool_sizes.input_size * sizeof(DataType));
    hip_check(hipMalloc<DataType>, &out_gpu,
              pool_sizes.output_size * sizeof(DataType));

    hip_check(hipMemcpy, inp_gpu, inp_vec.data(),
              sizeof(DataType) * pool_sizes.input_size, hipMemcpyHostToDevice);
    hip_check(hipMemset, out_gpu, 0, size(out_shape) * sizeof(DataType));

    SNN_ON_SCOPE_EXIT {
      // Unchecked as the checker throws
      miopenDestroyTensorDescriptor(inp_desc);
      miopenDestroyTensorDescriptor(out_desc);
      miopenDestroyPoolingDescriptor(pool_desc);
      hipFree(inp_gpu);
      hipFree(out_gpu);
    };

    const auto miopen_dtype = get_miopen_dtype<DataType>();
    constexpr auto miopen_pool_mode =
        get_miopen_pool_mode<SNNPoolOperator>::pool_mode;

    miopen_check(miopenSet4dTensorDescriptor, inp_desc, miopen_dtype,
                 inp_shape[0], inp_shape[1], inp_shape[2], inp_shape[3]);
    miopen_check(miopenSet4dTensorDescriptor, out_desc, miopen_dtype,
                 out_shape[0], out_shape[1], out_shape[2], out_shape[3]);
    miopen_check(miopenSet2dPoolingDescriptor, pool_desc, miopen_pool_mode,
                 params.window_rows, params.window_cols, params.pad_rows,
                 params.pad_cols, params.stride_rows, params.stride_cols);

    hip_check(hipDeviceSynchronize);

    constexpr auto alpha = 1.f;
    constexpr auto beta = 0.f;
    constexpr bool do_backward = false;
    void* wkspace = nullptr;
    constexpr size_t wkspace_size = 0;
    miopen_check(miopenPoolingForward, this->handle_, pool_desc, &alpha,
                 inp_desc, inp_gpu, &beta, out_desc, out_gpu, do_backward,
                 wkspace, wkspace_size);

    hip_check(hipDeviceSynchronize);
    const DataType tolerance = 1e-5;
    this->validate_results(out_gpu, inp_vec, params, tolerance);

    for (auto _ : state) {
      this->start_timing();
      miopen_check(miopenPoolingForward, this->handle_, pool_desc, &alpha,
                   inp_desc, inp_gpu, &beta, out_desc, out_gpu, do_backward,
                   wkspace, wkspace_size);

      hip_check(hipDeviceSynchronize);
      this->end_timing();
      this->set_iteration_time(state);
    }

    benchmark.template set_items_processed<pooling::Forward>(state, params);
    benchmark.add_param_counters(state, params);

    benchmark.template add_bandwidth_counters<DataType>(state, pool_sizes);

    this->finish_benchmark(state);
  }
};

}  // namespace bench
}  // namespace sycldnn

extern const char* commit_date;
extern const char* commit_hash;

template <typename DataType, typename DataLayout, typename Direction,
          template <typename> class SNNPoolOperator>
class MIOpenPoolingBenchmark
    : public sycldnn::bench::MIOpenPoolingBenchExecutor<
          MIOpenPoolingBenchmark<DataType, DataLayout, Direction,
                                 SNNPoolOperator>,
          DataType, SNNPoolOperator, Direction>,
      public sycldnn::bench::StringReporter,
      public BasePoolingBenchmark {
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
    this->add_to_label("@operator",
                       sycldnn::bench::OperatorTypeName<SNNPoolOperator>::name);
    this->add_to_label("@direction", sycldnn::bench::TypeName<Direction>::name);
    this->add_to_label("@library", "MIOpen");
    this->add_to_label("short_name", "Pooling");
    this->add_to_label("git_hash", commit_hash);
    this->add_to_label("vendor_name", "AMD");
    this->add_to_label("device_name", properties_.name);
    this->add_to_label("device_version",
                       std::to_string(version_ / 1000) + "." +
                           std::to_string((version_ % 1000) / 10));
    this->add_to_label("driver_version", "n/a");
    this->set_label(state);
  };

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
  /** Deinitialise miopen */
  void TearDown(const State&) { miopen_check(miopenDestroy, this->handle_); }
};

#define POOLING_BENCHMARK(name, ...)                                     \
  BENCHMARK_TEMPLATE_DEFINE_F(MIOpenPoolingBenchmark, name, __VA_ARGS__) \
  (benchmark::State & state) {                                           \
    this->set_model(get_benchmark_name());                               \
    this->run(state);                                                    \
  }                                                                      \
  BENCHMARK_REGISTER_F(MIOpenPoolingBenchmark, name)                     \
      ->UseManualTime()                                                  \
      ->Unit(benchmark::kNanosecond)                                     \
      ->Apply(RunForAllParamSets);

#endif  // SYCLDNN_BENCH_POOL_MIOPEN_POOL_EXECUTOR_H_
