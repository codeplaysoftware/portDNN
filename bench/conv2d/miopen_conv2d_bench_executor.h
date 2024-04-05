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
#ifndef SYCLDNN_BENCH_CONV2D_MIOPEN_CONV2D_EXECUTOR_H_
#define SYCLDNN_BENCH_CONV2D_MIOPEN_CONV2D_EXECUTOR_H_

#include "src/backend/snn_backend_provider.h"
#include "sycldnn/backend/snn_backend.h"
#include "sycldnn/conv2d/conv_type.h"
#include "sycldnn/conv2d/params.h"
#include "sycldnn/conv2d/selector/direct_selector.h"
#include "sycldnn/conv2d/selector/im2col_selector.h"
#include "sycldnn/conv2d/sizes.h"
#include "sycldnn/helpers/scope_exit.h"

#include "base_convolution_fixture.h"
#include "benchmark_config.h"
#include "benchmark_params.h"

#include "bench/common/miopen_common.h"
#include "bench/conv2d/snn_conv2d_executor.h"
#include "bench/fixture/add_datatype_info.h"
#include "bench/fixture/base_bench_executor.h"
#include "bench/fixture/statistic.h"
#include "bench/fixture/string_reporter.h"
#include "bench/fixture/typenames.h"

#include "test/helpers/transpose.h"

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

template <typename Benchmark, typename DataType, miopenConvFwdAlgorithm_t Algo>
struct MIOpenConv2DBenchExecutor : public BaseBenchExecutor {
 private:
  using State = ::benchmark::State;
  using Conv2DParams = conv2d::Conv2DParams;

  /** Get a reference to the underlying benchmark fixture. */
  Benchmark& underlying_benchmark() { return static_cast<Benchmark&>(*this); }

  template <typename ConvType>
  void validate_results(void* miopen_gpu, std::vector<DataType> const& inp_vec,
                        std::vector<DataType> const& fil_vec,
                        Conv2DParams const& params, DataType tolerance = 1e-5) {
    using Selector = conv2d::DirectSelector;
    using Executor = SNNConv2DExecutor<DataType, conv2d::conv_type::Forward,
                                       Selector, backend::SNNBackend>;

    auto selector = Selector{};
    Executor snn_conv2d(inp_vec, fil_vec, params, selector);

    SNN_ON_SCOPE_EXIT { snn_conv2d.teardown(); };

    // Allocate SNN device memory
    snn_conv2d.setup();

    // Execute SNN convolution
    auto status = snn_conv2d.execute();
    if (status.status != StatusCode::OK) {
      throw std::runtime_error(
          "Unexpected issue executing SNN convolution for MIOpen benchmark "
          "validation.");
    }

    status.event.wait_and_throw();

    const auto conv_sizes = conv2d::get_sizes<ConvType>(params);
    const auto output_size = conv_sizes.output_size;

    std::vector<DataType> snn_output(output_size);
    std::vector<DataType> miopen_output(output_size);

    // Copy SNN results to host
    snn_conv2d.copy_output_to_host(snn_output);
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

  /**
   * This function transposes the filter from HWCF to the MIOpen supported
   * format FHWC
   */
  std::vector<DataType> transpose_nhwc_filter(
      std::vector<DataType> const& filter, Conv2DParams const& params) {
    std::vector<DataType> output;
    // [1, HWC, F] -->  [1, F, HWC]
    ::transpose(output, filter, 1,
                params.window_rows * params.window_cols * params.channels,
                params.features);
    return output;
  }

 protected:
  miopenHandle_t handle_;

 public:
  /** Execute a conv2d benchmark with the given parameters and selector. */
  void execute(State& state, Conv2DParams const& params) {
    using ConvType = conv2d::conv_type::Forward;
    auto& benchmark = underlying_benchmark();

    miopenTensorDescriptor_t inp_desc;
    miopenTensorDescriptor_t out_desc;
    miopenTensorDescriptor_t fil_desc;
    miopenConvolutionDescriptor_t conv_desc;
    miopen_check(miopenCreateTensorDescriptor, &inp_desc);
    miopen_check(miopenCreateTensorDescriptor, &out_desc);
    miopen_check(miopenCreateTensorDescriptor, &fil_desc);
    miopen_check(miopenCreateConvolutionDescriptor, &conv_desc);

    using TensorShape = std::array<int32_t, 4>;
    TensorShape inp_shape = {params.batch, params.channels, params.in_rows,
                             params.in_cols};
    TensorShape out_shape = {params.batch, params.features, params.out_rows,
                             params.out_cols};
    TensorShape fil_shape = {params.features, params.channels,
                             params.window_rows, params.window_cols};

    const auto conv_sizes = sycldnn::conv2d::get_sizes<ConvType>(params);

    const DataType max_val = 4;
    std::vector<DataType> inp_vec =
        internal::iota_initialised_data(conv_sizes.input_size, max_val);
    std::vector<DataType> fil_vec =
        internal::iota_initialised_data(conv_sizes.filter_size, max_val);

    DataType* inp_gpu;
    DataType* out_gpu;
    DataType* fil_gpu;
    hip_check(hipMalloc<DataType>, &inp_gpu,
              conv_sizes.input_size * sizeof(DataType));
    hip_check(hipMalloc<DataType>, &out_gpu,
              conv_sizes.output_size * sizeof(DataType));
    hip_check(hipMalloc<DataType>, &fil_gpu,
              conv_sizes.filter_size * sizeof(DataType));

    hip_check(hipMemcpy, inp_gpu, inp_vec.data(),
              sizeof(DataType) * conv_sizes.input_size, hipMemcpyHostToDevice);
    hip_check(hipMemcpy, fil_gpu,
              params.filter_format != FilterFormat::HWCF
                  ? fil_vec.data()
                  : this->transpose_nhwc_filter(fil_vec, params).data(),
              sizeof(DataType) * conv_sizes.filter_size, hipMemcpyHostToDevice);
    hip_check(hipMemset, out_gpu, 0, conv_sizes.output_size * sizeof(DataType));

    SNN_ON_SCOPE_EXIT {
      // Unchecked as the checker throws
      miopenDestroyTensorDescriptor(inp_desc);
      miopenDestroyTensorDescriptor(out_desc);
      miopenDestroyTensorDescriptor(fil_desc);
      miopenDestroyConvolutionDescriptor(conv_desc);
      hip_check(hipFree, inp_gpu);
      hip_check(hipFree, out_gpu);
      hip_check(hipFree, fil_gpu);
    };

    // Initialise MIOpen descriptors
    auto miopen_dtype = get_miopen_dtype<DataType>();
    miopen_check(miopenSet4dTensorDescriptor, inp_desc, miopen_dtype,
                 inp_shape[0], inp_shape[1], inp_shape[2], inp_shape[3]);
    miopen_check(miopenSet4dTensorDescriptor, out_desc, miopen_dtype,
                 out_shape[0], out_shape[1], out_shape[2], out_shape[3]);
    miopen_check(miopenSet4dTensorDescriptor, fil_desc, miopen_dtype,
                 fil_shape[0], fil_shape[1], fil_shape[2], fil_shape[3]);
    miopen_check(miopenInitConvolutionDescriptor, conv_desc, miopenConvolution,
                 params.pad_rows, params.pad_cols, params.stride_rows,
                 params.stride_cols, params.dilation_rows,
                 params.dilation_cols);

    // Initialise MIOpen workspace.
    // NOTE: MIOpen *requires* the user to call FindConvolution*Algorithm which
    // requires a workspace. Therefore, calling
    // miopenConvolutionForwardGetWorkSpaceSize is also required in all cases.
    DataType* wspace = nullptr;
    size_t wspace_size = 0;
    miopen_check(miopenConvolutionForwardGetWorkSpaceSize, handle_, fil_desc,
                 inp_desc, conv_desc, out_desc, &wspace_size);
    if (wspace_size) {
      hip_check(hipMalloc<DataType>, &wspace, wspace_size);
    }
    SNN_ON_SCOPE_EXIT {
      if (wspace) {
        hip_check(hipFree, wspace);
      }
    };
    hip_check(hipDeviceSynchronize);

    constexpr int requested_algo_count = 1;
    constexpr bool exhaustive_search = true;
    int ret_algo_count = 0;
    std::vector<miopenConvAlgoPerf_t> perf_results(requested_algo_count);
    miopen_check(miopenFindConvolutionForwardAlgorithm, this->handle_, inp_desc,
                 inp_gpu, fil_desc, fil_gpu, conv_desc, out_desc, out_gpu,
                 requested_algo_count, &ret_algo_count, perf_results.data(),
                 wspace, wspace_size, exhaustive_search);

    if (ret_algo_count < 1) {
      state.SkipWithError("No supported algorithms for MIOpen found");
      return;
    }

    if (Algo != perf_results[0].fwd_algo) {
      state.SkipWithError("Unsupported configuration for MIOpen, skipping");
      return;
    }
    const size_t actual_wspace_size = perf_results[0].memory;

    DataType* actual_wspace = nullptr;
    if (actual_wspace_size) {
      hip_check(hipMalloc<DataType>, &actual_wspace, actual_wspace_size);
    }
    SNN_ON_SCOPE_EXIT {
      if (actual_wspace) {
        hip_check(hipFree, actual_wspace);
      }
    };
    hip_check(hipDeviceSynchronize);

    // Execute MIOpen convolution to warmup and validate results.
    auto alpha = 1.f;
    auto beta = 0.f;
    miopen_check(miopenConvolutionForward, this->handle_, &alpha, inp_desc,
                 inp_gpu, fil_desc, fil_gpu, conv_desc, Algo, &beta, out_desc,
                 out_gpu, actual_wspace, actual_wspace_size);

    hip_check(hipDeviceSynchronize);

    // Validate that MIOpen results equal the SNN results for the same
    // parameters. Throw `std::runtime_error` if not.
    constexpr DataType tolerance = 1e-4;
    validate_results<ConvType>(out_gpu, inp_vec, fil_vec, params, tolerance);

    for (auto _ : state) {
      this->start_timing();
      miopen_check(miopenConvolutionForward, this->handle_, &alpha, inp_desc,
                   inp_gpu, fil_desc, fil_gpu, conv_desc, Algo, &beta, out_desc,
                   out_gpu, actual_wspace, actual_wspace_size);

      hip_check(hipDeviceSynchronize);
      this->end_timing();
      this->set_iteration_time(state);
    }

    benchmark.template set_items_processed<conv2d::conv_type::Forward>(state,
                                                                       params);
    benchmark.add_param_counters(state, params);

    benchmark.template add_bandwidth_counters<DataType>(state, conv_sizes);

    this->finish_benchmark(state);
  }
};

}  // namespace bench
}  // namespace sycldnn

extern const char* commit_date;
extern const char* commit_hash;

template <typename DataType, typename DataLayout, miopenConvFwdAlgorithm_t Algo>
class MIOpenConvolutionBenchmark
    : public sycldnn::bench::MIOpenConv2DBenchExecutor<
          MIOpenConvolutionBenchmark<DataType, DataLayout, Algo>, DataType,
          Algo>,
      public sycldnn::bench::StringReporter,
      public BaseConvolutionBenchmark {
  using State = benchmark::State;
  hipDeviceProp_t properties_;
  int version_;

 protected:
  void run(State& state) {
    auto params = benchmark_params::deserialize(state);
    params.input_format = DataLayout::input_layout;
    params.filter_format = DataLayout::filter_layout;
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
    this->add_to_label("@conv_type", "Forward");
    this->add_to_label("@selector", "MIOpen");
    this->add_to_label("@library", "MIOpen");
    this->add_to_label("short_name", "Convolution");
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
  /** Deinitialise MIOpen */
  void TearDown(const State&) { miopen_check(miopenDestroy, this->handle_); }
};

#define CONVOLUTION_BENCHMARK(name, ...)                                     \
  BENCHMARK_TEMPLATE_DEFINE_F(MIOpenConvolutionBenchmark, name, __VA_ARGS__) \
  (benchmark::State & state) {                                               \
    this->set_model(get_benchmark_name());                                   \
    this->run(state);                                                        \
  }                                                                          \
  BENCHMARK_REGISTER_F(MIOpenConvolutionBenchmark, name)                     \
      ->UseManualTime()                                                      \
      ->Unit(benchmark::kNanosecond)                                         \
      ->Apply(RunForAllParamSets);

#endif  // SYCLDNN_BENCH_CONV2D_MIOPEN_CONV2D_EXECUTOR_H_
