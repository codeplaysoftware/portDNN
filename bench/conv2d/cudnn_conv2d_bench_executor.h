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
#ifndef SYCLDNN_BENCH_CONV2D_CUDNN_CONV2D_BENCH_EXECUTOR_H_
#define SYCLDNN_BENCH_CONV2D_CUDNN_CONV2D_BENCH_EXECUTOR_H_

#include "src/backend/snn_backend_provider.h"
#include "sycldnn/backend/snn_backend.h"

#include "sycldnn/conv2d/conv_type.h"
#include "sycldnn/conv2d/params.h"
#include "sycldnn/conv2d/selector/direct_selector.h"
#include "sycldnn/conv2d/selector/im2col_selector.h"
#include "sycldnn/conv2d/sizes.h"

#include "sycldnn/helpers/handle_exception.h"
#include "sycldnn/helpers/scope_exit.h"

#include "base_convolution_fixture.h"
#include "benchmark_config.h"
#include "benchmark_params.h"

#include "bench/common/cudnn_common.h"
#include "bench/conv2d/cudnn_conv2d_algorithms.h"
#include "bench/conv2d/snn_conv2d_executor.h"
#include "bench/fixture/add_datatype_info.h"
#include "bench/fixture/base_bench_executor.h"
#include "bench/fixture/statistic.h"
#include "bench/fixture/string_reporter.h"
#include "bench/fixture/typenames.h"

#include "test/helpers/transpose.h"

#include <benchmark/benchmark.h>

#include <cuda.h>
#include <cudnn.h>
#include <cudnn_cnn_infer.h>
#include <cudnn_ops_infer.h>

#include <array>
#include <functional>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <string>

namespace sycldnn {
namespace bench {

template <typename Benchmark, typename DataType, typename ConvType,
          cudnnConvolutionAlgo_t Algo, bool UseTensorOp>
struct CUDNNConv2DBenchExecutor : public BaseBenchExecutor {
 private:
  using State = ::benchmark::State;
  using Conv2DParams = conv2d::Conv2DParams;

  /** Get a reference to the underlying benchmark fixture. */
  Benchmark& underlying_benchmark() { return static_cast<Benchmark&>(*this); }

  void validate_results_impl(void* cudnn_gpu,
                             std::vector<DataType> const& inp_vec,
                             std::vector<DataType> const& fil_vec,
                             Conv2DParams const& params,
                             DataType tolerance = 1e-5) {
    using Selector = conv2d::DirectSelector;
    using Executor =
        SNNConv2DExecutor<DataType, ConvType, Selector, backend::SNNBackend>;

    auto selector = Selector{};
    Executor snn_conv2d(inp_vec, fil_vec, params, selector);

    SNN_ON_SCOPE_EXIT { snn_conv2d.teardown(); };

    // Allocate SNN device memory
    snn_conv2d.setup();

    // Execute SNN convolution
    auto status = snn_conv2d.execute();
    if (status.status != StatusCode::OK) {
      throw std::runtime_error(
          "Unexpected issue executing SNN convolution for CuDNN benchmark "
          "validation.");
    }

    status.event.wait_and_throw();

    const auto conv_sizes = conv2d::get_sizes<ConvType>(params);
    const auto output_size = conv_sizes.output_size;

    std::vector<DataType> snn_output(output_size);
    std::vector<DataType> cudnn_output(output_size);

    // Copy SNN results to host
    snn_conv2d.copy_output_to_host(snn_output);
    if (std::is_same_v<ConvType, conv2d::conv_type::FilterBackprop> &&
        params.input_format == sycldnn::DataFormat::NHWC) {
      snn_output = transpose_nhwc_filter(snn_output, params);
    }
    // Copy CuDNN results to host
    cuda_check(cudaMemcpy, cudnn_output.data(), cudnn_gpu,
               sizeof(DataType) * output_size, cudaMemcpyDeviceToHost);
    cuda_check(cudaDeviceSynchronize);

    // Assert that results are equal.
    if (!internal::compare_results(snn_output, cudnn_output, tolerance)) {
      throw std::runtime_error(
          "CuDNN results do not match SNN results, exiting benchmark.");
    }
  }

  void validate_results(void* inp_gpu, void* fil_gpu, void* out_gpu,
                        std::vector<DataType> const& inp_vec,
                        std::vector<DataType> const& fil_vec,
                        std::vector<DataType> const& out_vec,
                        Conv2DParams const& params, DataType tolerance = 1e-5) {
    if constexpr (std::is_same_v<ConvType, conv2d::conv_type::Forward>) {
      this->validate_results_impl(out_gpu, inp_vec, fil_vec, params, tolerance);
    } else if constexpr (std::is_same_v<ConvType,
                                        conv2d::conv_type::InputBackprop>) {
      this->validate_results_impl(inp_gpu, out_vec, fil_vec, params, tolerance);
    } else {
      this->validate_results_impl(fil_gpu, inp_vec, out_vec, params, tolerance);
    }
  }

  Conv2DParams get_snn_params(Conv2DParams params) {
    if (params.input_format == sycldnn::DataFormat::NHWC) {
      params.filter_format = sycldnn::FilterFormat::HWCF;
    }
    return params;
  }

  /**
   * This function transposes the filter from HWCF to the cudnn supported
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

  void skip_unsupported(Conv2DParams const& params) {
    if ((Algo == cudnnConvolutionAlgo_t::GEMM ||
         Algo == cudnnConvolutionAlgo_t::IMPLICIT_GEMM_PRECOMP ||
         Algo == cudnnConvolutionAlgo_t::FFT ||
         Algo == cudnnConvolutionAlgo_t::WINOGRAD) &&
        params.input_format == sycldnn::DataFormat::NHWC) {
      throw CUDNNExpectedFailure("Executing convolution forward algorithm " +
                                 get_cudnn_conv_algo_str(Algo) +
                                 " for NHWC layout is not supported.");
    }
  }

  void cudnn_conv_check(cudnnStatus_t status) {
    if (status == CUDNN_STATUS_LICENSE_ERROR ||
        status == CUDNN_STATUS_NOT_SUPPORTED ||
        status == CUDNN_STATUS_BAD_PARAM) {
      throw CUDNNExpectedFailure(status);
    }

    cudnn_check([&]() { return status; });
  }

  bool needs_workspace() {
    return Algo != cudnnConvolutionAlgo_t::IMPLICIT_GEMM_NONDETERMINISTIC &&
           Algo != cudnnConvolutionAlgo_t::IMPLICIT_GEMM_DETERMINISTIC;
  }

  void init_workspace(cudnnTensorDescriptor_t const& inp_desc,
                      cudnnFilterDescriptor_t const& fil_desc,
                      cudnnTensorDescriptor_t const& out_desc,
                      cudnnConvolutionDescriptor_t const& conv_desc,
                      size_t* wspace_size_ptr, DataType** wspace_gpu_ptr) {
    if (!this->needs_workspace()) {
      return;
    }
    if constexpr (std::is_same_v<ConvType, conv2d::conv_type::Forward>) {
      auto constexpr algo = get_cudnn_fwd_conv_algo<Algo>::value;
      auto ws_status = cudnnGetConvolutionForwardWorkspaceSize(
          this->handle_, inp_desc, fil_desc, conv_desc, out_desc, algo,
          wspace_size_ptr);
      this->cudnn_conv_check(ws_status);
      cuda_check(cudaMalloc<DataType>, wspace_gpu_ptr, *wspace_size_ptr);
    } else if constexpr (std::is_same_v<ConvType,
                                        conv2d::conv_type::InputBackprop>) {
      auto constexpr algo = get_cudnn_bwd_data_conv_algo<Algo>::value;
      auto ws_status = cudnnGetConvolutionBackwardDataWorkspaceSize(
          this->handle_, fil_desc, out_desc, conv_desc, inp_desc, algo,
          wspace_size_ptr);
      this->cudnn_conv_check(ws_status);
      cuda_check(cudaMalloc<DataType>, wspace_gpu_ptr, *wspace_size_ptr);
    } else {
      auto constexpr algo = get_cudnn_bwd_filter_conv_algo<Algo>::value;
      auto ws_status = cudnnGetConvolutionBackwardFilterWorkspaceSize(
          this->handle_, inp_desc, out_desc, conv_desc, fil_desc, algo,
          wspace_size_ptr);
      this->cudnn_conv_check(ws_status);
      cuda_check(cudaMalloc<DataType>, wspace_gpu_ptr, *wspace_size_ptr);
    }
  }

  DataType get_tolerance() {
    if constexpr (std::is_same_v<ConvType, conv2d::conv_type::FilterBackprop>) {
      return Algo == cudnnConvolutionAlgo_t::FFT ? 1 : 1e-4;
    } else {
      return Algo == cudnnConvolutionAlgo_t::FFT ? 1e-1 : 1e-4;
    }
  }

  static cudnnStatus_t execute_cudnn_conv(
      cudnnHandle_t handle, float alpha, float beta,
      cudnnTensorDescriptor_t const& inp_desc,
      cudnnFilterDescriptor_t const& fil_desc,
      cudnnTensorDescriptor_t const& out_desc,
      cudnnConvolutionDescriptor_t const& conv_desc, size_t wspace_size,
      void* inp_gpu, void* fil_gpu, void* out_gpu, void* wspace_gpu) {
    if constexpr (std::is_same_v<ConvType,
                                 sycldnn::conv2d::conv_type::Forward>) {
      auto constexpr algo = get_cudnn_fwd_conv_algo<Algo>::value;
      return cudnnConvolutionForward(
          handle, &alpha, inp_desc, inp_gpu, fil_desc, fil_gpu, conv_desc, algo,
          wspace_gpu, wspace_size, &beta, out_desc, out_gpu);
    } else if constexpr (std::is_same_v<
                             ConvType,
                             sycldnn::conv2d::conv_type::InputBackprop>) {
      auto constexpr algo = get_cudnn_bwd_data_conv_algo<Algo>::value;
      return cudnnConvolutionBackwardData(
          handle, &alpha, fil_desc, fil_gpu, out_desc, out_gpu, conv_desc, algo,
          wspace_gpu, wspace_size, &beta, inp_desc, inp_gpu);

    } else {
      auto constexpr algo = get_cudnn_bwd_filter_conv_algo<Algo>::value;
      return cudnnConvolutionBackwardFilter(
          handle, &alpha, inp_desc, inp_gpu, out_desc, out_gpu, conv_desc, algo,
          wspace_gpu, wspace_size, &beta, fil_desc, fil_gpu);
    }
  }

 protected:
  cudnnHandle_t handle_;

 public:
  /** Execute a conv2d benchmark with the given parameters and selector. */
  void execute(State& state, Conv2DParams const& params) {
    auto& benchmark = underlying_benchmark();

    // Skip unsupported algorithms and layouts
    this->skip_unsupported(params);

    cudnnTensorDescriptor_t inp_desc;
    cudnnTensorDescriptor_t out_desc;
    cudnnFilterDescriptor_t fil_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnn_check(cudnnCreateTensorDescriptor, &inp_desc);
    cudnn_check(cudnnCreateTensorDescriptor, &out_desc);
    cudnn_check(cudnnCreateFilterDescriptor, &fil_desc);
    cudnn_check(cudnnCreateConvolutionDescriptor, &conv_desc);

    using TensorShape = std::array<int32_t, 4>;
    TensorShape inp_shape = {params.batch, params.channels, params.in_rows,
                             params.in_cols};
    TensorShape out_shape = {params.batch, params.features, params.out_rows,
                             params.out_cols};
    TensorShape fil_shape = {params.features, params.channels,
                             params.window_rows, params.window_cols};

    const auto conv_sizes =
        sycldnn::conv2d::get_sizes<sycldnn::conv2d::conv_type::Forward>(params);

    const DataType max_val = 4;
    std::vector<DataType> inp_vec =
        internal::iota_initialised_data(conv_sizes.input_size, max_val);
    std::vector<DataType> fil_vec =
        internal::iota_initialised_data(conv_sizes.filter_size, max_val);
    std::vector<DataType> out_vec =
        internal::iota_initialised_data(conv_sizes.output_size, max_val);

    DataType* inp_gpu;
    DataType* out_gpu;
    DataType* fil_gpu;
    cuda_check(cudaMalloc<DataType>, &inp_gpu,
               conv_sizes.input_size * sizeof(DataType));
    cuda_check(cudaMalloc<DataType>, &out_gpu,
               conv_sizes.output_size * sizeof(DataType));
    cuda_check(cudaMalloc<DataType>, &fil_gpu,
               conv_sizes.filter_size * sizeof(DataType));

    cuda_check(cudaMemcpy, inp_gpu, inp_vec.data(),
               sizeof(DataType) * conv_sizes.input_size,
               cudaMemcpyHostToDevice);
    cuda_check(cudaMemcpy, fil_gpu,
               params.filter_format == FilterFormat::FCHW
                   ? fil_vec.data()
                   : this->transpose_nhwc_filter(fil_vec, params).data(),
               sizeof(DataType) * conv_sizes.filter_size,
               cudaMemcpyHostToDevice);

    if constexpr (std::is_same_v<ConvType,
                                 sycldnn::conv2d::conv_type::Forward>) {
      cuda_check(cudaMemset, out_gpu, 0,
                 conv_sizes.output_size * sizeof(DataType));
    } else {
      cuda_check(cudaMemcpy, out_gpu, out_vec.data(),
                 sizeof(DataType) * conv_sizes.output_size,
                 cudaMemcpyHostToDevice);
    }

    SNN_ON_SCOPE_EXIT {
      // Unchecked as the checker throws
      cudnnDestroyTensorDescriptor(inp_desc);
      cudnnDestroyTensorDescriptor(out_desc);
      cudnnDestroyFilterDescriptor(fil_desc);
      cudnnDestroyConvolutionDescriptor(conv_desc);
      cudaFree(inp_gpu);
      cudaFree(out_gpu);
      cudaFree(fil_gpu);
    };

    // Initialise CuDNN tensor descriptors
    auto cudnn_dtype = get_cudnn_dtype<DataType>();
    auto cudnn_tensor_format = get_cudnn_tensor_format(params.input_format);
    cudnn_check(cudnnSetTensor4dDescriptor, inp_desc, cudnn_tensor_format,
                cudnn_dtype, inp_shape[0], inp_shape[1], inp_shape[2],
                inp_shape[3]);
    cudnn_check(cudnnSetTensor4dDescriptor, out_desc, cudnn_tensor_format,
                cudnn_dtype, out_shape[0], out_shape[1], out_shape[2],
                out_shape[3]);
    cudnn_check(cudnnSetFilter4dDescriptor, fil_desc, cudnn_dtype,
                cudnn_tensor_format, fil_shape[0], fil_shape[1], fil_shape[2],
                fil_shape[3]);

    // Initialise CuDNN convolution descriptors
    auto conv_desc_status = cudnnSetConvolution2dDescriptor(
        conv_desc, params.pad_rows, params.pad_cols, params.stride_rows,
        params.stride_cols, params.dilation_rows, params.dilation_cols,
        CUDNN_CROSS_CORRELATION, cudnn_dtype);

    // Handle expected failures or throw if unexpected.
    this->cudnn_conv_check(conv_desc_status);

    // Initialise CuDNN workspace if needed.
    DataType* wspace_gpu = nullptr;
    size_t wspace_size = 0;
    this->init_workspace(inp_desc, fil_desc, out_desc, conv_desc, &wspace_size,
                         &wspace_gpu);

    // Free workspace on exit
    SNN_ON_SCOPE_EXIT {
      if (wspace_gpu) {
        cudaFree(wspace_gpu);
      }
    };

    // Set CuDNN math type (Explicitly turn on/off tensor core operations)
    cudnn_check(cudnnSetConvolutionMathType, conv_desc,
                get_cudnn_math_type<UseTensorOp>::value);

    cuda_check(cudaDeviceSynchronize);

    // Execute CuDNN convolution to warmup and validate results.
    auto alpha = 1.f;
    auto beta = 0.f;
    // Handle expected failures or throw if unexpected.
    this->cudnn_conv_check(this->execute_cudnn_conv(
        this->handle_, alpha, beta, inp_desc, fil_desc, out_desc, conv_desc,
        wspace_size, inp_gpu, fil_gpu, out_gpu, wspace_gpu));
    cuda_check(cudaDeviceSynchronize);

    // Validate that CuDNN results equal the SNN results for the same
    // parameters. Throw `std::runtime_error` if not.
    const DataType tolerance = this->get_tolerance();
    auto snn_params = this->get_snn_params(params);
    this->validate_results(inp_gpu, fil_gpu, out_gpu, inp_vec, fil_vec, out_vec,
                           snn_params, tolerance);

    for (auto _ : state) {
      this->start_timing();
      cudnn_check(this->execute_cudnn_conv, this->handle_, alpha, beta,
                  inp_desc, fil_desc, out_desc, conv_desc, wspace_size, inp_gpu,
                  fil_gpu, out_gpu, wspace_gpu);

      cuda_check(cudaDeviceSynchronize);
      this->end_timing();
      this->set_iteration_time(state);
    }

    benchmark.template set_items_processed<ConvType>(state, params);
    benchmark.add_param_counters(state, params);

    benchmark.template add_bandwidth_counters<DataType>(state, conv_sizes);

    this->finish_benchmark(state);
  }

 protected:
  void handle_exception(CUDNNFailure& e, State& state) {
    if (CUDNNExpectedFailure* expected =
            dynamic_cast<CUDNNExpectedFailure*>(&e)) {
      state.SkipWithError(
          (std::string(e.what()) +
           " Failed while executing convolution forward algorithm " +
           get_cudnn_conv_algo_str(Algo) + ". " + UnsupportedFailure)
              .c_str());
    } else {
      state.SkipWithError(
          (std::string(e.what()) +
           " Failed while executing convolution forward algorithm " +
           get_cudnn_conv_algo_str(Algo) + ". " + UnexpectedFailure)
              .c_str());
    }
  }
};

}  // namespace bench
}  // namespace sycldnn

extern const char* commit_date;
extern const char* commit_hash;

template <typename DataType, typename DataLayout, typename ConvType,
          sycldnn::bench::cudnnConvolutionAlgo_t Algo, bool UseTensorOp>
class CUDNNConvolutionBenchmark
    : public sycldnn::bench::CUDNNConv2DBenchExecutor<
          CUDNNConvolutionBenchmark<DataType, DataLayout, ConvType, Algo,
                                    UseTensorOp>,
          DataType, ConvType, Algo, UseTensorOp>,
      public sycldnn::bench::StringReporter,
      public BaseConvolutionBenchmark {
  using State = benchmark::State;
  cudaDeviceProp properties_;
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
    this->add_to_label("@algo", sycldnn::bench::get_cudnn_conv_algo_str(Algo));
    this->add_to_label("@use_tensor_op", UseTensorOp ? "true" : "false");
    this->add_to_label("@conv_type", sycldnn::bench::TypeName<ConvType>::name);
    this->add_to_label("@selector", "cuDNN");
    this->add_to_label("@library", "cuDNN");
    this->add_to_label("@backend", "cuDNN");
    this->add_to_label("short_name", "Convolution");
    this->add_to_label("git_hash", commit_hash);
    this->add_to_label("vendor_name", "NVIDIA");
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
    cuda_check(cudaSetDevice, 0);
    cuda_check(cudaGetDeviceProperties, &this->properties_, 0);
    cuda_check(cudaRuntimeGetVersion, &this->version_);
    cudnn_check(cudnnCreate, &this->handle_);
  }
  /** Deinitialise cudnn */
  void TearDown(const State&) { cudnn_check(cudnnDestroy, this->handle_); }
};

#define CONVOLUTION_BENCHMARK(name, ...)                                    \
  BENCHMARK_TEMPLATE_DEFINE_F(CUDNNConvolutionBenchmark, name, __VA_ARGS__) \
  (benchmark::State & state) {                                              \
    this->set_model(get_benchmark_name());                                  \
    try {                                                                   \
      this->run(state);                                                     \
    } catch (CUDNNFailure & e) {                                            \
      this->handle_exception(e, state);                                     \
    }                                                                       \
  }                                                                         \
  BENCHMARK_REGISTER_F(CUDNNConvolutionBenchmark, name)                     \
      ->UseManualTime()                                                     \
      ->Unit(benchmark::kNanosecond)                                        \
      ->Apply(RunForAllParamSets);

#endif  // SYCLDNN_BENCH_CONV2D_CUDNN_CONV2D_BENCH_EXECUTOR_H_
