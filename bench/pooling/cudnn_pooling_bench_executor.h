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
#ifndef SYCLDNN_BENCH_POOL_CUDNN_POOL_EXECUTOR_H_
#define SYCLDNN_BENCH_POOL_CUDNN_POOL_EXECUTOR_H_

#include <benchmark/benchmark.h>

#include <cudnn_cnn_infer.h>
#include <cudnn_ops_infer.h>

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

#include "bench/common/cudnn_common.h"
#include "bench/common/verification_utils.h"

#include "src/backend/snn_backend_provider.h"

namespace sycldnn {
namespace bench {

template <template <typename> class SNNPoolOperator>
struct get_cudnn_pool_mode;

template <>
struct get_cudnn_pool_mode<pooling::Max> {
  static constexpr auto pool_mode = cudnnPoolingMode_t::CUDNN_POOLING_MAX;
  static constexpr auto nan_propagation =
      cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN;
};

template <>
struct get_cudnn_pool_mode<pooling::MaxWithNan> {
  static constexpr auto pool_mode = cudnnPoolingMode_t::CUDNN_POOLING_MAX;
  static constexpr auto nan_propagation =
      cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN;
};

template <>
struct get_cudnn_pool_mode<pooling::Average> {
  static constexpr auto pool_mode =
      cudnnPoolingMode_t::CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  static constexpr auto nan_propagation =
      cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN;
};

template <typename Benchmark, typename DataType,
          template <typename> class SNNPoolOperator, typename Direction>
struct CUDNNPoolingBenchExecutor : public BaseBenchExecutor {
 private:
  using State = ::benchmark::State;
  using PoolingParams = pooling::PoolingParams;
  using Executor = SNNPoolingExecutor<DataType, SNNPoolOperator, Direction,
                                      backend::SNNBackend, false>;
  static constexpr bool IsForward =
      std::is_same_v<Direction, sycldnn::pooling::Forward>;
  static constexpr bool IsMax = std::is_same_v<SNNPoolOperator<DataType>,
                                               sycldnn::pooling::Max<DataType>>;
  static constexpr bool IsMaxGrad = !IsForward && IsMax;

  /** Get a reference to the underlying benchmark fixture. */
  Benchmark& underlying_benchmark() { return static_cast<Benchmark&>(*this); }

 protected:
  cudnnHandle_t handle_;

 public:
  void validate_results(void* cudnn_gpu, std::vector<DataType> const& inp_vec,
                        std::vector<DataType> const& out_grad_vec,
                        PoolingParams const& params,
                        DataType tolerance = 1e-5) {
    if constexpr (IsMaxGrad) {
      Executor snn_pool(inp_vec, out_grad_vec, params);

      SNN_ON_SCOPE_EXIT { snn_pool.teardown(); };

      // Allocate SNN device memory
      snn_pool.setup();

      // Execute SNN pooling
      auto status = snn_pool.execute();
      if (status.status != StatusCode::OK) {
        throw std::runtime_error(
            "Unexpected issue executing SNN Pooling for CuDNN benchmark "
            "validation.");
      }

      status.event.wait_and_throw();

      const auto pool_sizes = pooling::get_sizes<Direction>(params);
      const auto output_size = pool_sizes.output_size;

      std::vector<DataType> snn_output(output_size);
      std::vector<DataType> cudnn_output(output_size);

      // Copy SNN results to host
      snn_pool.copy_output_to_host(snn_output);
      // Copy CuDNN results to host
      cuda_check(cudaMemcpy, cudnn_output.data(), cudnn_gpu,
                 sizeof(DataType) * output_size, cudaMemcpyDeviceToHost);
      cuda_check(cudaDeviceSynchronize);

      // Assert that results are equal.
      if (!internal::compare_results(snn_output, cudnn_output, tolerance)) {
        throw std::runtime_error(
            "CuDNN results do not match SNN results, exiting benchmark.");
      }
    } else {
      SNN_UNUSED_VAR(out_grad_vec);

      Executor snn_pool(inp_vec, params);

      SNN_ON_SCOPE_EXIT { snn_pool.teardown(); };

      // Allocate SNN device memory
      snn_pool.setup();

      // Execute SNN pooling
      auto status = snn_pool.execute();
      if (status.status != StatusCode::OK) {
        throw std::runtime_error(
            "Unexpected issue executing SNN Pooling for CuDNN benchmark "
            "validation.");
      }

      status.event.wait_and_throw();

      const auto pool_sizes = pooling::get_sizes<Direction>(params);
      const auto output_size = pool_sizes.output_size;

      std::vector<DataType> snn_output(output_size);
      std::vector<DataType> cudnn_output(output_size);

      // Copy SNN results to host
      snn_pool.copy_output_to_host(snn_output);
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
  }

  /** Execute a pool benchmark with the given parameters and selector. */
  void execute(State& state, PoolingParams const& params) {
    if (params.input_format == sycldnn::DataFormat::NCHW &&
        std::is_same_v<Direction, pooling::Backpropagate>) {
      state.SkipWithError(
          (std::string("Backprop does not support NCHW.") + UnsupportedFailure)
              .c_str());
      return;
    }
    auto& benchmark = underlying_benchmark();
    if constexpr (IsForward) {
      cudnnTensorDescriptor_t inp_desc;
      cudnnTensorDescriptor_t out_desc;
      cudnnPoolingDescriptor_t pool_desc;
      cudnn_check(cudnnCreateTensorDescriptor, &inp_desc);
      cudnn_check(cudnnCreateTensorDescriptor, &out_desc);
      cudnn_check(cudnnCreatePoolingDescriptor, &pool_desc);

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

      cuda_check(cudaMalloc<DataType>, &inp_gpu,
                 pool_sizes.input_size * sizeof(DataType));
      cuda_check(cudaMalloc<DataType>, &out_gpu,
                 pool_sizes.output_size * sizeof(DataType));

      cuda_check(cudaMemcpy, inp_gpu, inp_vec.data(),
                 sizeof(DataType) * pool_sizes.input_size,
                 cudaMemcpyHostToDevice);
      cuda_check(cudaMemset, out_gpu, 0, size(out_shape) * sizeof(DataType));

      SNN_ON_SCOPE_EXIT {
        // Unchecked as the checker throws
        cudnnDestroyTensorDescriptor(inp_desc);
        cudnnDestroyTensorDescriptor(out_desc);
        cudnnDestroyPoolingDescriptor(pool_desc);
        cudaFree(inp_gpu);
        cudaFree(out_gpu);
      };

      const auto cudnn_dtype = get_cudnn_dtype<DataType>();
      const auto cudnn_tensor_format =
          get_cudnn_tensor_format(params.input_format);
      constexpr auto cudnn_pool_mode =
          get_cudnn_pool_mode<SNNPoolOperator>::pool_mode;
      constexpr auto cudnn_nan_prop =
          get_cudnn_pool_mode<SNNPoolOperator>::nan_propagation;

      cudnn_check(cudnnSetTensor4dDescriptor, inp_desc, cudnn_tensor_format,
                  cudnn_dtype, inp_shape[0], inp_shape[1], inp_shape[2],
                  inp_shape[3]);
      cudnn_check(cudnnSetTensor4dDescriptor, out_desc, cudnn_tensor_format,
                  cudnn_dtype, out_shape[0], out_shape[1], out_shape[2],
                  out_shape[3]);
      cudnn_check(cudnnSetPooling2dDescriptor, pool_desc, cudnn_pool_mode,
                  cudnn_nan_prop, params.window_rows, params.window_cols,
                  params.pad_rows, params.pad_cols, params.stride_rows,
                  params.stride_cols);

      cuda_check(cudaDeviceSynchronize);

      auto alpha = 1.f;
      auto beta = 0.f;
      cudnn_check(cudnnPoolingForward, this->handle_, pool_desc, &alpha,
                  inp_desc, inp_gpu, &beta, out_desc, out_gpu);

      cuda_check(cudaDeviceSynchronize);
      const DataType tolerance = 1e-5;
      this->validate_results(out_gpu, inp_vec, inp_vec, params, tolerance);

      for (auto _ : state) {
        this->start_timing();
        cudnn_check(cudnnPoolingForward, this->handle_, pool_desc, &alpha,
                    inp_desc, inp_gpu, &beta, out_desc, out_gpu);

        cuda_check(cudaDeviceSynchronize);
        this->end_timing();
        this->set_iteration_time(state);
      }

      benchmark.template set_items_processed<pooling::Forward>(state, params);
      benchmark.add_param_counters(state, params);

      benchmark.template add_bandwidth_counters<DataType>(state, pool_sizes);

      this->finish_benchmark(state);
    } else {
      cudnnTensorDescriptor_t inp_desc;
      cudnnTensorDescriptor_t inp_grad_desc;
      cudnnTensorDescriptor_t out_desc;
      cudnnPoolingDescriptor_t pool_desc;
      cudnn_check(cudnnCreateTensorDescriptor, &inp_desc);
      cudnn_check(cudnnCreateTensorDescriptor, &inp_grad_desc);
      cudnn_check(cudnnCreateTensorDescriptor, &out_desc);
      cudnn_check(cudnnCreatePoolingDescriptor, &pool_desc);

      using TensorShape = std::array<int32_t, 4>;
      TensorShape inp_shape = {params.batch, params.channels, params.in_rows,
                               params.in_cols};
      TensorShape out_shape = {params.batch, params.channels, params.out_rows,
                               params.out_cols};
      const auto back_pool_sizes =
          sycldnn::pooling::get_sizes<Direction>(params);
      const auto forward_pool_sizes =
          sycldnn::pooling::get_sizes<sycldnn::pooling::Forward>(params);
      const DataType max_val = 4;
      std::vector<DataType> inp_vec = internal::iota_initialised_data(
          forward_pool_sizes.input_size, max_val);
      std::vector<DataType> out_grad_vec =
          internal::iota_initialised_data(back_pool_sizes.input_size, max_val);
      // This is the output
      std::vector<DataType> in_grad_vec =
          internal::iota_initialised_data(back_pool_sizes.output_size, max_val);

      DataType* inp_gpu = nullptr;
      cuda_check(cudaMalloc<DataType>, &inp_gpu,
                 forward_pool_sizes.input_size * sizeof(DataType));
      cuda_check(cudaMemcpy, inp_gpu, inp_vec.data(),
                 sizeof(DataType) * forward_pool_sizes.input_size,
                 cudaMemcpyHostToDevice);

      DataType* out_gpu = nullptr;
      cuda_check(cudaMalloc<DataType>, &out_gpu,
                 forward_pool_sizes.output_size * sizeof(DataType));
      cuda_check(cudaMemcpy, out_gpu, out_grad_vec.data(),
                 sizeof(DataType) * forward_pool_sizes.output_size,
                 cudaMemcpyHostToDevice);

      DataType* inp_grad_gpu = nullptr;
      cuda_check(cudaMalloc<DataType>, &inp_grad_gpu,
                 back_pool_sizes.output_size * sizeof(DataType));
      cuda_check(cudaMemcpy, inp_grad_gpu, in_grad_vec.data(),
                 sizeof(DataType) * back_pool_sizes.output_size,
                 cudaMemcpyHostToDevice);

      DataType* out_grad_gpu = nullptr;
      cuda_check(cudaMalloc<DataType>, &out_grad_gpu,
                 back_pool_sizes.input_size * sizeof(DataType));
      cuda_check(cudaMemcpy, out_grad_gpu, in_grad_vec.data(),
                 sizeof(DataType) * back_pool_sizes.input_size,
                 cudaMemcpyHostToDevice);

      SNN_ON_SCOPE_EXIT {
        // Unchecked as the checker throws
        cudnnDestroyTensorDescriptor(inp_desc);
        cudnnDestroyTensorDescriptor(out_desc);
        cudnnDestroyPoolingDescriptor(pool_desc);
        cudaFree(inp_gpu);
        cudaFree(out_grad_gpu);
        cudaFree(inp_grad_gpu);
      };

      const auto cudnn_dtype = get_cudnn_dtype<DataType>();
      const auto cudnn_tensor_format =
          get_cudnn_tensor_format(params.input_format);
      constexpr auto cudnn_pool_mode =
          get_cudnn_pool_mode<SNNPoolOperator>::pool_mode;
      constexpr auto cudnn_nan_prop =
          get_cudnn_pool_mode<SNNPoolOperator>::nan_propagation;

      cudnn_check(cudnnSetTensor4dDescriptor, inp_desc, cudnn_tensor_format,
                  cudnn_dtype, inp_shape[0], inp_shape[1], inp_shape[2],
                  inp_shape[3]);
      cudnn_check(cudnnSetTensor4dDescriptor, out_desc, cudnn_tensor_format,
                  cudnn_dtype, out_shape[0], out_shape[1], out_shape[2],
                  out_shape[3]);
      cudnn_check(cudnnSetPooling2dDescriptor, pool_desc, cudnn_pool_mode,
                  cudnn_nan_prop, params.window_rows, params.window_cols,
                  params.pad_rows, params.pad_cols, params.stride_rows,
                  params.stride_cols);

      cuda_check(cudaDeviceSynchronize);
      auto alpha = 1.f;
      auto beta = 0.f;

      cudnn_check(cudnnPoolingForward, this->handle_, pool_desc, &alpha,
                  inp_desc, inp_gpu, &beta, out_desc, out_gpu);
      cudnn_check(cudnnPoolingBackward, this->handle_, pool_desc, &alpha,
                  out_desc, out_gpu, out_desc, out_grad_gpu, inp_desc, inp_gpu,
                  &beta, inp_desc, inp_grad_gpu);

      cuda_check(cudaDeviceSynchronize);
      const DataType tolerance = 1e-5;
      this->validate_results(inp_grad_gpu, inp_vec, out_grad_vec, params,
                             tolerance);
      cuda_check(cudaDeviceSynchronize);

      for (auto _ : state) {
        this->start_timing();
        cudnn_check(cudnnPoolingBackward, this->handle_, pool_desc, &alpha,
                    out_desc, out_gpu, out_desc, out_grad_gpu, inp_desc,
                    inp_gpu, &beta, inp_desc, inp_grad_gpu);

        cuda_check(cudaDeviceSynchronize);
        this->end_timing();
        this->set_iteration_time(state);
      }

      benchmark.template set_items_processed<Direction>(state, params);
      benchmark.add_param_counters(state, params);

      benchmark.template add_bandwidth_counters<DataType>(state,
                                                          back_pool_sizes);

      this->finish_benchmark(state);
    }
  }

 protected:
  void handle_exception(CUDNNFailure& e, State& state) {
    if (CUDNNExpectedFailure* expected =
            dynamic_cast<CUDNNExpectedFailure*>(&e)) {
      state.SkipWithError(
          (std::string(e.what()) + " Failed while executing pooling with " +
           sycldnn::bench::OperatorTypeName<SNNPoolOperator>::name +
           " operation. " + UnsupportedFailure)
              .c_str());
    } else {
      state.SkipWithError(
          (std::string(e.what()) + " Failed while executing pooling with " +
           sycldnn::bench::OperatorTypeName<SNNPoolOperator>::name +
           " operation. " + UnexpectedFailure)
              .c_str());
    }
  }
};

}  // namespace bench
}  // namespace sycldnn

extern const char* commit_date;
extern const char* commit_hash;

template <typename DataType, typename DataLayout, typename Direction,
          template <typename> class SNNPoolOperator>
class CUDNNPoolingBenchmark
    : public sycldnn::bench::CUDNNPoolingBenchExecutor<
          CUDNNPoolingBenchmark<DataType, DataLayout, Direction,
                                SNNPoolOperator>,
          DataType, SNNPoolOperator, Direction>,
      public sycldnn::bench::StringReporter,
      public BasePoolingBenchmark {
  using State = benchmark::State;
  cudaDeviceProp properties_;
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
    } catch (CUDNNFailure& e) {
      this->handle_exception(e, state);
    } catch (const std::runtime_error& e) {
      state.SkipWithError(e.what());
    }

    sycldnn::bench::datatype_info::add_datatype_info<DataType>(*this);

    this->add_to_label("@layout", sycldnn::bench::TypeName<DataLayout>::name);
    this->add_to_label("@operator",
                       sycldnn::bench::OperatorTypeName<SNNPoolOperator>::name);
    this->add_to_label("@direction", sycldnn::bench::TypeName<Direction>::name);
    this->add_to_label("@library", "cuDNN");
    this->add_to_label("@backend", "cuDNN");
    this->add_to_label("short_name", "Pooling");
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

#define POOLING_BENCHMARK(name, ...)                                    \
  BENCHMARK_TEMPLATE_DEFINE_F(CUDNNPoolingBenchmark, name, __VA_ARGS__) \
  (benchmark::State & state) {                                          \
    this->set_model(get_benchmark_name());                              \
    this->run(state);                                                   \
  }                                                                     \
  BENCHMARK_REGISTER_F(CUDNNPoolingBenchmark, name)                     \
      ->UseManualTime()                                                 \
      ->Unit(benchmark::kNanosecond)                                    \
      ->Apply(RunForAllParamSets);

#endif  // SYCLDNN_BENCH_POOL_CUDNN_POOL_EXECUTOR_H_
