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
#ifndef PORTDNN_BENCH_CONV2D_CUDNN_CONV2D_EXECUTOR_H_
#define PORTDNN_BENCH_CONV2D_CUDNN_CONV2D_EXECUTOR_H_

#include "portdnn/conv2d/conv_type.h"
#include "portdnn/conv2d/params.h"
#include "portdnn/conv2d/sizes.h"
#include "portdnn/helpers/scope_exit.h"

#include "base_convolution_fixture.h"
#include "benchmark_config.h"
#include "benchmark_params.h"

#include "bench/fixture/add_datatype_info.h"
#include "bench/fixture/base_executor.h"
#include "bench/fixture/statistic.h"
#include "bench/fixture/string_reporter.h"
#include "bench/fixture/typenames.h"

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

template <typename F, typename... Args>
void cuda_check(F f, Args... args) {
  auto st = f(args...);
  if (cudaSuccess != st) {
    throw std::runtime_error("Unexpected CUDA function failure: " +
                             std::to_string(st));
  }
}

template <template <typename T> class F, typename U, typename... Args>
void cuda_check(F<U> f, U u, size_t s) {
  auto st = f(u, s);
  if (cudaSuccess != st) {
    throw std::runtime_error("Unexpected CUDA function failure: " +
                             std::to_string(st));
  }
}

template <typename F, typename... Args>
void cudnn_check(F f, Args... args) {
  auto st = f(args...);
  if (CUDNN_STATUS_SUCCESS != st) {
    throw std::runtime_error("Unexpected cuDNN function failure: " +
                             std::to_string(st));
  }
}

namespace sycldnn {
namespace bench {

template <typename Benchmark, cudnnConvolutionFwdAlgo_t Algo>
struct CUDNNConv2DExecutor : public BaseExecutor {
 private:
  using State = ::benchmark::State;
  using Conv2DParams = conv2d::Conv2DParams;

  /** Get a reference to the underlying benchmark fixture. */
  Benchmark& underlying_benchmark() { return static_cast<Benchmark&>(*this); }

 protected:
  cudnnHandle_t handle_;

 public:
  /** Execute a conv2d benchmark with the given parameters and selector. */
  void execute(State& state, Conv2DParams const& params) {
    auto& benchmark = underlying_benchmark();

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

    auto size = [](TensorShape& shape) {
      return static_cast<size_t>(std::accumulate(
          std::begin(shape), std::end(shape), 1, std::multiplies<int32_t>()));
    };

    float* inp_gpu;
    float* out_gpu;
    float* fil_gpu;
    cuda_check(cudaMalloc<float>, &inp_gpu, size(inp_shape) * sizeof(float));
    cuda_check(cudaMalloc<float>, &out_gpu, size(out_shape) * sizeof(float));
    cuda_check(cudaMalloc<float>, &fil_gpu, size(fil_shape) * sizeof(float));
    cuda_check(cudaMemset, inp_gpu, 0, size(inp_shape) * sizeof(float));
    cuda_check(cudaMemset, out_gpu, 0, size(out_shape) * sizeof(float));
    cuda_check(cudaMemset, fil_gpu, 0, size(fil_shape) * sizeof(float));

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

    cudnn_check(cudnnSetTensor4dDescriptor, inp_desc, CUDNN_TENSOR_NCHW,
                CUDNN_DATA_FLOAT, inp_shape[0], inp_shape[1], inp_shape[2],
                inp_shape[3]);
    cudnn_check(cudnnSetTensor4dDescriptor, out_desc, CUDNN_TENSOR_NCHW,
                CUDNN_DATA_FLOAT, out_shape[0], out_shape[1], out_shape[2],
                out_shape[3]);
    cudnn_check(cudnnSetFilter4dDescriptor, fil_desc, CUDNN_DATA_FLOAT,
                CUDNN_TENSOR_NCHW, fil_shape[0], fil_shape[1], fil_shape[2],
                fil_shape[3]);
    cudnn_check(cudnnSetConvolution2dDescriptor, conv_desc, params.pad_rows,
                params.pad_cols, params.stride_rows, params.stride_cols,
                params.dilation_rows, params.dilation_cols, CUDNN_CONVOLUTION,
                CUDNN_DATA_FLOAT);

    float* wspace = nullptr;
    size_t wspace_size = 0;
    if (Algo != CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM) {
      cudnn_check(cudnnGetConvolutionForwardWorkspaceSize, handle_, inp_desc,
                  fil_desc, conv_desc, out_desc, Algo, &wspace_size);
      cuda_check(cudaMalloc<float>, &wspace, wspace_size);
    }
    SNN_ON_SCOPE_EXIT {
      if (wspace) {
        cudaFree(wspace);
      }
    };
    cuda_check(cudaDeviceSynchronize);

    auto alpha = 1.f;
    auto beta = 0.f;
    cudnn_check(cudnnConvolutionForward, this->handle_, &alpha, inp_desc,
                inp_gpu, fil_desc, fil_gpu, conv_desc, Algo, wspace,
                wspace_size, &beta, out_desc, out_gpu);

    cuda_check(cudaDeviceSynchronize);

    for (auto _ : state) {
      this->start_timing();
      cudnn_check(cudnnConvolutionForward, this->handle_, &alpha, inp_desc,
                  inp_gpu, fil_desc, fil_gpu, conv_desc, Algo, wspace,
                  wspace_size, &beta, out_desc, out_gpu);

      cuda_check(cudaDeviceSynchronize);
      this->end_timing();
      this->set_iteration_time(state);
    }

    benchmark.template set_items_processed<conv2d::conv_type::Forward>(state,
                                                                       params);
    benchmark.add_param_counters(state, params);

    sycldnn::conv2d::ConvSizes conv_sizes{size(inp_shape), size(fil_shape),
                                          size(out_shape)};
    benchmark.template add_bandwidth_counters<float>(state, conv_sizes);

    this->finish_benchmark(state);
  }
};

}  // namespace bench
}  // namespace sycldnn

extern const char* commit_date;
extern const char* commit_hash;

template <typename DataType, cudnnConvolutionFwdAlgo_t Algo>
class CUDNNConvolutionBenchmark
    : public sycldnn::bench::CUDNNConv2DExecutor<
          CUDNNConvolutionBenchmark<DataType, Algo>, Algo>,
      public sycldnn::bench::StringReporter,
      public BaseConvolutionBenchmark {
  using State = benchmark::State;
  cudaDeviceProp properties_;
  int version_;

 protected:
  void run(State& state) {
    auto params = benchmark_params::deserialize(state);
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

    this->add_to_label("@conv_type", "Forward");
    this->add_to_label("@selector", "cuDNN");
    this->add_to_label("@library", "cuDNN");
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
    cuda_check(cudaInitDevice, 0, 0, 0);
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
    this->run(state);                                                       \
  }                                                                         \
  BENCHMARK_REGISTER_F(CUDNNConvolutionBenchmark, name)                     \
      ->UseManualTime()                                                     \
      ->Unit(benchmark::kNanosecond)                                        \
      ->Apply(RunForAllParamSets);

#endif  // PORTDNN_BENCH_CONV2D_CUDNN_CONV2D_EXECUTOR_H_
