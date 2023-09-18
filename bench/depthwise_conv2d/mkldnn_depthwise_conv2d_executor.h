/*
 * Copyright Codeplay Software Ltd.
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
#ifndef PORTDNN_BENCH_DEPTHWISE_CONV2D_MKLDNN_DEPTHWISE_CONV2D_EXECUTOR_H_
#define PORTDNN_BENCH_DEPTHWISE_CONV2D_MKLDNN_DEPTHWISE_CONV2D_EXECUTOR_H_

#include <benchmark/benchmark.h>

#include "portdnn/conv2d/conv_type.h"
#include "portdnn/depthwise_conv2d/params.h"
#include "portdnn/depthwise_conv2d/sizes.h"

#include "base_depthwise_convolution_fixture.h"
#include "benchmark_config.h"
#include "benchmark_params.h"

#include "bench/fixture/add_datatype_info.h"
#include "bench/fixture/base_executor.h"
#include "bench/fixture/statistic.h"
#include "bench/fixture/string_reporter.h"
#include "bench/fixture/typenames.h"

#include <mkldnn.hpp>

#include <numeric>
#include <type_traits>

namespace sycldnn {
namespace bench {

/** Executor to perform the Depthwise Conv2d benchmark using MKL-DNN. */
template <typename DataType, typename Benchmark>
struct MKLDepthwiseConv2DExecutor : public BaseExecutor {
 private:
  using State = ::benchmark::State;
  using DepthwiseConv2DParams = depthwise_conv2d::DepthwiseConv2DParams;

  /** Get a reference to the underlying benchmark fixture. */
  Benchmark& underlying_benchmark() { return static_cast<Benchmark&>(*this); }

 public:
  /** Execute a depthwise conv2d benchmark with the given parameters. */
  void execute(State& state, DepthwiseConv2DParams const& params) {
    if (params.channel_multiplier != 1) {
      state.SkipWithError(
          "Channel multiplier must be one for MKL-DNN depthwise convolutions");
    }
    if (!(std::is_same<DataType, float>::value)) {
      state.SkipWithError(
          "Data format must be 32-bit float for MKL-DNN depthwise "
          "convolutions");
    }
    // Allocate tensors.
    mkldnn::engine engine{mkldnn::engine::cpu, 0};
    mkldnn::stream stream{engine};

    mkldnn::memory::dims in_shape = {params.batch, params.channels,
                                     params.in_rows, params.in_cols};
    // MKL-DNN uses 'groups' to implement depthwise-convolutions,
    // with the number of groups being the number of channels.
    // For each group, there is then 1 input channel, and 1 output channel.
    mkldnn::memory::dims fil_shape = {params.channels, /* groups */
                                      1,               /* output channels */
                                      1,               /* input channels */
                                      params.window_rows, params.window_cols};
    mkldnn::memory::dims bias_shape = {params.channels};
    mkldnn::memory::dims out_shape = {params.batch, params.channels,
                                      params.out_rows, params.out_cols};
    mkldnn::memory::dims stride = {params.stride_rows, params.stride_cols};
    mkldnn::memory::dims padding_before = {params.pad_rows, params.pad_cols};
    const int s_pad_end_rows = (params.out_rows - 1) * params.stride_rows +
                               params.window_rows - params.in_rows -
                               params.pad_rows;
    const unsigned pad_end_rows = std::max(s_pad_end_rows, 0);
    const int s_pad_end_cols = (params.out_cols - 1) * params.stride_cols +
                               params.window_cols - params.in_cols -
                               params.pad_cols;
    const unsigned pad_end_cols = std::max(s_pad_end_cols, 0);
    mkldnn::memory::dims padding_after = {pad_end_rows, pad_end_cols};

    auto product = [](mkldnn::memory::dims const& dims) {
      return std::accumulate(begin(dims), end(dims), 1,
                             std::multiplies<size_t>());
    };

    std::vector<float> in_vals(product(in_shape));
    std::vector<float> fil_vals(product(fil_shape));
    std::vector<float> bias_vals(product(bias_shape));
    std::vector<float> out_vals(product(out_shape));

    auto in_mem_desc = mkldnn::memory::desc{{in_shape},
                                            mkldnn::memory::data_type::f32,
                                            mkldnn::memory::format_tag::nchw};
    auto fil_mem_desc = mkldnn::memory::desc{{fil_shape},
                                             mkldnn::memory::data_type::f32,
                                             mkldnn::memory::format_tag::goihw};
    auto bias_mem_desc = mkldnn::memory::desc{{bias_shape},
                                              mkldnn::memory::data_type::f32,
                                              mkldnn::memory::format_tag::x};
    auto out_mem_desc = mkldnn::memory::desc{{out_shape},
                                             mkldnn::memory::data_type::f32,
                                             mkldnn::memory::format_tag::nchw};

    auto conv_desc =
        mkldnn::convolution_forward::desc{mkldnn::prop_kind::forward_inference,
                                          mkldnn::convolution_direct,
                                          in_mem_desc,
                                          fil_mem_desc,
                                          bias_mem_desc,
                                          out_mem_desc,
                                          stride,
                                          padding_before,
                                          padding_after,
                                          mkldnn::padding_kind::zero};

    auto conv_prim_desc =
        mkldnn::convolution_forward::primitive_desc{conv_desc, engine};
    auto conv = mkldnn::convolution_forward{conv_prim_desc};

    auto in_mem = mkldnn::memory{in_mem_desc, engine, in_vals.data()};
    auto fil_mem = mkldnn::memory{fil_mem_desc, engine, fil_vals.data()};
    auto bias_mem = mkldnn::memory{bias_mem_desc, engine, bias_vals.data()};
    auto out_mem = mkldnn::memory{out_mem_desc, engine};

    // Run the layer once to eliminate lazy behaviour.
    conv.execute(stream, {{MKLDNN_ARG_SRC, in_mem},
                          {MKLDNN_ARG_WEIGHTS, fil_mem},
                          {MKLDNN_ARG_BIAS, bias_mem},
                          {MKLDNN_ARG_DST, out_mem}});

    for (auto _ : state) {
      this->start_timing();
      conv.execute(stream, {{MKLDNN_ARG_SRC, in_mem},
                            {MKLDNN_ARG_WEIGHTS, fil_mem},
                            {MKLDNN_ARG_BIAS, bias_mem},
                            {MKLDNN_ARG_DST, out_mem}});
      this->end_timing();

      this->set_iteration_time(state);
    }

    auto& benchmark = underlying_benchmark();
    benchmark.template set_items_processed<sycldnn::conv2d::conv_type::Forward>(
        state, params);
    benchmark.add_param_counters(state, params);
    benchmark.template add_bandwidth_counters<float>(
        state, sycldnn::depthwise_conv2d::get_sizes<
                   sycldnn::conv2d::conv_type::Forward>(params));
    this->finish_benchmark(state);
  }
};

}  // namespace bench
}  // namespace sycldnn

extern const char* commit_date;
extern const char* commit_hash;

template <typename DataType>
class MKLDepthwiseConvolutionBenchmark
    : public sycldnn::bench::MKLDepthwiseConv2DExecutor<
          DataType, MKLDepthwiseConvolutionBenchmark<DataType>>,
      public sycldnn::bench::StringReporter,
      public BaseDepthwiseConvolutionBenchmark {
 private:
  using State = benchmark::State;

 protected:
  void run(State& state) {
    auto params = benchmark_params::deserialize(state);
    this->add_statistic(std::unique_ptr<sycldnn::bench::Statistic>{
        new sycldnn::bench::MaxStatistic{}});
    this->add_statistic(std::unique_ptr<sycldnn::bench::Statistic>{
        new sycldnn::bench::MinStatistic{}});
    this->add_statistic(std::unique_ptr<sycldnn::bench::Statistic>{
        new sycldnn::bench::StdDevStatistic{}});
    this->execute(state, params);

    sycldnn::bench::datatype_info::add_datatype_info<float>(*this);

    this->add_to_label("@conv_type", "Forward");
    this->add_to_label("@selector", "MKL-DNN");
    this->add_to_label("@library", "MKL-DNN");
    this->add_to_label("short_name", "Depthwise Convolution");
    this->add_to_label("git_hash", commit_hash);
    this->add_to_label("vendor_name", "Intel");
    this->add_to_label("device_name", "MKL-DNN");
    this->add_to_label("device_version", "N/A");
    this->add_to_label("driver_version", "N/A");
    this->set_label(state);
  };

  void set_model(const char* model_name) {
    this->add_to_label("@model_name", model_name);
  }
};

#define DEPTHWISE_CONVOLUTION_BENCHMARK(name, ...)                    \
  BENCHMARK_TEMPLATE_DEFINE_F(MKLDepthwiseConvolutionBenchmark, name, \
                              __VA_ARGS__)                            \
  (benchmark::State & state) {                                        \
    this->set_model(get_benchmark_name());                            \
    this->run(state);                                                 \
  }                                                                   \
  BENCHMARK_REGISTER_F(MKLDepthwiseConvolutionBenchmark, name)        \
      ->UseManualTime()                                               \
      ->Unit(benchmark::kNanosecond)                                  \
      ->Apply(RunForAllParamSets);

#endif  // PORTDNN_BENCH_DEPTHWISE_CONV2D_MKLDNN_DEPTHWISE_CONV2D_EXECUTOR_H_
