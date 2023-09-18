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
#ifndef PORTDNN_BENCH_CONV2D_MKLDNN_CONV2D_EXECUTOR_H_
#define PORTDNN_BENCH_CONV2D_MKLDNN_CONV2D_EXECUTOR_H_

#include <benchmark/benchmark.h>

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

#include <mkldnn.hpp>

#include <numeric>

namespace sycldnn {
namespace bench {

/** Executor to perform the Conv2d benchmark using MKL-DNN.  */
template <typename Benchmark>
struct MKLConv2DExecutor : public BaseExecutor {
 private:
  using State = ::benchmark::State;
  using Conv2DParams = conv2d::Conv2DParams;

  /** Get a reference to the underlying benchmark fixture. */
  Benchmark& underlying_benchmark() { return static_cast<Benchmark&>(*this); }

 public:
  /** Execute a conv2d benchmark with the given parameters. */
  void execute(State& state, Conv2DParams const& params) {
    // Allocate tensors.
    mkldnn::engine engine{mkldnn::engine::cpu, 0};
    mkldnn::stream stream{engine};

    mkldnn::memory::dims in_shape = {params.batch, params.channels,
                                     params.in_rows, params.in_cols};
    mkldnn::memory::dims fil_shape = {params.features, params.channels,
                                      params.window_rows, params.window_cols};
    mkldnn::memory::dims bias_shape = {params.features};
    mkldnn::memory::dims out_shape = {params.batch, params.features,
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
                                             mkldnn::memory::format_tag::oihw};
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
        state, sycldnn::conv2d::get_sizes<sycldnn::conv2d::conv_type::Forward>(
                   params));
    this->finish_benchmark(state);
  }
};

}  // namespace bench
}  // namespace sycldnn

extern const char* commit_date;
extern const char* commit_hash;

template <typename DataType>
class MKLConvolutionBenchmark : public sycldnn::bench::MKLConv2DExecutor<
                                    MKLConvolutionBenchmark<DataType>>,
                                public sycldnn::bench::StringReporter,
                                public BaseConvolutionBenchmark {
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

    sycldnn::bench::datatype_info::add_datatype_info<DataType>(*this);

    this->add_to_label("@conv_type", "Forward");
    this->add_to_label("@selector", "MKL-DNN");
    this->add_to_label("@library", "MKL-DNN");
    this->add_to_label("short_name", "Convolution");
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

#define CONVOLUTION_BENCHMARK(name, ...)                                  \
  BENCHMARK_TEMPLATE_DEFINE_F(MKLConvolutionBenchmark, name, __VA_ARGS__) \
  (benchmark::State & state) {                                            \
    this->set_model(get_benchmark_name());                                \
    this->run(state);                                                     \
  }                                                                       \
  BENCHMARK_REGISTER_F(MKLConvolutionBenchmark, name)                     \
      ->UseManualTime()                                                   \
      ->Unit(benchmark::kNanosecond)                                      \
      ->Apply(RunForAllParamSets);

#endif  // PORTDNN_BENCH_CONV2D_MKLDNN_CONV2D_EXECUTOR_H_
