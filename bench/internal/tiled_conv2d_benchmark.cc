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
#include <benchmark/benchmark.h>

// TODO(jwlawson): Remove cassert when no longer needed before Eigen include
#include <cassert>
#include <unsupported/Eigen/CXX11/Tensor>

#include "sycldnn/accessor_types.h"
#include "sycldnn/status.h"

#include "sycldnn/backend/eigen_backend.h"

#include "sycldnn/conv2d/launch.h"
#include "sycldnn/conv2d/params.h"
#include "sycldnn/conv2d/selector/selector.h"

#include "src/conv2d/tiled/kernel_params.h"
#include "src/conv2d/tiled/output_size.h"
#include "src/conv2d/tiled/queue_tiled_kernel_impl.h"

#include "bench/conv2d/fixture.h"

namespace {

template <typename T, typename Index, typename ConvType, int TileRows,
          int TileCols, int ChannelVectorWidth, int FeatureVectorWidth,
          bool UseFastDiv, int WindowRows, int WindowCols, int Stride,
          typename Backend>
sycldnn::SNNStatus launch_kernel(
    typename Backend::template pointer_type<const T> input,
    typename Backend::template pointer_type<const T> filter,
    typename Backend::template pointer_type<T> output,
    sycldnn::conv2d::Conv2DParams const& params,
    sycldnn::conv2d::ConvSizes const& sizes, Backend& backend) {
  auto in_buff = backend.template get_buffer(input, sizes.input_size);
  sycldnn::ReadAccessor<const T> in_acc{in_buff};

  auto fil_buff = backend.template get_buffer(filter, sizes.filter_size);
  sycldnn::ReadAccessor<const T> fil_acc{fil_buff};

  auto out_buff = backend.template get_buffer(output, sizes.output_size);
  sycldnn::WriteAccessor<T> out_acc{out_buff};

  auto queue = backend.get_queue();
  Index output_size = sycldnn::conv2d::internal::TiledOutputSize<
      ConvType, TileRows, TileCols, ChannelVectorWidth,
      FeatureVectorWidth>::get(params);

  auto kernel_params =
      sycldnn::conv2d::internal::get_kernel_params<ConvType>(params);

  auto status = sycldnn::conv2d::internal::queue_tiled_kernel<
      T, Index, ConvType, TileRows, TileCols, ChannelVectorWidth,
      FeatureVectorWidth, UseFastDiv, WindowRows, WindowCols, Stride>(
      in_acc, fil_acc, out_acc, kernel_params, output_size, queue);
  return status;
}

template <typename ParamGen, typename ConvType, int TileRows, int TileCols,
          int ChannelVectorWidth, int FeatureVectorWidth, bool UseFastDiv,
          int WindowRows, int WindowCols, int Stride>
class TiledConvolutionBenchmark : public BaseConvolutionBenchmark {
 private:
  using State = benchmark::State;
  using Conv2DParams = sycldnn::conv2d::Conv2DParams;

 protected:
  void add_opencl_device_info(cl::sycl::device const& device);
  void execute(State& state, Conv2DParams const& params);

  void run(State& state) {
    auto params = ParamGen()();
    if (params.window_rows != WindowRows || params.window_cols != WindowCols ||
        params.stride_rows != Stride || params.stride_cols != Stride) {
      state.SkipWithError(
          "Runtime paramters don't match the compile time kernel sizes.");
      return;
    }
    this->execute(state, params);
  };
};

/**
 * Get a pointer to a cached Eigen SYCL queue.
 *
 * By making the Eigen queue a static object it is kept alive between bechmark
 * runs, and so any kernels which have been built once do not need to be built
 * again for later benchmarks. This ensures that running multiple benchmarks in
 * the same binary does not waste time continually rebuilding kernels.
 */
Eigen::QueueInterface* get_eigen_queue() {
  static Eigen::QueueInterface queue{cl::sycl::default_selector{}};
  return &queue;
}

template <typename ParamGen, typename ConvType, int TileRows, int TileCols,
          int ChannelVectorWidth, int FeatureVectorWidth, bool UseFastDiv,
          int WindowRows, int WindowCols, int Stride>
void TiledConvolutionBenchmark<
    ParamGen, ConvType, TileRows, TileCols, ChannelVectorWidth,
    FeatureVectorWidth, UseFastDiv, WindowRows, WindowCols,
    Stride>::execute(benchmark::State& state,
                     sycldnn::conv2d::Conv2DParams const& params) {
  Eigen::SyclDevice device{get_eigen_queue()};
  sycldnn::backend::EigenBackend backend{device};

  auto conv_sizes = sycldnn::conv2d::get_sizes<ConvType>(params);

  size_t inp_bytes = conv_sizes.input_size * sizeof(float);
  float* inp_gpu = static_cast<float*>(device.allocate(inp_bytes));
  size_t fil_bytes = conv_sizes.filter_size * sizeof(float);
  float* fil_gpu = static_cast<float*>(device.allocate(fil_bytes));
  size_t out_bytes = conv_sizes.output_size * sizeof(float);
  float* out_gpu = static_cast<float*>(device.allocate(out_bytes));

  {  // Ensure the kernel is built before benchmarking
    auto status = launch_kernel<float, int, ConvType, TileRows, TileCols,
                                ChannelVectorWidth, FeatureVectorWidth,
                                UseFastDiv, WindowRows, WindowCols, Stride>(
        inp_gpu, fil_gpu, out_gpu, params, conv_sizes, backend);
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
    auto status = launch_kernel<float, int, ConvType, TileRows, TileCols,
                                ChannelVectorWidth, FeatureVectorWidth,
                                UseFastDiv, WindowRows, WindowCols, Stride>(
        inp_gpu, fil_gpu, out_gpu, params, conv_sizes, backend);

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

  set_items_processed<ConvType>(state, params);
  add_param_counters(state, params);
  state.counters["tile_rows"] = TileRows;
  state.counters["tile_cols"] = TileCols;
  state.counters["ch_vect"] = ChannelVectorWidth;
  state.counters["feat_vect"] = FeatureVectorWidth;
  state.counters["fast_div"] = UseFastDiv;
  add_bandwidth_counters<float>(state, conv_sizes);

  key_value_map["selector"] = "TiledSelector";
  key_value_map["git_hash"] = commit_hash;
  set_label(state);
}

template <typename ParamGen, typename ConvType, int TileRows, int TileCols,
          int ChannelVectorWidth, int FeatureVectorWidth, bool UseFastDiv,
          int WindowRows, int WindowCols, int Stride>
void TiledConvolutionBenchmark<
    ParamGen, ConvType, TileRows, TileCols, ChannelVectorWidth,
    FeatureVectorWidth, UseFastDiv, WindowRows, WindowCols,
    Stride>::add_opencl_device_info(const cl::sycl::device& device) {
  // OpenCL is unclear whether strings returned from clGet*Info() should be
  // null terminated, and ComputeCpp currently copies embedded nulls.
  // On some OpenCL implemntations this results in strings that behave
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

#define TILED_BENCHMARK(name, ...)                                          \
  BENCHMARK_TEMPLATE_DEFINE_F(TiledConvolutionBenchmark, name, __VA_ARGS__) \
  (benchmark::State & state) { this->run(state); }                          \
  BENCHMARK_REGISTER_F(TiledConvolutionBenchmark, name)                     \
      ->UseManualTime()                                                     \
      ->Unit(benchmark::kNanosecond);

template <int WindowRows, int WindowCols, int Stride>
struct ParamGenerator {
  sycldnn::conv2d::Conv2DParams operator()() {
    sycldnn::conv2d::Conv2DParams params;
    params.channels = 196;
    params.features = 384;
    params.batch = 16;
    params.in_rows = 27;
    params.in_cols = 27;
    params.window_rows = WindowCols;
    params.window_cols = WindowRows;
    params.stride_rows = Stride;
    params.stride_cols = Stride;
    params.out_rows = 27;
    params.out_cols = 27;
    params.pad_rows = 0;
    params.pad_cols = 0;
    params.dilation_rows = 1;
    params.dilation_cols = 1;
    return params;
  }
};

}  // namespace

#define PARAM_BENCHMARK(name, direction, tile_row, tile_col, ch_vect,        \
                        feat_vect, fast_div, window_row, window_col, stride) \
  TILED_BENCHMARK(                                                           \
      name##_##tile_row##_##tile_col##_##ch_vect##_##feat_vect##_##fast_div, \
      ParamGenerator<window_row, window_col, stride>, direction, tile_row,   \
      tile_col, ch_vect, feat_vect, fast_div, window_row, window_col, stride);

#define BENCH_WITH_TILES(name, direction, tile_row, tile_col, fast_div, \
                         window_row, window_col, stride)                \
  PARAM_BENCHMARK(name, direction, tile_row, tile_col, 1, 1, fast_div,  \
                  window_row, window_col, stride)                       \
  PARAM_BENCHMARK(name, direction, tile_row, tile_col, 1, 2, fast_div,  \
                  window_row, window_col, stride)                       \
  PARAM_BENCHMARK(name, direction, tile_row, tile_col, 1, 4, fast_div,  \
                  window_row, window_col, stride)                       \
  PARAM_BENCHMARK(name, direction, tile_row, tile_col, 1, 8, fast_div,  \
                  window_row, window_col, stride)                       \
  PARAM_BENCHMARK(name, direction, tile_row, tile_col, 2, 1, fast_div,  \
                  window_row, window_col, stride)                       \
  PARAM_BENCHMARK(name, direction, tile_row, tile_col, 2, 2, fast_div,  \
                  window_row, window_col, stride)                       \
  PARAM_BENCHMARK(name, direction, tile_row, tile_col, 2, 4, fast_div,  \
                  window_row, window_col, stride)                       \
  PARAM_BENCHMARK(name, direction, tile_row, tile_col, 4, 1, fast_div,  \
                  window_row, window_col, stride)                       \
  PARAM_BENCHMARK(name, direction, tile_row, tile_col, 4, 2, fast_div,  \
                  window_row, window_col, stride)                       \
  PARAM_BENCHMARK(name, direction, tile_row, tile_col, 4, 4, fast_div,  \
                  window_row, window_col, stride)

#define BENCH_WITH_FAST_DIV(name, direction, fast_div, window_row, window_col, \
                            stride)                                            \
  BENCH_WITH_TILES(name, direction, 1, 4, fast_div, window_row, window_col,    \
                   stride)                                                     \
  BENCH_WITH_TILES(name, direction, 2, 2, fast_div, window_row, window_col,    \
                   stride)                                                     \
  BENCH_WITH_TILES(name, direction, 2, 4, fast_div, window_row, window_col,    \
                   stride)                                                     \
  BENCH_WITH_TILES(name, direction, 2, 3, fast_div, window_row, window_col,    \
                   stride)                                                     \
  BENCH_WITH_TILES(name, direction, 3, 2, fast_div, window_row, window_col,    \
                   stride)                                                     \
  BENCH_WITH_TILES(name, direction, 3, 3, fast_div, window_row, window_col,    \
                   stride)                                                     \
  BENCH_WITH_TILES(name, direction, 3, 4, fast_div, window_row, window_col,    \
                   stride)                                                     \
  BENCH_WITH_TILES(name, direction, 4, 1, fast_div, window_row, window_col,    \
                   stride)                                                     \
  BENCH_WITH_TILES(name, direction, 4, 2, fast_div, window_row, window_col,    \
                   stride)                                                     \
  BENCH_WITH_TILES(name, direction, 4, 3, fast_div, window_row, window_col,    \
                   stride)                                                     \
  BENCH_WITH_TILES(name, direction, 4, 4, fast_div, window_row, window_col,    \
                   stride)

#define BENCH_BASE(name, direction, window_row, window_col, stride)           \
  BENCH_WITH_FAST_DIV(name, direction, false, window_row, window_col, stride) \
  BENCH_WITH_FAST_DIV(name, direction, true, window_row, window_col, stride)

BENCH_BASE(Forward, sycldnn::conv2d::conv_type::Forward, 3, 3, 1);
BENCH_BASE(InputBackprop, sycldnn::conv2d::conv_type::InputBackprop, 3, 3, 1);
