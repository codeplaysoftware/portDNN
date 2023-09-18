/*
 * Copyright Codeplay Software Ltd.
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
#ifndef PORTDNN_BENCH_DEPTHWISE_CONV2D_SNN_CONV2D_EXECUTOR_H_
#define PORTDNN_BENCH_DEPTHWISE_CONV2D_SNN_CONV2D_EXECUTOR_H_

#include "portdnn/depthwise_conv2d/launch.h"
#include "portdnn/depthwise_conv2d/params.h"

#include "portdnn/helpers/handle_exception.h"
#include "portdnn/helpers/scope_exit.h"

#include "bench/fixture/base_executor.h"

namespace sycldnn {
namespace bench {

/** Executor to perform the DepthwiseConv2d benchmark using portDNN.  */
template <typename Benchmark, typename ConvType>
struct SNNDepthwiseConv2DExecutor : public BaseExecutor {
 private:
  using State = ::benchmark::State;
  using DepthwiseConv2DParams = depthwise_conv2d::DepthwiseConv2DParams;

  /** Get a reference to the underlying benchmark fixture. */
  Benchmark& underlying_benchmark() { return static_cast<Benchmark&>(*this); }

 public:
  /** Execute a depthwise_conv2d benchmark with the given parameters. */
  void execute(State& state, DepthwiseConv2DParams const& params) {
    auto& benchmark = underlying_benchmark();
    auto backend = benchmark.get_backend();

    auto conv_sizes = sycldnn::depthwise_conv2d::get_sizes<ConvType>(params);

    std::vector<float> inp_vec(conv_sizes.input_size);
    std::vector<float> fil_vec(conv_sizes.filter_size);
    std::vector<float> out_vec(conv_sizes.output_size);

    auto inp_gpu =
        benchmark.get_initialised_device_memory(inp_vec.size(), inp_vec);
    auto fil_gpu =
        benchmark.get_initialised_device_memory(fil_vec.size(), fil_vec);
    auto out_gpu =
        benchmark.get_initialised_device_memory(out_vec.size(), out_vec);

    SNN_ON_SCOPE_EXIT {
      benchmark.deallocate_ptr(out_gpu);
      benchmark.deallocate_ptr(fil_gpu);
      benchmark.deallocate_ptr(inp_gpu);
    };

    {  // Ensure the kernel is built before benchmarking
      SNNStatus status;
      try {
        status = sycldnn::depthwise_conv2d::launch<float, ConvType>(
            inp_gpu, fil_gpu, out_gpu, params, backend);
      } catch (cl::sycl::exception const& e) {
        helpers::handle_exception(e, [&](std::string& msg) {
          state.SkipWithError((msg + UnexpectedFailure).c_str());
        });
        return;
      }

      if (sycldnn::StatusCode::OK != status.status) {
        state.SkipWithError(UnsupportedFailure);
        return;
      }

      try {
        status.event.wait_and_throw();
      } catch (cl::sycl::exception const& e) {
        helpers::handle_exception(e, [&](std::string& msg) {
          state.SkipWithError((msg + UnexpectedFailure).c_str());
        });
        return;
      } catch (std::exception const& e) {
        helpers::handle_exception(e, [&](std::string& msg) {
          state.SkipWithError((msg + UnexpectedFailure).c_str());
        });
        return;
      }
    }

    for (auto _ : state) {
      this->start_timing();
      try {
        auto status = sycldnn::depthwise_conv2d::launch<float, ConvType>(
            inp_gpu, fil_gpu, out_gpu, params, backend);

        status.event.wait_and_throw();
      } catch (cl::sycl::exception const& e) {
        helpers::handle_exception(e, [&](std::string& msg) {
          state.SkipWithError((msg + UnexpectedFailure).c_str());
        });
        return;
      }

      this->end_timing();
      this->set_iteration_time(state);
    }

    benchmark.template set_items_processed<ConvType>(state, params);
    benchmark.add_param_counters(state, params);
    benchmark.template add_bandwidth_counters<float>(state, conv_sizes);

    this->finish_benchmark(state);
  }
};

}  // namespace bench
}  // namespace sycldnn

#endif  // PORTDNN_BENCH_DEPTHWISE_CONV2D_SNN_CONV2D_EXECUTOR_H_
