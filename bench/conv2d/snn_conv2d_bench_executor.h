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
#ifndef SYCLDNN_BENCH_CONV2D_SNN_CONV2D_BENCH_EXECUTOR_H_
#define SYCLDNN_BENCH_CONV2D_SNN_CONV2D_BENCH_EXECUTOR_H_

#include <stdlib.h>
#include <vector>

#include "sycldnn/helpers/handle_exception.h"
#include "sycldnn/helpers/scope_exit.h"

#include "bench/common/event_handling.h"

#include "bench/fixture/base_bench_executor.h"

#include "bench/conv2d/snn_conv2d_executor.h"

#include "sycldnn/conv2d/selector/direct_selector.h"
#include "sycldnn/conv2d/selector/winograd_selector.h"

#ifdef SNN_BENCH_PORTBLAS
#include "sycldnn/backend/sycl_blas_backend.h"
#endif

#ifdef SNN_ENABLE_USM
#include "sycldnn/backend/snn_usm_backend.h"
#endif

namespace sycldnn {
namespace bench {

/** BenchExecutor to perform the Conv2d benchmark using SYCL-DNN.  */
template <typename Benchmark, typename ConvType, typename Backend,
          bool UseTensorOp, bool InOrderQueue, bool UseTranspose>
struct SNNConv2DBenchExecutor : public BaseBenchExecutor {
 private:
  using State = ::benchmark::State;
  using Conv2DParams = conv2d::Conv2DParams;
  using Selector = conv2d::Selector;
  using Executor = SNNConv2DExecutor<float, ConvType, Selector, Backend,
                                     InOrderQueue, UseTranspose>;

  /** Get a reference to the underlying benchmark fixture. */
  Benchmark& underlying_benchmark() { return static_cast<Benchmark&>(*this); }

  void validate_results(Executor& bench_conv2d,
                        std::vector<float> const& inp_vec,
                        std::vector<float> const& fil_vec,
                        Conv2DParams const& params, float tolerance = 1e-5) {
    using RefExecutor =
        SNNConv2DExecutor<float, ConvType, conv2d::DirectSelector,
                          backend::SNNBackend>;

    auto selector = conv2d::DirectSelector{};
    RefExecutor ref_conv2d(inp_vec, fil_vec, params, selector);

    SNN_ON_SCOPE_EXIT { ref_conv2d.teardown(); };

    // Allocate SNN device memory
    ref_conv2d.setup();

    // Execute SNN convolution
    auto status = ref_conv2d.execute();
    if (status.status != StatusCode::OK) {
      throw std::runtime_error(
          "Unexpected issue executing SNN convolution for CuDNN benchmark "
          "validation.");
    }

    status.event.wait_and_throw();

    const auto conv_sizes = conv2d::get_sizes<ConvType>(params);
    const auto output_size = conv_sizes.output_size;

    std::vector<float> ref_output(output_size);
    std::vector<float> bench_output(output_size);

    // Copy reference results to host
    ref_conv2d.copy_output_to_host(ref_output);
    // Copy benchmark results to host
    bench_conv2d.copy_output_to_host(bench_output);

    // Assert that results are equal.
    if (!internal::compare_results(ref_output, bench_output, tolerance)) {
      throw std::runtime_error(
          "Benchmark SNN results do not match reference results, exiting "
          "benchmark.");
    }
  }

 public:
  /** Execute a conv2d benchmark with the given parameters and selector. */
  void execute(State& state, Conv2DParams const& params, Selector& selector) {
    // Initialise host data
    const float max_val = 2.0;
    const auto conv2d_sizes = conv2d::get_sizes<ConvType>(params);
    const auto inp_vec =
        internal::iota_initialised_data(conv2d_sizes.input_size, max_val);
    const auto fil_vec =
        internal::iota_initialised_data(conv2d_sizes.filter_size, max_val);

    // Initialise executor
    Executor conv2d_executor(inp_vec, fil_vec, params, selector);

    SNN_ON_SCOPE_EXIT { conv2d_executor.teardown(); };

    // Allocate and initialise device memory
    try {
      conv2d_executor.setup();
    } catch (...) {
      state.SkipWithError(AllocationFailure);
      return;
    }

    // Setup portBlas tensor op
    if constexpr (UseTensorOp) {
      setenv("SB_ENABLE_JOINT_MATRIX", "1", 1);
    } else {
      setenv("SB_ENABLE_JOINT_MATRIX", "0", 1);
    }

    auto queue = conv2d_executor.get_queue();
    {  // Ensure the kernel is built before benchmarking
      SNNStatus status;
      try {
        status = conv2d_executor.execute();
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

    // Tolerance needs to be reduced for winograd large input backprop
    bool const is_wino_large =
        dynamic_cast<conv2d::WinogradLargeSelector*>(&selector) != nullptr;
    const float tolerance = is_wino_large ? 1e-1 : 1e-4;

    // Validate benchmark output against reference impl (Direct)
    this->validate_results(conv2d_executor, inp_vec, fil_vec, params,
                           tolerance);

    for (auto _ : state) {
      this->start_timing();
      try {
        auto status = conv2d_executor.execute();
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

      this->end_timing();
      this->set_iteration_time(state);
    }

    auto& benchmark = underlying_benchmark();
    benchmark.template set_items_processed<ConvType>(state, params);
    benchmark.add_param_counters(state, params);
    benchmark.template add_bandwidth_counters<float>(state, conv2d_sizes);

    this->finish_benchmark(state);
  }
};

}  // namespace bench
}  // namespace sycldnn

#endif  // SYCLDNN_BENCH_CONV2D_SNN_CONV2D_EXECUTOR_H_
