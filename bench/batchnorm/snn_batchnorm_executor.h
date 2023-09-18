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
#ifndef PORTDNN_BENCH_BATCHNORM_SNN_BATCHNORM_EXECUTOR_H_
#define PORTDNN_BENCH_BATCHNORM_SNN_BATCHNORM_EXECUTOR_H_

#include <benchmark/benchmark.h>

#include "portdnn/helpers/handle_exception.h"
#include "portdnn/helpers/scope_exit.h"

#include "portdnn/batchnorm/direction.h"
#include "portdnn/batchnorm/launch.h"

#include "bench/fixture/base_executor.h"

namespace sycldnn {
namespace bench {

/** Executor to perform the batchnorm benchmark using portDNN.  */
template <typename Benchmark, typename DataType, typename Backend>
struct SNNBatchnormExecutor : public BaseExecutor {
 private:
  using State = ::benchmark::State;
  using Params = batchnorm::BatchNormParams;

  /** Get a reference to the underlying benchmark fixture. */
  Benchmark& underlying_benchmark() { return static_cast<Benchmark&>(*this); }

 public:
  /** Execute the batchnorm benchmark for the given parameters. */
  void execute(State& state, Params const& params) {
    auto& benchmark = underlying_benchmark();
    auto& backend = benchmark.get_backend();
    using Direction = sycldnn::batchnorm::Forward;

    auto input_size =
        params.batch * params.rows * params.cols * params.channels;
    std::vector<float> input_vec(input_size);
    std::vector<float> beta_vec(params.channels);
    std::vector<float> gamma_vec(params.channels);
    std::vector<float> input_mean_vec(params.channels);
    std::vector<float> input_variance_vec(params.channels);
    std::vector<float> out_vec(input_size);

    auto input_gpu =
        benchmark.get_initialised_device_memory(input_vec.size(), input_vec);
    auto beta_gpu =
        benchmark.get_initialised_device_memory(beta_vec.size(), beta_vec);
    auto gamma_gpu =
        benchmark.get_initialised_device_memory(gamma_vec.size(), gamma_vec);
    auto input_mean_gpu = benchmark.get_initialised_device_memory(
        input_mean_vec.size(), input_mean_vec);
    auto input_variance_gpu = benchmark.get_initialised_device_memory(
        input_variance_vec.size(), input_variance_vec);
    auto out_gpu =
        benchmark.get_initialised_device_memory(out_vec.size(), out_vec);

    SNN_ON_SCOPE_EXIT {
      benchmark.deallocate_ptr(input_gpu);
      benchmark.deallocate_ptr(beta_gpu);
      benchmark.deallocate_ptr(gamma_gpu);
      benchmark.deallocate_ptr(input_mean_gpu);
      benchmark.deallocate_ptr(input_variance_gpu);
      benchmark.deallocate_ptr(out_gpu);
    };

    {  // Ensure the kernel is built before benchmarking
      SNNStatus status;
      try {
        status = sycldnn::batchnorm::launch<DataType, Backend, Direction>(
            input_gpu, beta_gpu, gamma_gpu, input_mean_gpu, input_variance_gpu,
            out_gpu, params, backend);
        status.event.wait_and_throw();
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
    }

    for (auto _ : state) {
      this->start_timing();
      try {
        auto status = sycldnn::batchnorm::launch<DataType, Backend, Direction>(
            input_gpu, beta_gpu, gamma_gpu, input_mean_gpu, input_variance_gpu,
            out_gpu, params, backend);

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

    benchmark.set_items_processed(state, params);
    benchmark.add_param_counters(state, params);
    benchmark.template add_bandwidth_counters<float>(state, params);

    this->finish_benchmark(state);
  }
};

}  // namespace bench
}  // namespace sycldnn

#endif  // PORTDNN_BENCH_BATCHNORM_SNN_BATCHNORM_EXECUTOR_H_
