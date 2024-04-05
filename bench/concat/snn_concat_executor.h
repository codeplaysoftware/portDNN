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
#ifndef SYCLDNN_BENCH_CONCAT_SNN_CONCAT_EXECUTOR_H_
#define SYCLDNN_BENCH_CONCAT_SNN_CONCAT_EXECUTOR_H_

#include <benchmark/benchmark.h>

#include <math.h>

#include "sycldnn/helpers/handle_exception.h"
#include "sycldnn/helpers/scope_exit.h"

#include "sycldnn/concat/launch.h"
#include "sycldnn/concat/params.h"

#include "bench/fixture/base_bench_executor.h"

namespace sycldnn {
namespace bench {

/** Executor to perform the concat benchmark using SYCL-DNN. */
template <typename Backend, typename Benchmark>
struct SNNConcatExecutor : public BaseBenchExecutor {
 private:
  using State = ::benchmark::State;

  /** Get a reference to the underlying benchmark fixture. */
  Benchmark& underlying_benchmark() { return static_cast<Benchmark&>(*this); }

 public:
  /** Execute the concat benchmark for the given parameters. */
  void execute(State& state, std::vector<std::vector<int>>& tensor_descs,
               int rank, int axis) {
    auto& benchmark = underlying_benchmark();

    cl::sycl::queue q{cl::sycl::gpu_selector(),
                      cl::sycl::property::queue::in_order()};
    Backend backend(q);

    std::vector<typename Benchmark::template Pointer<const float>>
        input_vec_gpu;
    for (size_t i = 0; i < tensor_descs.size(); i++) {
      int tensor_size =
          std::accumulate(tensor_descs[i].begin(), tensor_descs[i].end(), 1,
                          std::multiplies<int>());
      std::vector<float> tensor(tensor_size, 0.0);
      input_vec_gpu.push_back(
          benchmark.get_initialised_device_memory(tensor_size, tensor));
    }

    sycldnn::concat::ConcatParams params;
    params.rank = rank;
    params.axis = axis;
    params.input_descriptors = tensor_descs;

    sycldnn::concat::internal::check_dimensions(params);

    int output_size = sycldnn::concat::internal::get_concat_output_size(params);
    std::vector<float> out_vec(output_size);
    auto out_gpu =
        benchmark.get_initialised_device_memory(out_vec.size(), out_vec);

    benchmark.template set_bytes_processed<float>(state, output_size);
    benchmark.add_param_counters(state, params);
    benchmark.template add_bandwidth_counters<float>(state, output_size);

    {  // Ensure the kernel is built before benchmarking
      SNNStatus status;
      try {
        status = sycldnn::concat::launch<float>(input_vec_gpu, out_gpu, params,
                                                backend);
      } catch (cl::sycl::exception const& e) {
        helpers::handle_exception(e, [&](std::string& msg) {
          state.SkipWithError((msg + UnexpectedFailure).c_str());
        });
        return;
      }

      if (status.status != sycldnn::StatusCode::OK) {
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

      auto status = sycldnn::concat::launch<float>(input_vec_gpu, out_gpu,
                                                   params, backend);
      status.event.wait_and_throw();

      this->end_timing();

      this->set_iteration_time(state);
    }

    this->finish_benchmark(state);
  }
};

}  // namespace bench
}  // namespace sycldnn

#endif
