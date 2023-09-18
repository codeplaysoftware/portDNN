/*
 * Copyright Codeplay Software Ltd
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
#ifndef PORTDNN_BENCH_MATMUL_SNN_MATMUL_EXECUTOR_H_
#define PORTDNN_BENCH_MATMUL_SNN_MATMUL_EXECUTOR_H_

#include "portdnn/matmul/launch.h"

#include "portdnn/helpers/handle_exception.h"
#include "portdnn/helpers/scope_exit.h"

#include "bench/fixture/base_executor.h"

namespace sycldnn {
namespace bench {

/** Helper function that checks if portDNN can wait on events directly, or
 * has to wait on the queue. This is because Eigen cannot return us the events
 * corresponding to the kernel launch directly. */
/* TODO: SD-404 Remove queue::wait_and_throw workaround when Eigen removed */
inline void wait_for_event(cl::sycl::event& ev, cl::sycl::queue q) {
  if (ev.is_host()) {
    q.wait_and_throw();
  } else {
    ev.wait_and_throw();
  }
}

/** Executor to perform a matrix multiply benchmark using portDNN.  */
template <typename Benchmark>
struct SNNMatmulExecutor : public BaseExecutor {
 private:
  using State = ::benchmark::State;

  /** Get a reference to the underlying benchmark fixture. */
  Benchmark& underlying_benchmark() { return static_cast<Benchmark&>(*this); }

 public:
  /** Execute a conv2d benchmark with the given parameters and selector. */
  void execute(State& state, int m, int k, int n, int batch, bool transpose_lhs,
               bool transpose_rhs) {
    auto& benchmark = underlying_benchmark();
    auto& backend = benchmark.get_backend();

    auto lhs_size = batch * m * k;
    auto rhs_size = batch * k * n;
    auto out_size = batch * m * n;

    std::vector<float> lhs_vec(lhs_size);
    std::vector<float> rhs_vec(rhs_size);
    std::vector<float> out_vec(out_size);

    auto lhs_gpu =
        benchmark.get_initialised_device_memory(lhs_vec.size(), lhs_vec);
    auto rhs_gpu =
        benchmark.get_initialised_device_memory(rhs_vec.size(), rhs_vec);
    auto out_gpu =
        benchmark.get_initialised_device_memory(out_vec.size(), out_vec);

    SNN_ON_SCOPE_EXIT {
      benchmark.deallocate_ptr(out_gpu);
      benchmark.deallocate_ptr(rhs_gpu);
      benchmark.deallocate_ptr(lhs_gpu);
    };

    auto do_matmul = [&]() {
      if (!transpose_lhs && !transpose_rhs) {
        return backend.template batch_matmul<false, false, float>(
            lhs_gpu, rhs_gpu, out_gpu, batch, m, k, n);
      } else if (transpose_lhs && !transpose_rhs) {
        return backend.template batch_matmul<true, false, float>(
            lhs_gpu, rhs_gpu, out_gpu, batch, m, k, n);
      } else if (!transpose_lhs && transpose_rhs) {
        return backend.template batch_matmul<false, true, float>(
            lhs_gpu, rhs_gpu, out_gpu, batch, m, k, n);
      } else {  // transpose_lhs && transpose_rhs
        return backend.template batch_matmul<true, true, float>(
            lhs_gpu, rhs_gpu, out_gpu, batch, m, k, n);
      }
    };

    {  // Ensure the kernel is built before benchmarking
      cl::sycl::event ev;
      try {
        ev = do_matmul();
      } catch (cl::sycl::exception const& e) {
        helpers::handle_exception(e, [&](std::string& msg) {
          state.SkipWithError((msg + UnexpectedFailure).c_str());
        });
        return;
      }

      try {
        wait_for_event(ev, backend.get_queue());
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
        auto ev = do_matmul();
        wait_for_event(ev, backend.get_queue());
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

    state.SetItemsProcessed(state.iterations() * 2 * batch * m * k * n);
    state.counters["m"] = m;
    state.counters["k"] = k;
    state.counters["n"] = n;
    state.counters["batch"] = batch;
    state.counters["transpose_lhs"] = transpose_lhs;
    state.counters["transpose_rhs"] = transpose_rhs;

    this->finish_benchmark(state);
  }
};

}  // namespace bench
}  // namespace sycldnn

#endif  // PORTDNN_BENCH_MATMUL_SNN_MATMUL_EXECUTOR_H_
