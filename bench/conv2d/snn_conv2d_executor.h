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
#ifndef PORTDNN_BENCH_CONV2D_SNN_CONV2D_EXECUTOR_H_
#define PORTDNN_BENCH_CONV2D_SNN_CONV2D_EXECUTOR_H_

#include "portdnn/conv2d/launch.h"
#include "portdnn/conv2d/params.h"
#include "portdnn/conv2d/selector/selector.h"
#include "portdnn/conv2d/workspace_size.h"

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

/** Executor to perform the Conv2d benchmark using portDNN.  */
template <typename Benchmark, typename ConvType>
struct SNNConv2DExecutor : public BaseExecutor {
 private:
  using State = ::benchmark::State;
  using Conv2DParams = conv2d::Conv2DParams;

  /** Get a reference to the underlying benchmark fixture. */
  Benchmark& underlying_benchmark() { return static_cast<Benchmark&>(*this); }

 public:
  /** Execute a conv2d benchmark with the given parameters and selector. */
  void execute(State& state, Conv2DParams const& params,
               conv2d::Selector& selector) {
    auto& benchmark = underlying_benchmark();
    auto& backend = benchmark.get_backend();

    auto conv_sizes = sycldnn::conv2d::get_sizes<ConvType>(params);

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

    auto workspace_size = compute_workspace_size(
        params, backend.get_queue().get_device(), selector);
    std::vector<float> workspace_vals(workspace_size);

    typename Benchmark::template Pointer<float> workspace{};
    try {
      workspace = benchmark.get_initialised_device_memory(workspace_size,
                                                          workspace_vals);
    } catch (...) {
      state.SkipWithError(AllocationFailure);
      return;
    }
    SNN_ON_SCOPE_EXIT { benchmark.deallocate_ptr(workspace); };

    {  // Ensure the kernel is built before benchmarking
      SNNStatus status;
      try {
        status = sycldnn::conv2d::launch<float, ConvType>(
            inp_gpu, fil_gpu, out_gpu, params, selector, backend, workspace,
            workspace_size);
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
        wait_for_event(status.event, backend.get_queue());
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
        auto status = sycldnn::conv2d::launch<float, ConvType>(
            inp_gpu, fil_gpu, out_gpu, params, selector, backend, workspace,
            workspace_size);

        wait_for_event(status.event, backend.get_queue());
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

    benchmark.template set_items_processed<ConvType>(state, params);
    benchmark.add_param_counters(state, params);
    benchmark.template add_bandwidth_counters<float>(state, conv_sizes);

    this->finish_benchmark(state);
  }

  /**
   * Get the required size for the workspace buffer.
   *
   * Query the required workspace size and the available allocation size on the
   * device, then choose a size based on this. The size is chosen to be as large
   * as it can be while still fitting in memory. If the smallest size is still
   * too large to be allocated then 0 is returned and fall back to using
   * separate temporary buffers.
   */
  size_t compute_workspace_size(Conv2DParams const& params,
                                cl::sycl::device device,
                                conv2d::Selector& selector) {
    auto workspace_size_struct =
        sycldnn::conv2d::query_workspace_size<ConvType>(params, selector);
    // Query the max allocation size on the device. As the input, output and
    // filter tensors also need to be allocated we conservatively divide the
    // alloc size by 4.
    auto max_alloc_size =
        device.get_info<cl::sycl::info::device::max_mem_alloc_size>() / 4;

    if (workspace_size_struct.recommended_size < max_alloc_size) {
      return workspace_size_struct.recommended_size;
    } else {
      if (workspace_size_struct.required_size < max_alloc_size) {
        // If the recommended size is too large but the required size is small
        // enough to be allocated then choose a size in between.
        auto n_multiples = max_alloc_size / workspace_size_struct.required_size;
        return workspace_size_struct.required_size * n_multiples;
      } else {
        // If even the required size is too large then we cannot use a
        // workspace.
        return 0u;
      }
    }
  }
};

}  // namespace bench
}  // namespace sycldnn

#endif  // PORTDNN_BENCH_CONV2D_SNN_CONV2D_EXECUTOR_H_
