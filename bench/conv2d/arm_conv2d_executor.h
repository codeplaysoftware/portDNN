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
#ifndef PORTDNN_BENCH_CONV2D_ARM_CONV2D_EXECUTOR_H_
#define PORTDNN_BENCH_CONV2D_ARM_CONV2D_EXECUTOR_H_

#include <benchmark/benchmark.h>

#include "bench/fixture/add_arm_opencl_device_info.h"
#include "portdnn/conv2d/params.h"
#include "portdnn/conv2d/sizes.h"
#include "portdnn/helpers/scope_exit.h"

#include "bench/fixture/base_executor.h"

#include <arm_compute/runtime/CL/CLFunctions.h>
#include <arm_compute/runtime/CL/CLScheduler.h>
#include <arm_compute/runtime/CL/CLTensor.h>

#include <arm_compute/runtime/NEON/NEFunctions.h>
#include <arm_compute/runtime/Tensor.h>

namespace sycldnn {
namespace bench {

namespace arm = arm_compute;

/** ACL Executor wrapper to abstract NEON tensors and convolution layer. */
struct ACLNeonExecutor {
  /** NEON execution is synchronous, so wait is a no-op. */
  void wait() {}

  /** Add NEON description to benchmark label. */
  void add_device_info(sycldnn::bench::StringReporter& reporter) {
    reporter.add_to_label("vendor_name", "ARM");
    reporter.add_to_label("device_name", "NEON");
    reporter.add_to_label("device_version", "N/A");
    reporter.add_to_label("driver_version", "N/A");
  }

  arm::Tensor filter, input, output, bias;
  arm::NEConvolutionLayer conv1;
};

/** ACL Executor wrapper to abstract OpenCL tensors and convolution layer. */
struct ACLOpenCLExecutor {
  ACLOpenCLExecutor() : scheduler{arm::CLScheduler::get()} {
    scheduler.default_init();
  }

  /** Wait for OpenCL execution to finish. */
  void wait() { scheduler.sync(); }

  /** Get the OpenCL device, and add device and driver info to the benchmark. */
  void add_device_info(sycldnn::bench::StringReporter& reporter) {
    auto device = cl::Device::getDefault();
    sycldnn::bench::device_info::add_opencl_device_info(device, reporter);
  }

  arm::CLTensor filter, input, output, bias;
  arm::CLScheduler& scheduler;
  arm::CLConvolutionLayer conv1;
};

/** Executor to perform the Conv2d benchmark using ARM Compute Library.  */
template <typename Benchmark, typename ACLExecutor>
struct ARMConv2DExecutor : public BaseExecutor {
 private:
  using State = ::benchmark::State;
  using Conv2DParams = conv2d::Conv2DParams;

  /** Get a reference to the underlying benchmark fixture. */
  Benchmark& underlying_benchmark() { return static_cast<Benchmark&>(*this); }

 public:
  /** Execute a conv2d benchmark with the given parameters. */
  void execute(State& state, Conv2DParams const& params) {
    // Allocate tensors.
    ACLExecutor ex;
    ex.input.allocator()->init(
        arm::TensorInfo(arm::TensorShape(params.in_rows, params.in_cols,
                                         params.channels, params.batch),
                        arm::Format::F32));
    ex.output.allocator()->init(
        arm::TensorInfo(arm::TensorShape(params.out_rows, params.out_cols,
                                         params.features, params.batch),
                        arm::Format::F32));
    ex.filter.allocator()->init(
        arm::TensorInfo(arm::TensorShape(params.window_rows, params.window_cols,
                                         params.channels, params.features),
                        arm::Format::F32));
    ex.bias.allocator()->init(
        arm::TensorInfo(arm::TensorShape(params.features), arm::Format::F32));

    // Construct a convolution layer.
    const int s_pad_end_rows = (params.out_rows - 1) * params.stride_rows +
                               params.window_rows - params.in_rows -
                               params.pad_rows;
    const unsigned pad_end_rows = std::max(s_pad_end_rows, 0);
    const int s_pad_end_cols = (params.out_cols - 1) * params.stride_cols +
                               params.window_cols - params.in_cols -
                               params.pad_cols;
    const unsigned pad_end_cols = std::max(s_pad_end_cols, 0);

    arm::PadStrideInfo psi(params.stride_cols, params.stride_rows,
                           params.pad_cols, pad_end_cols, params.pad_rows,
                           pad_end_rows, arm::DimensionRoundingType::FLOOR);
    ex.conv1.configure(&ex.input, &ex.filter, &ex.bias, &ex.output, psi);

    // Validate the configuration.
    auto status = ex.conv1.validate(ex.input.info(), ex.filter.info(),
                                    ex.bias.info(), ex.output.info(), psi);
    if (!status) {
      state.SkipWithError(status.error_description().c_str());
      return;
    }

    // Allocate the tensors themselves.
    ex.input.allocator()->allocate();
    ex.output.allocator()->allocate();
    ex.filter.allocator()->allocate();
    ex.bias.allocator()->allocate();
    SNN_ON_SCOPE_EXIT {
      ex.input.allocator()->free();
      ex.output.allocator()->free();
      ex.filter.allocator()->free();
      ex.bias.allocator()->free();
    };

    // Run the layer once to eliminate lazy behaviour.
    ex.conv1.run();
    ex.wait();

    for (auto _ : state) {
      this->start_timing();
      ex.conv1.run();
      ex.wait();
      this->end_timing();

      this->set_iteration_time(state);
    }

    auto& benchmark = underlying_benchmark();
    ex.add_device_info(benchmark);
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

#endif  // PORTDNN_BENCH_CONV2D_ARM_CONV2D_EXECUTOR_H_
