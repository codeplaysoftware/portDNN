/*
 * Copyright Codeplay Software Ltd
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

/**
 * \file
 * A sample containing benchmarks of convolutions to compare performance when
 * using on chip memory and when just using global memory.
 *
 * Both benchmarks run the same convolution using two different portDNN
 * kernels:
 *   a) Direct is a naive convolution implementation
 *   b) Tiled is a more complex implementation which introduces data re-use
 *      within threads, and so improves performance.
 *
 * The first convolution is very small, such that all of the input, filter and
 * output tensors are likely to fit in on chip memory. This gives a comparison
 * between the two cases:
 *   a) All loads and stores are from global memory.
 *   b) All loads and stores are from on chip memory.
 *
 * The second convolution is more representative of the sizes used in
 * contemporary image recognition networks. As the tensors are much larger, it
 * is likely that only the filter tensor can fit in on chip memory, while the
 * data tensors are left in global memory. This gives a comparison between:
 *   a) All loads and stores are from global memory
 *   b) All data loads and stores are from global memory, but filter loads are
 *      from on chip memory.
 */

#include <stddef.h>
#include <chrono>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <ratio>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

// This sample makes use of the Eigen backend, and so we need to include the
// relevant Eigen header.
#include <unsupported/Eigen/CXX11/Tensor>

#include "portdnn/backend/eigen_backend.h"
#include "portdnn/conv2d/conv_type.h"
#include "portdnn/conv2d/launch.h"
#include "portdnn/conv2d/params.h"
#include "portdnn/conv2d/selector/direct_selector.h"
#include "portdnn/conv2d/selector/selector.h"
#include "portdnn/conv2d/selector/tiled_selector.h"
#include "portdnn/conv2d/sizes.h"
#include "portdnn/conv2d/workspace_size.h"
#include "portdnn/status.h"

// Include the codeplay specific onchip_memory buffer property
#include <SYCL/codeplay.hpp>

#include <CL/sycl.hpp>

/**
 * Run and time the convolution specified by the `params` on the given input
 * tensors.
 *
 * The timing of the kernel will be printed out to `cout`.
 *
 * \param input    Pointer to the convolution data input.
 * \param filter   Pointer to the convolution filters.
 * \param output   Pointer to the output data.
 * \param params   Convolution parameters including tensor sizes and filter
 *                 sizes.
 * \param backend  Eigen backend which provides the SYCL buffers and queue.
 * \param selector A selector to choose the type of convolution algorithm to
 *                 run.
 */
void time_convolution(float const* const input, float const* const filter,
                      float* output,
                      sycldnn::conv2d::Conv2DParams const& params,
                      sycldnn::backend::EigenBackend& backend,
                      sycldnn::conv2d::Selector& selector) {
  static constexpr int warmup_iterations = 64;
  static constexpr int num_iterations = 128;

  auto workspace_size = sycldnn::conv2d::query_workspace_size<
      sycldnn::conv2d::conv_type::Forward>(params, selector);
  auto workspace = backend.allocate<float>(workspace_size.recommended_size);

  // Run once to make sure the kernel runs without error
  auto status =
      sycldnn::conv2d::launch<float, sycldnn::conv2d::conv_type::Forward>(
          input, filter, output, params, selector, backend, workspace,
          workspace_size.recommended_size);
  if (sycldnn::StatusCode::OK != status.status) {
    throw std::runtime_error(
        std::string{"Error launching initial kernel for "} + selector.name());
  }
  status.event.wait_and_throw();

  for (int i = 0; i < warmup_iterations; ++i) {
    status =
        sycldnn::conv2d::launch<float, sycldnn::conv2d::conv_type::Forward>(
            input, filter, output, params, selector, backend, workspace,
            workspace_size.recommended_size);
    if (sycldnn::StatusCode::OK != status.status) {
      throw std::runtime_error(
          std::string{"Error launching warmup kernel for "} + selector.name());
    }
  }
  status.event.wait_and_throw();

  auto start_time = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_iterations; ++i) {
    status =
        sycldnn::conv2d::launch<float, sycldnn::conv2d::conv_type::Forward>(
            input, filter, output, params, selector, backend, workspace,
            workspace_size.recommended_size);
    if (sycldnn::StatusCode::OK != status.status) {
      throw std::runtime_error(std::string{"Error launching kernel for "} +
                               selector.name());
    }
  }
  status.event.wait_and_throw();
  auto end_time = std::chrono::high_resolution_clock::now();
  auto time_taken = (end_time - start_time) / num_iterations;
  auto time_taken_ms =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
          time_taken);
  std::cout << "Convolution took: " << std::setw(11) << std::fixed
            << time_taken_ms.count() << "ms for " << selector.name() << "\n";
}

/**
 * Convert the provided buffers to Eigen pointers, then time the specified
 * convolution.
 *
 * \param in_buffer  SYCL Buffer containing the convolution data input.
 * \param fil_buffer SYCL Buffer containing the convolution filters.
 * \param out_buffer SYCL Buffer containing the output data.
 * \param params     Convolution parameters including tensor sizes and filter
 *                   sizes.
 * \param backend    Eigen backend to add the SYCL buffers to and use to run the
 *                   convolution.
 */
void time_conv2d_for_buffers(cl::sycl::buffer<uint8_t> in_buffer,
                             cl::sycl::buffer<uint8_t> fil_buffer,
                             cl::sycl::buffer<uint8_t> out_buffer,
                             sycldnn::conv2d::Conv2DParams const& params,
                             sycldnn::backend::EigenBackend& backend) {
  auto device = backend.get_eigen_device();
  auto* input_gpu_buffer = static_cast<float*>(device.attach_buffer(in_buffer));
  auto* filter_gpu_buffer =
      static_cast<float*>(device.attach_buffer(fil_buffer));
  auto* output_gpu_buffer =
      static_cast<float*>(device.attach_buffer(out_buffer));

  auto direct_algo_selector = sycldnn::conv2d::DirectSelector{};
  time_convolution(input_gpu_buffer, filter_gpu_buffer, output_gpu_buffer,
                   params, backend, direct_algo_selector);

  auto tiled_algo_selector = sycldnn::conv2d::TiledSelector{};
  time_convolution(input_gpu_buffer, filter_gpu_buffer, output_gpu_buffer,
                   params, backend, tiled_algo_selector);

  device.detach_buffer(output_gpu_buffer);
  device.detach_buffer(filter_gpu_buffer);
  device.detach_buffer(input_gpu_buffer);
}

/**
 * Construct a SYCL buffer from the given host data pointer.
 *
 * \tparam OnChip Whether to use on chip memory for the buffer.
 *
 * \param data    Pointer to host data to use in the buffer.
 * \param n_bytes The number of bytes of data to use in the buffer.
 *
 * \return A SYCL buffer of size `n_bytes` containing the host data from `data`.
 */
template <bool OnChip = false>
cl::sycl::buffer<uint8_t> get_buffer(void* data, size_t n_bytes) {
  if (OnChip) {
    return cl::sycl::buffer<uint8_t>{
        static_cast<uint8_t*>(data),
        cl::sycl::range<1>{n_bytes},
        {cl::sycl::codeplay::property::buffer::use_onchip_memory(
            cl::sycl::codeplay::property::prefer)}};
  } else {
    return cl::sycl::buffer<uint8_t>{static_cast<uint8_t*>(data),
                                     cl::sycl::range<1>{n_bytes}};
  }
}

/**
 * Run the convolution specified by the given `params` both with and without the
 * use of on chip memory.
 *
 * Set up the initial data tensors, and provide the SYCL buffers to be used wen
 * running the convolution. Dependin g on the value of `AllOnChip` either all
 * buffers will use on chip memory, or just the filter buffer will.
 *
 * Running the same convolution with and without on chip memory gives a
 * comparison to show the possible performance improvements.
 *
 * \tparam AllOnChip Whether all the convolution parameters should be placed in
 *                   on chip memory. If false only the filter tensor will be
 *                   places in on chip memory.
 *
 * \param params  Convolution parameters including tensor sizes and filter
 *                sizes.
 * \param backend Eigen backend to add the SYCL buffers to and use to run the
 *                convolution.
 */
template <bool AllOnChip = true>
void time_conv2d_with_onchip_and_without(
    sycldnn::conv2d::Conv2DParams const& params,
    sycldnn::backend::EigenBackend& backend) {
  auto conv_sizes =
      sycldnn::conv2d::get_sizes<sycldnn::conv2d::conv_type::Forward>(params);

  size_t input_nbytes = conv_sizes.input_size * sizeof(float);
  size_t output_nbytes = conv_sizes.output_size * sizeof(float);
  size_t filter_nbytes = conv_sizes.filter_size * sizeof(float);

  std::vector<float> input(conv_sizes.input_size);
  std::iota(std::begin(input), std::end(input), 0);

  std::vector<float> filter(conv_sizes.filter_size);
  std::iota(std::begin(filter), std::end(filter), 0);

  std::vector<float> output(conv_sizes.output_size);
  std::iota(std::begin(output), std::end(output), 0);

  {
    std::cout << "Without using on chip memory:\n";
    auto in_buffer = get_buffer(input.data(), input_nbytes);
    auto fil_buffer = get_buffer(filter.data(), filter_nbytes);
    auto out_buffer = get_buffer(output.data(), output_nbytes);

    time_conv2d_for_buffers(in_buffer, fil_buffer, out_buffer, params, backend);
  }
  backend.get_queue().wait_and_throw();
  {
    std::cout << "Using on chip memory:\n";
    auto in_buffer = get_buffer<AllOnChip>(input.data(), input_nbytes);
    auto fil_buffer = get_buffer<true>(filter.data(), filter_nbytes);
    auto out_buffer = get_buffer<AllOnChip>(output.data(), output_nbytes);

    time_conv2d_for_buffers(in_buffer, fil_buffer, out_buffer, params, backend);
  }
  backend.get_queue().wait_and_throw();
}

/**
 * Get parameters for a convolution which is small enough for all input, filter
 * and output tensors to be in onchip memory at once.
 */
sycldnn::conv2d::Conv2DParams params_entirely_in_onchip() {
  sycldnn::conv2d::Conv2DParams params{};
  params.channels = 32;
  params.features = 64;
  params.batch = 1;
  params.in_rows = 28;
  params.in_cols = 28;
  params.window_rows = 3;
  params.window_cols = 3;
  params.stride_rows = 1;
  params.stride_cols = 1;
  params.out_rows = 28;
  params.out_cols = 28;
  params.pad_rows = 1;
  params.pad_cols = 1;
  params.dilation_rows = 1;
  params.dilation_cols = 1;
  return params;
}

/**
 * Get parameters for a convolution where the filter tensor is small enough to
 * be in onchip memory.
 *
 * The sizes used here are modelled on the 3rd layer of the VGG model.
 */
sycldnn::conv2d::Conv2DParams params_filter_in_onchip() {
  sycldnn::conv2d::Conv2DParams params{};
  params.channels = 64;
  params.features = 128;
  params.batch = 1;
  params.in_rows = 56;
  params.in_cols = 56;
  params.window_rows = 3;
  params.window_cols = 3;
  params.stride_rows = 1;
  params.stride_cols = 1;
  params.out_rows = 56;
  params.out_cols = 56;
  params.pad_rows = 1;
  params.pad_cols = 1;
  params.dilation_rows = 1;
  params.dilation_cols = 1;
  return params;
}

int main() {
  auto device_selector = cl::sycl::default_selector{};

  auto queue = std::unique_ptr<Eigen::QueueInterface>(
      new Eigen::QueueInterface{device_selector});
  auto device = Eigen::SyclDevice{queue.get()};
  auto backend = sycldnn::backend::EigenBackend{device};

  try {
    std::cout << "Launching a convolution with all input, filter and output "
                 "tensors in onchip memory\n";
    auto params_entirely = params_entirely_in_onchip();
    time_conv2d_with_onchip_and_without(params_entirely, backend);

    std::cout << "Launching a larger convolution with filter tensor in onchip "
                 "memory\n";
    auto params_filter = params_filter_in_onchip();
    time_conv2d_with_onchip_and_without<false>(params_filter, backend);

  } catch (cl::sycl::exception const& e) {
    std::cerr << "SYCL Exception caught:\n" << e.what() << "\n";
    throw;
  } catch (std::exception const& e) {
    std::cerr << "Runtime exception caught:\n" << e.what() << "\n";
    throw;
  }

  return 0;
}
