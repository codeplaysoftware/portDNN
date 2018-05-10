/*
 * Copyright 2018 Codeplay Software Ltd
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

// This sample makes use of the Eigen backend, and so we need to include the
// relevant Eigen header. Dependent on the build settings, Eigen also requires
// assert() to be declared, and doesn't pull in the required header itself, so
// we do that as well.
#include <cassert>
#include <memory>
#include <unsupported/Eigen/CXX11/Tensor>

#include <sycldnn/backend/eigen_backend.h>
#include <sycldnn/conv2d/launch.h>
#include <sycldnn/conv2d/params.h>
#include <sycldnn/conv2d/selector/direct_selector.h>
#include <sycldnn/conv2d/sizes.h>

int main(int, char**) {
  // A SYCL device selector is a C++ class responsible for selecting what OpenCL
  // device to bind a dispatch queue to, and consequently execute OpenCL kernels
  // on.
  //
  // Users can provide their own custom selectors for more fine-grained control,
  // but here we simply use the default selector.
  auto device_selector = cl::sycl::default_selector{};

  // This sample relies upon SYCL-DNN's Eigen backend. Here, we construct the
  // necessary Eigen objects, a dispatch queue and associated device.
  auto queue = std::unique_ptr<Eigen::QueueInterface>(
      new Eigen::QueueInterface{device_selector});
  auto device = Eigen::SyclDevice{queue.get()};

  // Construct a SYCL-DNN Eigen backend instance, which provides memory
  // allocation routines and an accelerated matrix multiply.
  //
  // To accompilsh this, we first construct an Eigen::SyclDevice, which is an
  // Eigen-specific abstraction over a compute accelerator, such as a GPU. We
  // then use this to instantiate a SYCL-DNN backend, based on the device.
  auto backend = sycldnn::backend::EigenBackend{device};

  // In order to execute a convolution we construct a parameters object, which
  // describes the shape of the convolution operands, along with strides and
  // padding. For this particular example the parameters are configured to
  // generate 12 feature maps per-input image by applying 12 5 x 5 filters to a
  // batch of 128 256 x 256 3-channel images.
  sycldnn::conv2d::Conv2DParams params{};
  params.channels = 3;
  params.features = 12;
  params.batch = 128;
  params.in_rows = 256;
  params.in_cols = 256;
  params.window_rows = 5;
  params.window_cols = 5;
  params.stride_rows = 1;
  params.stride_cols = 1;
  params.out_rows = 252;
  params.out_cols = 252;
  params.pad_rows = 0;
  params.pad_cols = 0;
  params.dilation_rows = 1;
  params.dilation_cols = 1;

  // We can derive the sizes of the tensors from the convolution params. A real
  // application/framework likely already has this information.
  auto conv_sizes =
      sycldnn::conv2d::get_sizes<sycldnn::conv2d::conv_type::Forward>(params);

  // A 2D convolution requires an input tensor representing a batch of images,
  // a filter tensor containing a filter kernel for each feature, and an output
  // tensor to hold the generated feature maps.
  //
  // Here we calculate the storage requirements for these tensors, and then
  // allocate storage for them via Eigen's GPU device memory allocator.
  using value_type = float;
  auto input_nbytes = conv_sizes.input_size * sizeof(value_type);
  auto output_nbytes = conv_sizes.output_size * sizeof(value_type);
  auto filter_nbytes = conv_sizes.filter_size * sizeof(value_type);

  auto* input_gpu_buffer =
      static_cast<value_type*>(device.allocate(input_nbytes));
  auto* output_gpu_buffer =
      static_cast<value_type*>(device.allocate(output_nbytes));
  auto* filter_gpu_buffer =
      static_cast<value_type*>(device.allocate(filter_nbytes));

  // The GPU buffers are initially unpopulated. Here we fill the input and
  // filter tensors. The output tensor is left undefined.
  std::vector<value_type> input;
  input.resize(conv_sizes.input_size);
  std::iota(begin(input), end(input), 0);

  std::vector<value_type> filter;
  filter.resize(conv_sizes.filter_size);
  std::iota(begin(filter), end(filter), 0);

  device.memcpyHostToDevice(input_gpu_buffer, input.data(), input_nbytes);
  device.memcpyHostToDevice(filter_gpu_buffer, filter.data(), filter_nbytes);

  // Now that all of our buffers are populated, and parameters configured, we
  // can execute the convolution itself. This happens asynchronously, so we
  // follow the launch of the convolution kernel with a blocking wait.
  auto algo_selector = sycldnn::conv2d::DirectSelector{};
  auto status =
      sycldnn::conv2d::launch<value_type, sycldnn::conv2d::conv_type::Forward>(
          input_gpu_buffer, filter_gpu_buffer, output_gpu_buffer, params,
          algo_selector, backend);
  if (sycldnn::StatusCode::OK != status.status) {
    // If the launch failed, then clean up our GPU buffers and return failure.
    device.deallocate(input_gpu_buffer);
    device.deallocate(output_gpu_buffer);
    device.deallocate(filter_gpu_buffer);
    return -1;
  }

  // The convolution is now executing. While it runs, we can allocate a
  // host-accessible vector, then wait for completion and trigger a copy via
  // Eigen to return the results to system memory.
  std::vector<value_type> output;
  output.resize(conv_sizes.output_size);

  // Wait for completetion, then copy results to system memory.
  status.event.wait();
  device.memcpyDeviceToHost(output.data(), output_gpu_buffer, output_nbytes);

  // The convolution results are now available in host-accessible system memory.

  // We can now deallocate the Eigen GPU buffers.
  device.deallocate(input_gpu_buffer);
  device.deallocate(output_gpu_buffer);
  device.deallocate(filter_gpu_buffer);

  return 0;
}
