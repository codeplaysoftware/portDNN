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

// This sample makes use of the Eigen backend, and so we need to include the
// relevant Eigen header.
#include <unsupported/Eigen/CXX11/Tensor>

#include "portdnn/backend/eigen_backend.h"

#include "portdnn/conv2d/conv_type.h"
#include "portdnn/conv2d/launch.h"
#include "portdnn/conv2d/params.h"
#include "portdnn/conv2d/selector/direct_selector.h"
#include "portdnn/conv2d/selector/im2col_selector.h"
#include "portdnn/conv2d/selector/winograd_selector.h"
#include "portdnn/conv2d/sizes.h"
#include "portdnn/conv2d/workspace_size.h"
#include "portdnn/status.h"

#include <iterator>
#include <memory>
#include <numeric>
#include <vector>

#include <CL/sycl.hpp>

int main() {
  // A SYCL device selector is a C++ class responsible for selecting what OpenCL
  // device to bind a dispatch queue to, and consequently execute OpenCL kernels
  // on.
  //
  // Users can provide their own custom selectors for more fine-grained control,
  // but here we simply use the default selector.
  auto device_selector = cl::sycl::default_selector{};

  // Construct different algorithm selector algorithms.
  auto direct_algo_selector = sycldnn::conv2d::DirectSelector{};
  auto im2col_algo_selector = sycldnn::conv2d::Im2colSelector{};
  auto winograd_algo_selector = sycldnn::conv2d::WinogradSelector{};

  // This sample relies upon portDNN's Eigen backend. Here, we construct the
  // necessary Eigen objects, a dispatch queue and associated device.
  auto queue = std::unique_ptr<Eigen::QueueInterface>(
      new Eigen::QueueInterface{device_selector});
  auto device = Eigen::SyclDevice{queue.get()};

  // Construct a portDNN Eigen backend instance, which provides memory
  // allocation routines and an accelerated matrix multiply.
  //
  // To accomplish this, we first construct an Eigen::SyclDevice, which is an
  // Eigen-specific abstraction over a compute accelerator, such as a GPU. We
  // then use this to instantiate a portDNN backend, based on the device.
  auto backend = sycldnn::backend::EigenBackend{device};

  // In order to execute a convolution we construct a parameters object, which
  // describes the shape of the convolution operands, along with strides and
  // padding. For this particular example the first set of parameters are
  // configured to generate 12 feature maps per-input image by applying 12
  // 5 x 5 filters to a batch of 32 256 x 256 3-channel images.
  sycldnn::conv2d::Conv2DParams conv1_params{};
  conv1_params.channels = 3;
  conv1_params.features = 12;
  conv1_params.batch = 32;
  conv1_params.in_rows = 256;
  conv1_params.in_cols = 256;
  conv1_params.window_rows = 5;
  conv1_params.window_cols = 5;
  conv1_params.stride_rows = 1;
  conv1_params.stride_cols = 1;
  conv1_params.out_rows = 252;
  conv1_params.out_cols = 252;
  conv1_params.pad_rows = 0;
  conv1_params.pad_cols = 0;
  conv1_params.dilation_rows = 1;
  conv1_params.dilation_cols = 1;

  // This second set of parameters are configured to consume the output of the
  // previous layer, and apply a second set of 5 x 5 convolution layers.
  sycldnn::conv2d::Conv2DParams conv2_params{};
  conv2_params.channels = conv1_params.features;
  conv2_params.features = 32;
  conv2_params.batch = conv1_params.batch;
  conv2_params.in_rows = conv1_params.out_rows;
  conv2_params.in_cols = conv1_params.out_cols;
  conv2_params.window_rows = 5;
  conv2_params.window_cols = 5;
  conv2_params.stride_rows = 1;
  conv2_params.stride_cols = 1;
  conv2_params.out_rows = 252;
  conv2_params.out_cols = 252;
  conv2_params.pad_rows = 2;
  conv2_params.pad_cols = 2;
  conv2_params.dilation_rows = 1;
  conv2_params.dilation_cols = 1;

  // Similarly we make a third set of parameters to consume the output of the
  // previous layer, and apply a 3 x 3 convolution layer.
  sycldnn::conv2d::Conv2DParams conv3_params{};
  conv3_params.channels = conv2_params.features;
  conv3_params.features = 32;
  conv3_params.batch = conv2_params.batch;
  conv3_params.in_rows = conv2_params.out_rows;
  conv3_params.in_cols = conv2_params.out_cols;
  conv3_params.window_rows = 3;
  conv3_params.window_cols = 3;
  conv3_params.stride_rows = 1;
  conv3_params.stride_cols = 1;
  conv3_params.out_rows = 252;
  conv3_params.out_cols = 252;
  conv3_params.pad_rows = 2;
  conv3_params.pad_cols = 2;
  conv3_params.dilation_rows = 1;
  conv3_params.dilation_cols = 1;

  // We can derive the sizes of the tensors from the convolution params. A real
  // application/framework likely already has this information.
  auto conv1_sizes =
      sycldnn::conv2d::get_sizes<sycldnn::conv2d::conv_type::Forward>(
          conv1_params);
  auto conv2_sizes =
      sycldnn::conv2d::get_sizes<sycldnn::conv2d::conv_type::Forward>(
          conv2_params);
  auto conv3_sizes =
      sycldnn::conv2d::get_sizes<sycldnn::conv2d::conv_type::Forward>(
          conv3_params);

  // We can get the workspace memory sizes from query_workspace_size.
  auto conv1_workspace_size = sycldnn::conv2d::query_workspace_size<
      sycldnn::conv2d::conv_type::Forward>(conv1_params, direct_algo_selector);
  auto conv2_workspace_size = sycldnn::conv2d::query_workspace_size<
      sycldnn::conv2d::conv_type::Forward>(conv2_params, im2col_algo_selector);
  auto conv3_workspace_size = sycldnn::conv2d::query_workspace_size<
      sycldnn::conv2d::conv_type::Forward>(conv3_params,
                                           winograd_algo_selector);

  // A 2D convolution requires an input tensor representing a batch of images,
  // a filter tensor containing a filter kernel for each feature, and an output
  // tensor to hold the generated feature maps.
  //
  // Here we calculate the storage requirements for these tensors, and then
  // allocate storage for them via Eigen's GPU device memory allocator.
  using value_type = float;
  auto input_nbytes = conv1_sizes.input_size * sizeof(value_type);
  auto intermediate1_nbytes = conv1_sizes.output_size * sizeof(value_type);
  auto intermediate2_nbytes = conv2_sizes.output_size * sizeof(value_type);
  auto output_nbytes = conv3_sizes.output_size * sizeof(value_type);

  auto filter1_nbytes = conv1_sizes.filter_size * sizeof(value_type);
  auto filter2_nbytes = conv2_sizes.filter_size * sizeof(value_type);
  auto filter3_nbytes = conv3_sizes.filter_size * sizeof(value_type);

  auto workspace1_nbytes =
      conv1_workspace_size.recommended_size * sizeof(value_type);
  auto workspace2_nbytes =
      conv2_workspace_size.recommended_size * sizeof(value_type);
  auto workspace3_nbytes =
      conv3_workspace_size.recommended_size * sizeof(value_type);

  auto* input_gpu_buffer =
      static_cast<value_type*>(device.allocate(input_nbytes));
  auto* intermediate1_gpu_buffer =
      static_cast<value_type*>(device.allocate(intermediate1_nbytes));
  auto* intermediate2_gpu_buffer =
      static_cast<value_type*>(device.allocate(intermediate2_nbytes));
  auto* output_gpu_buffer =
      static_cast<value_type*>(device.allocate(output_nbytes));

  auto* filter1_gpu_buffer =
      static_cast<value_type*>(device.allocate(filter1_nbytes));
  auto* filter2_gpu_buffer =
      static_cast<value_type*>(device.allocate(filter2_nbytes));
  auto* filter3_gpu_buffer =
      static_cast<value_type*>(device.allocate(filter3_nbytes));

  auto* workspace1_gpu_buffer =
      static_cast<value_type*>(device.allocate(workspace1_nbytes));
  auto* workspace2_gpu_buffer =
      static_cast<value_type*>(device.allocate(workspace2_nbytes));
  auto* workspace3_gpu_buffer =
      static_cast<value_type*>(device.allocate(workspace3_nbytes));

  // The GPU buffers are initially unpopulated. Here we fill the input and
  // filter tensors. The output tensors are left undefined.
  std::vector<value_type> input;
  input.resize(conv1_sizes.input_size);
  std::iota(begin(input), end(input), 0);

  std::vector<value_type> filter1;
  filter1.resize(conv1_sizes.filter_size);
  std::iota(begin(filter1), end(filter1), 0);

  std::vector<value_type> filter2;
  filter2.resize(conv2_sizes.filter_size);
  std::iota(begin(filter2), end(filter2), 0);

  std::vector<value_type> filter3;
  filter3.resize(conv3_sizes.filter_size);
  std::iota(begin(filter3), end(filter3), 0);

  device.memcpyHostToDevice(input_gpu_buffer, input.data(), input_nbytes);
  device.memcpyHostToDevice(filter1_gpu_buffer, filter1.data(), filter1_nbytes);
  device.memcpyHostToDevice(filter2_gpu_buffer, filter2.data(), filter2_nbytes);
  device.memcpyHostToDevice(filter3_gpu_buffer, filter3.data(), filter3_nbytes);

  // Now that all of our buffers are populated, and parameters configured, we
  // can execute the convolution itself. This happens asynchronously, so we
  // follow the launch of the convolution kernel with a blocking wait.
  auto status =
      sycldnn::conv2d::launch<value_type, sycldnn::conv2d::conv_type::Forward>(
          input_gpu_buffer, filter1_gpu_buffer, intermediate1_gpu_buffer,
          conv1_params, direct_algo_selector, backend, workspace1_gpu_buffer,
          conv1_workspace_size.recommended_size);
  if (sycldnn::StatusCode::OK != status.status) {
    // If the launch failed, then clean up our GPU buffers and return failure.
    device.deallocate(input_gpu_buffer);
    device.deallocate(intermediate1_gpu_buffer);
    device.deallocate(intermediate2_gpu_buffer);
    device.deallocate(output_gpu_buffer);
    device.deallocate(filter1_gpu_buffer);
    device.deallocate(filter2_gpu_buffer);
    device.deallocate(filter3_gpu_buffer);
    return -1;
  }

  // We can now launch the second layer. We use a different algorithm selector
  // to force the use of the im2col algorithm rather than the direct convolution
  // algorithm.
  status =
      sycldnn::conv2d::launch<value_type, sycldnn::conv2d::conv_type::Forward>(
          intermediate1_gpu_buffer, filter2_gpu_buffer,
          intermediate2_gpu_buffer, conv2_params, im2col_algo_selector, backend,
          workspace2_gpu_buffer, conv2_workspace_size.recommended_size);
  if (sycldnn::StatusCode::OK != status.status) {
    // If the launch failed, then clean up our GPU buffers and return failure.
    device.deallocate(input_gpu_buffer);
    device.deallocate(intermediate1_gpu_buffer);
    device.deallocate(intermediate2_gpu_buffer);
    device.deallocate(output_gpu_buffer);
    device.deallocate(filter1_gpu_buffer);
    device.deallocate(filter2_gpu_buffer);
    device.deallocate(filter3_gpu_buffer);
    return -1;
  }

  // For the third convolution, we use a Winograd selector to ensure that the
  // convolution is computed using the Winograd algorithm. This is currently
  // implemented for filters of size 3. If this selector were used on one of the
  // previous 5 x 5 convolutions then the launch function would return a status
  // which contains an InvalidAlgorithm StatusCode.
  status =
      sycldnn::conv2d::launch<value_type, sycldnn::conv2d::conv_type::Forward>(
          intermediate2_gpu_buffer, filter3_gpu_buffer, output_gpu_buffer,
          conv3_params, winograd_algo_selector, backend, workspace3_gpu_buffer,
          conv3_workspace_size.recommended_size);
  if (sycldnn::StatusCode::OK != status.status) {
    // If the launch failed, then clean up our GPU buffers and return failure.
    device.deallocate(input_gpu_buffer);
    device.deallocate(intermediate1_gpu_buffer);
    device.deallocate(intermediate2_gpu_buffer);
    device.deallocate(output_gpu_buffer);
    device.deallocate(filter1_gpu_buffer);
    device.deallocate(filter2_gpu_buffer);
    device.deallocate(filter3_gpu_buffer);
    return -1;
  }

  // The convolutions are now executing. While they run, we can allocate a
  // host-accessible vector, then wait for completion and trigger a copy via
  // Eigen to return the results to system memory.
  std::vector<value_type> output;
  output.resize(conv2_sizes.output_size);

  // Wait for completion, then copy results to system memory.
  status.event.wait_and_throw();
  device.memcpyDeviceToHost(output.data(), output_gpu_buffer, output_nbytes);

  // The convolution results are now available in host-accessible system memory.

  // We can now deallocate the Eigen GPU buffers.
  device.deallocate(input_gpu_buffer);
  device.deallocate(intermediate1_gpu_buffer);
  device.deallocate(intermediate2_gpu_buffer);
  device.deallocate(output_gpu_buffer);
  device.deallocate(filter1_gpu_buffer);
  device.deallocate(filter2_gpu_buffer);
  device.deallocate(filter3_gpu_buffer);

  return 0;
}
