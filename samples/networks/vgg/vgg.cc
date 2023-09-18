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

#if defined(SNN_TEST_SYCLBLAS)
#include "portdnn/backend/sycl_blas_backend.h"
#else
#include "portdnn/backend/snn_backend.h"
#endif

#include "tools/network.h"

#include <fstream>
#include <iostream>

using DType = float;

#if defined(SNN_TEST_SYCLBLAS)
using Backend = sycldnn::backend::SyclBLASBackend;
#else
using Backend = sycldnn::backend::SNNBackend;
#endif
using DeviceMem = Backend::pointer_type<DType>;

// Helper function that reads binary data produced by h5tobin.py into a vector
std::vector<char> read_binary_data(std::string const& name) {
  std::ifstream file(name, std::ios_base::binary | std::ios_base::in);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file " + name);
  }
  std::vector<char> output{std::istreambuf_iterator<char>{file}, {}};
  return output;
}

// read image data from disk
DeviceMem read_image_data(std::string const& name, Backend& backend) {
  cl::sycl::range<1> r{224 * 224 * 3};  // vgg input size
  cl::sycl::buffer<DType> b{r};
  auto data = read_binary_data(name);
  assert(data.size() == 224 * 224 * 3 * sizeof(DType));
  {
    auto char_data = b.reinterpret<char>(r * sizeof(DType));
    auto event = backend.get_queue().submit([&](cl::sycl::handler& cgh) {
      auto acc =
          char_data.get_access<cl::sycl::access::mode::discard_write>(cgh);
      cgh.copy(data.data(), acc);
    });
    event.wait_and_throw();
  }
  return DeviceMem{b, 0};
}

// make conv layer parameters
inline sycldnn::conv2d::Conv2DParams make_conv_params(
    int batch, int input, int channels, int features, int window, int stride,
    sycldnn::PaddingMode pad) {
  sycldnn::conv2d::Conv2DParams params = {
      channels, features, batch, input, input, window, window,
      stride,   stride,   0,     0,     0,     0};
  params = sycldnn::helpers::add_padding_to(params, pad);
  return params;
}

// make conv layer
template <typename T>
inline sycldnn::ConvolutionLayer<T, Backend>* create_conv_layer(
    DeviceMem const input, Backend& backend, std::string const& data_dir,
    sycldnn::conv2d::Selector& selector,
    sycldnn::conv2d::Conv2DParams const& params) {
  DeviceMem weights;
  DeviceMem output;
  DeviceMem workspace;
  auto new_size = sycldnn::conv2d::query_workspace_size<
                      sycldnn::conv2d::conv_type::Forward>(params, selector)
                      .recommended_size;
  if (new_size > 0) {
    workspace = backend.template allocate<T>(new_size);
  }
  auto sizes =
      sycldnn::conv2d::get_sizes<sycldnn::conv2d::conv_type::Forward>(params);
  weights = backend.template allocate<T>(sizes.filter_size);
  output = backend.template allocate<T>(sizes.output_size);

  std::vector<char> filter(sizes.filter_size * sizeof(T));
  if (data_dir == "")
    std::fill(filter.begin(), filter.end(), 'a');
  else
    filter = read_binary_data(data_dir);
  assert(filter.size() == sizes.filter_size * sizeof(T));
  auto data_size = cl::sycl::range<1>{sizes.filter_size};
  auto queue = backend.get_queue();
  auto buf = weights.get_buffer();
  auto char_buf = buf.template reinterpret<char>(data_size * sizeof(T));
  auto copy_event = queue.submit([&](cl::sycl::handler& cgh) {
    auto acc =
        char_buf.template get_access<cl::sycl::access::mode::discard_write>(
            cgh);
    cgh.copy(filter.data(), acc);
  });
  copy_event.wait_and_throw();
  return new sycldnn::ConvolutionLayer<T, Backend>(
      params, input, weights, output, workspace, new_size, backend, selector);
}

// make bias layer parameters
inline sycldnn::binaryop::BinaryParams make_bias_params(int batch, int spatial,
                                                        int channels) {
  sycldnn::binaryop::BinaryParams params;
  params.lhs_dims = {batch, spatial, spatial, channels};
  params.rhs_dims = {channels};
  return params;
}

// make bias-add layer
template <typename T>
inline sycldnn::BiasAddLayer<T, Backend>* create_bias_layer(
    DeviceMem const input, Backend& backend, std::string const& data_dir,
    sycldnn::binaryop::BinaryParams const& params) {
  DeviceMem bias, output;
  auto lhs_size = sycldnn::helpers::get_total_size(params.lhs_dims);
  auto rhs_size = sycldnn::helpers::get_total_size(params.rhs_dims);
  bias = backend.allocate<T>(rhs_size);
  output = backend.allocate<T>(lhs_size);

  std::vector<char> biases(rhs_size * sizeof(T));
  if (data_dir == "")
    std::fill(biases.begin(), biases.end(), 'a');
  else
    biases = read_binary_data(data_dir);
  assert(biases.size() == rhs_size * sizeof(T));
  auto data_size = cl::sycl::range<1>(rhs_size);
  auto buf = bias.get_buffer();
  auto char_buf = buf.reinterpret<char>(data_size * sizeof(T));
  auto queue = backend.get_queue();
  auto copy_event = queue.submit([&](cl::sycl::handler& h) {
    auto acc = char_buf.get_access<cl::sycl::access::mode::discard_write>(h);
    h.copy(biases.data(), acc);
  });
  copy_event.wait_and_throw();
  return new sycldnn::BiasAddLayer<T, Backend>(params, input, bias, output,
                                               backend);
}

// make activation layer parameters
inline sycldnn::pointwise::PointwiseParams make_pointwise_params(int size) {
  sycldnn::pointwise::PointwiseParams params = {size};
  return params;
}

// make activation layer
template <typename T, template <typename> class ActivationFunc>
inline sycldnn::ActivationLayer<T, Backend, ActivationFunc>*
create_activation_layer(DeviceMem const input, Backend& backend,
                        sycldnn::pointwise::PointwiseParams const& params) {
  DeviceMem output;
  output = backend.template allocate<T>(params.size);
  return new sycldnn::ActivationLayer<T, Backend, ActivationFunc>(
      params, input, output, backend);
}

// make pooling layer parameters
inline sycldnn::pooling::PoolingParams make_pooling_params(
    int batch, int input, int channels, int window, int stride,
    sycldnn::PaddingMode pad) {
  sycldnn::pooling::PoolingParams params = {input,  input,    0,      0,
                                            window, window,   stride, stride,
                                            batch,  channels, 0,      0};
  params = sycldnn::helpers::add_padding_to(params, pad);
  return params;
}

// make pooling layer
template <typename T, template <typename> class PoolingType>
inline sycldnn::PoolingLayer<T, Backend, PoolingType>* create_pooling_layer(
    DeviceMem const input, Backend& backend,
    sycldnn::pooling::PoolingParams const& params) {
  DeviceMem output;
  auto sizes = sycldnn::pooling::get_sizes<sycldnn::pooling::Forward>(params);
  output = backend.allocate<T>(sizes.output_size);
  return new sycldnn::PoolingLayer<T, Backend, PoolingType>(params, input,
                                                            output, backend);
}

// make fully connected layer parameters
template <typename T>
inline sycldnn::matmul::MatmulParams make_fc_params(int input, int output) {
  sycldnn::matmul::MatmulParams params = {1, 1, input, output, T{0}};
  return params;
}

// make fully-connected layer
template <typename T>
inline sycldnn::FCLayer<T, Backend>* create_fc_layer(
    DeviceMem const input, Backend& backend, std::string const& data_dir,
    sycldnn::matmul::MatmulParams const& params) {
  DeviceMem filter, output;
  auto filter_size = params.k * params.n;
  filter = backend.allocate<T>(filter_size);
  output = backend.allocate<T>(params.n);

  std::vector<char> weights(filter_size * sizeof(T));
  if (data_dir == "")
    std::fill(weights.begin(), weights.end(), 'a');
  else
    weights = read_binary_data(data_dir);
  assert(weights.size() == filter_size * sizeof(T));
  auto data_size = cl::sycl::range<1>{static_cast<size_t>(filter_size)};
  auto buf = filter.get_buffer();
  auto char_buf = buf.reinterpret<char>(data_size * sizeof(T));
  auto queue = backend.get_queue();
  auto copy_event = queue.submit([&](cl::sycl::handler& h) {
    auto acc = char_buf.get_access<cl::sycl::access::mode::discard_write>(h);
    h.copy(weights.data(), acc);
  });
  // keeps weights alive and accessible
  copy_event.wait_and_throw();
  return new sycldnn::FCLayer<T, Backend>(params, input, filter, output,
                                          backend);
}

// make softmax layer parameters
inline sycldnn::softmax::SoftmaxParams make_softmax_params(int batch, int rows,
                                                           int cols,
                                                           int channels) {
  sycldnn::softmax::SoftmaxParams params = {channels, batch, rows, cols};
  return params;
}

// make softmax layer
template <typename T>
inline sycldnn::SoftmaxLayer<T, Backend>* create_softmax_layer(
    DeviceMem const input, Backend& backend,
    sycldnn::softmax::SoftmaxParams const& params) {
  DeviceMem workspace, output;
  workspace = backend.allocate<T>(params.batch * params.rows * params.cols);
  output = backend.allocate<T>(params.batch * params.rows * params.cols *
                               params.channels);
  return new sycldnn::SoftmaxLayer<T, Backend>(params, input, workspace, output,
                                               backend);
}

std::string get_path_to_layer_weights(std::string const& data_dir,
                                      int const& layer_number) {
  return data_dir + "layer_" + std::to_string(layer_number) + "-weights.bin";
}

std::string get_path_to_layer_biases(std::string const& data_dir,
                                     int const& layer_number) {
  return data_dir + "layer_" + std::to_string(layer_number) + "-biases.bin";
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout << "USAGE: vgg <directory> <image>\n";
    return 1;
  }

  cl::sycl::queue q([](cl::sycl::exception_list l) {
    for (auto e : l) {
      try {
        std::rethrow_exception(e);
      } catch (cl::sycl::exception& e) {
        std::cout << e.what() << " " << e.get_cl_code() << "\n";
      }
    }
  });
  Backend backend(q);
  auto selector = sycldnn::conv2d::get_default_selector(q.get_device());
  std::vector<DType> output;
  std::string data_dir{argv[1]};
  auto input = read_image_data(argv[2], backend);
  sycldnn::Network<DType, Backend> network(backend, output);

  network.add_layer(create_conv_layer<DType>(
      input, backend, get_path_to_layer_weights(data_dir, 1), *selector,
      make_conv_params(1, 224, 3, 64, 3, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, get_path_to_layer_biases(data_dir, 1),
      make_bias_params(1, 224, 64)));

  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(224 * 224 * 64)));

  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend, get_path_to_layer_weights(data_dir, 2),
      *selector,
      make_conv_params(1, 224, 64, 64, 3, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, get_path_to_layer_biases(data_dir, 2),
      make_bias_params(1, 224, 64)));

  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(224 * 224 * 64)));

  network.add_layer(create_pooling_layer<DType, sycldnn::pooling::Max>(
      network.get_output(), backend,
      make_pooling_params(1, 224, 64, 2, 2, sycldnn::PaddingMode::VALID)));

  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend, get_path_to_layer_weights(data_dir, 3),
      *selector,
      make_conv_params(1, 112, 64, 128, 3, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, get_path_to_layer_biases(data_dir, 3),
      make_bias_params(1, 112, 128)));

  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(112 * 112 * 128)));

  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend, get_path_to_layer_weights(data_dir, 4),
      *selector,
      make_conv_params(1, 112, 128, 128, 3, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, get_path_to_layer_biases(data_dir, 4),
      make_bias_params(1, 112, 128)));

  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(112 * 112 * 128)));

  network.add_layer(create_pooling_layer<DType, sycldnn::pooling::Max>(
      network.get_output(), backend,
      make_pooling_params(1, 112, 128, 2, 2, sycldnn::PaddingMode::VALID)));

  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend, get_path_to_layer_weights(data_dir, 5),
      *selector,
      make_conv_params(1, 56, 128, 256, 3, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, get_path_to_layer_biases(data_dir, 5),
      make_bias_params(1, 56, 256)));

  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(56 * 56 * 256)));

  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend, get_path_to_layer_weights(data_dir, 6),
      *selector,
      make_conv_params(1, 56, 256, 256, 3, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, get_path_to_layer_biases(data_dir, 6),
      make_bias_params(1, 56, 256)));

  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(56 * 56 * 256)));

  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend, get_path_to_layer_weights(data_dir, 7),
      *selector,
      make_conv_params(1, 56, 256, 256, 3, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, get_path_to_layer_biases(data_dir, 7),
      make_bias_params(1, 56, 256)));

  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(56 * 56 * 256)));

  network.add_layer(create_pooling_layer<DType, sycldnn::pooling::Max>(
      network.get_output(), backend,
      make_pooling_params(1, 56, 256, 2, 2, sycldnn::PaddingMode::VALID)));

  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend, get_path_to_layer_weights(data_dir, 8),
      *selector,
      make_conv_params(1, 28, 256, 512, 3, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, get_path_to_layer_biases(data_dir, 8),
      make_bias_params(1, 28, 512)));

  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(28 * 28 * 512)));

  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend, get_path_to_layer_weights(data_dir, 9),
      *selector,
      make_conv_params(1, 28, 512, 512, 3, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, get_path_to_layer_biases(data_dir, 9),
      make_bias_params(1, 28, 512)));

  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(28 * 28 * 512)));

  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend, get_path_to_layer_weights(data_dir, 10),
      *selector,
      make_conv_params(1, 28, 512, 512, 3, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, get_path_to_layer_biases(data_dir, 10),
      make_bias_params(1, 28, 512)));

  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(28 * 28 * 512)));

  network.add_layer(create_pooling_layer<DType, sycldnn::pooling::Max>(
      network.get_output(), backend,
      make_pooling_params(1, 28, 512, 2, 2, sycldnn::PaddingMode::VALID)));

  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend, get_path_to_layer_weights(data_dir, 11),
      *selector,
      make_conv_params(1, 14, 512, 512, 3, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, get_path_to_layer_biases(data_dir, 11),
      make_bias_params(1, 14, 512)));

  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(14 * 14 * 512)));

  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend, get_path_to_layer_weights(data_dir, 12),
      *selector,
      make_conv_params(1, 14, 512, 512, 3, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, get_path_to_layer_biases(data_dir, 12),
      make_bias_params(1, 14, 512)));

  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(14 * 14 * 512)));

  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend, get_path_to_layer_weights(data_dir, 13),
      *selector,
      make_conv_params(1, 14, 512, 512, 3, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, get_path_to_layer_biases(data_dir, 13),
      make_bias_params(1, 14, 512)));

  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(14 * 14 * 512)));

  network.add_layer(create_pooling_layer<DType, sycldnn::pooling::Max>(
      network.get_output(), backend,
      make_pooling_params(1, 14, 512, 2, 2, sycldnn::PaddingMode::VALID)));

  network.add_layer(create_fc_layer<DType>(
      network.get_output(), backend, get_path_to_layer_weights(data_dir, 14),
      make_fc_params<DType>(7 * 7 * 512, 4096)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, get_path_to_layer_biases(data_dir, 14),
      make_bias_params(1, 1, 4096)));

  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(4096)));

  network.add_layer(create_fc_layer<DType>(
      network.get_output(), backend, get_path_to_layer_weights(data_dir, 15),
      make_fc_params<DType>(4096, 4096)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, get_path_to_layer_biases(data_dir, 15),
      make_bias_params(1, 1, 4096)));

  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(4096)));

  network.add_layer(create_fc_layer<DType>(
      network.get_output(), backend, get_path_to_layer_weights(data_dir, 16),
      make_fc_params<DType>(4096, 1000)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, get_path_to_layer_biases(data_dir, 16),
      make_bias_params(1, 1, 1000)));

  network.add_layer(create_softmax_layer<DType>(
      network.get_output(), backend, make_softmax_params(1, 1, 1, 1000)));

  auto test_status = network.test();
  test_status.event.wait_and_throw();
  auto index = std::max_element(output.begin(), output.end());
  std::cout << "classed as " << std::distance(output.begin(), index)
            << ", value " << (index != std::end(output) ? *index : 0.f)
            << std::endl;

  int loops = 8;
  do {
    auto st = std::chrono::high_resolution_clock::now();
    auto status = network.run();
    status.event.wait_and_throw();
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << (end - st).count() << " ns\n";
  } while (--loops);

  q.wait_and_throw();
  return 0;
}
