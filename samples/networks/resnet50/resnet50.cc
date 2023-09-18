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
  cl::sycl::range<1> r{224 * 224 * 3};  // resnet input size
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

// make batchnorm layer parameters
inline sycldnn::batchnorm::BatchNormParams make_batchnorm_params(int batch,
                                                                 int rows,
                                                                 int channels) {
  bool is_training = false;
  float epsilon = 1.001e-5;
  sycldnn::batchnorm::BatchNormParams params{batch,    rows,        rows,
                                             channels, is_training, epsilon};
  return params;
}

// create batchnorm layer
template <typename T>
inline sycldnn::BatchNormFrozenLayer<T, Backend>* create_batchnorm_layer(
    DeviceMem const input, Backend& backend, std::string const& beta_file,
    std::string const& gamma_file, std::string const& mean_file,
    std::string const& variance_file,
    sycldnn::batchnorm::BatchNormParams const& params) {
  DeviceMem beta, gamma, mean, variance, output;
  beta = backend.template allocate<T>(params.channels);
  gamma = backend.template allocate<T>(params.channels);
  mean = backend.template allocate<T>(params.channels);
  variance = backend.template allocate<T>(params.channels);
  output = backend.template allocate<T>(params.batch * params.rows *
                                        params.cols * params.channels);

  std::vector<char> beta_vec(params.channels * sizeof(T));
  std::vector<char> gamma_vec(params.channels * sizeof(T));
  std::vector<char> mean_vec(params.channels * sizeof(T));
  std::vector<char> variance_vec(params.channels * sizeof(T));

  if (beta_file == "")
    std::fill(beta_vec.begin(), beta_vec.end(), 'a');
  else
    beta_vec = read_binary_data(beta_file);

  if (gamma_file == "")
    std::fill(gamma_vec.begin(), gamma_vec.end(), 'a');
  else
    gamma_vec = read_binary_data(gamma_file);

  if (mean_file == "")
    std::fill(mean_vec.begin(), mean_vec.end(), 'a');
  else
    mean_vec = read_binary_data(mean_file);

  if (variance_file == "")
    std::fill(variance_vec.begin(), variance_vec.end(), 'a');
  else
    variance_vec = read_binary_data(variance_file);

  assert(beta_vec.size() == static_cast<size_t>(params.channels) * sizeof(T));
  assert(gamma_vec.size() == static_cast<size_t>(params.channels) * sizeof(T));
  assert(mean_vec.size() == static_cast<size_t>(params.channels) * sizeof(T));
  assert(variance_vec.size() ==
         static_cast<size_t>(params.channels) * sizeof(T));

  auto data_size = cl::sycl::range<1>(params.channels);

  auto beta_buf = beta.get_buffer();
  auto gamma_buf = gamma.get_buffer();
  auto mean_buf = mean.get_buffer();
  auto variance_buf = variance.get_buffer();

  auto beta_char_buf = beta_buf.reinterpret<char>(data_size * sizeof(T));
  auto gamma_char_buf = gamma_buf.reinterpret<char>(data_size * sizeof(T));
  auto mean_char_buf = mean_buf.reinterpret<char>(data_size * sizeof(T));
  auto variance_char_buf =
      variance_buf.reinterpret<char>(data_size * sizeof(T));

  auto queue = backend.get_queue();
  auto beta_event = queue.submit([&](cl::sycl::handler& h) {
    auto beta_acc =
        beta_char_buf.get_access<cl::sycl::access::mode::discard_write>(h);

    h.copy(beta_vec.data(), beta_acc);
  });

  auto gamma_event = queue.submit([&](cl::sycl::handler& h) {
    auto gamma_acc =
        gamma_char_buf.get_access<cl::sycl::access::mode::discard_write>(h);

    h.copy(gamma_vec.data(), gamma_acc);
  });

  auto mean_event = queue.submit([&](cl::sycl::handler& h) {
    auto mean_acc =
        mean_char_buf.get_access<cl::sycl::access::mode::discard_write>(h);

    h.copy(mean_vec.data(), mean_acc);
  });

  auto variance_event = queue.submit([&](cl::sycl::handler& h) {
    auto variance_acc =
        variance_char_buf.get_access<cl::sycl::access::mode::discard_write>(h);

    h.copy(variance_vec.data(), variance_acc);
  });

  beta_event.wait_and_throw();
  gamma_event.wait_and_throw();
  mean_event.wait_and_throw();
  variance_event.wait_and_throw();

  return new sycldnn::BatchNormFrozenLayer<T, Backend>(
      params, input, beta, gamma, mean, variance, output, backend);
}

// create ResidualAdd layer
template <typename T>
inline sycldnn::BiasAddLayer<T, Backend>* create_residual_layer(
    DeviceMem const input, DeviceMem output, Backend& backend,
    sycldnn::binaryop::BinaryParams const& params) {
  return new sycldnn::BiasAddLayer<T, Backend>(params, input, output, output,
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

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout << "USAGE: resnet <directory> <image>\n";
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
      input, backend, data_dir + "conv1_conv_kernel.bin", *selector,
      make_conv_params(1, 224, 3, 64, 7, 2, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(network.get_output(), backend,
                                             data_dir + "conv1_conv_bias.bin",
                                             make_bias_params(1, 112, 64)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv1_bn_beta.bin",
      data_dir + "conv1_bn_gamma.bin", data_dir + "conv1_bn_moving_mean.bin",
      data_dir + "conv1_bn_moving_variance.bin",
      make_batchnorm_params(1, 112, 64)));

  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(112 * 112 * 64)));

  network.add_layer(create_pooling_layer<DType, sycldnn::pooling::Max>(
      network.get_output(), backend,
      make_pooling_params(1, 112, 64, 3, 2, sycldnn::PaddingMode::SAME)));

  int layer_before_residual_connection = network.get_network_size() - 1;
  // Residual Block start
  // Residual Conv start
  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend,
      data_dir + "conv2_block1_0_conv_kernel.bin", *selector,
      make_conv_params(1, 56, 64, 256, 1, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv2_block1_0_conv_bias.bin",
      make_bias_params(1, 56, 256)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv2_block1_0_bn_beta.bin",
      data_dir + "conv2_block1_0_bn_gamma.bin",
      data_dir + "conv2_block1_0_bn_moving_mean.bin",
      data_dir + "conv2_block1_0_bn_moving_variance.bin",
      make_batchnorm_params(1, 56, 256)));
  // Residual Conv end
  int residual_connection_reference = network.get_network_size() - 1;

  network.add_layer(create_conv_layer<DType>(
      network.get_output(layer_before_residual_connection), backend,
      data_dir + "conv2_block1_1_conv_kernel.bin", *selector,
      make_conv_params(1, 56, 64, 64, 1, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv2_block1_1_conv_bias.bin",
      make_bias_params(1, 56, 64)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv2_block1_1_bn_beta.bin",
      data_dir + "conv2_block1_1_bn_gamma.bin",
      data_dir + "conv2_block1_1_bn_moving_mean.bin",
      data_dir + "conv2_block1_1_bn_moving_variance.bin",
      make_batchnorm_params(1, 56, 64)));

  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(56 * 56 * 64)));

  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend,
      data_dir + "conv2_block1_2_conv_kernel.bin", *selector,
      make_conv_params(1, 56, 64, 64, 3, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv2_block1_2_conv_bias.bin",
      make_bias_params(1, 56, 64)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv2_block1_2_bn_beta.bin",
      data_dir + "conv2_block1_2_bn_gamma.bin",
      data_dir + "conv2_block1_2_bn_moving_mean.bin",
      data_dir + "conv2_block1_2_bn_moving_variance.bin",
      make_batchnorm_params(1, 56, 64)));

  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(56 * 56 * 64)));

  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend,
      data_dir + "conv2_block1_3_conv_kernel.bin", *selector,
      make_conv_params(1, 56, 64, 256, 1, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv2_block1_3_conv_bias.bin",
      make_bias_params(1, 56, 256)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv2_block1_3_bn_beta.bin",
      data_dir + "conv2_block1_3_bn_gamma.bin",
      data_dir + "conv2_block1_3_bn_moving_mean.bin",
      data_dir + "conv2_block1_3_bn_moving_variance.bin",
      make_batchnorm_params(1, 56, 256)));

  // perform residual addition
  network.add_layer(create_residual_layer<DType>(
      network.get_output(residual_connection_reference), network.get_output(),
      backend, make_bias_params(1, 1, 56 * 56 * 256)));
  // residual addition complete, move to activation layer
  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(56 * 56 * 256)));
  // Residual Block end
  residual_connection_reference = network.get_network_size() - 1;
  // Residual Block start
  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend,
      data_dir + "conv2_block2_1_conv_kernel.bin", *selector,
      make_conv_params(1, 56, 256, 64, 1, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv2_block2_1_conv_bias.bin",
      make_bias_params(1, 56, 64)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv2_block2_1_bn_beta.bin",
      data_dir + "conv2_block2_1_bn_gamma.bin",
      data_dir + "conv2_block2_1_bn_moving_mean.bin",
      data_dir + "conv2_block2_1_bn_moving_variance.bin",
      make_batchnorm_params(1, 56, 64)));

  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(56 * 56 * 64)));

  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend,
      data_dir + "conv2_block2_2_conv_kernel.bin", *selector,
      make_conv_params(1, 56, 64, 64, 3, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv2_block2_2_conv_bias.bin",
      make_bias_params(1, 56, 64)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv2_block2_2_bn_beta.bin",
      data_dir + "conv2_block2_2_bn_gamma.bin",
      data_dir + "conv2_block2_2_bn_moving_mean.bin",
      data_dir + "conv2_block2_2_bn_moving_variance.bin",
      make_batchnorm_params(1, 56, 64)));

  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(56 * 56 * 64)));

  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend,
      data_dir + "conv2_block2_3_conv_kernel.bin", *selector,
      make_conv_params(1, 56, 64, 256, 1, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv2_block2_3_conv_bias.bin",
      make_bias_params(1, 56, 256)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv2_block2_3_bn_beta.bin",
      data_dir + "conv2_block2_3_bn_gamma.bin",
      data_dir + "conv2_block2_3_bn_moving_mean.bin",
      data_dir + "conv2_block2_3_bn_moving_variance.bin",
      make_batchnorm_params(1, 56, 256)));

  // perform residual addition
  network.add_layer(create_residual_layer<DType>(
      network.get_output(residual_connection_reference), network.get_output(),
      backend, make_bias_params(1, 1, 56 * 56 * 256)));
  // residual addition complete, move to activation layer
  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(56 * 56 * 256)));
  // Residual Block end
  residual_connection_reference = network.get_network_size() - 1;
  // Residual Block start
  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend,
      data_dir + "conv2_block3_1_conv_kernel.bin", *selector,
      make_conv_params(1, 56, 256, 64, 1, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv2_block3_1_conv_bias.bin",
      make_bias_params(1, 56, 64)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv2_block3_1_bn_beta.bin",
      data_dir + "conv2_block3_1_bn_gamma.bin",
      data_dir + "conv2_block3_1_bn_moving_mean.bin",
      data_dir + "conv2_block3_1_bn_moving_variance.bin",
      make_batchnorm_params(1, 56, 64)));

  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(56 * 56 * 64)));

  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend,
      data_dir + "conv2_block3_2_conv_kernel.bin", *selector,
      make_conv_params(1, 56, 64, 64, 3, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv2_block3_2_conv_bias.bin",
      make_bias_params(1, 56, 64)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv2_block3_2_bn_beta.bin",
      data_dir + "conv2_block3_2_bn_gamma.bin",
      data_dir + "conv2_block3_2_bn_moving_mean.bin",
      data_dir + "conv2_block3_2_bn_moving_variance.bin",
      make_batchnorm_params(1, 56, 64)));

  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(56 * 56 * 64)));

  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend,
      data_dir + "conv2_block3_3_conv_kernel.bin", *selector,
      make_conv_params(1, 56, 64, 256, 1, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv2_block3_3_conv_bias.bin",
      make_bias_params(1, 56, 256)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv2_block3_3_bn_beta.bin",
      data_dir + "conv2_block3_3_bn_gamma.bin",
      data_dir + "conv2_block3_3_bn_moving_mean.bin",
      data_dir + "conv2_block3_3_bn_moving_variance.bin",
      make_batchnorm_params(1, 56, 256)));

  // perform residual addition
  network.add_layer(create_residual_layer<DType>(
      network.get_output(residual_connection_reference), network.get_output(),
      backend, make_bias_params(1, 1, 56 * 56 * 256)));
  // residual addition complete, move to activation layer
  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(56 * 56 * 256)));
  // Residual Block end
  layer_before_residual_connection = network.get_network_size() - 1;
  // Residual Block start
  // Residual Conv start
  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend,
      data_dir + "conv3_block1_0_conv_kernel.bin", *selector,
      make_conv_params(1, 56, 256, 512, 1, 2, sycldnn::PaddingMode::VALID)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv3_block1_0_conv_bias.bin",
      make_bias_params(1, 28, 512)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv3_block1_0_bn_beta.bin",
      data_dir + "conv3_block1_0_bn_gamma.bin",
      data_dir + "conv3_block1_0_bn_moving_mean.bin",
      data_dir + "conv3_block1_0_bn_moving_variance.bin",
      make_batchnorm_params(1, 28, 512)));
  // Residual Conv end
  residual_connection_reference = network.get_network_size() - 1;
  network.add_layer(create_conv_layer<DType>(
      network.get_output(layer_before_residual_connection), backend,
      data_dir + "conv3_block1_1_conv_kernel.bin", *selector,
      make_conv_params(1, 56, 256, 128, 1, 2, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv3_block1_1_conv_bias.bin",
      make_bias_params(1, 28, 128)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv3_block1_1_bn_beta.bin",
      data_dir + "conv3_block1_1_bn_gamma.bin",
      data_dir + "conv3_block1_1_bn_moving_mean.bin",
      data_dir + "conv3_block1_1_bn_moving_variance.bin",
      make_batchnorm_params(1, 28, 128)));

  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(28 * 28 * 128)));

  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend,
      data_dir + "conv3_block1_2_conv_kernel.bin", *selector,
      make_conv_params(1, 28, 128, 128, 3, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv3_block1_2_conv_bias.bin",
      make_bias_params(1, 28, 128)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv3_block1_2_bn_beta.bin",
      data_dir + "conv3_block1_2_bn_gamma.bin",
      data_dir + "conv3_block1_2_bn_moving_mean.bin",
      data_dir + "conv3_block1_2_bn_moving_variance.bin",
      make_batchnorm_params(1, 28, 128)));

  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(28 * 28 * 128)));

  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend,
      data_dir + "conv3_block1_3_conv_kernel.bin", *selector,
      make_conv_params(1, 28, 128, 512, 1, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv3_block1_3_conv_bias.bin",
      make_bias_params(1, 28, 512)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv3_block1_3_bn_beta.bin",
      data_dir + "conv3_block1_3_bn_gamma.bin",
      data_dir + "conv3_block1_3_bn_moving_mean.bin",
      data_dir + "conv3_block1_3_bn_moving_variance.bin",
      make_batchnorm_params(1, 28, 512)));

  // perform residual addition
  network.add_layer(create_residual_layer<DType>(
      network.get_output(residual_connection_reference), network.get_output(),
      backend, make_bias_params(1, 1, 28 * 28 * 512)));
  // residual addition complete, move to activation layer
  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(28 * 28 * 512)));
  // Residual Block end
  residual_connection_reference = network.get_network_size() - 1;
  // Residual Block start
  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend,
      data_dir + "conv3_block2_1_conv_kernel.bin", *selector,
      make_conv_params(1, 28, 512, 128, 1, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv3_block2_1_conv_bias.bin",
      make_bias_params(1, 28, 128)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv3_block2_1_bn_beta.bin",
      data_dir + "conv3_block2_1_bn_gamma.bin",
      data_dir + "conv3_block2_1_bn_moving_mean.bin",
      data_dir + "conv3_block2_1_bn_moving_variance.bin",
      make_batchnorm_params(1, 28, 128)));

  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(28 * 28 * 128)));

  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend,
      data_dir + "conv3_block2_2_conv_kernel.bin", *selector,
      make_conv_params(1, 28, 128, 128, 3, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv3_block2_2_conv_bias.bin",
      make_bias_params(1, 28, 128)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv3_block2_2_bn_beta.bin",
      data_dir + "conv3_block2_2_bn_gamma.bin",
      data_dir + "conv3_block2_2_bn_moving_mean.bin",
      data_dir + "conv3_block2_2_bn_moving_variance.bin",
      make_batchnorm_params(1, 28, 128)));

  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(28 * 28 * 128)));

  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend,
      data_dir + "conv3_block2_3_conv_kernel.bin", *selector,
      make_conv_params(1, 28, 128, 512, 1, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv3_block2_3_conv_bias.bin",
      make_bias_params(1, 28, 512)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv3_block2_3_bn_beta.bin",
      data_dir + "conv3_block2_3_bn_gamma.bin",
      data_dir + "conv3_block2_3_bn_moving_mean.bin",
      data_dir + "conv3_block2_3_bn_moving_variance.bin",
      make_batchnorm_params(1, 28, 512)));

  // perform residual addition
  network.add_layer(create_residual_layer<DType>(
      network.get_output(residual_connection_reference), network.get_output(),
      backend, make_bias_params(1, 1, 28 * 28 * 512)));
  // residual addition complete, move to activation layer
  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(28 * 28 * 512)));
  // Residual Block end
  residual_connection_reference = network.get_network_size() - 1;
  // Residual Block start
  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend,
      data_dir + "conv3_block3_1_conv_kernel.bin", *selector,
      make_conv_params(1, 28, 512, 128, 1, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv3_block3_1_conv_bias.bin",
      make_bias_params(1, 28, 128)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv3_block3_1_bn_beta.bin",
      data_dir + "conv3_block3_1_bn_gamma.bin",
      data_dir + "conv3_block3_1_bn_moving_mean.bin",
      data_dir + "conv3_block3_1_bn_moving_variance.bin",
      make_batchnorm_params(1, 28, 128)));

  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(28 * 28 * 128)));

  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend,
      data_dir + "conv3_block3_2_conv_kernel.bin", *selector,
      make_conv_params(1, 28, 128, 128, 3, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv3_block3_2_conv_bias.bin",
      make_bias_params(1, 28, 128)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv3_block3_2_bn_beta.bin",
      data_dir + "conv3_block3_2_bn_gamma.bin",
      data_dir + "conv3_block3_2_bn_moving_mean.bin",
      data_dir + "conv3_block3_2_bn_moving_variance.bin",
      make_batchnorm_params(1, 28, 128)));

  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(28 * 28 * 128)));

  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend,
      data_dir + "conv3_block3_3_conv_kernel.bin", *selector,
      make_conv_params(1, 28, 128, 512, 1, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv3_block3_3_conv_bias.bin",
      make_bias_params(1, 28, 512)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv3_block3_3_bn_beta.bin",
      data_dir + "conv3_block3_3_bn_gamma.bin",
      data_dir + "conv3_block3_3_bn_moving_mean.bin",
      data_dir + "conv3_block3_3_bn_moving_variance.bin",
      make_batchnorm_params(1, 28, 512)));

  // perform residual addition
  network.add_layer(create_residual_layer<DType>(
      network.get_output(residual_connection_reference), network.get_output(),
      backend, make_bias_params(1, 1, 28 * 28 * 512)));
  // residual addition complete, move to activation layer
  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(28 * 28 * 512)));
  // Residual Block end
  residual_connection_reference = network.get_network_size() - 1;
  // Residual Block start
  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend,
      data_dir + "conv3_block4_1_conv_kernel.bin", *selector,
      make_conv_params(1, 28, 512, 128, 1, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv3_block4_1_conv_bias.bin",
      make_bias_params(1, 28, 128)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv3_block4_1_bn_beta.bin",
      data_dir + "conv3_block4_1_bn_gamma.bin",
      data_dir + "conv3_block4_1_bn_moving_mean.bin",
      data_dir + "conv3_block4_1_bn_moving_variance.bin",
      make_batchnorm_params(1, 28, 128)));

  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(28 * 28 * 128)));

  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend,
      data_dir + "conv3_block4_2_conv_kernel.bin", *selector,
      make_conv_params(1, 28, 128, 128, 3, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv3_block4_2_conv_bias.bin",
      make_bias_params(1, 28, 128)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv3_block4_2_bn_beta.bin",
      data_dir + "conv3_block4_2_bn_gamma.bin",
      data_dir + "conv3_block4_2_bn_moving_mean.bin",
      data_dir + "conv3_block4_2_bn_moving_variance.bin",
      make_batchnorm_params(1, 28, 128)));

  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(28 * 28 * 128)));

  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend,
      data_dir + "conv3_block4_3_conv_kernel.bin", *selector,
      make_conv_params(1, 28, 128, 512, 1, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv3_block4_3_conv_bias.bin",
      make_bias_params(1, 28, 512)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv3_block4_3_bn_beta.bin",
      data_dir + "conv3_block4_3_bn_gamma.bin",
      data_dir + "conv3_block4_3_bn_moving_mean.bin",
      data_dir + "conv3_block4_3_bn_moving_variance.bin",
      make_batchnorm_params(1, 28, 512)));

  // perform residual addition
  network.add_layer(create_residual_layer<DType>(
      network.get_output(residual_connection_reference), network.get_output(),
      backend, make_bias_params(1, 1, 28 * 28 * 512)));
  // residual addition complete, move to activation layer
  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(28 * 28 * 512)));
  // Residual Block end
  layer_before_residual_connection = network.get_network_size() - 1;
  // Residual Block start
  // Residual Conv start
  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend,
      data_dir + "conv4_block1_0_conv_kernel.bin", *selector,
      make_conv_params(1, 28, 512, 1024, 1, 2, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv4_block1_0_conv_bias.bin",
      make_bias_params(1, 14, 1024)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv4_block1_0_bn_beta.bin",
      data_dir + "conv4_block1_0_bn_gamma.bin",
      data_dir + "conv4_block1_0_bn_moving_mean.bin",
      data_dir + "conv4_block1_0_bn_moving_variance.bin",
      make_batchnorm_params(1, 14, 1024)));
  // Residual Conv end
  residual_connection_reference = network.get_network_size() - 1;

  network.add_layer(create_conv_layer<DType>(
      network.get_output(layer_before_residual_connection), backend,
      data_dir + "conv4_block1_1_conv_kernel.bin", *selector,
      make_conv_params(1, 28, 512, 256, 1, 2, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv4_block1_1_conv_bias.bin",
      make_bias_params(1, 14, 256)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv4_block1_1_bn_beta.bin",
      data_dir + "conv4_block1_1_bn_gamma.bin",
      data_dir + "conv4_block1_1_bn_moving_mean.bin",
      data_dir + "conv4_block1_1_bn_moving_variance.bin",
      make_batchnorm_params(1, 14, 256)));

  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(14 * 14 * 256)));

  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend,
      data_dir + "conv4_block1_2_conv_kernel.bin", *selector,
      make_conv_params(1, 14, 256, 256, 3, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv4_block1_2_conv_bias.bin",
      make_bias_params(1, 14, 256)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv4_block1_2_bn_beta.bin",
      data_dir + "conv4_block1_2_bn_gamma.bin",
      data_dir + "conv4_block1_2_bn_moving_mean.bin",
      data_dir + "conv4_block1_2_bn_moving_variance.bin",
      make_batchnorm_params(1, 14, 256)));

  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(14 * 14 * 256)));

  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend,
      data_dir + "conv4_block1_3_conv_kernel.bin", *selector,
      make_conv_params(1, 14, 256, 1024, 1, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv4_block1_3_conv_bias.bin",
      make_bias_params(1, 14, 1024)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv4_block1_3_bn_beta.bin",
      data_dir + "conv4_block1_3_bn_gamma.bin",
      data_dir + "conv4_block1_3_bn_moving_mean.bin",
      data_dir + "conv4_block1_3_bn_moving_variance.bin",
      make_batchnorm_params(1, 14, 1024)));

  // perform residual addition
  network.add_layer(create_residual_layer<DType>(
      network.get_output(residual_connection_reference), network.get_output(),
      backend, make_bias_params(1, 1, 14 * 14 * 1024)));
  // residual addition complete, move to activation layer
  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(14 * 14 * 1024)));
  // Residual Block end
  residual_connection_reference = network.get_network_size() - 1;
  // Residual Block start
  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend,
      data_dir + "conv4_block2_1_conv_kernel.bin", *selector,
      make_conv_params(1, 14, 1024, 256, 1, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv4_block2_1_conv_bias.bin",
      make_bias_params(1, 14, 256)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv4_block2_1_bn_beta.bin",
      data_dir + "conv4_block2_1_bn_gamma.bin",
      data_dir + "conv4_block2_1_bn_moving_mean.bin",
      data_dir + "conv4_block2_1_bn_moving_variance.bin",
      make_batchnorm_params(1, 14, 256)));

  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(14 * 14 * 256)));

  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend,
      data_dir + "conv4_block2_2_conv_kernel.bin", *selector,
      make_conv_params(1, 14, 256, 256, 3, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv4_block2_2_conv_bias.bin",
      make_bias_params(1, 14, 256)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv4_block2_2_bn_beta.bin",
      data_dir + "conv4_block2_2_bn_gamma.bin",
      data_dir + "conv4_block2_2_bn_moving_mean.bin",
      data_dir + "conv4_block2_2_bn_moving_variance.bin",
      make_batchnorm_params(1, 14, 256)));

  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(14 * 14 * 256)));

  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend,
      data_dir + "conv4_block2_3_conv_kernel.bin", *selector,
      make_conv_params(1, 14, 256, 1024, 1, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv4_block2_3_conv_bias.bin",
      make_bias_params(1, 14, 1024)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv4_block2_3_bn_beta.bin",
      data_dir + "conv4_block2_3_bn_gamma.bin",
      data_dir + "conv4_block2_3_bn_moving_mean.bin",
      data_dir + "conv4_block2_3_bn_moving_variance.bin",
      make_batchnorm_params(1, 14, 1024)));

  // perform residual addition
  network.add_layer(create_residual_layer<DType>(
      network.get_output(residual_connection_reference), network.get_output(),
      backend, make_bias_params(1, 1, 14 * 14 * 1024)));
  // residual addition complete, move to activation layer
  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(14 * 14 * 1024)));
  // Residual Block end
  residual_connection_reference = network.get_network_size() - 1;
  // Residual Block start
  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend,
      data_dir + "conv4_block3_1_conv_kernel.bin", *selector,
      make_conv_params(1, 14, 1024, 256, 1, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv4_block3_1_conv_bias.bin",
      make_bias_params(1, 14, 256)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv4_block3_1_bn_beta.bin",
      data_dir + "conv4_block3_1_bn_gamma.bin",
      data_dir + "conv4_block3_1_bn_moving_mean.bin",
      data_dir + "conv4_block3_1_bn_moving_variance.bin",
      make_batchnorm_params(1, 14, 256)));

  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(14 * 14 * 256)));

  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend,
      data_dir + "conv4_block3_2_conv_kernel.bin", *selector,
      make_conv_params(1, 14, 256, 256, 3, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv4_block3_2_conv_bias.bin",
      make_bias_params(1, 14, 256)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv4_block3_2_bn_beta.bin",
      data_dir + "conv4_block3_2_bn_gamma.bin",
      data_dir + "conv4_block3_2_bn_moving_mean.bin",
      data_dir + "conv4_block3_2_bn_moving_variance.bin",
      make_batchnorm_params(1, 14, 256)));

  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(14 * 14 * 256)));

  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend,
      data_dir + "conv4_block3_3_conv_kernel.bin", *selector,
      make_conv_params(1, 14, 256, 1024, 1, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv4_block3_3_conv_bias.bin",
      make_bias_params(1, 14, 1024)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv4_block3_3_bn_beta.bin",
      data_dir + "conv4_block3_3_bn_gamma.bin",
      data_dir + "conv4_block3_3_bn_moving_mean.bin",
      data_dir + "conv4_block3_3_bn_moving_variance.bin",
      make_batchnorm_params(1, 14, 1024)));

  // perform residual addition
  network.add_layer(create_residual_layer<DType>(
      network.get_output(residual_connection_reference), network.get_output(),
      backend, make_bias_params(1, 1, 14 * 14 * 1024)));
  // residual addition complete, move to activation layer
  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(14 * 14 * 1024)));
  // Residual Block end
  residual_connection_reference = network.get_network_size() - 1;
  // Residual Block start
  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend,
      data_dir + "conv4_block4_1_conv_kernel.bin", *selector,
      make_conv_params(1, 14, 1024, 256, 1, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv4_block4_1_conv_bias.bin",
      make_bias_params(1, 14, 256)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv4_block4_1_bn_beta.bin",
      data_dir + "conv4_block4_1_bn_gamma.bin",
      data_dir + "conv4_block4_1_bn_moving_mean.bin",
      data_dir + "conv4_block4_1_bn_moving_variance.bin",
      make_batchnorm_params(1, 14, 256)));

  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(14 * 14 * 256)));

  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend,
      data_dir + "conv4_block4_2_conv_kernel.bin", *selector,
      make_conv_params(1, 14, 256, 256, 3, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv4_block4_2_conv_bias.bin",
      make_bias_params(1, 14, 256)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv4_block4_2_bn_beta.bin",
      data_dir + "conv4_block4_2_bn_gamma.bin",
      data_dir + "conv4_block4_2_bn_moving_mean.bin",
      data_dir + "conv4_block4_2_bn_moving_variance.bin",
      make_batchnorm_params(1, 14, 256)));

  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(14 * 14 * 256)));

  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend,
      data_dir + "conv4_block4_3_conv_kernel.bin", *selector,
      make_conv_params(1, 14, 256, 1024, 1, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv4_block4_3_conv_bias.bin",
      make_bias_params(1, 14, 1024)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv4_block4_3_bn_beta.bin",
      data_dir + "conv4_block4_3_bn_gamma.bin",
      data_dir + "conv4_block4_3_bn_moving_mean.bin",
      data_dir + "conv4_block4_3_bn_moving_variance.bin",
      make_batchnorm_params(1, 14, 1024)));

  // perform residual addition
  network.add_layer(create_residual_layer<DType>(
      network.get_output(residual_connection_reference), network.get_output(),
      backend, make_bias_params(1, 1, 14 * 14 * 1024)));
  // residual addition complete, move to activation layer
  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(14 * 14 * 1024)));
  // Residual Block end
  residual_connection_reference = network.get_network_size() - 1;
  // Residual Block start
  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend,
      data_dir + "conv4_block5_1_conv_kernel.bin", *selector,
      make_conv_params(1, 14, 1024, 256, 1, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv4_block5_1_conv_bias.bin",
      make_bias_params(1, 14, 256)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv4_block5_1_bn_beta.bin",
      data_dir + "conv4_block5_1_bn_gamma.bin",
      data_dir + "conv4_block5_1_bn_moving_mean.bin",
      data_dir + "conv4_block5_1_bn_moving_variance.bin",
      make_batchnorm_params(1, 14, 256)));

  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(14 * 14 * 256)));

  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend,
      data_dir + "conv4_block5_2_conv_kernel.bin", *selector,
      make_conv_params(1, 14, 256, 256, 3, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv4_block5_2_conv_bias.bin",
      make_bias_params(1, 14, 256)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv4_block5_2_bn_beta.bin",
      data_dir + "conv4_block5_2_bn_gamma.bin",
      data_dir + "conv4_block5_2_bn_moving_mean.bin",
      data_dir + "conv4_block5_2_bn_moving_variance.bin",
      make_batchnorm_params(1, 14, 256)));

  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(14 * 14 * 256)));

  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend,
      data_dir + "conv4_block5_3_conv_kernel.bin", *selector,
      make_conv_params(1, 14, 256, 1024, 1, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv4_block5_3_conv_bias.bin",
      make_bias_params(1, 14, 1024)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv4_block5_3_bn_beta.bin",
      data_dir + "conv4_block5_3_bn_gamma.bin",
      data_dir + "conv4_block5_3_bn_moving_mean.bin",
      data_dir + "conv4_block5_3_bn_moving_variance.bin",
      make_batchnorm_params(1, 14, 1024)));

  // perform residual addition
  network.add_layer(create_residual_layer<DType>(
      network.get_output(residual_connection_reference), network.get_output(),
      backend, make_bias_params(1, 1, 14 * 14 * 1024)));
  // residual addition complete, move to activation layer
  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(14 * 14 * 1024)));
  // Residual Block end
  residual_connection_reference = network.get_network_size() - 1;
  // Residual Block start
  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend,
      data_dir + "conv4_block6_1_conv_kernel.bin", *selector,
      make_conv_params(1, 14, 1024, 256, 1, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv4_block6_1_conv_bias.bin",
      make_bias_params(1, 14, 256)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv4_block6_1_bn_beta.bin",
      data_dir + "conv4_block6_1_bn_gamma.bin",
      data_dir + "conv4_block6_1_bn_moving_mean.bin",
      data_dir + "conv4_block6_1_bn_moving_variance.bin",
      make_batchnorm_params(1, 14, 256)));

  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(14 * 14 * 256)));

  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend,
      data_dir + "conv4_block6_2_conv_kernel.bin", *selector,
      make_conv_params(1, 14, 256, 256, 3, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv4_block6_2_conv_bias.bin",
      make_bias_params(1, 14, 256)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv4_block6_2_bn_beta.bin",
      data_dir + "conv4_block6_2_bn_gamma.bin",
      data_dir + "conv4_block6_2_bn_moving_mean.bin",
      data_dir + "conv4_block6_2_bn_moving_variance.bin",
      make_batchnorm_params(1, 14, 256)));

  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(14 * 14 * 256)));

  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend,
      data_dir + "conv4_block6_3_conv_kernel.bin", *selector,
      make_conv_params(1, 14, 256, 1024, 1, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv4_block6_3_conv_bias.bin",
      make_bias_params(1, 14, 1024)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv4_block6_3_bn_beta.bin",
      data_dir + "conv4_block6_3_bn_gamma.bin",
      data_dir + "conv4_block6_3_bn_moving_mean.bin",
      data_dir + "conv4_block6_3_bn_moving_variance.bin",
      make_batchnorm_params(1, 14, 1024)));

  // perform residual addition
  network.add_layer(create_residual_layer<DType>(
      network.get_output(residual_connection_reference), network.get_output(),
      backend, make_bias_params(1, 1, 14 * 14 * 1024)));
  // residual addition complete, move to activation layer
  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(14 * 14 * 1024)));
  // Residual Block end
  layer_before_residual_connection = network.get_network_size() - 1;
  // Residual Block start
  // Residual Conv start
  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend,
      data_dir + "conv5_block1_0_conv_kernel.bin", *selector,
      make_conv_params(1, 14, 1024, 2048, 1, 2, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv5_block1_0_conv_bias.bin",
      make_bias_params(1, 7, 2048)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv5_block1_0_bn_beta.bin",
      data_dir + "conv5_block1_0_bn_gamma.bin",
      data_dir + "conv5_block1_0_bn_moving_mean.bin",
      data_dir + "conv5_block1_0_bn_moving_variance.bin",
      make_batchnorm_params(1, 7, 2048)));
  // Residual Conv end
  residual_connection_reference = network.get_network_size() - 1;
  network.add_layer(create_conv_layer<DType>(
      network.get_output(layer_before_residual_connection), backend,
      data_dir + "conv5_block1_1_conv_kernel.bin", *selector,
      make_conv_params(1, 14, 1024, 512, 1, 2, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv5_block1_1_conv_bias.bin",
      make_bias_params(1, 7, 512)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv5_block1_1_bn_beta.bin",
      data_dir + "conv5_block1_1_bn_gamma.bin",
      data_dir + "conv5_block1_1_bn_moving_mean.bin",
      data_dir + "conv5_block1_1_bn_moving_variance.bin",
      make_batchnorm_params(1, 7, 512)));

  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(7 * 7 * 512)));

  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend,
      data_dir + "conv5_block1_2_conv_kernel.bin", *selector,
      make_conv_params(1, 7, 512, 512, 3, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv5_block1_2_conv_bias.bin",
      make_bias_params(1, 7, 512)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv5_block1_2_bn_beta.bin",
      data_dir + "conv5_block1_2_bn_gamma.bin",
      data_dir + "conv5_block1_2_bn_moving_mean.bin",
      data_dir + "conv5_block1_2_bn_moving_variance.bin",
      make_batchnorm_params(1, 7, 512)));

  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(7 * 7 * 512)));

  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend,
      data_dir + "conv5_block1_3_conv_kernel.bin", *selector,
      make_conv_params(1, 7, 512, 2048, 1, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv5_block1_3_conv_bias.bin",
      make_bias_params(1, 7, 2048)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv5_block1_3_bn_beta.bin",
      data_dir + "conv5_block1_3_bn_gamma.bin",
      data_dir + "conv5_block1_3_bn_moving_mean.bin",
      data_dir + "conv5_block1_3_bn_moving_variance.bin",
      make_batchnorm_params(1, 7, 2048)));

  // perform residual addition
  network.add_layer(create_residual_layer<DType>(
      network.get_output(residual_connection_reference), network.get_output(),
      backend, make_bias_params(1, 1, 7 * 7 * 2048)));
  // residual addition complete, move to activation layer
  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(7 * 7 * 2048)));
  // Residual Block end
  residual_connection_reference = network.get_network_size() - 1;
  // Residual Block start
  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend,
      data_dir + "conv5_block2_1_conv_kernel.bin", *selector,
      make_conv_params(1, 7, 2048, 512, 1, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv5_block2_1_conv_bias.bin",
      make_bias_params(1, 7, 512)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv5_block2_1_bn_beta.bin",
      data_dir + "conv5_block2_1_bn_gamma.bin",
      data_dir + "conv5_block2_1_bn_moving_mean.bin",
      data_dir + "conv5_block2_1_bn_moving_variance.bin",
      make_batchnorm_params(1, 7, 512)));

  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(7 * 7 * 512)));

  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend,
      data_dir + "conv5_block2_2_conv_kernel.bin", *selector,
      make_conv_params(1, 7, 512, 512, 3, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv5_block2_2_conv_bias.bin",
      make_bias_params(1, 7, 512)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv5_block2_2_bn_beta.bin",
      data_dir + "conv5_block2_2_bn_gamma.bin",
      data_dir + "conv5_block2_2_bn_moving_mean.bin",
      data_dir + "conv5_block2_2_bn_moving_variance.bin",
      make_batchnorm_params(1, 7, 512)));

  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(7 * 7 * 512)));

  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend,
      data_dir + "conv5_block2_3_conv_kernel.bin", *selector,
      make_conv_params(1, 7, 512, 2048, 1, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv5_block2_3_conv_bias.bin",
      make_bias_params(1, 7, 2048)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv5_block2_3_bn_beta.bin",
      data_dir + "conv5_block2_3_bn_gamma.bin",
      data_dir + "conv5_block2_3_bn_moving_mean.bin",
      data_dir + "conv5_block2_3_bn_moving_variance.bin",
      make_batchnorm_params(1, 7, 2048)));

  // perform residual addition
  network.add_layer(create_residual_layer<DType>(
      network.get_output(residual_connection_reference), network.get_output(),
      backend, make_bias_params(1, 1, 7 * 7 * 2048)));
  // residual addition complete, move to activation layer
  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(7 * 7 * 2048)));
  // Residual Block end
  residual_connection_reference = network.get_network_size() - 1;
  // Residual Block start
  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend,
      data_dir + "conv5_block3_1_conv_kernel.bin", *selector,
      make_conv_params(1, 7, 2048, 512, 1, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv5_block3_1_conv_bias.bin",
      make_bias_params(1, 7, 512)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv5_block3_1_bn_beta.bin",
      data_dir + "conv5_block3_1_bn_gamma.bin",
      data_dir + "conv5_block3_1_bn_moving_mean.bin",
      data_dir + "conv5_block3_1_bn_moving_variance.bin",
      make_batchnorm_params(1, 7, 512)));

  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(7 * 7 * 512)));

  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend,
      data_dir + "conv5_block3_2_conv_kernel.bin", *selector,
      make_conv_params(1, 7, 512, 512, 3, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv5_block3_2_conv_bias.bin",
      make_bias_params(1, 7, 512)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv5_block3_2_bn_beta.bin",
      data_dir + "conv5_block3_2_bn_gamma.bin",
      data_dir + "conv5_block3_2_bn_moving_mean.bin",
      data_dir + "conv5_block3_2_bn_moving_variance.bin",
      make_batchnorm_params(1, 7, 512)));

  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(7 * 7 * 512)));

  network.add_layer(create_conv_layer<DType>(
      network.get_output(), backend,
      data_dir + "conv5_block3_3_conv_kernel.bin", *selector,
      make_conv_params(1, 7, 512, 2048, 1, 1, sycldnn::PaddingMode::SAME)));

  network.add_layer(create_bias_layer<DType>(
      network.get_output(), backend, data_dir + "conv5_block3_3_conv_bias.bin",
      make_bias_params(1, 7, 2048)));

  network.add_layer(create_batchnorm_layer<DType>(
      network.get_output(), backend, data_dir + "conv5_block3_3_bn_beta.bin",
      data_dir + "conv5_block3_3_bn_gamma.bin",
      data_dir + "conv5_block3_3_bn_moving_mean.bin",
      data_dir + "conv5_block3_3_bn_moving_variance.bin",
      make_batchnorm_params(1, 7, 2048)));

  // perform residual addition
  network.add_layer(create_residual_layer<DType>(
      network.get_output(residual_connection_reference), network.get_output(),
      backend, make_bias_params(1, 1, 7 * 7 * 2048)));
  // residual addition complete, move to activation layer
  network.add_layer(create_activation_layer<DType, sycldnn::pointwise::Relu>(
      network.get_output(), backend, make_pointwise_params(7 * 7 * 2048)));
  // Residual Block end

  network.add_layer(create_pooling_layer<DType, sycldnn::pooling::Average>(
      network.get_output(), backend,
      make_pooling_params(1, 7, 2048, 7, 1, sycldnn::PaddingMode::VALID)));

  network.add_layer(create_fc_layer<DType>(network.get_output(), backend,
                                           data_dir + "probs_kernel.bin",
                                           make_fc_params<DType>(2048, 1000)));

  network.add_layer(create_bias_layer<DType>(network.get_output(), backend,
                                             data_dir + "probs_bias.bin",
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
