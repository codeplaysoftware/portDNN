#include <gtest/gtest.h>

#include <sycldnn/compat/batchnorm.hpp>
#include <type_traits>
#include "test/gen/iota_initialised_data.h"
#include "test/helpers/float_comparison.h"

using namespace sycldnn;
using namespace sycldnn::compat;

class BatchnormCompatTest : public ::testing::Test {
 protected:
  SNNHandle handle;
  using DataType = float;

  void SetUp() override { SNNCreate(handle); }

  int flatten_index(int batch, int channel, int hw,
                    sycldnn::batchnorm::BatchNormParams const& params) {
    if (params.input_format == sycldnn::DataFormat::NHWC) {
      return (batch * params.rows * params.cols * params.channels) +
             (hw * params.channels) + channel;
    } else {
      return (batch * params.rows * params.cols * params.channels) +
             (channel * params.rows * params.cols) + hw;
    }
  }

  template <typename T>
  std::vector<T> compute_expected_mean(
      std::vector<T> const& input,
      sycldnn::batchnorm::BatchNormParams const& params) {
    std::vector<T> mean(params.channels, 0);
    for (int batch = 0; batch < params.batch; ++batch) {
      for (int hw = 0; hw < params.rows * params.cols; ++hw) {
        for (int channel = 0; channel < params.channels; ++channel) {
          mean[channel] += input[flatten_index(batch, channel, hw, params)];
        }
      }
    }

    for (size_t i = 0; i < mean.size(); ++i) {
      mean[i] /= (params.batch * params.rows * params.cols);
    }

    return mean;
  }

  template <typename T>
  std::vector<T> compute_expected_inv_variance(
      std::vector<T> const& input, std::vector<T> const& mean,
      sycldnn::batchnorm::BatchNormParams const& params) {
    std::vector<T> inv_var(params.channels, 0);
    for (int batch = 0; batch < params.batch; ++batch) {
      for (int hw = 0; hw < params.rows * params.cols; ++hw) {
        for (int channel = 0; channel < params.channels; ++channel) {
          auto const in_val = input[flatten_index(batch, channel, hw, params)];
          auto const mean_val = mean[channel];
          inv_var[channel] += (in_val - mean_val) * (in_val - mean_val);
        }
      }
    }

    for (size_t i = 0; i < inv_var.size(); ++i) {
      inv_var[i] /= (params.batch * params.rows * params.cols);
      inv_var[i] = T{1} / std::sqrt(inv_var[i] + params.epsilon);
    }

    return inv_var;
  }

  auto getPtr(const std::vector<DataType>& data) {
    const int size = data.size();
    DataType* ptr = sycl::malloc_device<DataType>(size, handle.getQueue());
    handle.getQueue().memcpy(ptr, data.data(), size * sizeof(DataType)).wait();
    return ptr;
  };

  void test_backward(sycldnn::DataFormat format,
                     const std::array<int, 4>& in_shape,
                     std::vector<DataType> const& exp_out_grad,
                     std::vector<DataType> const& exp_beta_grad,
                     std::vector<DataType> const& exp_gamma_grad,
                     DataType max_input_val, DataType max_gradient_val,
                     DataType max_gamma_val, DataType max_pop_mean_val,
                     DataType max_pop_var_val, double epsilon,
                     float alphaDataDiff = 1.f, float betaDataDiff = 0.f,
                     float alphaParamDiff = 1.f, float betaParamDiff = 0.f) {
    const int n = in_shape[0];
    const int c = in_shape[3];
    const int h = in_shape[1];
    const int w = in_shape[2];
    auto input_size = n * c * h * w;
    const auto size = exp_out_grad.size();
    float max_value = 2048;
    std::vector<DataType> inputData =
        iota_initialised_data<DataType>(input_size, max_input_val);
    std::vector<DataType> gradientData =
        iota_initialised_data<DataType>(input_size, max_gradient_val);
    std::vector<DataType> gamma =
        iota_initialised_data<DataType>(c, max_gamma_val);
    std::vector<DataType> pop_mean =
        iota_initialised_data<DataType>(c, max_pop_mean_val);
    std::vector<DataType> pop_var =
        iota_initialised_data<DataType>(c, max_pop_var_val);
    std::vector<DataType> beta_grad =
        iota_initialised_data<DataType>(c, max_value);
    std::vector<DataType> gamma_grad =
        iota_initialised_data<DataType>(c, max_value);
    std::vector<DataType> outputData =
        iota_initialised_data<DataType>(size, max_value);

    auto q = handle.getQueue();

    auto in_ptr = getPtr(inputData);
    auto in_grad_ptr = getPtr(gradientData);
    auto gamma_ptr = getPtr(gamma);
    auto pop_mean_ptr = getPtr(pop_mean);
    auto pop_var_ptr = getPtr(pop_var);
    auto beta_grad_ptr = getPtr(beta_grad);
    auto gamma_grad_ptr = getPtr(gamma_grad);
    auto out_ptr = getPtr(outputData);

    TensorDescriptor x_desc;
    x_desc.set4d(format, n, c, h, w);
    TensorDescriptor mean_desc;
    mean_desc.set4d(format, 1, c, 1, 1);

    auto status = batchNormalizationBackward(
        handle, BatchNormMode::BATCHNORM_SPATIAL, &alphaDataDiff, &betaDataDiff,
        &alphaParamDiff, &betaParamDiff, x_desc, in_ptr, x_desc, in_grad_ptr,
        x_desc, out_ptr, mean_desc, gamma_ptr, gamma_grad_ptr, beta_grad_ptr,
        epsilon, pop_mean_ptr, pop_var_ptr);
    status.event.wait();

    auto beta_copy_event =
        q.memcpy(beta_grad.data(), beta_grad_ptr, c * sizeof(DataType));
    auto gamma_copy_event =
        q.memcpy(gamma_grad.data(), gamma_grad_ptr, c * sizeof(DataType));
    auto out_copy_event =
        q.memcpy(outputData.data(), out_ptr, size * sizeof(DataType));

    beta_copy_event.wait();
    for (int i = 0; i < c; i++) {
      SNN_ALMOST_EQUAL_EPS(exp_beta_grad[i], beta_grad[i], 10u, 1e-5);
    }

    gamma_copy_event.wait();
    for (int i = 0; i < c; i++) {
      SNN_ALMOST_EQUAL_EPS(exp_gamma_grad[i], gamma_grad[i], 30u, 1e-2);
    }

    out_copy_event.wait();
    for (size_t i = 0; i < size; i++) {
      SNN_ALMOST_EQUAL_EPS(exp_out_grad[i], outputData[i], 30u, 1e-2);
    }

    sycl::free(in_ptr, q);
    sycl::free(in_grad_ptr, q);
    sycl::free(gamma_ptr, q);
    sycl::free(pop_mean_ptr, q);
    sycl::free(pop_var_ptr, q);
    sycl::free(beta_grad_ptr, q);
    sycl::free(gamma_grad_ptr, q);
    sycl::free(out_ptr, q);
  }
  void test_forward(sycldnn::DataFormat format,
                    const std::array<int, 4>& in_shape,
                    const std::vector<DataType>& exp_out,
                    const std::vector<DataType>& exp_mean,
                    const std::vector<DataType>& exp_var,
                    const DataType max_input_val, const DataType max_beta_val,
                    const DataType max_gamma_val,
                    const DataType max_input_mean_val,
                    const DataType max_input_var_val, const double momentum,
                    const double epsilon, bool is_training, bool use_cache,
                    float alpha = 1.0, float beta_scalar = 0.0) {
    ASSERT_TRUE(!use_cache || is_training);
    const int n = in_shape[0];
    const int c = in_shape[3];
    const int h = in_shape[1];
    const int w = in_shape[2];
    auto input_size = n * c * h * w;

    std::vector<DataType> inputData =
        iota_initialised_data<DataType>(input_size, max_input_val);
    std::vector<DataType> beta =
        iota_initialised_data<DataType>(c, max_beta_val);
    std::vector<DataType> gamma =
        iota_initialised_data<DataType>(c, max_gamma_val);
    std::vector<DataType> input_mean =
        iota_initialised_data<DataType>(c, max_input_mean_val);
    std::vector<DataType> input_var =
        iota_initialised_data<DataType>(c, max_input_var_val);
    std::vector<DataType> mean_var_zeros(c, 0);
    const auto size = exp_out.size();
    float max_value = 2048;
    std::vector<DataType> outputData =
        iota_initialised_data<DataType>(size, max_value);

    const auto mean_var_size = c;
    std::vector<DataType> outputMean(mean_var_size);
    std::vector<DataType> outputVar(mean_var_size);

    std::vector<DataType> trInputData;

    TensorDescriptor x_desc;
    x_desc.set4d(format, n, c, h, w);
    TensorDescriptor mean_desc;
    mean_desc.set4d(format, 1, c, 1, 1);
    auto q = handle.getQueue();

    auto in_ptr = getPtr(inputData);
    auto scale_ptr = getPtr(gamma);
    auto bias_ptr = getPtr(beta);
    auto running_mean_ptr = getPtr(is_training ? mean_var_zeros : input_mean);
    auto running_var_ptr = getPtr(is_training ? mean_var_zeros : input_var);
    auto saved_mean_ptr =
        use_cache ? sycl::malloc_device<DataType>(c, q) : nullptr;
    auto saved_var_ptr =
        use_cache ? sycl::malloc_device<DataType>(c, q) : nullptr;
    auto out_ptr = getPtr(outputData);

    if (is_training) {
      auto status = batchNormalizationForwardTraining(
          handle, BatchNormMode::BATCHNORM_SPATIAL, &alpha, &beta_scalar,
          x_desc, in_ptr, x_desc, out_ptr, mean_desc, scale_ptr, bias_ptr,
          momentum, running_mean_ptr, running_var_ptr, epsilon, saved_mean_ptr,
          saved_var_ptr);

      status.event.wait();
      auto running_mean_copy_event =
          q.memcpy(outputMean.data(), running_mean_ptr,
                   mean_var_size * sizeof(DataType));
      auto running_var_copy_event = q.memcpy(outputVar.data(), running_var_ptr,
                                             mean_var_size * sizeof(DataType));

      running_mean_copy_event.wait();
      for (int i = 0; i < mean_var_size; i++) {
        SCOPED_TRACE("Element: " + std::to_string(i));
        SNN_ALMOST_EQUAL_EPS(exp_mean[i], outputMean[i], 10u, 2e-5);
      }

      running_var_copy_event.wait();
      for (int i = 0; i < mean_var_size; i++) {
        SCOPED_TRACE("Element: " + std::to_string(i));
        SNN_ALMOST_EQUAL_EPS(exp_var[i], outputVar[i], 30u, 1e-5);
      }

      if (use_cache) {
        auto saved_mean_copy_event = q.memcpy(outputMean.data(), saved_mean_ptr,
                                              mean_var_size * sizeof(DataType));
        auto saved_var_copy_event = q.memcpy(outputVar.data(), saved_var_ptr,
                                             mean_var_size * sizeof(DataType));

        saved_mean_copy_event.wait();
        sycldnn::batchnorm::BatchNormParams test_params;
        test_params.batch = in_shape[0];
        test_params.rows = in_shape[1];
        test_params.cols = in_shape[2];
        test_params.channels = in_shape[3];
        test_params.input_format = format;
        const auto expected_saved_mean =
            compute_expected_mean(inputData, test_params);
        for (int i = 0; i < mean_var_size; i++) {
          SCOPED_TRACE("Element: " + std::to_string(i));
          SNN_ALMOST_EQUAL_EPS(expected_saved_mean[i], outputMean[i], 10u,
                               1e-5);
        }

        saved_var_copy_event.wait();
        const auto expected_saved_invvar = compute_expected_inv_variance(
            inputData, expected_saved_mean, test_params);
        for (int i = 0; i < mean_var_size; i++) {
          SCOPED_TRACE("Element: " + std::to_string(i));
          SNN_ALMOST_EQUAL_EPS(expected_saved_invvar[i], outputVar[i], 30u,
                               1e-5);
        }
      }
    } else {
      auto status = batchNormalizationForwardInference(
          handle, BatchNormMode::BATCHNORM_SPATIAL, &alpha, &beta_scalar,
          x_desc, in_ptr, x_desc, out_ptr, mean_desc, scale_ptr, bias_ptr,
          running_mean_ptr, running_var_ptr, epsilon);
      status.event.wait();
    }

    q.memcpy(outputData.data(), out_ptr, size * sizeof(DataType)).wait();
    for (size_t i = 0; i < size; i++) {
      SCOPED_TRACE("Element: " + std::to_string(i));
      SNN_ALMOST_EQUAL_EPS(exp_out[i], outputData[i], 30u, 1e-2);
    }

    sycl::free(in_ptr, q);
    sycl::free(scale_ptr, q);
    sycl::free(bias_ptr, q);
    sycl::free(running_mean_ptr, q);
    sycl::free(running_var_ptr, q);
    sycl::free(saved_mean_ptr, q);
    sycl::free(saved_var_ptr, q);
    sycl::free(out_ptr, q);
  }
};

TEST_F(BatchnormCompatTest, ForwardTr1x1x1x8Cache) {
  const std::vector<DataType> exp_running_mean = {
      0.010000000000000009, 0.020000000000000018, 0.030000000000000027,
      0.040000000000000036, 0.050000000000000044, 0.010000000000000009,
      0.020000000000000018, 0.030000000000000027};
  const std::vector<DataType> exp_running_var = {0., 0., 0., 0.,
                                                 0., 0., 0., 0.};
  const std::vector<DataType> exp_out = {1., 2., 3., 4., 1., 2., 3., 4.};
  const std::array<int, 4> in_shape = {{1, 1, 1, 8}};
  const bool is_training = true;
  const float momentum = 1 - 0.99;
  const float epsilon = 0.001;
  const DataType max_input_val = 5.0;
  const DataType max_beta_val = 4.0;
  const DataType max_gamma_val = 5.0;
  const DataType max_input_mean_val = 6.0;
  const DataType max_input_var_val = 7.0;
  bool use_cache = true;

  this->test_forward(sycldnn::DataFormat::NHWC, in_shape, exp_out,
                     exp_running_mean, exp_running_var, max_input_val,
                     max_beta_val, max_gamma_val, max_input_mean_val,
                     max_input_var_val, momentum, epsilon, is_training,
                     use_cache);
}

TEST_F(BatchnormCompatTest, ForwardTr1x1x1x8NoCache) {
  const std::vector<DataType> exp_running_mean = {
      0.010000000000000009, 0.020000000000000018, 0.030000000000000027,
      0.040000000000000036, 0.050000000000000044, 0.010000000000000009,
      0.020000000000000018, 0.030000000000000027};
  const std::vector<DataType> exp_running_var = {0., 0., 0., 0.,
                                                 0., 0., 0., 0.};
  const std::vector<DataType> exp_out = {1., 2., 3., 4., 1., 2., 3., 4.};
  const std::array<int, 4> in_shape = {{1, 1, 1, 8}};

  const DataType max_input_val = 5.0;
  const DataType max_beta_val = 4.0;
  const DataType max_gamma_val = 5.0;
  const DataType max_input_mean_val = 6.0;
  const DataType max_input_var_val = 7.0;
  const float momentum = 0.01;  // 1 - 0.99;
  const float epsilon = 0.001;
  const bool is_training = true;
  const bool use_cache = false;

  this->test_forward(sycldnn::DataFormat::NHWC, in_shape, exp_out,
                     exp_running_mean, exp_running_var, max_input_val,
                     max_beta_val, max_gamma_val, max_input_mean_val,
                     max_input_var_val, momentum, epsilon, is_training,
                     use_cache);
}

TEST_F(BatchnormCompatTest, ForwardTr1x8x8x1) {
  const std::vector<DataType> exp_running_mean = {0.029687500000000026};
  const std::vector<DataType> exp_running_var = {0.019677734375000016};
  const std::vector<DataType> exp_out = {
      -0.4031131682009884, 0.3095792346947517,  1.0222716375904919,
      1.7349640404862319,  2.447656443381972,   -0.4031131682009884,
      0.3095792346947517,  1.0222716375904919,  1.7349640404862319,
      2.447656443381972,   -0.4031131682009884, 0.3095792346947517,
      1.0222716375904919,  1.7349640404862319,  2.447656443381972,
      -0.4031131682009884, 0.3095792346947517,  1.0222716375904919,
      1.7349640404862319,  2.447656443381972,   -0.4031131682009884,
      0.3095792346947517,  1.0222716375904919,  1.7349640404862319,
      2.447656443381972,   -0.4031131682009884, 0.3095792346947517,
      1.0222716375904919,  1.7349640404862319,  2.447656443381972,
      -0.4031131682009884, 0.3095792346947517,  1.0222716375904919,
      1.7349640404862319,  2.447656443381972,   -0.4031131682009884,
      0.3095792346947517,  1.0222716375904919,  1.7349640404862319,
      2.447656443381972,   -0.4031131682009884, 0.3095792346947517,
      1.0222716375904919,  1.7349640404862319,  2.447656443381972,
      -0.4031131682009884, 0.3095792346947517,  1.0222716375904919,
      1.7349640404862319,  2.447656443381972,   -0.4031131682009884,
      0.3095792346947517,  1.0222716375904919,  1.7349640404862319,
      2.447656443381972,   -0.4031131682009884, 0.3095792346947517,
      1.0222716375904919,  1.7349640404862319,  2.447656443381972,
      -0.4031131682009884, 0.3095792346947517,  1.0222716375904919,
      1.7349640404862319};
  const std::array<int, 4> in_shape = {{1, 8, 8, 1}};
  const float momentum = 1 - 0.99;
  const float epsilon = 0.001;
  const DataType max_input_val = 5.0;
  const DataType max_beta_val = 4.0;
  const DataType max_gamma_val = 5.0;
  const DataType max_input_mean_val = 6.0;
  const DataType max_input_var_val = 7.0;
  this->test_forward(sycldnn::DataFormat::NHWC, in_shape, exp_out,
                     exp_running_mean, exp_running_var, max_input_val,
                     max_beta_val, max_gamma_val, max_input_mean_val,
                     max_input_var_val, momentum, epsilon, true, true);
}

TEST_F(BatchnormCompatTest, ForwardInf1x1x1x5) {
  using DataType = float;
  const std::vector<DataType> exp_running_mean = {};
  const std::vector<DataType> exp_running_var = {};
  const std::vector<DataType> exp_out = {1., 2., 3., 4., 1.};
  const std::array<int, 4> in_shape = {{1, 1, 1, 5}};
  const bool is_training = false;
  const bool use_cache = false;
  const float momentum = 1 - 0.99;
  const float epsilon = 0.001;
  const DataType max_input_val = 5.0;
  const DataType max_beta_val = 4.0;
  const DataType max_gamma_val = 5.0;
  const DataType max_input_mean_val = 6.0;
  const DataType max_input_var_val = 7.0;
  this->test_forward(sycldnn::DataFormat::NHWC, in_shape, exp_out,
                     exp_running_mean, exp_running_var, max_input_val,
                     max_beta_val, max_gamma_val, max_input_mean_val,
                     max_input_var_val, momentum, epsilon, is_training,
                     use_cache);
}

TEST_F(BatchnormCompatTest, ForwardInf1x1x8x8) {
  using DataType = float;
  const std::vector<DataType> exp_running_mean = {};
  const std::vector<DataType> exp_running_var = {};
  const std::vector<DataType> exp_out = {1.,
                                         2.,
                                         3.,
                                         4.,
                                         1.,
                                         -0.04107137012493428,
                                         3.7558749568782206,
                                         6.998501124063319,
                                         3.9985011240633197,
                                         6.241580424529413,
                                         -0.4635244091660504,
                                         0.0004999062695274503,
                                         -3.4716888084749407,
                                         1.1835714519500262,
                                         6.023499827512882,
                                         1.0014988759366803,
                                         1.999500374687773,
                                         3.413860141509805,
                                         4.731762204583026,
                                         5.999750046865236,
                                         -7.9433776169498795,
                                         0.36714290390005255,
                                         4.511749913756441,
                                         9.99700224812664,
                                         4.998001498751092,
                                         0.5861398584901953,
                                         1.268237795416975,
                                         2.000249953134764,
                                         -1.2358444042374703,
                                         1.5917857259750132,
                                         3.,
                                         4.,
                                         2.9990007493755466,
                                         4.827720283019609,
                                         6.463524409166052,
                                         -1.9992501405957088,
                                         -5.70753321271241,
                                         0.7753571779250394,
                                         5.267624870634663,
                                         12.99550337218996,
                                         1.,
                                         2.,
                                         3.,
                                         4.,
                                         1.,
                                         -0.04107137012493428,
                                         3.7558749568782206,
                                         6.998501124063319,
                                         3.9985011240633197,
                                         6.241580424529413,
                                         -0.4635244091660504,
                                         0.0004999062695274503,
                                         -3.4716888084749407,
                                         1.1835714519500262,
                                         6.023499827512882,
                                         1.0014988759366803,
                                         1.999500374687773,
                                         3.413860141509805,
                                         4.731762204583026,
                                         5.999750046865236,
                                         -7.9433776169498795,
                                         0.36714290390005255,
                                         4.511749913756441,
                                         9.99700224812664};
  const std::array<int, 4> in_shape = {{1, 1, 8, 8}};
  const bool is_training = false;
  const bool use_cache = false;
  const float momentum = 1 - 0.99;
  const float epsilon = 0.001;
  const DataType max_input_val = 5.0;
  const DataType max_beta_val = 4.0;
  const DataType max_gamma_val = 5.0;
  const DataType max_input_mean_val = 6.0;
  const DataType max_input_var_val = 7.0;
  this->test_forward(sycldnn::DataFormat::NHWC, in_shape, exp_out,
                     exp_running_mean, exp_running_var, max_input_val,
                     max_beta_val, max_gamma_val, max_input_mean_val,
                     max_input_var_val, momentum, epsilon, is_training,
                     use_cache);
}

TEST_F(BatchnormCompatTest, Backward1x1x8x1) {
  using DataType = float;
  const std::vector<DataType> exp_grad = {1., 2., 3., 4., 1., 2., 3., 4.};
  const std::vector<DataType> beta_grad = {20.};
  const std::vector<DataType> gamma_grad = {35.};
  const std::array<int, 4> in_shape = {{1, 1, 8, 1}};
  const float epsilon = 0.001;
  const DataType max_input_val = 5.0;
  const DataType max_gradient_val = 4.0;
  const DataType max_gamma_val = 5.0;
  const DataType max_pop_mean_val = 6.0;
  const DataType max_pop_var_val = 7.0;
  this->test_backward(sycldnn::DataFormat::NCHW, in_shape, exp_grad, beta_grad,
                      gamma_grad, max_input_val, max_gradient_val,
                      max_gamma_val, max_pop_mean_val, max_pop_var_val,
                      epsilon);
}

TEST_F(BatchnormCompatTest, ForwardInf1x1x1x5_alpha_0_beta_0) {
  using DataType = float;
  const std::vector<DataType> exp_running_mean = {};
  const std::vector<DataType> exp_running_var = {};
  const std::vector<DataType> exp_out = {0., 0., 0., 0., 0.};
  const std::array<int, 4> in_shape = {{1, 1, 1, 5}};
  const bool is_training = false;
  const bool use_cache = false;
  const float momentum = 1 - 0.99;
  const float epsilon = 0.001;
  const DataType max_input_val = 5.0;
  const DataType max_beta_val = 4.0;
  const DataType max_gamma_val = 5.0;
  const DataType max_input_mean_val = 6.0;
  const DataType max_input_var_val = 7.0;
  const float alpha = 0.f;
  const float beta = 0.f;
  this->test_forward(sycldnn::DataFormat::NHWC, in_shape, exp_out,
                     exp_running_mean, exp_running_var, max_input_val,
                     max_beta_val, max_gamma_val, max_input_mean_val,
                     max_input_var_val, momentum, epsilon, is_training,
                     use_cache, alpha, beta);
}

TEST_F(BatchnormCompatTest, ForwardInf1x1x1x5_alpha_0_beta_1) {
  using DataType = float;
  const std::vector<DataType> exp_running_mean = {};
  const std::vector<DataType> exp_running_var = {};
  const std::vector<DataType> exp_out = {1., 2., 3., 4., 5.};
  const std::array<int, 4> in_shape = {{1, 1, 1, 5}};
  const bool is_training = false;
  const bool use_cache = false;
  const float momentum = 1 - 0.99;
  const float epsilon = 0.001;
  const DataType max_input_val = 5.0;
  const DataType max_beta_val = 4.0;
  const DataType max_gamma_val = 5.0;
  const DataType max_input_mean_val = 6.0;
  const DataType max_input_var_val = 7.0;
  const float alpha = 0.f;
  const float beta = 1.f;
  this->test_forward(sycldnn::DataFormat::NHWC, in_shape, exp_out,
                     exp_running_mean, exp_running_var, max_input_val,
                     max_beta_val, max_gamma_val, max_input_mean_val,
                     max_input_var_val, momentum, epsilon, is_training,
                     use_cache, alpha, beta);
}

TEST_F(BatchnormCompatTest, ForwardInf1x1x1x5_alpha_1_beta_1) {
  using DataType = float;
  const std::vector<DataType> exp_running_mean = {};
  const std::vector<DataType> exp_running_var = {};
  const std::vector<DataType> exp_out = {2., 4., 6., 8., 6.};
  const std::array<int, 4> in_shape = {{1, 1, 1, 5}};
  const bool is_training = false;
  const bool use_cache = false;
  const float momentum = 1 - 0.99;
  const float epsilon = 0.001;
  const DataType max_input_val = 5.0;
  const DataType max_beta_val = 4.0;
  const DataType max_gamma_val = 5.0;
  const DataType max_input_mean_val = 6.0;
  const DataType max_input_var_val = 7.0;
  const float alpha = 1.f;
  const float beta = 1.f;
  this->test_forward(sycldnn::DataFormat::NHWC, in_shape, exp_out,
                     exp_running_mean, exp_running_var, max_input_val,
                     max_beta_val, max_gamma_val, max_input_mean_val,
                     max_input_var_val, momentum, epsilon, is_training,
                     use_cache, alpha, beta);
}

TEST_F(BatchnormCompatTest, ForwardInf1x1x1x5_alpha_2_beta_0) {
  using DataType = float;
  const std::vector<DataType> exp_running_mean = {};
  const std::vector<DataType> exp_running_var = {};
  const std::vector<DataType> exp_out = {2., 4., 6., 8., 2.};
  const std::array<int, 4> in_shape = {{1, 1, 1, 5}};
  const bool is_training = false;
  const bool use_cache = false;
  const float momentum = 1 - 0.99;
  const float epsilon = 0.001;
  const DataType max_input_val = 5.0;
  const DataType max_beta_val = 4.0;
  const DataType max_gamma_val = 5.0;
  const DataType max_input_mean_val = 6.0;
  const DataType max_input_var_val = 7.0;
  const float alpha = 2.f;
  const float beta = 0.f;
  this->test_forward(sycldnn::DataFormat::NHWC, in_shape, exp_out,
                     exp_running_mean, exp_running_var, max_input_val,
                     max_beta_val, max_gamma_val, max_input_mean_val,
                     max_input_var_val, momentum, epsilon, is_training,
                     use_cache, alpha, beta);
}

TEST_F(BatchnormCompatTest, ForwardInf1x1x1x5_alpha_neg_2_beta_0) {
  using DataType = float;
  const std::vector<DataType> exp_running_mean = {};
  const std::vector<DataType> exp_running_var = {};
  const std::vector<DataType> exp_out = {-2., -4., -6., -8., -2.};
  const std::array<int, 4> in_shape = {{1, 1, 1, 5}};
  const bool is_training = false;
  const bool use_cache = false;
  const float momentum = 1 - 0.99;
  const float epsilon = 0.001;
  const DataType max_input_val = 5.0;
  const DataType max_beta_val = 4.0;
  const DataType max_gamma_val = 5.0;
  const DataType max_input_mean_val = 6.0;
  const DataType max_input_var_val = 7.0;
  const float alpha = -2.f;
  const float beta = 0.f;
  this->test_forward(sycldnn::DataFormat::NHWC, in_shape, exp_out,
                     exp_running_mean, exp_running_var, max_input_val,
                     max_beta_val, max_gamma_val, max_input_mean_val,
                     max_input_var_val, momentum, epsilon, is_training,
                     use_cache, alpha, beta);
}

TEST_F(BatchnormCompatTest, ForwardInf1x1x8x8_alpha_0_beta_1) {
  using DataType = float;
  const std::vector<DataType> exp_running_mean = {};
  const std::vector<DataType> exp_running_var = {};
  const std::vector<DataType> exp_out = {
      1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  10., 11., 12., 13.,
      14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26.,
      27., 28., 29., 30., 31., 32., 33., 34., 35., 36., 37., 38., 39.,
      40., 41., 42., 43., 44., 45., 46., 47., 48., 49., 50., 51., 52.,
      53., 54., 55., 56., 57., 58., 59., 60., 61., 62., 63., 64.};
  const std::array<int, 4> in_shape = {{1, 1, 8, 8}};
  const bool is_training = false;
  const bool use_cache = false;
  const float momentum = 1 - 0.99;
  const float epsilon = 0.001;
  const DataType max_input_val = 5.0;
  const DataType max_beta_val = 4.0;
  const DataType max_gamma_val = 5.0;
  const DataType max_input_mean_val = 6.0;
  const DataType max_input_var_val = 7.0;
  const float alpha = 0.f;
  const float beta = 1.f;
  this->test_forward(sycldnn::DataFormat::NHWC, in_shape, exp_out,
                     exp_running_mean, exp_running_var, max_input_val,
                     max_beta_val, max_gamma_val, max_input_mean_val,
                     max_input_var_val, momentum, epsilon, is_training,
                     use_cache, alpha, beta);
}

TEST_F(BatchnormCompatTest, ForwardTr1x1x1x8Cache_alpha_0_beta_0) {
  const std::vector<DataType> exp_running_mean = {
      0.010000000000000009, 0.020000000000000018, 0.030000000000000027,
      0.040000000000000036, 0.050000000000000044, 0.010000000000000009,
      0.020000000000000018, 0.030000000000000027};
  const std::vector<DataType> exp_running_var = {0., 0., 0., 0.,
                                                 0., 0., 0., 0.};
  const std::vector<DataType> exp_out = {0., 0., 0., 0., 0., 0., 0., 0.};
  const std::array<int, 4> in_shape = {{1, 1, 1, 8}};

  const DataType max_input_val = 5.0;
  const DataType max_beta_val = 4.0;
  const DataType max_gamma_val = 5.0;
  const DataType max_input_mean_val = 6.0;
  const DataType max_input_var_val = 7.0;
  const float momentum = 0.01;  // 1 - 0.99;
  const float epsilon = 0.001;
  const bool is_training = true;
  const bool use_cache = true;
  const float alpha = 0.f;
  const float beta = 0.f;

  this->test_forward(sycldnn::DataFormat::NHWC, in_shape, exp_out,
                     exp_running_mean, exp_running_var, max_input_val,
                     max_beta_val, max_gamma_val, max_input_mean_val,
                     max_input_var_val, momentum, epsilon, is_training,
                     use_cache, alpha, beta);
}

TEST_F(BatchnormCompatTest,
       Backward1x1x8x1_alpha_data_2_beta_data_0_alpha_param_2_beta_param_0) {
  using DataType = float;
  const std::vector<DataType> exp_grad = {2., 4., 6., 8., 2., 4., 6., 8.};
  const std::vector<DataType> beta_grad = {2 * 20.};
  const std::vector<DataType> gamma_grad = {2 * 35.};
  const std::array<int, 4> in_shape = {{1, 1, 8, 1}};
  const float epsilon = 0.001;
  const DataType max_input_val = 5.0;
  const DataType max_gradient_val = 4.0;
  const DataType max_gamma_val = 5.0;
  const DataType max_pop_mean_val = 6.0;
  const DataType max_pop_var_val = 7.0;
  float alphaDataDiff = 2.f;
  float betaDataDiff = 0.f;
  float alphaParamDiff = 2.0;
  float betaParamDiff = 0.f;
  this->test_backward(sycldnn::DataFormat::NCHW, in_shape, exp_grad, beta_grad,
                      gamma_grad, max_input_val, max_gradient_val,
                      max_gamma_val, max_pop_mean_val, max_pop_var_val, epsilon,
                      alphaDataDiff, betaDataDiff, alphaParamDiff,
                      betaParamDiff);
}

TEST_F(BatchnormCompatTest,
       Backward1x1x8x1_alpha_data_2_beta_data_0_alpha_param_0_beta_param_0) {
  using DataType = float;
  const std::vector<DataType> exp_grad = {2., 4., 6., 8., 2., 4., 6., 8.};
  const std::vector<DataType> beta_grad = {0.};
  const std::vector<DataType> gamma_grad = {0.};
  const std::array<int, 4> in_shape = {{1, 1, 8, 1}};
  const float epsilon = 0.001;
  const DataType max_input_val = 5.0;
  const DataType max_gradient_val = 4.0;
  const DataType max_gamma_val = 5.0;
  const DataType max_pop_mean_val = 6.0;
  const DataType max_pop_var_val = 7.0;
  float alphaDataDiff = 2.f;
  float betaDataDiff = 0.f;
  float alphaParamDiff = 0.0;
  float betaParamDiff = 0.f;
  this->test_backward(sycldnn::DataFormat::NCHW, in_shape, exp_grad, beta_grad,
                      gamma_grad, max_input_val, max_gradient_val,
                      max_gamma_val, max_pop_mean_val, max_pop_var_val, epsilon,
                      alphaDataDiff, betaDataDiff, alphaParamDiff,
                      betaParamDiff);
}

TEST_F(BatchnormCompatTest,
       Backward1x1x8x1_alpha_data_0_beta_data_0_alpha_param_2_beta_param_0) {
  using DataType = float;
  const std::vector<DataType> exp_grad = {0., 0., 0., 0., 0., 0., 0., 0.};
  const std::vector<DataType> beta_grad = {2 * 20.};
  const std::vector<DataType> gamma_grad = {2 * 35.};
  const std::array<int, 4> in_shape = {{1, 1, 8, 1}};
  const float epsilon = 0.001;
  const DataType max_input_val = 5.0;
  const DataType max_gradient_val = 4.0;
  const DataType max_gamma_val = 5.0;
  const DataType max_pop_mean_val = 6.0;
  const DataType max_pop_var_val = 7.0;
  float alphaDataDiff = 0.f;
  float betaDataDiff = 0.f;
  float alphaParamDiff = 2.0;
  float betaParamDiff = 0.f;
  this->test_backward(sycldnn::DataFormat::NCHW, in_shape, exp_grad, beta_grad,
                      gamma_grad, max_input_val, max_gradient_val,
                      max_gamma_val, max_pop_mean_val, max_pop_var_val, epsilon,
                      alphaDataDiff, betaDataDiff, alphaParamDiff,
                      betaParamDiff);
}

TEST_F(
    BatchnormCompatTest,
    Backward1x1x8x1_alpha_data_0_beta_data_0_alpha_param_0_5_beta_param_0_5) {
  using DataType = float;
  const std::vector<DataType> exp_grad = {0., 0., 0., 0., 0., 0., 0., 0.};
  const std::vector<DataType> beta_grad = {(0.5 * 20.) + (0.5 * 1.)};
  const std::vector<DataType> gamma_grad = {(0.5 * 35.) + (0.5 * 1.)};
  const std::array<int, 4> in_shape = {{1, 1, 8, 1}};
  const float epsilon = 0.001;
  const DataType max_input_val = 5.0;
  const DataType max_gradient_val = 4.0;
  const DataType max_gamma_val = 5.0;
  const DataType max_pop_mean_val = 6.0;
  const DataType max_pop_var_val = 7.0;
  float alphaDataDiff = 0.f;
  float betaDataDiff = 0.f;
  float alphaParamDiff = 0.5f;
  float betaParamDiff = 0.5f;
  this->test_backward(sycldnn::DataFormat::NCHW, in_shape, exp_grad, beta_grad,
                      gamma_grad, max_input_val, max_gradient_val,
                      max_gamma_val, max_pop_mean_val, max_pop_var_val, epsilon,
                      alphaDataDiff, betaDataDiff, alphaParamDiff,
                      betaParamDiff);
}

TEST_F(
    BatchnormCompatTest,
    Backward1x1x8x1_alpha_data_0_5_beta_data_0_5_alpha_param_0_beta_param_0) {
  using DataType = float;
  const std::vector<DataType> exp_grad = {
      (1. * 0.5) + (1. * 0.5), (2. * 0.5) + (2. * 0.5), (3. * 0.5) + (3. * 0.5),
      (4. * 0.5) + (4. * 0.5), (1. * 0.5) + (5. * 0.5), (2. * 0.5) + (6. * 0.5),
      (3. * 0.5) + (7. * 0.5), (4. * 0.5) + (8. * 0.5)};
  const std::vector<DataType> beta_grad = {0.};
  const std::vector<DataType> gamma_grad = {0.};
  const std::array<int, 4> in_shape = {{1, 1, 8, 1}};
  const float epsilon = 0.001;
  const DataType max_input_val = 5.0;
  const DataType max_gradient_val = 4.0;
  const DataType max_gamma_val = 5.0;
  const DataType max_pop_mean_val = 6.0;
  const DataType max_pop_var_val = 7.0;
  float alphaDataDiff = 0.5f;
  float betaDataDiff = 0.5f;
  float alphaParamDiff = 0.f;
  float betaParamDiff = 0.f;
  this->test_backward(sycldnn::DataFormat::NCHW, in_shape, exp_grad, beta_grad,
                      gamma_grad, max_input_val, max_gradient_val,
                      max_gamma_val, max_pop_mean_val, max_pop_var_val, epsilon,
                      alphaDataDiff, betaDataDiff, alphaParamDiff,
                      betaParamDiff);
}

TEST_F(
    BatchnormCompatTest,
    Backward1x1x8x1_alpha_data_0_5_beta_data_0_5_alpha_param_0_5_beta_param_0_5) {
  using DataType = float;
  const std::vector<DataType> exp_grad = {
      (1. * 0.5) + (1. * 0.5), (2. * 0.5) + (2. * 0.5), (3. * 0.5) + (3. * 0.5),
      (4. * 0.5) + (4. * 0.5), (1. * 0.5) + (5. * 0.5), (2. * 0.5) + (6. * 0.5),
      (3. * 0.5) + (7. * 0.5), (4. * 0.5) + (8. * 0.5)};
  const std::vector<DataType> beta_grad = {(0.5 * 20) + (0.5 * 1.)};
  const std::vector<DataType> gamma_grad = {(0.5 * 35.) + (0.5 * 1.)};
  const std::array<int, 4> in_shape = {{1, 1, 8, 1}};
  const float epsilon = 0.001;
  const DataType max_input_val = 5.0;
  const DataType max_gradient_val = 4.0;
  const DataType max_gamma_val = 5.0;
  const DataType max_pop_mean_val = 6.0;
  const DataType max_pop_var_val = 7.0;
  float alphaDataDiff = 0.5f;
  float betaDataDiff = 0.5f;
  float alphaParamDiff = 0.5f;
  float betaParamDiff = 0.5f;
  this->test_backward(sycldnn::DataFormat::NCHW, in_shape, exp_grad, beta_grad,
                      gamma_grad, max_input_val, max_gradient_val,
                      max_gamma_val, max_pop_mean_val, max_pop_var_val, epsilon,
                      alphaDataDiff, betaDataDiff, alphaParamDiff,
                      betaParamDiff);
}
