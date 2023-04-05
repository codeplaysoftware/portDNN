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
                     DataType max_pop_var_val, double epsilon) {
    const int n = in_shape[0];
    const int c = in_shape[3];
    const int h = in_shape[1];
    const int w = in_shape[2];
    auto input_size = n * c * h * w;
    const int size = exp_out_grad.size();
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
    std::vector<DataType> beta_grad(c);
    std::vector<DataType> gamma_grad(c);
    std::vector<DataType> outputData(size);

    auto q = handle.getQueue();

    auto in_ptr = getPtr(inputData);
    auto in_grad_ptr = getPtr(gradientData);
    auto gamma_ptr = getPtr(gamma);
    auto pop_mean_ptr = getPtr(pop_mean);
    auto pop_var_ptr = getPtr(pop_var);
    auto beta_grad_ptr = sycl::malloc_device<DataType>(c, q);
    auto gamma_grad_ptr = sycl::malloc_device<DataType>(c, q);
    auto out_ptr = sycl::malloc_device<DataType>(size, q);

    TensorDescriptor x_desc;
    x_desc.set4d(format, n, c, h, w);
    TensorDescriptor mean_desc;
    mean_desc.set4d(format, 1, c, 1, 1);

    auto status = batchNormalizationBackward(
        handle, BatchNormMode::BATCHNORM_SPATIAL, nullptr, nullptr, nullptr,
        nullptr, x_desc, in_ptr, x_desc, in_grad_ptr, x_desc, out_ptr,
        mean_desc, pop_mean_ptr, beta_grad_ptr, gamma_grad_ptr, epsilon,
        nullptr, nullptr);
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
    for (int i = 0; i < size; i++) {
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
                    const double epsilon, bool is_training, bool use_cache) {
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
    const auto size = exp_out.size();
    std::vector<DataType> outputData(size);

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
    auto scale_ptr = getPtr(beta);
    auto bias_ptr = getPtr(gamma);
    auto running_mean_ptr = getPtr(input_mean);
    auto running_var_ptr = getPtr(input_var);
    auto saved_mean_ptr =
        use_cache ? sycl::malloc_device<DataType>(c, q) : nullptr;
    auto saved_var_ptr =
        use_cache ? sycl::malloc_device<DataType>(c, q) : nullptr;
    auto out_ptr = sycl::malloc_device<DataType>(size, q);

    if (is_training) {
      auto status = batchNormalizationForwardTraining(
          handle, BatchNormMode::BATCHNORM_SPATIAL, nullptr, nullptr, x_desc,
          in_ptr, x_desc, out_ptr, mean_desc, scale_ptr, bias_ptr, momentum,
          running_mean_ptr, running_var_ptr, epsilon, saved_mean_ptr,
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
        for (int i = 0; i < mean_var_size; i++) {
          SCOPED_TRACE("Element: " + std::to_string(i));
          SNN_ALMOST_EQUAL_EPS(exp_mean[i], outputMean[i], 10u, 1e-5);
        }

        saved_var_copy_event.wait();
        for (int i = 0; i < mean_var_size; i++) {
          SCOPED_TRACE("Element: " + std::to_string(i));
          SNN_ALMOST_EQUAL_EPS(exp_var[i], outputVar[i], 30u, 1e-5);
        }
      }
    } else {
      auto status = batchNormalizationForwardInference(
          handle, BatchNormMode::BATCHNORM_SPATIAL, nullptr, nullptr, x_desc,
          in_ptr, x_desc, out_ptr, mean_desc, scale_ptr, bias_ptr,
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
      1., 2., 3., 4., 5., 5.949999999999999, 1.01, 2.01};
  const std::vector<DataType> exp_running_var = {0.99, 1.98, 2.9699999999999998,
                                                 3.96, 4.95, 5.9399999999999995,
                                                 6.93, 0.99};
  const std::vector<DataType> exp_out = {1.,
                                         2.,
                                         3.,
                                         4.,
                                         1.,
                                         -0.04107137012493428,
                                         3.7558749568782206,
                                         6.998501124063319};
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

  this->test_forward(sycldnn::DataFormat::NHWC, in_shape, exp_out,
                     exp_running_mean, exp_running_var, max_input_val,
                     max_beta_val, max_gamma_val, max_input_mean_val,
                     max_input_var_val, momentum, epsilon, is_training,
                     use_cache);
}

TEST_F(BatchnormCompatTest, ForwardTr1x1x1x8NoCache) {
  const std::vector<DataType> exp_running_mean = {
      1., 2., 3., 4., 5., 5.949999999999999, 1.01, 2.01};
  const std::vector<DataType> exp_running_var = {0.99, 1.98, 2.9699999999999998,
                                                 3.96, 4.95, 5.9399999999999995,
                                                 6.93, 0.99};
  const std::vector<DataType> exp_out = {1.,
                                         2.,
                                         3.,
                                         4.,
                                         1.,
                                         -0.04107137012493428,
                                         3.7558749568782206,
                                         6.998501124063319};
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
  const std::vector<DataType> exp_running_mean = {1.0196875};
  const std::vector<DataType> exp_running_var = {1.009677734375};
  const std::vector<DataType> exp_out = {1.,
                                         1.999500374687773,
                                         2.9990007493755466,
                                         3.9985011240633197,
                                         4.998001498751092,
                                         1.,
                                         1.999500374687773,
                                         2.9990007493755466,
                                         3.9985011240633197,
                                         4.998001498751092,
                                         1.,
                                         1.999500374687773,
                                         2.9990007493755466,
                                         3.9985011240633197,
                                         4.998001498751092,
                                         1.,
                                         1.999500374687773,
                                         2.9990007493755466,
                                         3.9985011240633197,
                                         4.998001498751092,
                                         1.,
                                         1.999500374687773,
                                         2.9990007493755466,
                                         3.9985011240633197,
                                         4.998001498751092,
                                         1.,
                                         1.999500374687773,
                                         2.9990007493755466,
                                         3.9985011240633197,
                                         4.998001498751092,
                                         1.,
                                         1.999500374687773,
                                         2.9990007493755466,
                                         3.9985011240633197,
                                         4.998001498751092,
                                         1.,
                                         1.999500374687773,
                                         2.9990007493755466,
                                         3.9985011240633197,
                                         4.998001498751092,
                                         1.,
                                         1.999500374687773,
                                         2.9990007493755466,
                                         3.9985011240633197,
                                         4.998001498751092,
                                         1.,
                                         1.999500374687773,
                                         2.9990007493755466,
                                         3.9985011240633197,
                                         4.998001498751092,
                                         1.,
                                         1.999500374687773,
                                         2.9990007493755466,
                                         3.9985011240633197,
                                         4.998001498751092,
                                         1.,
                                         1.999500374687773,
                                         2.9990007493755466,
                                         3.9985011240633197,
                                         4.998001498751092,
                                         1.,
                                         1.999500374687773,
                                         2.9990007493755466,
                                         3.9985011240633197};
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
  const std::vector<DataType> exp_grad = {
      -0.9165282260134587, -0.2941179649043309, 0.328292296204797,
      0.9507025573139248,  -1.4633175268209155, -0.1574206397024667,
      0.46498962140666117, 1.087399882515789};
  const std::vector<DataType> beta_grad = {20.};
  const std::vector<DataType> gamma_grad = {1.89776896577748};
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
