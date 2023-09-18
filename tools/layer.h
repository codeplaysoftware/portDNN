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

#include "portdnn/conv2d/launch.h"
#include "portdnn/conv2d/selector/default_selector.h"
#include "portdnn/conv2d/workspace_size.h"

#include "portdnn/helpers/dims.h"
#include "portdnn/helpers/padding.h"
#include "portdnn/helpers/ratio.h"

#include "portdnn/pointwise/launch.h"
#include "portdnn/pointwise/params.h"

#include "portdnn/pooling/launch.h"

#include "portdnn/binaryop/launch.h"
#include "portdnn/binaryop/operators.h"

#include "portdnn/batchnorm/launch.h"

#include "portdnn/matmul/launch.h"
#include "portdnn/matmul/params.h"

#include "portdnn/softmax/launch.h"
#include "portdnn/softmax/sizes.h"

#include "portdnn/padding_mode.h"
#include "portdnn/status.h"

#include <CL/sycl.hpp>

namespace sycldnn {

// Base class of all layer types to present unified interface and construction
template <typename DType, typename Backend>
struct Layer {
  using DeviceMem = typename Backend::template pointer_type<DType>;
  Backend& backend_;

  Layer(Backend& b) : backend_(b) {}
  virtual ~Layer() = default;

  virtual DeviceMem get_output() = 0;
  virtual size_t get_output_size() const = 0;
  virtual sycldnn::SNNStatus run() = 0;
};

template <typename DType, typename Backend>
struct ConvolutionLayer : Layer<DType, Backend> {
  using DeviceMem = typename Backend::template pointer_type<DType>;
  sycldnn::conv2d::Conv2DParams params_;
  sycldnn::conv2d::ConvSizes sizes_;
  DeviceMem input_;
  DeviceMem filter_;
  DeviceMem output_;
  DeviceMem workspace_;
  size_t workspace_size_;
  sycldnn::conv2d::Selector& selector_;

  // Sets parameters and copies data into filter buffer
  ConvolutionLayer(sycldnn::conv2d::Conv2DParams const& params,
                   DeviceMem const input, DeviceMem const weights,
                   DeviceMem output, DeviceMem workspace, size_t workspace_size,
                   Backend& b, sycldnn::conv2d::Selector& selector)
      : Layer<DType, Backend>(b),
        params_{params},
        sizes_{sycldnn::conv2d::get_sizes<sycldnn::conv2d::conv_type::Forward>(
            params_)},
        input_{input},
        filter_{weights},
        output_{output},
        workspace_{workspace},
        workspace_size_{workspace_size},
        selector_{selector} {}

  DeviceMem get_output() override { return output_; }
  size_t get_output_size() const override { return sizes_.output_size; }

  sycldnn::SNNStatus run() override {
    return sycldnn::conv2d::launch<DType, sycldnn::conv2d::conv_type::Forward>(
        input_, filter_, output_, params_, selector_, this->backend_,
        workspace_, workspace_size_);
  }
};
template <typename DType, typename Backend>
struct BiasAddLayer : Layer<DType, Backend> {
  using DeviceMem = typename Backend::template pointer_type<DType>;
  sycldnn::binaryop::BinaryParams params_;
  DeviceMem input_;
  DeviceMem biases_;
  DeviceMem output_;

  BiasAddLayer(sycldnn::binaryop::BinaryParams const& params,
               DeviceMem const input, DeviceMem const bias, DeviceMem output,
               Backend& b)
      : Layer<DType, Backend>(b),
        params_{params},
        input_{input},
        biases_{bias},
        output_{output} {}

  DeviceMem get_output() override { return output_; }
  size_t get_output_size() const override {
    return helpers::get_total_size(params_.lhs_dims);
  }

  sycldnn::SNNStatus run() override {
    return sycldnn::binaryop::launch<DType, sycldnn::binaryop::Add>(
        input_, biases_, output_, params_, this->backend_);
  }
};

template <typename DType, typename Backend>
struct BatchNormTrainingLayer : Layer<DType, Backend> {
  using DeviceMem = typename Backend::template pointer_type<DType>;
  sycldnn::batchnorm::BatchNormParams params_;
  DeviceMem input_;
  DeviceMem beta_;
  DeviceMem gamma_;
  DeviceMem input_mean_, running_mean_;
  DeviceMem input_variance_, running_variance_;
  DeviceMem output_;

  BatchNormTrainingLayer(sycldnn::batchnorm::BatchNormParams const& params,
                         DeviceMem const input, DeviceMem const beta,
                         DeviceMem const gamma, DeviceMem const input_mean,
                         DeviceMem const input_variance, DeviceMem running_mean,
                         DeviceMem running_variance, DeviceMem output,
                         Backend& b)
      : Layer<DType, Backend>(b),
        params_{params},
        input_{input},
        beta_{beta},
        gamma_{gamma},
        input_mean_{input_mean},
        input_variance_{input_variance},
        running_mean_{running_mean},
        running_variance_{running_variance},
        output_{output} {}

  DeviceMem get_output() override { return output_; }
  size_t get_output_size() const override {
    return params_.batch * params_.rows * params_.cols * params_.channels;
  }

  sycldnn::SNNStatus run() override {
    return sycldnn::batchnorm::launch<DType, Backend,
                                      sycldnn::batchnorm::Forward>(
        input_, beta_, gamma_, input_mean_, input_variance_, running_mean_,
        running_variance_, output_, params_, this->backend_);
  }
};

template <typename DType, typename Backend>
struct BatchNormFrozenLayer : Layer<DType, Backend> {
  using DeviceMem = typename Backend::template pointer_type<DType>;
  sycldnn::batchnorm::BatchNormParams params_;
  DeviceMem input_;
  DeviceMem beta_;
  DeviceMem gamma_;
  DeviceMem mean_;
  DeviceMem variance_;
  DeviceMem output_;

  BatchNormFrozenLayer(sycldnn::batchnorm::BatchNormParams const& params,
                       DeviceMem const input, DeviceMem const beta,
                       DeviceMem const gamma, DeviceMem const mean,
                       DeviceMem const variance, DeviceMem output, Backend& b)
      : Layer<DType, Backend>(b),
        params_{params},
        input_{input},
        beta_{beta},
        gamma_{gamma},
        mean_{mean},
        variance_{variance},
        output_{output} {}

  DeviceMem get_output() override { return output_; }
  size_t get_output_size() const override {
    return params_.batch * params_.rows * params_.cols * params_.channels;
  }

  sycldnn::SNNStatus run() override {
    return sycldnn::batchnorm::launch<DType, Backend,
                                      sycldnn::batchnorm::Forward>(
        input_, beta_, gamma_, mean_, variance_, output_, params_,
        this->backend_);
  }
};

template <typename DType, typename Backend,
          template <typename> class ActivationType>
struct ActivationLayer : Layer<DType, Backend> {
  using DeviceMem = typename Backend::template pointer_type<DType>;
  sycldnn::pointwise::PointwiseParams params_;
  DeviceMem input_;
  DeviceMem output_;

  ActivationLayer(sycldnn::pointwise::PointwiseParams const& params,
                  DeviceMem const input, DeviceMem output, Backend& b)
      : Layer<DType, Backend>(b),
        params_{params},
        input_{input},
        output_{output} {}

  DeviceMem get_output() override { return output_; }
  size_t get_output_size() const override { return params_.size; }

  sycldnn::SNNStatus run() override {
    return sycldnn::pointwise::launch<DType, ActivationType,
                                      sycldnn::pointwise::Forward>(
        input_, output_, params_.size, this->backend_);
  }
};

template <typename DType, typename Backend,
          template <typename> class PoolingType>
struct PoolingLayer : Layer<DType, Backend> {
  using DeviceMem = typename Backend::template pointer_type<DType>;
  sycldnn::pooling::PoolingParams params_;
  sycldnn::pooling::PoolingSizes sizes_;
  DeviceMem input_;
  DeviceMem output_;

  PoolingLayer(sycldnn::pooling::PoolingParams const& params,
               DeviceMem const input, DeviceMem output, Backend& b)
      : Layer<DType, Backend>(b),
        params_{params},
        sizes_{sycldnn::pooling::get_sizes<sycldnn::pooling::Forward>(params_)},
        input_{input},
        output_{output} {}

  DeviceMem get_output() override { return output_; }
  size_t get_output_size() const override { return sizes_.output_size; }

  sycldnn::SNNStatus run() override {
    return sycldnn::pooling::launch<DType, PoolingType,
                                    sycldnn::pooling::Forward>(
        input_, output_, params_, this->backend_);
  }
};

template <typename DType, typename Backend>
struct FCLayer : Layer<DType, Backend> {
  using DeviceMem = typename Backend::template pointer_type<DType>;
  sycldnn::matmul::MatmulParams params_;
  DeviceMem input_;
  DeviceMem weights_;
  DeviceMem output_;

  FCLayer(sycldnn::matmul::MatmulParams const& p, DeviceMem const input,
          DeviceMem const weights, DeviceMem output, Backend& b)
      : Layer<DType, Backend>(b),
        params_(p),
        input_{input},
        weights_{weights},
        output_{output} {}

  DeviceMem get_output() override { return output_; }
  size_t get_output_size() const override { return params_.n; }
  sycldnn::SNNStatus run() override {
    using ConstPointer = typename Backend::template pointer_type<DType const>;
    return {this->backend_.template matmul<false, false>(
                ConstPointer{input_}, ConstPointer{weights_}, output_,
                params_.beta, params_.m, params_.k, params_.n),
            sycldnn::StatusCode::OK};
  }
};

template <typename DType, typename Backend>
struct SoftmaxLayer : Layer<DType, Backend> {
  using DeviceMem = typename Backend::template pointer_type<DType>;
  sycldnn::softmax::SoftmaxParams params_;
  sycldnn::softmax::SoftmaxSizes sizes_;
  DeviceMem input_;
  DeviceMem workspace_;
  DeviceMem output_;

  SoftmaxLayer(sycldnn::softmax::SoftmaxParams const& params,
               DeviceMem const input, DeviceMem workspace, DeviceMem output,
               Backend& b)
      : Layer<DType, Backend>(b),
        params_{params},
        sizes_{sycldnn::softmax::get_sizes(params)},
        input_{input},
        workspace_{workspace},
        output_{output} {}

  DeviceMem get_output() override { return output_; }
  size_t get_output_size() const override { return sizes_.output_size; }

  sycldnn::SNNStatus run() override {
    return sycldnn::softmax::launch<DType, sycldnn::softmax::Forward, Backend>(
        input_, workspace_, output_, params_, this->backend_);
  }
};
}  // namespace sycldnn
