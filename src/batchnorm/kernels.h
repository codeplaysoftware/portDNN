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

#ifndef SYCLDNN_SRC_BATCHNORM_KERNELS_H_
#define SYCLDNN_SRC_BATCHNORM_KERNELS_H_

#include <CL/sycl.hpp>
#include <array>

#include "src/helpers/tensor_index.h"
#include "src/helpers/vector_io.h"
#include "src/helpers/vector_type.h"

#include "sycldnn/accessor_types.h"

#include "sycldnn/batchnorm/params.h"
#include "sycldnn/helpers/dims.h"
#include "sycldnn/helpers/ratio.h"

namespace sycldnn {
namespace batchnorm {

template <typename Index, int VectorWidth, bool IsNCHW>
struct flattened_index;

template <typename Index, int VectorWidth>
struct flattened_index<Index, VectorWidth, /*IsNCHW*/ true> {
  static Index value(Index n, Index c, Index hw, Index channel_size,
                     Index spatial_size) {
    return ((n * channel_size * spatial_size) + (c * spatial_size) +
            (hw * VectorWidth));
  }
};

template <typename Index, int VectorWidth>
struct flattened_index<Index, VectorWidth, /*IsNCHW*/ false> {
  static Index value(Index n, Index c, Index hw, Index channel_size,
                     Index spatial_size) {
    return ((n * channel_size * spatial_size) + (hw * channel_size) + c);
  }
};

template <typename Index, int VectorWidth, bool IsNCHW>
inline cl::sycl::nd_range<3> compute_range(Index batch, Index channels,
                                           Index spatial_size) {
  Index const wg_x = 1;
  Index const wg_y = IsNCHW ? 8 : 32;
  Index const wg_z = IsNCHW ? 32 : 8;
  Index global_size_x = helpers::round_up_to_nearest_multiple(batch, wg_x);
  Index global_size_y = helpers::round_up_to_nearest_multiple(
      IsNCHW ? channels : spatial_size, wg_y);
  Index global_size_z = helpers::round_up_to_nearest_multiple(
      (IsNCHW ? (spatial_size / VectorWidth) : channels), wg_z);

  return cl::sycl::nd_range<3>(
      cl::sycl::range<3>(global_size_x, global_size_y, global_size_z),
      cl::sycl::range<3>(std::min(global_size_x, wg_x),
                         std::min(global_size_y, wg_y),
                         std::min(global_size_z, wg_z)));
}

template <typename T, typename Index, int VectorWidth, bool IsNCHW,
          typename DevPointer, typename ConstDevPointer>
SNN_ALWAYS_INLINE void batchnorm_inf(Index channel_size, Index spatial_size,
                                     Index batch_idx, Index channel_idx,
                                     Index spatial_idx, ConstDevPointer input,
                                     T mean, T inv_var, T scale, T bias,
                                     DevPointer out) {
  using DataType = typename helpers::VectorType<T, VectorWidth>::type;
  using LoadData = helpers::io::Load<DataType>;
  using StoreData = helpers::io::Store<DataType>;

  Index const flat_idx = flattened_index<Index, VectorWidth, IsNCHW>::value(
      batch_idx, channel_idx, spatial_idx, channel_size, spatial_size);

  auto const in_val = LoadData()(input, flat_idx);
  auto const in_hat = (in_val - mean) * inv_var;
  auto const out_val = scale * in_hat + bias;
  StoreData()(out, flat_idx, out_val);
}

template <typename T, typename Index, int VectorWidth, bool IsNCHW, bool IsUSM>
class BatchnormInferenceOp {
  ReadMem<T const, IsUSM> input_, beta_, gamma_, mean_, variance_;
  WriteMem<T, IsUSM> out_;
  Index const batch_, channels_, spatial_size_;
  T const epsilon_, momentum_;

 public:
  BatchnormInferenceOp(ReadMem<T const, IsUSM> input,
                       ReadMem<T const, IsUSM> beta,
                       ReadMem<T const, IsUSM> gamma,
                       ReadMem<T const, IsUSM> mean,
                       ReadMem<T const, IsUSM> variance, WriteMem<T, IsUSM> out,
                       const BatchNormParams& params)
      : input_(input),
        beta_(beta),
        gamma_(gamma),
        mean_(mean),
        variance_(variance),
        out_(out),
        batch_(params.batch),
        channels_(params.channels),
        spatial_size_(params.rows * params.cols),
        epsilon_(params.epsilon),
        momentum_(params.momentum) {}

  cl::sycl::nd_range<3> get_range() {
    return compute_range<Index, VectorWidth, IsNCHW>(batch_, channels_,
                                                     spatial_size_);
  }

  SNN_ALWAYS_INLINE void operator()(cl::sycl::nd_item<3> item) const {
    Index const batch_idx = item.get_global_id(0);
    Index const channel_idx = item.get_global_id(IsNCHW ? 1 : 2);
    Index const spatial_idx = item.get_global_id(IsNCHW ? 2 : 1);

    if (!(batch_idx < batch_ && channel_idx < channels_ &&
          spatial_idx < (spatial_size_ / VectorWidth))) {
      return;
    }

    auto const mean = mean_.get_pointer()[channel_idx];
    auto const var = variance_.get_pointer()[channel_idx];
    auto const scale = gamma_.get_pointer()[channel_idx];
    auto const bias = beta_.get_pointer()[channel_idx];
    auto const inv_var = cl::sycl::rsqrt(var + epsilon_);

    batchnorm_inf<T, Index, VectorWidth, IsNCHW>(
        channels_, spatial_size_, batch_idx, channel_idx, spatial_idx,
        input_.get_pointer(), mean, inv_var, scale, bias, out_.get_pointer());
  }
};

template <typename T, typename Index, bool IsNCHW, bool IsUSM>
class BatchnormTrainingOp {
  static constexpr int VectorWidth = 1;

  ReadMem<T const, IsUSM> input_, beta_, gamma_;
  WriteMem<T, IsUSM> running_mean_, running_variance_, cache_mean_,
      cache_invvariance_, workspace_var_, out_;
  Index const batch_, channels_, spatial_size_;
  T const epsilon_, momentum_;

 public:
  BatchnormTrainingOp(
      ReadMem<T const, IsUSM> input, ReadMem<T const, IsUSM> beta,
      ReadMem<T const, IsUSM> gamma, WriteMem<T, IsUSM> running_mean,
      WriteMem<T, IsUSM> running_variance, WriteMem<T, IsUSM> cache_mean,
      WriteMem<T, IsUSM> cache_invvariance, WriteMem<T, IsUSM> workspace_var,
      WriteMem<T, IsUSM> out, const BatchNormParams& params)
      : input_(input),
        beta_(beta),
        gamma_(gamma),
        running_mean_(running_mean),
        running_variance_(running_variance),
        cache_mean_(cache_mean),
        cache_invvariance_(cache_invvariance),
        workspace_var_(workspace_var),
        out_(out),
        batch_(params.batch),
        channels_(params.channels),
        spatial_size_(params.rows * params.cols),
        epsilon_(params.epsilon),
        momentum_(params.momentum) {}

  cl::sycl::nd_range<3> get_range() {
    return compute_range<Index, VectorWidth, IsNCHW>(batch_, channels_,
                                                     spatial_size_);
  }

  SNN_ALWAYS_INLINE void operator()(cl::sycl::nd_item<3> item) const {
    Index const batch_idx = item.get_global_id(0);
    Index const channel_idx = item.get_global_id(IsNCHW ? 1 : 2);
    Index const spatial_idx = item.get_global_id(IsNCHW ? 2 : 1);

    if (!(batch_idx < batch_ && channel_idx < channels_ &&
          spatial_idx < (spatial_size_ / VectorWidth))) {
      return;
    }

    auto const mean = cache_mean_.get_pointer()[channel_idx];
    auto const var = workspace_var_.get_pointer()[channel_idx];
    auto const scale = gamma_.get_pointer()[channel_idx];
    auto const bias = beta_.get_pointer()[channel_idx];
    auto const inv_var = cl::sycl::rsqrt(var + epsilon_);

    batchnorm_inf<T, Index, VectorWidth, IsNCHW>(
        channels_, spatial_size_, batch_idx, channel_idx, spatial_idx,
        input_.get_pointer(), mean, inv_var, scale, bias, out_.get_pointer());

    if (batch_idx == 0 && spatial_idx == 0) {
      auto running_mean_ptr = running_mean_.get_pointer();
      auto running_var_ptr = running_variance_.get_pointer();
      running_mean_ptr[channel_idx] =
          (running_mean_ptr[channel_idx] * momentum_) +
          (mean * (T{1} - momentum_));
      running_var_ptr[channel_idx] =
          (running_var_ptr[channel_idx] * momentum_) +
          (var * (T{1} - momentum_));

      cache_invvariance_.get_pointer()[channel_idx] = inv_var;
    }
  }
};

template <typename T, typename Index, int VectorWidth, bool IsNCHW, bool IsUSM>
class BatchnormGradientOp {
  using DataType = typename helpers::VectorType<T, VectorWidth>::type;
  using LoadData = helpers::io::Load<DataType>;
  using StoreData = helpers::io::Store<DataType>;

  ReadMem<T const, IsUSM> input_, out_grad_, gamma_, cache_mean_, cache_invvar_;
  WriteMem<T, IsUSM> in_grad_, workspace_;
  Index const batch_, channels_, spatial_size_;

 public:
  BatchnormGradientOp(ReadMem<T const, IsUSM> input,
                      ReadMem<T const, IsUSM> out_grad,
                      WriteMem<T, IsUSM> in_grad, ReadMem<T const, IsUSM> gamma,
                      ReadMem<T const, IsUSM> cache_mean,
                      ReadMem<T const, IsUSM> cache_invvariance,
                      WriteMem<T, IsUSM> workspace,
                      const BatchNormParams& params)
      : input_(input),
        out_grad_(out_grad),
        gamma_(gamma),
        cache_mean_(cache_mean),
        cache_invvar_(cache_invvariance),
        in_grad_(in_grad),
        workspace_(workspace),
        batch_(params.batch),
        channels_(params.channels),
        spatial_size_(params.rows * params.cols) {}

  cl::sycl::nd_range<3> get_range() {
    return compute_range<Index, VectorWidth, IsNCHW>(batch_, channels_,
                                                     spatial_size_);
  }

  SNN_ALWAYS_INLINE void operator()(cl::sycl::nd_item<3> item) const {
    Index const batch_idx = item.get_global_id(0);
    Index const channel_idx = item.get_global_id(IsNCHW ? 1 : 2);
    Index const spatial_idx = item.get_global_id(IsNCHW ? 2 : 1);

    if (!(batch_idx < batch_ && channel_idx < channels_ &&
          spatial_idx < (spatial_size_ / VectorWidth))) {
      return;
    }

    auto const scale = gamma_.get_pointer()[channel_idx];
    auto const mean = cache_mean_.get_pointer()[channel_idx];
    auto const inv_var = cache_invvar_.get_pointer()[channel_idx];

    Index const flat_idx = flattened_index<Index, VectorWidth, IsNCHW>::value(
        batch_idx, channel_idx, spatial_idx, channels_, spatial_size_);

    auto const out_grad_val = LoadData()(out_grad_.get_pointer(), flat_idx);
    auto const out_val = scale * out_grad_val * inv_var;
    StoreData()(in_grad_.get_pointer(), flat_idx, out_val);

    auto const input_val = LoadData()(input_.get_pointer(), flat_idx);
    auto const workspace_val = out_grad_val * (input_val - mean) * inv_var;
    StoreData()(workspace_.get_pointer(), flat_idx, workspace_val);
  }
};

}  // namespace batchnorm
}  // namespace sycldnn

#endif  // SYCLDNN_SRC_BATCHNORM_KERNELS_H_
