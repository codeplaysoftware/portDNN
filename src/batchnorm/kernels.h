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

#include "src/helpers/tensor_index.h"
#include "src/helpers/vector_io.h"
#include "src/helpers/vector_type.h"

#include "sycldnn/accessor_types.h"

#include "sycldnn/batchnorm/params.h"

namespace sycldnn {
namespace batchnorm {

template <typename T, typename Index, int VectorWidth>
class VarianceOp;

template <typename T, typename Index, int VectorWidth>
class VarianceOp {
  using DType = typename helpers::VectorType<T, VectorWidth>::type;
  using Load = helpers::io::Load<DType>;
  using Store = helpers::io::Store<DType>;

  ReadAccessor<T const> input_, current_mean_;
  WriteAccessor<T> current_variance_;
  const Index n_items_;

 public:
  SNN_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    Index idx = item.get_id(0);

    if (idx < n_items_) {
      auto in_count = static_cast<Index>(input_.get_extent());
      auto mean_count = static_cast<Index>(current_mean_.get_extent());

      auto incr = in_count / mean_count;

      auto vec_idx1 = idx * VectorWidth;
      auto vec_idx2 = vec_idx1 % mean_count;

      const auto input = input_.get_pointer();
      const auto current_mean = current_mean_.get_pointer();
      auto current_variance = current_variance_.get_pointer();

      auto mean = Load()(current_mean, vec_idx2);
      auto out = static_cast<DType>(0);
      for (Index i = vec_idx1; i < in_count; i += mean_count) {
        out += cl::sycl::pow(Load()(input, i) - mean, static_cast<DType>(2));
      }

      if (incr == 1) {
        Store()(current_variance, vec_idx2, out);
      } else {
        Store()(current_variance, vec_idx2, out / static_cast<DType>(incr));
      }
    }
  }

  VarianceOp(ReadAccessor<T const> input, ReadAccessor<T const> current_mean,
             WriteAccessor<T> current_variance, Index const num_items)
      : input_(input),
        current_mean_(current_mean),
        current_variance_(current_variance),
        n_items_(num_items) {}
};

template <typename T, typename Index, int VectorWidth>
class RunningMeanVarianceOp {
  using DType = typename helpers::VectorType<T, VectorWidth>::type;
  using Load = helpers::io::Load<DType>;
  using Store = helpers::io::Store<DType>;

  ReadAccessor<T const> input_mean_, input_variance_;
  ReadWriteAccessor<T> output_mean_, output_variance_;
  const Index n_items_;
  const float momentum_;

 public:
  SNN_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    Index idx = item.get_id(0);

    if (idx < n_items_) {
      auto vec_idx = idx * VectorWidth;

      const auto input_mean_ptr = input_mean_.get_pointer();
      const auto input_variance_ptr = input_variance_.get_pointer();
      auto output_mean_ptr = output_mean_.get_pointer();
      auto output_variance_ptr = output_variance_.get_pointer();

      auto out_mean_val =
          Load()(input_mean_ptr, vec_idx) * static_cast<DType>(momentum_) +
          Load()(helpers::internal::as_const_ptr(output_mean_ptr), vec_idx) *
              static_cast<DType>(1 - momentum_);
      auto out_variance_val =
          Load()(input_variance_ptr, vec_idx) * static_cast<DType>(momentum_) +
          Load()(helpers::internal::as_const_ptr(output_variance_ptr),
                 vec_idx) *
              static_cast<DType>(1 - momentum_);

      Store()(output_mean_ptr, vec_idx, out_mean_val);
      Store()(output_variance_ptr, vec_idx, out_variance_val);
    }
  }

  RunningMeanVarianceOp(ReadAccessor<T const> input_mean,
                        ReadAccessor<T const> input_variance,
                        ReadWriteAccessor<T> output_mean,
                        ReadWriteAccessor<T> output_variance,
                        Index const num_items, float const momentum)
      : input_mean_(input_mean),
        input_variance_(input_variance),
        output_mean_(output_mean),
        output_variance_(output_variance),
        n_items_(num_items),
        momentum_(momentum) {}
};

template <typename T, typename Index, int VectorWidth>
class BatchNormOp;

template <typename T, typename Index, int VectorWidth>
class BatchNormOp {
  using DType = typename helpers::VectorType<T, VectorWidth>::type;
  using Load = helpers::io::Load<DType>;
  using Store = helpers::io::Store<DType>;

  ReadAccessor<T const> input_, beta_, gamma_, current_mean_, current_variance_;
  WriteAccessor<T> output_;
  const Index n_items_;
  BatchNormParams params_;

 public:
  SNN_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    Index idx = item.get_id(0);

    if (idx < n_items_) {
      auto vec_idx1 = idx * VectorWidth;
      auto vec_idx2 = vec_idx1 % params_.channels;

      const auto input = input_.get_pointer();
      const auto beta = beta_.get_pointer();
      const auto gamma = gamma_.get_pointer();
      const auto current_mean = current_mean_.get_pointer();
      const auto current_variance = current_variance_.get_pointer();
      auto output = output_.get_pointer();

      auto feature = Load()(input, vec_idx1);
      auto beta_val = Load()(beta, vec_idx2);
      auto gamma_val = Load()(gamma, vec_idx2);
      auto mean = Load()(current_mean, vec_idx2);
      auto variance = Load()(current_variance, vec_idx2);

      auto val = gamma_val * ((feature - mean) /
                              cl::sycl::sqrt(variance + params_.epsilon)) +
                 beta_val;
      Store()(output, vec_idx1, val);
    }
  }

  BatchNormOp(ReadAccessor<T const> input, ReadAccessor<T const> beta,
              ReadAccessor<T const> gamma, ReadAccessor<T const> current_mean,
              ReadAccessor<T const> current_variance, WriteAccessor<T> output,
              BatchNormParams const& pp)
      : input_(input),
        beta_(beta),
        gamma_(gamma),
        current_mean_(current_mean),
        current_variance_(current_variance),
        output_(output),
        n_items_(pp.batch * pp.rows * pp.cols * pp.channels / VectorWidth),
        params_(pp) {}
};

}  // namespace batchnorm
}  // namespace sycldnn

#endif  // SYCLDNN_SRC_BATCHNORM_KERNELS_H_
