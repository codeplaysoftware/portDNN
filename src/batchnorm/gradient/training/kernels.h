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

#ifndef SYCLDNN_SRC_BATCHNORM_GRADIENT_TRAINING_KERNELS_H_
#define SYCLDNN_SRC_BATCHNORM_GRADIENT_TRAINING_KERNELS_H_

#include <CL/sycl.hpp>

#include "src/helpers/tensor_index.h"
#include "src/helpers/vector_io.h"
#include "src/helpers/vector_type.h"

#include "sycldnn/accessor_types.h"

#include "sycldnn/batchnorm/params.h"

namespace sycldnn {
namespace batchnorm {

template <typename T, typename Index, int VectorWidth>
class InputGradientTraining {
  using DType = typename helpers::VectorType<T, VectorWidth>::type;
  using Load = helpers::io::Load<DType>;
  using Store = helpers::io::Store<DType>;

  ReadAccessor<T const> gamma_, variance_, mean_, x_offset_;
  ReadWriteAccessor<T> output_;
  const Index n_items_, n_iterations_, n_offset_;
  float const epsilon_;

 public:
  SNN_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    Index idx = item.get_id(0);

    if (idx < n_items_) {
      auto vec_idx = idx * VectorWidth;

      const auto gamma_ptr = gamma_.get_pointer();
      const auto variance_ptr = variance_.get_pointer();
      const auto mean_ptr = mean_.get_pointer();
      const auto x_offset_ptr = x_offset_.get_pointer();
      auto output_ptr = output_.get_pointer();

      auto gamma_val = Load()(gamma_ptr, vec_idx);
      auto variance_val = Load()(variance_ptr, vec_idx) + DType(epsilon_);
      auto mean_val = Load()(mean_ptr, vec_idx);

      auto gamma_variance = gamma_val / cl::sycl::sqrt(variance_val);
      auto mean_variance = mean_val / variance_val;

      auto vec_idx2 = vec_idx;

      for (Index i = 0; i < n_iterations_; i++, vec_idx2 += n_offset_) {
        auto x_offset_val = Load()(x_offset_ptr, vec_idx2);
        auto grad_y_offset_val =
            Load()(helpers::internal::as_const_ptr(output_ptr), vec_idx2);

        auto out =
            gamma_variance * (grad_y_offset_val - x_offset_val * mean_variance);
        Store()(output_ptr, vec_idx2, out);
      }
    }
  }

  InputGradientTraining(ReadAccessor<T const> gamma,
                        ReadAccessor<T const> variance,
                        ReadAccessor<T const> mean,
                        ReadAccessor<T const> x_offset,
                        ReadWriteAccessor<T> output, Index const num_items,
                        float const epsilon)
      : gamma_(gamma),
        variance_(variance),
        mean_(mean),
        x_offset_(x_offset),
        output_(output),
        n_items_(num_items),
        n_iterations_(x_offset.get_extent() / mean.get_extent()),
        n_offset_(num_items * VectorWidth),
        epsilon_(epsilon) {}
};

template <typename T, typename Index, int VectorWidth>
class GammaGradientTraining {
  using DType = typename helpers::VectorType<T, VectorWidth>::type;
  using Load = helpers::io::Load<DType>;
  using Store = helpers::io::Store<DType>;

  ReadAccessor<T const> variance_, grad_y_x_offset_;
  WriteAccessor<T> output_;
  const Index n_items_;
  const float epsilon_;

 public:
  SNN_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    Index idx = item.get_id(0);

    if (idx < n_items_) {
      auto vec_idx = idx * VectorWidth;

      const auto variance_ptr = variance_.get_pointer();
      const auto grad_y_x_offset_ptr = grad_y_x_offset_.get_pointer();
      auto output_ptr = output_.get_pointer();

      auto var_sqrt =
          cl::sycl::sqrt(Load()(variance_ptr, vec_idx) + DType(epsilon_));
      auto out_val = Load()(grad_y_x_offset_ptr, vec_idx) / var_sqrt;

      Store()(output_ptr, vec_idx, out_val);
    }
  }

  GammaGradientTraining(ReadAccessor<T const> variance,
                        ReadAccessor<T const> grad_y_x_offset,
                        WriteAccessor<T> output, Index const num_items,
                        float const epsilon)
      : variance_(variance),
        grad_y_x_offset_(grad_y_x_offset),
        output_(output),
        n_items_(num_items),
        epsilon_(epsilon) {}
};

}  // namespace batchnorm
}  // namespace sycldnn

#endif  // SYCLDNN_SRC_BATCHNORM_GRADIENT_TRAINING_KERNELS_H_
