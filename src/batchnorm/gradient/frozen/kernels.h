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

#ifndef SYCLDNN_SRC_BATCHNORM_GRADIENT_FROZEN_KERNELS_H_
#define SYCLDNN_SRC_BATCHNORM_GRADIENT_FROZEN_KERNELS_H_

#include <CL/sycl.hpp>

#include "src/helpers/tensor_index.h"
#include "src/helpers/vector_io.h"
#include "src/helpers/vector_type.h"

#include "sycldnn/accessor_types.h"

#include "sycldnn/batchnorm/params.h"

namespace sycldnn {
namespace batchnorm {

template <typename T, typename Index, int VectorWidth>
class InputGradientFrozen {
  using DType = typename helpers::VectorType<T, VectorWidth>::type;
  using Load = helpers::io::Load<DType>;
  using Store = helpers::io::Store<DType>;

  ReadAccessor<T const> gradient_, gamma_, variance_;
  WriteAccessor<T> output_;
  Index const n_items_, n_iterations_, n_offset_;
  float const epsilon_;

 public:
  SNN_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    Index idx = item.get_id(0);

    if (idx < n_items_) {
      auto vec_idx = idx * VectorWidth;

      const auto gradient_ptr = gradient_.get_pointer();
      const auto gamma_ptr = gamma_.get_pointer();
      const auto variance_ptr = variance_.get_pointer();
      auto output_ptr = output_.get_pointer();

      auto gamma_val = Load()(gamma_ptr, vec_idx);
      auto variance_val =
          Load()(variance_ptr, vec_idx) + static_cast<DType>(epsilon_);

      auto gamma_variance = gamma_val / cl::sycl::sqrt(variance_val);

      auto vec_idx2 = vec_idx;

      for (Index i = 0; i < n_iterations_; i++, vec_idx2 += n_offset_) {
        auto gradient_val = Load()(gradient_ptr, vec_idx2);
        auto out = gamma_variance * gradient_val;
        Store()(output_ptr, vec_idx2, out);
      }
    }
  }

  InputGradientFrozen(ReadAccessor<T const> gradient,
                      ReadAccessor<T const> gamma,
                      ReadAccessor<T const> variance, WriteAccessor<T> output,
                      Index const num_items, float const epsilon)
      : gradient_(gradient),
        gamma_(gamma),
        variance_(variance),
        output_(output),
        n_items_(num_items),
        n_iterations_(gradient.get_extent() / gamma.get_extent()),
        n_offset_(num_items * VectorWidth),
        epsilon_(epsilon) {}
};

template <typename T, typename Index, int VectorWidth>
class GammaGradientFrozen {
  using DType = typename helpers::VectorType<T, VectorWidth>::type;
  using Load = helpers::io::Load<DType>;
  using Store = helpers::io::Store<DType>;

  ReadAccessor<T const> gradient_, input_, mean_, variance_;
  WriteAccessor<T> output_;
  Index const n_items_, n_iterations_, n_offset_;
  float const epsilon_;

 public:
  SNN_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    Index idx = item.get_id(0);

    if (idx < n_items_) {
      auto vec_idx = idx * VectorWidth;

      const auto gradient_ptr = gradient_.get_pointer();
      const auto input_ptr = input_.get_pointer();
      const auto mean_ptr = mean_.get_pointer();
      const auto variance_ptr = variance_.get_pointer();
      auto output_ptr = output_.get_pointer();

      auto mean_val = Load()(mean_ptr, vec_idx);
      auto variance_val =
          Load()(variance_ptr, vec_idx) + static_cast<DType>(epsilon_);

      auto vec_idx2 = vec_idx;

      for (Index i = 0; i < n_iterations_; i++, vec_idx2 += n_offset_) {
        auto gradient_val = Load()(gradient_ptr, vec_idx2);
        auto input_val = Load()(input_ptr, vec_idx2);
        auto out = (gradient_val * (input_val - mean_val)) /
                   cl::sycl::sqrt(variance_val);
        Store()(output_ptr, vec_idx2, out);
      }
    }
  }

  GammaGradientFrozen(ReadAccessor<T const> gradient,
                      ReadAccessor<T const> input, ReadAccessor<T const> mean,
                      ReadAccessor<T const> variance, WriteAccessor<T> output,
                      Index const num_items, float const epsilon)
      : gradient_(gradient),
        input_(input),
        mean_(mean),
        variance_(variance),
        output_(output),
        n_items_(num_items),
        n_iterations_(gradient.get_extent() / mean.get_extent()),
        n_offset_(num_items * VectorWidth),
        epsilon_(epsilon) {}
};

}  // namespace batchnorm
}  // namespace sycldnn

#endif  // SYCLDNN_SRC_BATCHNORM_GRADIENT_FROZEN_KERNELS_H_
