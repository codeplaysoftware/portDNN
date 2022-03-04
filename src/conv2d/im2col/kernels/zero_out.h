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
#ifndef SYCLDNN_SRC_CONV2D_IM2COL_KERNELS_ZERO_OUT_H_
#define SYCLDNN_SRC_CONV2D_IM2COL_KERNELS_ZERO_OUT_H_

#include "sycldnn/accessor_types.h"

#include "sycldnn/conv2d/params.h"

#include "src/helpers/vector_io.h"
#include "src/helpers/vector_type.h"

namespace sycldnn {
namespace conv2d {
namespace internal {
namespace im2col {

/** Kernel to fill a buffer with zeros. */
template <typename T, int VectorWidth>
struct ZeroFunctor {
  using StoreType = typename helpers::VectorType<T, VectorWidth>::type;

  ZeroFunctor(size_t output_size, WriteAccessor<T> const& output)
      : output_size_{output_size}, output_{output} {}

  void SNN_ALWAYS_INLINE operator()(cl::sycl::item<1> item) {
    size_t const id = item.get_id(0) * VectorWidth;
    if (id < output_size_) {
      StoreType zeros{0};
      auto output_ptr = output_.get_pointer();
      helpers::io::Store<StoreType>()(output_ptr, id, zeros);
    }
  }

 private:
  /** Number of elements in the output buffer to set to zero. */
  size_t output_size_;
  /** Accessor to the output buffer. */
  WriteAccessor<T> output_;
};

}  // namespace im2col
}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn
#endif  // SYCLDNN_SRC_CONV2D_IM2COL_KERNELS_ZERO_OUT_H_
