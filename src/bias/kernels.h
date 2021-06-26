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

#ifndef SYCLDNN_SRC_BIAS_KERNELS_H_
#define SYCLDNN_SRC_BIAS_KERNELS_H_

#include <CL/sycl.hpp>

#include "src/helpers/fast_div.h"
#include "src/helpers/tensor_index.h"
#include "src/helpers/vector_io.h"
#include "src/helpers/vector_type.h"

#include "sycldnn/accessor_types.h"

#include "sycldnn/helpers/minmax.h"

#include "sycldnn/bias/params.h"

namespace sycldnn {
namespace bias {

template <typename T, typename Index, int VectorWidth>
class BiasOp;

template <typename T, typename Index, int VectorWidth>
class BiasOp {
  using DataT = typename helpers::VectorType<T, VectorWidth>::type;
  using Load = helpers::io::Load<DataT>;
  using Store = helpers::io::Store<DataT>;

  ReadAccessor<T const> in_data_, bias_;
  WriteAccessor<T> out_data_;
  const Index n_items_;
  BiasParams params_;

 public:
  SNN_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    Index index = item.get_id(0);
    Index bias_index = (index * VectorWidth) % params_.bias;

    const auto in = in_data_.get_pointer();
    const auto bias = bias_.get_pointer();
    auto out = out_data_.get_pointer();

    Store()(out, index * VectorWidth,
            Load()(in, index * VectorWidth) + Load()(bias, bias_index));
  }

  BiasOp(ReadAccessor<T const> in_data, ReadAccessor<T const> bias,
         WriteAccessor<T> out_data, BiasParams const& pp)
      : in_data_(in_data),
        bias_(bias),
        out_data_(out_data),
        n_items_(pp.batch * pp.in_rows * pp.in_cols * pp.channels /
                 VectorWidth),
        params_(pp) {}
};

}  // namespace bias
}  // namespace sycldnn

#endif  // SYCLDNN_SRC_BIAS_KERNELS_H_
