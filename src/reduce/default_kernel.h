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
#ifndef PORTDNN_SRC_REDUCE_DEFAULT_KERNEL_H_
#define PORTDNN_SRC_REDUCE_DEFAULT_KERNEL_H_

#include "portdnn/accessor_types.h"
#include "portdnn/reduce/operators.h"
#include "portdnn/status.h"

namespace sycldnn {
namespace reduce {

namespace internal {

template <typename T, typename Index, typename Op>
struct Reducer;

template <typename T, typename Index>
struct Reducer<T, Index, Add> {
  Reducer(T) : res_(0) {}

  SNN_ALWAYS_INLINE void reduce(T x) { res_ += x; }

  SNN_ALWAYS_INLINE T finalize(Index) { return res_; }

 private:
  T res_;
};

template <typename T, typename Index>
struct Reducer<T, Index, Mean> {
  Reducer(T) : res_(0) {}

  SNN_ALWAYS_INLINE void reduce(T x) { res_ += x; }

  SNN_ALWAYS_INLINE T finalize(Index outer_size) { return res_ / outer_size; }

 private:
  T res_;
};

template <typename T, typename Index>
struct Reducer<T, Index, Max> {
  Reducer(T init) : res_(init) {}

  SNN_ALWAYS_INLINE void reduce(T x) { res_ = cl::sycl::max(res_, x); }

  SNN_ALWAYS_INLINE T finalize(Index) { return res_; }

 private:
  T res_;
};

template <typename T, typename Index>
struct Reducer<T, Index, Min> {
  Reducer(T init) : res_(init) {}

  SNN_ALWAYS_INLINE void reduce(T x) { res_ = cl::sycl::min(res_, x); }

  SNN_ALWAYS_INLINE T finalize(Index) { return res_; }

 private:
  T res_;
};

}  // namespace internal

// TODO: Optimize and specialize kernel for certain sizes
template <typename T, typename Index, typename Op, bool IsUSM>
struct ReduceKernel {
  ReduceKernel(ReadMem<T const, IsUSM> const& input,
               WriteMem<T, IsUSM> const& output, Index batches, Index outer,
               Index inner, Index finalizeParam, T init)
      : input_{input},
        output_{output},
        batches_{batches},
        outer_{outer},
        inner_{inner},
        finalizeParam_{finalizeParam},
        init_{init} {}

  void SNN_ALWAYS_INLINE operator()(cl::sycl::item<2> item) const {
    Index batch = item.get_id(0);
    Index inner = item.get_id(1);

    const auto input = input_.get_pointer();
    auto output = output_.get_pointer();
    internal::Reducer<T, Index, Op> reducer(init_);

    const auto input_n = input + batch * outer_ * inner_ + inner;
    for (Index i = 0; i < outer_; ++i) {
      reducer.reduce(input_n[i * inner_]);
    }
    output[batch * inner_ + inner] = reducer.finalize(finalizeParam_);
  }

 private:
  ReadMem<T const, IsUSM> input_;
  WriteMem<T, IsUSM> output_;
  Index const batches_;
  Index const outer_;
  Index const inner_;
  Index const finalizeParam_;
  T const init_;
};

}  // namespace reduce
}  // namespace sycldnn
#endif  // PORTDNN_SRC_REDUCE_DEFAULT_KERNEL_H_
