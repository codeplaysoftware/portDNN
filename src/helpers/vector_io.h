/*
 * Copyright 2018 Codeplay Software Ltd
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
#ifndef SYCLDNN_SRC_HELPERS_VECTOR_IO_H_
#define SYCLDNN_SRC_HELPERS_VECTOR_IO_H_

#include <CL/sycl.hpp>

namespace sycldnn {
namespace helpers {
/**
 * Load and Store helper structs to load and store SYCL vectors and data types
 * to memory. When possible vload is used to load data into a vector and vstore
 * to store a vector in memory. Also provides operations for non vector types
 * so that a single interface can be used in kernels no matter what the data
 * type.
 */
namespace io {
template <typename T>
struct Load {
  template <typename _T, typename Index>
  T SNN_ALWAYS_INLINE operator()(_T const* const ptr, Index const offset) {
    return ptr[offset];
  }
};
template <typename T, int N>
struct Load<cl::sycl::vec<T, N>> {
  template <typename _T, typename Index>
  cl::sycl::vec<T, N> SNN_ALWAYS_INLINE operator()(_T const* const ptr,
                                                   Index const offset) {
    static constexpr auto address_space =
        cl::sycl::access::address_space::global_space;
    _T* non_const_ptr = const_cast<_T*>(ptr);
    cl::sycl::multi_ptr<T, address_space> mptr(non_const_ptr + offset);
    cl::sycl::vec<T, N> result;
    result.load(0, mptr);
    return result;
  }
};
template <typename T>
struct Load<cl::sycl::vec<T, 1>> {
  template <typename _T, typename Index>
  cl::sycl::vec<T, 1> SNN_ALWAYS_INLINE operator()(_T const* const ptr,
                                                   Index const offset) {
    cl::sycl::vec<T, 1> result(ptr[offset]);
    return result;
  }
};
template <typename T>
struct Store {
  template <typename _T, typename Index>
  void SNN_ALWAYS_INLINE operator()(_T* ptr, Index const offset, T const val) {
    static_assert(!std::is_const<_T>::value,
                  "Cannot store values in a pointer to const types.");
    ptr[offset] = val;
  }
};
template <typename T, int N>
struct Store<cl::sycl::vec<T, N>> {
  template <typename _T, typename Index>
  void SNN_ALWAYS_INLINE operator()(_T* ptr, Index const offset,
                                    cl::sycl::vec<T, N> const val) {
    static_assert(!std::is_const<_T>::value,
                  "Cannot store values in a pointer to const types.");
    static constexpr auto address_space =
        cl::sycl::access::address_space::global_space;
    cl::sycl::multi_ptr<T, address_space> mptr(ptr + offset);
    val.store(0, mptr);
  }
};
template <typename T>
struct Store<cl::sycl::vec<T, 1>> {
  template <typename _T, typename Index>
  void SNN_ALWAYS_INLINE operator()(_T* ptr, Index const offset,
                                    cl::sycl::vec<T, 1> val) {
    static_assert(!std::is_const<_T>::value,
                  "Cannot store values in a pointer to const types.");
    ptr[offset] = val.s0();
  }
};
}  // namespace io
}  // namespace helpers
}  // namespace sycldnn
#endif  // SYCLDNN_SRC_HELPERS_VECTOR_IO_H_
