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
#include <type_traits>

#include "sycldnn/helpers/macros.h"

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
  template <typename U, typename Index>
  T SNN_ALWAYS_INLINE operator()(U const* const ptr, Index const offset) {
    static_assert(std::is_convertible<U, T>::value,
                  "Type U must be convertible to type T.");
    return ptr[offset];
  }
  template <typename U, typename Index, cl::sycl::access::address_space Space>
  T SNN_ALWAYS_INLINE operator()(cl::sycl::multi_ptr<U, Space> ptr,
                                 Index const offset) {
    static_assert(std::is_convertible<U, T>::value,
                  "Type U must be convertible to type T.");
    return *(ptr + offset);
  }
};
template <typename T, int N>
struct Load<cl::sycl::vec<T, N>> {
  static_assert(!std::is_const<T>::value,
                "Cannot load values into a vector of const types.");
  template <typename Index, cl::sycl::access::address_space Space>
  cl::sycl::vec<T, N> SNN_ALWAYS_INLINE
  operator()(cl::sycl::multi_ptr<T, Space> ptr, Index const offset) {
    cl::sycl::vec<T, N> result;
    result.load(0, ptr + offset);
    return result;
  }
  template <typename U, typename Index, cl::sycl::access::address_space Space,
            typename DependentU = U,
            typename std::enable_if<std::is_same<U, DependentU>::value &&
                                        std::is_const<DependentU>::value,
                                    int>::type = 0>
  cl::sycl::vec<T, N> SNN_ALWAYS_INLINE
  operator()(cl::sycl::multi_ptr<U, Space> ptr, Index const offset) {
    static_assert(std::is_convertible<U, T>::value,
                  "Type U must be convertible to type T.");
    // When the pointer is of the form multi_ptr<T const> we need to cast away
    // the const as vec<T> will only load values from a multi_ptr<T>
    using NonConstT = typename std::remove_const<U>::type;
    auto non_const_ptr = const_cast<NonConstT*>(ptr.get());
    cl::sycl::multi_ptr<T, Space> multi_ptr(non_const_ptr);
    return operator()(multi_ptr, offset);
  }
  template <typename U, typename Index>
  cl::sycl::vec<T, N> SNN_ALWAYS_INLINE operator()(U const* const ptr,
                                                   Index const offset) {
    static_assert(std::is_convertible<U, T>::value,
                  "Type U must be convertible to type T.");
    static constexpr auto address_space =
        cl::sycl::access::address_space::global_space;
    auto* non_const_ptr = const_cast<U*>(ptr);
    cl::sycl::multi_ptr<T, address_space> mptr(non_const_ptr);
    return operator()(mptr, offset);
  }
};
template <typename T>
struct Load<cl::sycl::vec<T, 1>> {
  template <typename U, typename Index, cl::sycl::access::address_space Space>
  cl::sycl::vec<T, 1> SNN_ALWAYS_INLINE
  operator()(cl::sycl::multi_ptr<U, Space> ptr, Index const offset) {
    static_assert(std::is_convertible<U, T>::value,
                  "Type U must be convertible to type T.");
    cl::sycl::vec<T, 1> result(ptr[offset]);
    return result;
  }
  template <typename U, typename Index>
  cl::sycl::vec<T, 1> SNN_ALWAYS_INLINE operator()(U const* const ptr,
                                                   Index const offset) {
    static_assert(std::is_convertible<U, T>::value,
                  "Type U must be convertible to type T.");
    cl::sycl::vec<T, 1> result(ptr[offset]);
    return result;
  }
};
template <typename T>
struct Store {
  template <typename Index, cl::sycl::access::address_space Space>
  void SNN_ALWAYS_INLINE operator()(cl::sycl::multi_ptr<T, Space> ptr,
                                    Index const offset, T const val) {
    *(ptr + offset) = val;
  }
  template <typename U, typename Index>
  void SNN_ALWAYS_INLINE operator()(U* ptr, Index const offset, T const val) {
    static_assert(!std::is_const<U>::value,
                  "Cannot store values in a pointer to const types.");
    ptr[offset] = val;
  }
};
template <typename T, int N>
struct Store<cl::sycl::vec<T, N>> {
  template <typename Index, cl::sycl::access::address_space Space>
  void SNN_ALWAYS_INLINE operator()(cl::sycl::multi_ptr<T, Space> ptr,
                                    Index const offset,
                                    cl::sycl::vec<T, N> const val) {
    val.store(0, ptr + offset);
  }
  template <typename U, typename Index>
  void SNN_ALWAYS_INLINE operator()(U* ptr, Index const offset,
                                    cl::sycl::vec<T, N> const val) {
    static_assert(!std::is_const<U>::value,
                  "Cannot store values in a pointer to const types.");
    static constexpr auto address_space =
        cl::sycl::access::address_space::global_space;
    cl::sycl::multi_ptr<T, address_space> mptr(ptr);
    operator()(mptr, offset, val);
  }
};
template <typename T>
struct Store<cl::sycl::vec<T, 1>> {
  template <typename Index, cl::sycl::access::address_space Space>
  void SNN_ALWAYS_INLINE operator()(cl::sycl::multi_ptr<T, Space> ptr,
                                    Index const offset,
                                    cl::sycl::vec<T, 1> const val) {
    *(ptr + offset) = val.s0();
  }
  template <typename U, typename Index>
  void SNN_ALWAYS_INLINE operator()(U* ptr, Index const offset,
                                    cl::sycl::vec<T, 1> val) {
    static_assert(!std::is_const<U>::value,
                  "Cannot store values in a pointer to const types.");
    ptr[offset] = val.s0();
  }
};
}  // namespace io
}  // namespace helpers
}  // namespace sycldnn
#endif  // SYCLDNN_SRC_HELPERS_VECTOR_IO_H_
