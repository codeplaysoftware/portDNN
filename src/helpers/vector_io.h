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
#ifndef PORTDNN_SRC_HELPERS_VECTOR_IO_H_
#define PORTDNN_SRC_HELPERS_VECTOR_IO_H_

#include <CL/sycl.hpp>
#include <type_traits>

#include "portdnn/helpers/macros.h"

#if SNN_ENABLE_USM
#define MULTI_PTR_TEMPLATE_DECL          \
  cl::sycl::access::address_space Space, \
      cl::sycl::access::decorated DecorateAddress
#else
#define MULTI_PTR_TEMPLATE_DECL cl::sycl::access::address_space Space
#endif  // SNN_ENABLE_USM

#if SNN_ENABLE_USM
#define MULTI_PTR_TEMPLATE Space, DecorateAddress
#else
#define MULTI_PTR_TEMPLATE Space
#endif  // SNN_ENABLE_USM

namespace sycldnn {
namespace helpers {

namespace internal {
/**
 * Internal type to signal that an index should be treated as a vector index,
 * not a scalar index.
 *
 * This is implicitly convertible to an underlying Index, so can be passed to
 * functions in the same way as any index.
 */
template <typename Index>
struct AsVecIndex {
  /** Convert this wrapper to the underlying Index type. */
  operator Index() const { return value; }
  /** The actual index value. */
  Index value;
};

/**
 * Adds const to multi_ptr.
 */
template <typename T, MULTI_PTR_TEMPLATE_DECL>
cl::sycl::multi_ptr<T const, MULTI_PTR_TEMPLATE> as_const_ptr(
    cl::sycl::multi_ptr<T, MULTI_PTR_TEMPLATE> ptr) {
  return ptr;
}

}  // namespace internal

/**
 * Load and Store helper structs to load and store SYCL vectors and data types
 * to memory. When possible vload is used to load data into a vector and vstore
 * to store a vector in memory. Also provides operations for non vector types
 * so that a single interface can be used in kernels no matter what the data
 * type.
 */
namespace io {

/**
 * Identifier function to mark an index as a vector index, rather than a scalar
 * index. When used in a Load or Store operation these will use strides of
 * vector size, rather than strides of scalar size to compute offsets from the
 * given pointer.
 */
template <typename Index>
internal::AsVecIndex<Index> as_vec_index(Index val) {
  return {val};
}

/**
 * Load helper for general types.
 *
 * Provides an `operator()` to load a value from an offset to a pointer.
 */
template <typename T>
struct Load {
  template <typename U, typename Index>
  T SNN_ALWAYS_INLINE operator()(U const* const ptr, Index const offset) {
    static_assert(std::is_convertible<U, T>::value,
                  "Type U must be convertible to type T.");
    return ptr[offset];
  }

  template <typename U, typename Index, MULTI_PTR_TEMPLATE_DECL>
  T SNN_ALWAYS_INLINE operator()(cl::sycl::multi_ptr<U, MULTI_PTR_TEMPLATE> ptr,
                                 Index const offset) {
    static_assert(std::is_convertible<U, T>::value,
                  "Type U must be convertible to type T.");
    return *(ptr + offset);
  }
};

/** Load specialisation for vector types. */
template <typename T, int N>
struct Load<cl::sycl::vec<T, N>> {
  static_assert(!std::is_const<T>::value,
                "Cannot load values into a vector of const types.");
  template <typename Index, MULTI_PTR_TEMPLATE_DECL>
  cl::sycl::vec<T, N> SNN_ALWAYS_INLINE
  operator()(cl::sycl::multi_ptr<const T, MULTI_PTR_TEMPLATE> ptr,
             Index const offset) {
    cl::sycl::vec<T, N> result;
    result.load(0, ptr + offset);
    return result;
  }

  template <typename Index, MULTI_PTR_TEMPLATE_DECL>
  cl::sycl::vec<T, N> SNN_ALWAYS_INLINE
  operator()(cl::sycl::multi_ptr<const T, MULTI_PTR_TEMPLATE> ptr,
             internal::AsVecIndex<Index> const offset) {
    cl::sycl::vec<T, N> result;
    result.load(offset, ptr);
    return result;
  }
};

/** Load specialisation to treat one element vectors as scalars. */
template <typename T>
struct Load<cl::sycl::vec<T, 1>> {
  template <typename U, typename Index, MULTI_PTR_TEMPLATE_DECL>
  cl::sycl::vec<T, 1> SNN_ALWAYS_INLINE operator()(
      cl::sycl::multi_ptr<U, MULTI_PTR_TEMPLATE> ptr, Index const offset) {
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

/**
 * Store operator for general types.
 *
 * Provides an `operator()` to store a value to an offset from a pointer.
 */
template <typename T>
struct Store {
  template <typename Index, MULTI_PTR_TEMPLATE_DECL>
  void SNN_ALWAYS_INLINE
  operator()(cl::sycl::multi_ptr<T, MULTI_PTR_TEMPLATE> ptr, Index const offset,
             T const val) {
    *(ptr + offset) = val;
  }

  template <typename U, typename Index>
  void SNN_ALWAYS_INLINE operator()(U* ptr, Index const offset, T const val) {
    static_assert(!std::is_const<U>::value,
                  "Cannot store values in a pointer to const types.");
    ptr[offset] = val;
  }
};

/** Store specialisation for SYCL vectors. */
template <typename T, int N>
struct Store<cl::sycl::vec<T, N>> {
  template <typename Index, MULTI_PTR_TEMPLATE_DECL>
  void SNN_ALWAYS_INLINE
  operator()(cl::sycl::multi_ptr<T, MULTI_PTR_TEMPLATE> ptr, Index const offset,
             cl::sycl::vec<T, N> const val) {
    val.store(0, ptr + offset);
  }

  template <typename Index, MULTI_PTR_TEMPLATE_DECL>
  void SNN_ALWAYS_INLINE operator()(
      cl::sycl::multi_ptr<T, MULTI_PTR_TEMPLATE> ptr,
      internal::AsVecIndex<Index> const offset, cl::sycl::vec<T, N> const val) {
    val.store(offset, ptr);
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

/** Store specialisation to treat single element vectors as scalars. */
template <typename T>
struct Store<cl::sycl::vec<T, 1>> {
  template <typename Index, MULTI_PTR_TEMPLATE_DECL>
  void SNN_ALWAYS_INLINE
  operator()(cl::sycl::multi_ptr<T, MULTI_PTR_TEMPLATE> ptr, Index const offset,
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
#endif  // PORTDNN_SRC_HELPERS_VECTOR_IO_H_
