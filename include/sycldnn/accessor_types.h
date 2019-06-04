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
#ifndef SYCLDNN_INCLUDE_ACCESSOR_TYPES_H_
#define SYCLDNN_INCLUDE_ACCESSOR_TYPES_H_

/**
 * \file
 * Provides the \ref sycldnn::ReadAccessor, sycldnn::WriteAccessor and
 * sycldnn::ReadWriteAccessor aliases.
 */
#include <CL/sycl.hpp>

namespace sycldnn {
/** Local memory accessor for a given dimension of type T. */
template <typename T, int Dimension = 1>
using LocalAccessor =
    cl::sycl::accessor<T, Dimension, cl::sycl::access::mode::read_write,
                       cl::sycl::access::target::local>;

/**
 * SYCL Accessor wrapper.
 *
 * Provides a simple constructor for accessors, and a unified way of ensuring
 * that offsets into buffers are included in kernels.
 */
template <typename T, cl::sycl::access::mode Mode>
struct BaseAccessor {
 private:
  static auto constexpr GlobalTarget = cl::sycl::access::target::global_buffer;
  static auto constexpr GlobalSpace =
      cl::sycl::access::address_space::global_space;

  /** Alias for a global SYCL pointer. */
  using MultiPtr = cl::sycl::multi_ptr<T, GlobalSpace>;
  /** Alias for a SYCL command group handler. */
  using Handler = cl::sycl::handler;
  /** Alias for a SYCL buffer matching the accessor type. */
  template <typename Alloc>
  using Buffer = cl::sycl::buffer<T, 1, Alloc>;
  /** The underlying SYCL accessor type. */
  using Accessor = cl::sycl::accessor<T, 1, Mode, GlobalTarget>;

 public:
  /**
   * Contruct a BaseAccessor from a SYCL buffer and command group handler.
   * \param buf The SYCL buffer to construct an accessor from.
   * \param cgh The SYCL command group handler to bind the accessor to.
   * \param extent The number of elements in the buffer to provide access to.
   * \param offset The offset from the start of the buffer.
   */
  template <typename Alloc>
  BaseAccessor(Buffer<Alloc>& buf, Handler& cgh, size_t extent, size_t offset)
      : acc_{buf, cgh, cl::sycl::range<1>{extent}, cl::sycl::id<1>{offset}},
        offset_{offset} {}

  /**
   * Get the underlying pointer from the accessor.
   * \return A global pointer to the underlying memory.
   */
  MultiPtr get_pointer() const { return acc_.get_pointer() + offset_; }

  /**
   * Get a reference to the underlying SYCL accessor.
   * \return A reference to the underlying SYCL accessor.
   */
  Accessor& get_accessor() { return acc_; }

  /**
   * Get a const reference to the underlying SYCL accessor.
   * \return A const reference to the underlying SYCL accessor.
   */
  Accessor const& get_accessor() const { return acc_; }

 private:
  /** The SYCL accessor. */
  Accessor acc_;
  /**
   * The offset from the start of the SYCL buffer in elements.
   *
   * NB. The accessor stores these offsets itself, but it might store all
   * dimensions which means that when used in a kernel more registers are
   * needed than are actually required. By storing the offset separately we can
   * ensure that only a single offset value is used in the kernel.
   */
  size_t offset_;
};

/** Read only accessor for a 1D buffer of type T. */
template <typename T>
using ReadAccessor = BaseAccessor<T, cl::sycl::access::mode::read>;

/** Write only accessor for a 1D buffer of type T. */
template <typename T>
using WriteAccessor = BaseAccessor<T, cl::sycl::access::mode::discard_write>;

/** Read-write accessor for a 1D buffer of type T. */
template <typename T>
using ReadWriteAccessor = BaseAccessor<T, cl::sycl::access::mode::read_write>;

}  // namespace sycldnn
#endif  // SYCLDNN_INCLUDE_ACCESSOR_TYPES_H_
