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
#ifndef PORTDNN_INCLUDE_BACKEND_DEVICE_MEM_POINTER_H_
#define PORTDNN_INCLUDE_BACKEND_DEVICE_MEM_POINTER_H_

#include "portdnn/helpers/macros.h"

#include <CL/sycl.hpp>

/**
 * \file
 * Contains the DeviceMemPointer class which wraps a SYCL buffer and an offset.
 */

namespace sycldnn {
namespace backend {

/**
 * The DeviceMemPointer class mimics a pointer into a SYCL buffer.
 *
 * The pointer type supports simple arithmetic which changes the offset into the
 * buffer. Access to the buffer and offset are provided through get_buffer() and
 * get_offset().
 */
template <typename T>
struct DeviceMemPointer {
  /** The type of the SYCL buffer pointed to by this DeviceMemPointer. */
  using Buffer = cl::sycl::buffer<T, 1>;

  /**
   * Default constructor creates a DeviceMemPointer to a dummy SYCL buffer.
   *
   * Note that SYCL buffers are not default constructible, so a size of 1 must
   * be used. This will not actually cause any allocation on the device unless
   * the buffer is actually used in a kernel.
   */
  DeviceMemPointer() : buffer{cl::sycl::range<1>{1u}}, offset{} {}

  /** Default copy constructor and assignment. */
  SNN_DEFAULT_COPY(DeviceMemPointer);

  /** Default move constructor and assignment. */
  SNN_DEFAULT_MOVE(DeviceMemPointer);

  /**
   * Construct a DeviceMemPointer to a SYCL buffer containing the elements
   * currently available in the specified host memory. The DeviceMemPointer will
   * not take ownership of the pointer, but will assume that the data within the
   * pointer is available for the duration of its lifetime.
   *
   * \param host_pointer Pointer to host memory to copy into the buffer.
   * \param n_elements Size of buffer to create.
   */
  explicit DeviceMemPointer(T* host_pointer, size_t n_elements)
      : buffer{host_pointer,
               cl::sycl::range<1>{n_elements},
               {cl::sycl::property::buffer::use_host_ptr{}}},
        offset{} {}

  /**
   * Construct a DeviceMemPointer to a SYCL buffer containing the given number
   * of elements.
   *
   * \param n_elements Size of the buffer to create.
   */
  explicit DeviceMemPointer(size_t n_elements)
#ifdef SYCL_IMPLEMENTATION_ONEAPI
      : buffer{cl::sycl::range<1>{n_elements}}, offset{} {
  }
#else
      : offset{} {
    if (n_elements == 0)
      buffer = cl::sycl::buffer<T>();
    else
      buffer = cl::sycl::buffer<T>(cl::sycl::range<1>(n_elements));
  }
#endif

  /**
   * Construct a DeviceMemPointer to point to a known offset into the
   * given SYCL buffer.
   *
   * \param buffer SYCL buffer to point to.
   * \param offset Offset in number of elements into the SYCL buffer.
   */
  explicit DeviceMemPointer(Buffer buffer, size_t offset)
      : buffer{std::move(buffer)}, offset{offset} {}

  /**
   * Convert a DeviceMemPointer<T> into a DeviceMemPointer<T const>, to match
   * the conversion semantics of raw pointers.
   */
  operator DeviceMemPointer<T const>() const {
    return DeviceMemPointer<T const>{
        buffer.template reinterpret<T const, 1>(buffer.get_count()), offset};
  }

  /**
   * Increment the offset into this DeviceMemPointer's SYCL buffer.
   * \param increment Number of elements to increase the offset by.
   * \return A reference to this DeviceMemPointer.
   */
  DeviceMemPointer& operator+=(size_t increment) {
    offset += increment;
    return *this;
  }

  /**
   * Add an offset into this DeviceMemPointer's SYCL buffer.
   * \param lhs DeviceMemPointer to a SYCL buffer.
   * \param rhs Integer value to increment the offset.
   * \return A DeviceMemPointer to the original SYCL buffer, with an increased
   * offset.
   */
  friend DeviceMemPointer operator+(DeviceMemPointer lhs, size_t rhs) {
    lhs += rhs;
    return lhs;
  }

  /**
   * Add an offset into this DeviceMemPointer's SYCL buffer.
   * \param lhs Integer value to increment the offset.
   * \param rhs DeviceMemPointer to a SYCL buffer.
   * \return A DeviceMemPointer to the original SYCL buffer, with an increased
   * offset.
   */
  friend DeviceMemPointer operator+(size_t lhs, DeviceMemPointer rhs) {
    rhs += lhs;
    return rhs;
  }

  /**
   * Get a reference to the DeviceMemPointer's SYCL buffer.
   * \return A reference to the SYCL buffer.
   */
  Buffer& get_buffer() { return buffer; }

  /**
   * Get a const reference to the DeviceMemPointer's SYCL buffer.
   * \return A reference to the SYCL buffer.
   */
  Buffer const& get_buffer() const { return buffer; }

  /**
   * Get the number of elements offset into the SYCL buffer.
   * \return The offset into the SYCL buffer.
   */
  size_t get_offset() const { return offset; }

 private:
  /** The SYCL buffer that this DeviceMemPointer points to. */
  Buffer buffer;

  /** The offset into the SYCL buffer. */
  size_t offset;
};

}  // namespace backend
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_BACKEND_DEVICE_MEM_POINTER_H_
