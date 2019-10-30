/*
 * Copyright 2019 Codeplay Software Ltd
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
#ifndef SYCLDNN_INCLUDE_MEM_OBJECT_H_
#define SYCLDNN_INCLUDE_MEM_OBJECT_H_

/**
 * \file
 * Provides the \ref sycldnn::MemObject and \ref sycldnn::BaseMemObject classes,
 * along with the \ref sycldnn::make_mem_object helper function.
 */
#include "sycldnn/accessor_types.h"
#include "sycldnn/helpers/macros.h"

#include <CL/sycl.hpp>

namespace sycldnn {

/**
 * Abstract MemObject base class.
 *
 * Provides virtual functions to get different Accessor types from the
 * underlying memory object.
 */
template <typename T>
struct BaseMemObject {
  /** Alias for the SYCL command group handler. */
  using Handler = cl::sycl::handler;

  /** Virtual destructor. */
  virtual ~BaseMemObject() = default;

  /**
   * Get a read only accessor to the underlying memory object.
   *
   * \param cgh The SYCL command group handler to bind the buffer accessor to.
   * \return A ReadAccessor wrapper containing a SYCL accessor.
   */
  virtual ReadAccessor<T> read_accessor(Handler& cgh) = 0;

  /**
   * Get a read-write accessor to the underlying memory object.
   *
   * \param cgh The SYCL command group handler to bind the buffer accessor to.
   * \return A ReadWriteAccessor wrapper containing a SYCL accessor.
   */
  virtual ReadWriteAccessor<T> read_write_accessor(Handler& cgh) = 0;

  /**
   * Get a write only accessor to the underlying memory object.
   *
   * \param cgh The SYCL command group handler to bind the buffer accessor to.
   * \return A WriteAccessor wrapper containing a SYCL accessor.
   */
  virtual WriteAccessor<T> write_accessor(Handler& cgh) = 0;
};

/**
 * Specialisation of BaseMemObject abstract class for const datatypes.
 *
 * If the DataType of a memory object is `const` it cannot be written to, so
 * remove the ability to create write accessors to the memory object.
 */
template <typename T>
struct BaseMemObject<T const> {
  /** Alias for the SYCL command group handler. */
  using Handler = cl::sycl::handler;

  /** Virtual destructor. */
  virtual ~BaseMemObject() = default;

  /** \copydoc BaseMemObject<T>::read_accessor */
  virtual ReadAccessor<T const> read_accessor(Handler& cgh) = 0;
};

/**
 * The implementation of BaseMemObject for SYCL buffers.
 */
template <typename T, typename Alloc>
struct MemObject final : public BaseMemObject<T> {
  /** The datatype stored in the memory object. */
  using DataType = T;
  /** The allocator type of the underlying SYCL buffer. */
  using AllocType = Alloc;
  /** Alias for the underlying SYCL buffer type. */
  using Buffer = cl::sycl::buffer<DataType, 1, AllocType>;
  /** \copydoc BaseMemObject<T>::Handler */
  using typename BaseMemObject<DataType>::Handler;

 public:
  /**
   * Construct a MemObject wrapper around the given SYCL buffer.
   *
   * \param buffer SYCL buffer to use as underlying memory.
   * \param extent The overall number of elements in the buffer to provide
   *               access to.
   * \param offset The offset from the start of the buffer (in number of
   *               elements) to use as the initial index for the memory object.
   */
  MemObject(Buffer buffer, size_t extent, size_t offset)
      : buffer_{buffer}, extent_{extent}, offset_{offset} {}

  /** \copydoc BaseMemObject<T>::read_accessor */
  ReadAccessor<DataType> read_accessor(Handler& cgh) override {
    return {buffer_, cgh, extent_, offset_};
  }

  /** \copydoc BaseMemObject<T>::read_write_accessor */
  ReadWriteAccessor<DataType> read_write_accessor(Handler& cgh) override {
    return {buffer_, cgh, extent_, offset_};
  }

  /** \copydoc BaseMemObject<T>::write_accessor */
  WriteAccessor<DataType> write_accessor(Handler& cgh) override {
    return {buffer_, cgh, extent_, offset_};
  }

  /**
   * Get a reference to the SYCL buffer referred to by this MemObject.
   * \return A reference to the SYCL buffer.
   */
  Buffer const& get_buffer() const { return buffer_; }

  /**
   * Get the extent of this MemObject. This is the number of elements in the
   * SYCL buffer that are available to a user when a SYCL accessor is requested.
   * \return The extent of this MemObject.
   */
  size_t get_extent() const { return extent_; }

  /**
   * Get the offset of this MemObject into its Buffer.
   * \return The number of elements offset from the start of the Buffer.
   */
  size_t get_offset() const { return offset_; }

 private:
  /** The underlying SYCL buffer. */
  Buffer buffer_;
  /** The number of elements to expose in the SYCL buffer. */
  size_t extent_;
  /** The offset from the start of the buffer (in elements). */
  size_t offset_;
};

/**
 * Specialisation of \ref MemObject for `const` DataTypes.
 *
 * The specialisation restricts access to read only, as the underlying data type
 * is constant.
 */
template <typename T, typename Alloc>
struct MemObject<T const, Alloc> final : public BaseMemObject<T const> {
  /** The datatype stored in the memory object. */
  using DataType = T const;
  /** The allocator type of the underlying SYCL buffer. */
  using AllocType = Alloc;
  /** Alias for the underlying SYCL buffer type. */
  using Buffer = cl::sycl::buffer<DataType, 1, AllocType>;
  /** \copydoc BaseMemObject<T const>::Handler */
  using typename BaseMemObject<DataType>::Handler;

 public:
  /** \copydoc MemObject<T>::MemObject */
  MemObject(Buffer buffer, size_t extent, size_t offset)
      : buffer_{buffer}, extent_{extent}, offset_{offset} {}

  /** \copydoc BaseMemObject<T>::read_accessor */
  ReadAccessor<DataType> read_accessor(Handler& cgh) override {
    return {buffer_, cgh, extent_, offset_};
  }

  /** \copydoc MemObject<T>::get_buffer  */
  Buffer const& get_buffer() const { return buffer_; }

  /** \copydoc MemObject<T>::get_extent  */
  size_t get_extent() const { return extent_; }

  /** \copydoc MemObject<T>::get_offset  */
  size_t get_offset() const { return offset_; }

 private:
  /** The underlying SYCL buffer. */
  Buffer buffer_;
  /** The number of elements to expose in the SYCL buffer. */
  size_t extent_;
  /** The offset from the start of the buffer (in elements). */
  size_t offset_;
};

/**
 * Helper function to create MemObjects.
 *
 * This is useful as it can automatically deduce the template types, enabling
 * MemObjects to be constructed as simply as:
 * \code
 *   auto mem_object = make_mem_object(buffer, size, offset);
 * \endcode
 *
 * \param buffer The SYCL buffer to use as the underlying memory object.
 * \param extent The overall number of elements in the buffer to provide
 *               access to.
 * \param offset The offset from the start of the buffer (in number of
 *               elements) to use as the initial index for the memory object.
 *
 * \return A MemObject that provides access to the given SYCL buffer.
 */
template <typename T, typename Alloc>
MemObject<T, Alloc> make_mem_object(cl::sycl::buffer<T, 1, Alloc> buffer,
                                    size_t extent, size_t offset) {
  SNN_ASSERT(buffer.get_count() >= extent + offset,
             "Buffer must contain at least extent + offset elements");
  return MemObject<T, Alloc>{buffer, extent, offset};
}

}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_MEM_OBJECT_H_
