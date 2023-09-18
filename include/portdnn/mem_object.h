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
#ifndef PORTDNN_INCLUDE_MEM_OBJECT_H_
#define PORTDNN_INCLUDE_MEM_OBJECT_H_

/**
 * \file
 * Provides the \ref sycldnn::USMMemObject and \ref sycldnn::BufferMemObject
 * classes, along with the \ref sycldnn::make_mem_object helper function.
 */
#include <CL/sycl.hpp>
#include <type_traits>

#include "portdnn/accessor_types.h"
#include "portdnn/helpers/macros.h"
#include "portdnn/helpers/sycl_language_helpers.h"

namespace sycldnn {

/**
 * Forward decleration of USMMemObject.
 */
template <typename T>
class USMMemObject;

/**
 * Forward decleration of BufferMemObject/
 */
template <typename T>
class BufferMemObject;

/**
 * Templated struct to check if type T is of type USMMemObject specialized for
 * U.
 */
template <typename T, typename U>
struct is_usm_obj
    : std::integral_constant<
          bool, std::is_same<USMMemObject<U>,
                             typename std::remove_cv<T>::type>::value> {};

/**
 * Helper for adding template enable_if checks for is_usm_obj.
 */
template <typename T, typename U>
inline constexpr bool is_usm_obj_v = is_usm_obj<T, U>::value;

/**
 * Templated struct to check if type T is of type USMMemObject specialized for
 * U.
 */
template <typename T, typename U>
struct is_buffer_obj
    : std::integral_constant<
          bool, std::is_same<BufferMemObject<U>,
                             typename std::remove_cv<T>::type>::value> {};

/**
 * Helper for adding template enable_if checks for is_usm_obj.
 */
template <typename T, typename U>
inline constexpr bool is_buffer_obj_v = is_buffer_obj<T, U>::value;

/**
 * Templated struct to check if type T is of type USMMemObject or
 * BufferMemObject specialized for U.
 */
template <typename T, typename U>
struct is_mem_obj : std::integral_constant<bool, is_usm_obj_v<T, U> ||
                                                     is_buffer_obj_v<T, U>> {};

/**
 * Helper for adding template enable_if checks for is_mem_obj.
 */
template <typename T, typename U>
inline constexpr bool is_mem_obj_v = is_mem_obj<T, U>::value;

/**
 * Helper function to create BufferMemObjects.
 *
 * This is useful as it can automatically deduce the template types, enabling
 * BufferMemObjects to be constructed as simply as:
 * \code
 *   auto mem_object = make_buffer_mem_object(buffer, size, offset);
 * \endcode
 *
 * \param buffer The SYCL buffer to use as the underlying memory object.
 * \param extent The overall number of elements in the buffer to provide
 *               access to.
 * \param offset The offset from the start of the buffer (in number of
 *               elements) to use as the initial index for the memory object.
 *
 * \return A BufferMemObject that provides access to the given SYCL buffer.
 */
template <typename T>
BufferMemObject<T> make_buffer_mem_object(cl::sycl::buffer<T, 1> buffer,
                                          size_t extent, size_t offset = 0) {
  SNN_ASSERT(buffer.size() >= extent + offset,
             "Buffer must contain at least extent + offset elements");
  return BufferMemObject<T>{buffer, extent, offset};
}

/**
 * Helper function to create USMMemObjects.
 *
 * This is useful as it can automatically deduce the template types, enabling
 * USMMemObjects to be constructed as simply as:
 * \code
 *   auto mem_object = make_usm_mem_object(ptr, size, offset);
 * \endcode
 *
 * \param ptr The SYCL pointer to use as the underlying memory object.
 * \param extent The overall number of elements in the memory block.
 * \param offset The offset from the start of the USM address (in number of
 *               elements).
 *
 * \return A USMMemObject that provides access to the given SYCL USM pointer.
 */
template <typename T>
USMMemObject<T> make_usm_mem_object(T* ptr, size_t extent, size_t offset = 0) {
  return USMMemObject<T>{ptr, extent, offset};
}

/**
 * Helper function to create MemObjects.
 *
 * This is useful as it can automatically deduce the template types, enabling
 * BufferMemObjects to be constructed as simply as:
 * \code
 *   auto mem_object = make_buffer_mem_object(buffer, size, offset);
 * \endcode
 *
 * \param buffer The SYCL buffer to use as the underlying memory object.
 * \param extent The overall number of elements in the buffer to provide
 *               access to.
 * \param offset The offset from the start of the buffer (in number of
 *               elements) to use as the initial index for the memory object.
 *
 * \return A BufferMemObject that provides access to the given SYCL buffer.
 */
template <typename T>
BufferMemObject<T> make_mem_object(cl::sycl::buffer<T, 1> buffer, size_t extent,
                                   size_t offset = 0) {
  return make_buffer_mem_object(buffer, extent, offset);
}

/**
 * Helper function to create MemObjects.
 *
 * This is useful as it can automatically deduce the template types, enabling
 * BufferMemObjects to be constructed as simply as:
 * \code
 *   auto mem_object = make_buffer_mem_object(buffer, size, offset);
 * \endcode
 *
 * \param buffer The SYCL buffer to use as the underlying memory object.
 * \param extent The overall number of elements in the buffer to provide
 *               access to.
 * \param offset The offset from the start of the buffer (in number of
 *               elements) to use as the initial index for the memory object.
 *
 * \return A BufferMemObject that provides access to the given SYCL buffer.
 */
template <typename T, typename = std::enable_if<std::is_const_v<T>>>
BufferMemObject<T> make_mem_object(
    cl::sycl::buffer<typename std::remove_const<T>::type, 1> buffer,
    size_t extent, size_t offset = 0) {
  return make_buffer_mem_object(buffer.template reinterpret<T>(), extent,
                                offset);
}

/**
 * Helper function to create USMMemObjects.
 *
 * This is useful as it can automatically deduce the template types, enabling
 * USMMemObjects to be constructed as simply as:
 * \code
 *   auto mem_object = make_usm_mem_object(ptr, size, offset);
 * \endcode
 *
 * \param ptr The SYCL pointer to use as the underlying memory object.
 * \param extent The overall number of elements in the memory block.
 * \param offset The offset from the start of the USM address (in number of
 *               elements).
 *
 * \return A USMMemObject that provides access to the given SYCL USM pointer.
 */
template <typename T>
USMMemObject<T> make_mem_object(T* ptr, size_t extent, size_t offset = 0) {
  return make_usm_mem_object(ptr, extent, offset);
}

/**
 * The implementation of USMMemObject for SYCL pointers.
 */
template <typename T>
class USMMemObject {
 private:
  /** The datatype stored in the memory object. */
  using DataType = T;
  /** Alias for the SYCL command group handler. */
  using Handler = cl::sycl::handler;

 public:
  /**
   * Construct a USMMemObject wrapper around the given SYCL pointer.
   *
   * \param ptr SYCL pointer to use as underlying memory.
   * \param extent The overall number of elements the pointer spans.
   * \param offset The offset from the start of the pointer (in number of
   *               elements) to use as the initial index for the memory
   */
  USMMemObject(DataType* ptr, size_t extent, size_t offset)
      : ptr_(ptr), extent_(extent), offset_(offset){};

  /**
   * Returns the underlying USM pointer
   *
   * \return The underlying USM pointer.
   */
  DataType* get_pointer() const { return ptr_; }

  /**
   * Get the extent of this USMMemObject. This is the number of elements in the
   * SYCL pointer that have been allocated.
   * \return The extent of this USMMemObject.
   */
  size_t get_extent() const { return extent_; }

  /**
   * Get the offset of this USMMemObject.
   * \return The number of elements offset from the start of the pointer.
   */
  size_t get_offset() const { return offset_; }

  /**
   * Get a read only generic memory object to the underlying memory object.
   *
   * \param cgh The SYCL command group handler.
   * \return A ReadMem wrapper containing a SYCL pointer.
   */
  ReadMem<T, /*IsUSM*/ true> read_mem(Handler& cgh) {
    return {ptr_, cgh, extent_, offset_};
  }

  /**
   * Get a read-write generic memory object to the underlying memory object.
   *
   * \param cgh The SYCL command group handler.
   * \return A ReadMem wrapper containing a SYCL pointer.
   */
  template <typename U = DataType,
            typename = std::enable_if<std::is_same<U, DataType>::value>>
  typename std::enable_if_t<!std::is_const<U>::value,
                            ReadWriteMem<U, /*IsUSM*/ true>>
  read_write_mem(Handler& cgh) {
    return {ptr_, cgh, extent_, offset_};
  }

  /**
   * Get a write only generic memory object to the underlying memory object.
   *
   * \param cgh The SYCL command group handler.
   * \return A WriteMem wrapper containing a SYCL pointer.
   */
  template <typename U = DataType,
            typename = std::enable_if<std::is_same<U, DataType>::value>>
  typename std::enable_if_t<!std::is_const<U>::value,
                            WriteMem<U, /*IsUSM*/ true>>
  write_mem(Handler& cgh) {
    return {ptr_, cgh, extent_, offset_};
  }

  /**
   * Return a new USMMemObject with a pointer casted to a new type.
   * \return Casted USMMemObject.
   */
  template <typename NewDataType,
            typename std::enable_if<sizeof(NewDataType) == sizeof(DataType),
                                    int>::type = 0>
  USMMemObject<NewDataType> cast() {
    return USMMemObject<NewDataType>(reinterpret_cast<NewDataType*>(ptr_),
                                     extent_, offset_);
  }

  /**
   * Return the same USMMemObject as a read-only one.
   * \return Read-only USMMemObject.
   */
  USMMemObject<DataType const> as_const() {
    return this->cast<DataType const>();
  }

 private:
  /** The underlying SYCL pointer. */
  DataType* ptr_;
  /** The number of elements the pointer spans. */
  size_t extent_;
  /** The offset from the start of the pointer (in number of elements). */
  size_t offset_;
};

template <typename T>
class BufferMemObject {
 private:
  /** The datatype stored in the memory object. */
  using DataType = T;
  using Buffer = cl::sycl::buffer<T, 1>;
  using Handler = cl::sycl::handler;

 public:
  /**
   * Construct a BufferMemObject wrapper around the given SYCL buffer.
   *
   * \param buffer SYCL buffer to use as underlying memory.
   * \param extent The overall number of elements in the buffer to provide
   *               access to.
   * \param offset The offset from the start of the buffer (in number of
   *               elements) to use as the initial index for the memory
   * object.
   */
  BufferMemObject(Buffer buffer, size_t extent, size_t offset)
      : buffer_(buffer), extent_(extent), offset_(offset) {
    SNN_ASSERT(buffer_.size() >= extent_ + offset_,
               "Buffer must contain at least extent + offset elements");
  };

  /**
   * Get a reference to the SYCL buffer referred to by this MemObject.
   * \return A reference to the SYCL buffer.
   */
  Buffer const& get_buffer() const { return buffer_; }

  /**
   * Get the extent of this MemObject. This is the number of elements in the
   * SYCL buffer that are available to a user when a SYCL accessor is
   * requested.
   * \return The extent of this MemObject.
   */
  size_t get_extent() const { return extent_; }

  /**
   * Get the offset of this MemObject into its Buffer.
   * \return The number of elements offset from the start of the Buffer.
   */
  size_t get_offset() const { return offset_; }

  /**
   * Get a read only accessor to the underlying memory object.
   *
   * \param cgh The SYCL command group handler to bind the buffer accessor to.
   * \return A ReadAccessor wrapper containing a SYCL accessor.
   */
  ReadAccessor<DataType> read_accessor(Handler& cgh) {
    return {buffer_, cgh, extent_, offset_};
  }

  /**
   * Get a read-write accessor to the underlying memory object.
   *
   * \param cgh The SYCL command group handler to bind the buffer accessor to.
   * \return A ReadWriteAccessor wrapper containing a SYCL accessor.
   */
  template <typename U = DataType,
            typename = std::enable_if<std::is_same<U, DataType>::value>>
  typename std::enable_if_t<!std::is_const<U>::value, ReadWriteAccessor<U>>
  read_write_accessor(Handler& cgh) {
    return {buffer_, cgh, extent_, offset_};
  }

  /**
   * Get a write only accessor to the underlying memory object.
   *
   * \param cgh The SYCL command group handler to bind the buffer accessor to.
   * \return A WriteAccessor wrapper containing a SYCL accessor.
   */
  template <typename U = DataType,
            typename = std::enable_if<std::is_same<U, DataType>::value>>
  typename std::enable_if_t<!std::is_const<U>::value, WriteAccessor<U>>
  write_accessor(Handler& cgh) {
    return {buffer_, cgh, extent_, offset_};
  }

  /**
   * Get a read only generic memory object to the underlying memory object.
   *
   * \param cgh The SYCL command group handler.
   * \return A ReadMem wrapper containing a SYCL buffer.
   */
  ReadMem<T, /*IsUSM*/ false> read_mem(Handler& cgh) {
    return {buffer_, cgh, extent_, offset_};
  }

  /**
   * Get a read-write generic memory object to the underlying memory object.
   *
   * \param cgh The SYCL command group handler.
   * \return A ReadMem wrapper containing a SYCL buffer.
   */
  template <typename U = DataType,
            typename = std::enable_if<std::is_same<U, DataType>::value>>
  typename std::enable_if_t<!std::is_const<U>::value,
                            ReadWriteMem<U, /*IsUSM*/ false>>
  read_write_mem(Handler& cgh) {
    return {buffer_, cgh, extent_, offset_};
  }

  /**
   * Get a write only generic memory object to the underlying memory object.
   *
   * \param cgh The SYCL command group handler.
   * \return A ReadMem wrapper containing a SYCL buffer.
   */
  template <typename U = DataType,
            typename = std::enable_if<std::is_same<U, DataType>::value>>
  typename std::enable_if_t<!std::is_const<U>::value,
                            WriteMem<U, /*IsUSM*/ false>>
  write_mem(Handler& cgh) {
    return {buffer_, cgh, extent_, offset_};
  }

  /**
   * Return a new MemObject with a buffer casted to a new type.
   * \return Casted BufferMemObject.
   */
  template <typename NewDataType,
            typename std::enable_if<sizeof(NewDataType) == sizeof(DataType),
                                    int>::type = 0>
  BufferMemObject<NewDataType> cast() {
    return BufferMemObject<NewDataType>(
        buffer_.template reinterpret<NewDataType>(), extent_, offset_);
  }

  /**
   * Return a new MemObject with a buffer casted to a const of the DataType.
   * \return const BufferMemObject.
   */
  BufferMemObject<DataType const> as_const() {
    return this->cast<DataType const>();
  }

 private:
  /** The underlying SYCL buffer. */
  Buffer buffer_;
  /** The number of elements to expose in the SYCL buffer. */
  size_t extent_;
  /** The offset from the start of the buffer (in elements). */
  size_t offset_;
};

}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_MEM_OBJECT_H_
