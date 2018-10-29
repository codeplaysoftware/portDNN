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
/** Read only placeholder SYCL accessor for a 1D buffer of type T. */
template <typename T>
using ReadAccessor = cl::sycl::accessor<T, 1, cl::sycl::access::mode::read,
                                        cl::sycl::access::target::global_buffer,
                                        cl::sycl::access::placeholder::true_t>;
/** Write only placeholder SYCL accessor for a 1D buffer of type T. */
template <typename T>
using WriteAccessor =
    cl::sycl::accessor<T, 1, cl::sycl::access::mode::discard_write,
                       cl::sycl::access::target::global_buffer,
                       cl::sycl::access::placeholder::true_t>;
/** Read-write placeholder SYCL accessor for a 1D buffer of type T. */
template <typename T>
using ReadWriteAccessor =
    cl::sycl::accessor<T, 1, cl::sycl::access::mode::read_write,
                       cl::sycl::access::target::global_buffer,
                       cl::sycl::access::placeholder::true_t>;

/** Local memory accessor for a given dimension of type T. */
template <typename T, int Dimension = 1>
using LocalAccessor =
    cl::sycl::accessor<T, Dimension, cl::sycl::access::mode::read_write,
                       cl::sycl::access::target::local>;

}  // namespace sycldnn
#endif  // SYCLDNN_INCLUDE_ACCESSOR_TYPES_H_
