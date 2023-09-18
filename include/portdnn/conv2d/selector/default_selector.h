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
#ifndef PORTDNN_INCLUDE_CONV2D_DEFAULT_SELECTOR_H_
#define PORTDNN_INCLUDE_CONV2D_DEFAULT_SELECTOR_H_

/**
 * \file
 * Contains the declaration of the
 * \ref sycldnn::conv2d::get_default_selector(const cl::sycl::device &)
 * function.
 */
#include "portdnn/conv2d/selector/selector.h"

#include <CL/sycl.hpp>
#include <memory>

#include "portdnn/export.h"

namespace sycldnn {
namespace conv2d {

/**
 * Gets a suitable default algorithm selector for a particular SYCL device.
 * \param device The SYCL device to make algorithm selections for.
 * \return Returns a unique (owning) pointer to an algorithm selector.
 */
SNN_EXPORT std::unique_ptr<Selector> get_default_selector(
    const cl::sycl::device& device);

}  // namespace conv2d
}  // namespace sycldnn
#endif  // PORTDNN_INCLUDE_CONV2D_DEFAULT_SELECTOR_H_
