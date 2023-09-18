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
#ifndef PORTDNN_INCLUDE_CONV2D_LAUNCH_H_
#define PORTDNN_INCLUDE_CONV2D_LAUNCH_H_

/**
 * \file
 * Implements the \ref sycldnn::conv2d::launch() function, which specialises the
 * call of sycldnn::conv2d::sublaunch() based on the backend (USM/non-USM)
 */

#include "portdnn/backend/backend_helpers.h"
#include "portdnn/conv2d/params.h"
#include "portdnn/conv2d/selector/selector.h"
#include "portdnn/internal/conv2d/launch.h"
#include "portdnn/status.h"

namespace sycldnn {
namespace conv2d {
/**
 * Launch a 2D convolution, with the implementation chosen by the Selector.
 *
 * The selector will be used to select which implementation to use, and the
 * corresponding kernels will be launched. If any additional temporary memory is
 * required then it will be allocated through the backend.
 *
 * \param input A pointer to the memory representing the input tensor.
 * \param filter A pointer to the memory representing the tensor of filter
 *               coefficients.
 * \param output A pointer to the memory representing the output tensor.
 * \param params The convolution parameters, which describe the tensor shapes
 *               and convolution strides.
 * \param selector An instance of \ref sycldnn::conv2d::Selector, used to guide
 *                 the selection of the most appropriate convolution algorithm
 *                 for a specific target platform or problem size.
 * \param backend The backend implementation, used to provide optimized matrix
 *                multiplies and to map between pointer representations.
 * \param workspace Optional pointer to a workspace buffer for use whenever
 *                  temporary memory is required.
 * \param workspace_size The number of elements available in the workspace
 *                       buffer.
 * \return Returns an SNNStatus containing the SYCL event tied to the kernel
 * launches and a StatusCode enum showing if the launch was OK or whether it
 * encountered some problem.
 */
template <typename T, typename ConvType, typename Backend,
          typename = typename std::enable_if<
              sycldnn::backend::is_buffer_backend_v<Backend>>::type>
SNNStatus launch(typename Backend::template pointer_type<T const> input,
                 typename Backend::template pointer_type<T const> filter,
                 typename Backend::template pointer_type<T> output,
                 Conv2DParams const& params, Selector& selector,
                 Backend& backend,
                 typename Backend::template pointer_type<T> workspace,
                 size_t workspace_size) {
  return sublaunch<T, ConvType, Backend>(input, filter, output, params,
                                         selector, backend, workspace,
                                         workspace_size, {});
}

/**
 * Launch a 2D convolution, with the implementation chosen by the Selector.
 *
 * The selector will be used to select which implementation to use, and the
 * corresponding kernels will be launched. If any additional temporary memory is
 * required then it will be allocated through the backend.
 *
 * \param input A pointer to the memory representing the input tensor.
 * \param filter A pointer to the memory representing the tensor of filter
 *               coefficients.
 * \param output A pointer to the memory representing the output tensor.
 * \param params The convolution parameters, which describe the tensor shapes
 *               and convolution strides.
 * \param selector An instance of \ref sycldnn::conv2d::Selector, used to guide
 *                 the selection of the most appropriate convolution algorithm
 *                 for a specific target platform or problem size.
 * \param backend The backend implementation, used to provide optimized matrix
 *                multiplies and to map between pointer representations.
 * \param workspace Optional pointer to a workspace buffer for use whenever
 *                  temporary memory is required.
 * \param workspace_size The number of elements available in the workspace
 *                       buffer.
 * \param events Optional vector of
 *               events which the convolution will wait on before launching the
 *               kernels, required for USM
 * \return Returns an SNNStatus containing the SYCL
 * event tied to the kernel launches and a StatusCode enum showing if the launch
 * was OK or whether it encountered some problem.
 */

template <typename T, typename ConvType, typename Backend,
          typename = typename std::enable_if<
              sycldnn::backend::is_usm_backend_v<Backend>>::type>
SNNStatus launch(typename Backend::template pointer_type<T const> input,
                 typename Backend::template pointer_type<T const> filter,
                 typename Backend::template pointer_type<T> output,
                 Conv2DParams const& params, Selector& selector,
                 Backend& backend,
                 typename Backend::template pointer_type<T> workspace,
                 size_t workspace_size,
                 const std::vector<cl::sycl::event>& events = {}) {
  return sublaunch<T, ConvType, Backend>(input, filter, output, params,
                                         selector, backend, workspace,
                                         workspace_size, events);
}

}  // namespace conv2d
}  // namespace sycldnn
#endif  // PORTDNN_INCLUDE_CONV2D_LAUNCH_H_
