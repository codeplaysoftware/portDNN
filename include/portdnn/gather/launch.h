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
#ifndef PORTDNN_INCLUDE_GATHER_LAUNCH_H_
#define PORTDNN_INCLUDE_GATHER_LAUNCH_H_

/**
 * \file
 * Implements the \ref sycldnn::gather::launch() function, which
 * asynchronously dispatches a SYCL kernel to compute a gather operation
 * along a given single dimension of a N-dimensional tensor.
 */
#include "portdnn/backend/backend_helpers.h"
#include "portdnn/gather/params.h"
#include "portdnn/gather/sizes.h"
#include "portdnn/helpers/macros.h"
#include "portdnn/internal/gather/launch.h"
#include "portdnn/mem_object.h"
#include "portdnn/status.h"

namespace sycldnn {
/** Namespace containing the gather operator. */
namespace gather {

/**
 * Launch the gather operation kernel.
 *
 * Gather is applied on a given axis of an input of any rank r>=1 given a set of
 * indices of rank q>=1. It takes the input entries along the axis indexed by
 * the indices values and concatenates them in an output tensor of rank q +
 * (r-1).
 *
 * \tparam T          The data type of the input tensor.
 * \tparam Index      The data type of the indices tensor.
 * \tparam Backend    The type of backend.
 * \param input       A pointer to memory representing the input tensor.
 * \param indices     A pointer to memory representing the indices tensor.
 * \param output      A pointer to memory representing the output tensor.
 * \param params      The gather params.
 * \param backend     The backend for mapping between pointer representations.
 * \return SNNStatus  Returns a SNNStatus containing the SYCL event tied to
 *                    the kernel launches and a StatusCode enum showing if the
 *                    launch was OK or whether it encountered some problem.
 */
template <typename T, typename Index, typename Backend,
          typename = typename std::enable_if<
              sycldnn::backend::is_buffer_backend_v<Backend>>::type>
SNNStatus launch(typename Backend::template pointer_type<T const> input,
                 typename Backend::template pointer_type<Index const> indices,
                 typename Backend::template pointer_type<T> output,
                 const GatherParams& params, Backend& backend) {
  return internal::sublaunch<T, Index>(input, indices, output, params, backend,
                                       {});
}

/**
 * Launch the gather operation kernel.
 *
 * Gather is applied on a given axis of an input of any rank r>=1 given a set of
 * indices of rank q>=1. It takes the input entries along the axis indexed by
 * the indices values and concatenates them in an output tensor of rank q +
 * (r-1).
 *
 * \tparam T          The data type of the input tensor.
 * \tparam Index      The data type of the indices tensor.
 * \tparam Backend    The type of backend.
 * \param input       A pointer to memory representing the input tensor.
 * \param indices     A pointer to memory representing the indices tensor.
 * \param output      A pointer to memory representing the output tensor.
 * \param params      The gather params.
 * \param backend     The backend for mapping between pointer representations.
 * \param events     Events which should be completed before the operation
 * \return SNNStatus  Returns a SNNStatus containing the SYCL event tied to
 *                    the kernel launches and a StatusCode enum showing if the
 *                    launch was OK or whether it encountered some problem.
 */
template <typename T, typename Index, typename Backend,
          typename = typename std::enable_if<
              sycldnn::backend::is_usm_backend_v<Backend>>::type>
SNNStatus launch(typename Backend::template pointer_type<T const> input,
                 typename Backend::template pointer_type<Index const> indices,
                 typename Backend::template pointer_type<T> output,
                 const GatherParams& params, Backend& backend,
                 const std::vector<cl::sycl::event>& events = {}) {
  return internal::sublaunch<T, Index>(input, indices, output, params, backend,
                                       events);
}

}  // namespace gather
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_GATHER_LAUNCH_H_
