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
#ifndef SYCLDNN_INCLUDE_GATHER_LAUNCH_H_
#define SYCLDNN_INCLUDE_GATHER_LAUNCH_H_

/**
 * \file
 * Implements the \ref sycldnn::gather::launch() function, which
 * asynchronously dispatches a SYCL kernel to compute a gather operation
 * along a given single dimension of a N-dimensional tensor.
 */
#include "sycldnn/gather/params.h"
#include "sycldnn/gather/sizes.h"
#include "sycldnn/helpers/macros.h"
#include "sycldnn/internal/gather/launch.h"
#include "sycldnn/mem_object.h"
#include "sycldnn/status.h"

namespace sycldnn {
/** Namespace containing the gather operator. */
namespace gather {
/** Namespace containing internal implementation details for gather. */
namespace internal {

/**
 * Validate that the user-provided gather parameters are consistent with what
 * is expected by SYCL-DNN.
 *
 * If compiled with asserts, any invalid parameter will fail with an assert.
 * Otherwise a status code \ref StatusCode::InvalidParameter will be returned.
 *
 * \param params  Gather parameters to validate.
 * \return        A SNNStatus object containing either \ref StatusCode::OK if
 * all parameters are valid, or \ref StatusCode::InvalidParameter otherwise.
 */
SNNStatus inline validate_params(GatherParams const& params) {
  int input_rank = static_cast<int>(params.input_dims.size());

  SNN_VALIDATE_PARAM(params.axis < input_rank,
                     "The axis should be < input rank");
  SNN_VALIDATE_PARAM(params.axis >= -input_rank,
                     "The axis should be >= -input rank");
  SNN_VALIDATE_PARAM(params.indices_dims.size() != 0,
                     "The indices should be of dimension >=1");

  return StatusCode::OK;
}

}  // namespace internal

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
template <typename T, typename Index, typename Backend>
SNNStatus launch(typename Backend::template pointer_type<T const> input,
                 typename Backend::template pointer_type<Index const> indices,
                 typename Backend::template pointer_type<T> output,
                 const GatherParams& params, Backend& backend) {
  auto validation_status = internal::validate_params(params);
  if (validation_status.status != StatusCode::OK) {
    return validation_status;
  }

  GatherSizes sizes = get_sizes(params);

  auto in_mem = backend.get_mem_object(input, sizes.input_size);
  auto indices_mem = backend.get_mem_object(indices, sizes.indices_size);
  auto out_mem = backend.get_mem_object(output, sizes.output_size);

  auto queue = backend.get_queue();

  return internal::launch<T, Index>(in_mem, indices_mem, out_mem, sizes, queue);
}

}  // namespace gather
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_GATHER_LAUNCH_H_
