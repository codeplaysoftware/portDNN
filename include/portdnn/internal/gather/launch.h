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
#ifndef PORTDNN_INCLUDE_INTERNAL_GATHER_LAUNCH_H_
#define PORTDNN_INCLUDE_INTERNAL_GATHER_LAUNCH_H_

#include <CL/sycl.hpp>

#include "portdnn/export.h"
#include "portdnn/helpers/sycl_language_helpers.h"
#include "portdnn/mem_object.h"
#include "portdnn/status.h"

#include "portdnn/gather/sizes.h"

namespace sycldnn {
namespace gather {
namespace internal {

/**
 * Validate that the user-provided gather parameters are consistent with what
 * is expected by portDNN.
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

/**
 * The internal Gather launcher.
 * Implemented in the compiled SYCL DNN library.
 */
template <typename T, typename Index, template <typename> class MemObj>
SNN_EXPORT SNNStatus launch_impl(MemObj<T const>& input,
                                 MemObj<Index const>& indices,
                                 MemObj<T>& output, const GatherSizes& sizes,
                                 cl::sycl::queue& queue,
                                 const std::vector<cl::sycl::event>& events);

/**
 * Internal gather launcher that casts tensor types to the
 * implemented types when needed.
 */
template <typename SrcT, typename DstT, typename Index,
          template <typename> class MemObj>
SNNStatus launch_cast(MemObj<SrcT const>& input, MemObj<Index const>& indices,
                      MemObj<SrcT>& output, const GatherSizes& sizes,
                      cl::sycl::queue& queue,
                      const std::vector<cl::sycl::event>& events) {
  if (std::is_same<SrcT, DstT>::value) {
    return launch_impl(input, indices, output, sizes, queue, events);
  }
  auto input_int_mem = input.template cast<DstT const>();
  auto output_int_mem = output.template cast<DstT>();
  return launch_impl(input_int_mem, indices, output_int_mem, sizes, queue,
                     events);
}

#define SNN_LAUNCH_CAST(DST_T, MEM_OBJ)                                       \
  template <typename T, typename Index,                                       \
            typename std::enable_if<sizeof(T) == sizeof(DST_T), int>::type =  \
                0>                                                            \
  SNNStatus launch(MEM_OBJ<T const>& input, MEM_OBJ<Index const>& indices,    \
                   MEM_OBJ<T>& output, const GatherSizes& sizes,              \
                   cl::sycl::queue& queue,                                    \
                   const std::vector<cl::sycl::event>& events) {              \
    return launch_cast<T, DST_T, Index>(input, indices, output, sizes, queue, \
                                        events);                              \
  }

SNN_LAUNCH_CAST(uint8_t, USMMemObject);
SNN_LAUNCH_CAST(uint16_t, USMMemObject);
SNN_LAUNCH_CAST(uint32_t, USMMemObject);
SNN_LAUNCH_CAST(uint64_t, USMMemObject);

SNN_LAUNCH_CAST(uint8_t, BufferMemObject);
SNN_LAUNCH_CAST(uint16_t, BufferMemObject);
SNN_LAUNCH_CAST(uint32_t, BufferMemObject);
SNN_LAUNCH_CAST(uint64_t, BufferMemObject);
#undef SNN_LAUNCH_CAST

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
SNNStatus sublaunch(
    typename Backend::template pointer_type<T const> input,
    typename Backend::template pointer_type<Index const> indices,
    typename Backend::template pointer_type<T> output,
    const GatherParams& params, Backend& backend,
    const std::vector<cl::sycl::event>& events) {
  auto validation_status = internal::validate_params(params);
  if (validation_status.status != StatusCode::OK) {
    return validation_status;
  }

  GatherSizes sizes = get_sizes(params);

  auto in_mem = backend.get_mem_object(input, sizes.input_size);
  auto indices_mem = backend.get_mem_object(indices, sizes.indices_size);
  auto out_mem = backend.get_mem_object(output, sizes.output_size);

  auto queue = backend.get_queue();

  return internal::launch<T, Index>(in_mem, indices_mem, out_mem, sizes, queue,
                                    events);
}

}  // namespace internal
}  // namespace gather
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_INTERNAL_GATHER_LAUNCH_H_
