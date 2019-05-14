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
#ifndef SYCLDNN_INCLUDE_CONV2D_DIRECT_H_
#define SYCLDNN_INCLUDE_CONV2D_DIRECT_H_

#include "sycldnn/conv2d/conv_type.h"
#include "sycldnn/conv2d/params.h"
#include "sycldnn/conv2d/sizes.h"

#include "sycldnn/internal/conv2d/direct.h"

namespace sycldnn {
namespace conv2d {
/**
 * Launch the direct implementation of a 2D convolution.
 *
 * Will extract the SYCL buffers and SYCL queue from the backend and forward
 * these on to the precompiled kernels.
 *
 * Returns an SNNStatus containing the SYCL event tied to the kernel launch.
 */
template <typename T, typename ConvType, typename Backend>
inline SNNStatus launch_direct(
    typename Backend::template pointer_type<T const> input,
    typename Backend::template pointer_type<T const> filter,
    typename Backend::template pointer_type<T> output,
    Conv2DParams const& params, Backend& backend) {
  auto conv_sizes = get_sizes<ConvType>(params);
  auto inp_buff = backend.get_buffer(input, conv_sizes.input_size);
  auto fil_buff = backend.get_buffer(filter, conv_sizes.filter_size);
  auto out_buff = backend.get_buffer(output, conv_sizes.output_size);

  auto const inp_offset = backend.get_offset(input);
  auto const fil_offset = backend.get_offset(filter);
  auto const out_offset = backend.get_offset(output);

  ReadAccessor<T const> inp_access{inp_buff,
                                   cl::sycl::range<1>{conv_sizes.input_size},
                                   cl::sycl::id<1>{inp_offset}};
  ReadAccessor<T const> fil_access{fil_buff,
                                   cl::sycl::range<1>{conv_sizes.filter_size},
                                   cl::sycl::id<1>{fil_offset}};
  WriteAccessor<T> out_access{out_buff,
                              cl::sycl::range<1>{conv_sizes.output_size},
                              cl::sycl::id<1>{out_offset}};
  cl::sycl::queue queue = backend.get_queue();
  return internal::launch_direct<T, ConvType>(inp_access, fil_access,
                                              out_access, params, queue);
}
}  // namespace conv2d
}  // namespace sycldnn
#endif  // SYCLDNN_INCLUDE_CONV2D_DIRECT_H_
