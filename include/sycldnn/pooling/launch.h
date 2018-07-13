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

#ifndef SYCLDNN_INCLUDE_POOLING_LAUNCH_H_
#define SYCLDNN_INCLUDE_POOLING_LAUNCH_H_

#include "sycldnn/accessor_types.h"
#include "sycldnn/internal/pooling/launch_internal.h"
#include "sycldnn/pooling/params.h"
#include "sycldnn/status.h"

namespace sycldnn {
namespace pooling {

/*
 * Returns an SNNStatus containing the SYCL event tied to the kernel launches
 * and a StatusCode enum showing if the launch was OK or whether it encountered
 * some problem.
 */
template <typename T, template <typename U> class PoolType, typename Direction,
          typename Backend>
SNNStatus launch(typename Backend::template pointer_type<T const> input,
                 typename Backend::template pointer_type<T> output,
                 const PoolingParams& pp, Backend& backend) {
  auto inp_buf = backend.get_buffer(input, pp.in_rows * pp.in_cols);
  auto outp_buf = backend.get_buffer(output, pp.out_rows * pp.out_cols);
  ReadAccessor<T const> inp_access{inp_buf};
  WriteAccessor<T> outp_access{outp_buf};
  auto queue = backend.get_queue();
  return internal::launch_pooling<T, PoolType, Direction>(
      inp_access, outp_access, pp, queue);
}

}  // namespace pooling
}  // namespace sycldnn
#endif  // SYCLDNN_INCLUDE_POOLING_LAUNCH_H_
