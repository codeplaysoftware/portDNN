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
#ifndef SYCLDNN_BENCH_COMMON_EVENT_HANDLING_H_
#define SYCLDNN_BENCH_COMMON_EVENT_HANDLING_H_

#include <CL/sycl.hpp>

namespace sycldnn {
namespace bench {
namespace internal {
/** Helper function that checks if SYCL-DNN can wait on events directly, or
 * has to wait on the queue. This is because Eigen cannot return us the events
 * corresponding to the kernel launch directly. */
/* TODO: SD-404 Remove queue::wait_and_throw workaround when Eigen removed */
inline void wait_for_event(cl::sycl::event& ev, cl::sycl::queue q) {
  if (ev.is_host()) {
    q.wait_and_throw();
  } else {
    ev.wait_and_throw();
  }
}
}  // namespace internal
}  // namespace bench
}  // namespace sycldnn

#endif
