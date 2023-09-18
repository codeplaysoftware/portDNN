/*
 * Copyright Codeplay Software Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use these files except in compliance with the License.
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

#ifndef INCLUDE_PORTDNN_HELPERS_EVENT_HANDLING_H_
#define INCLUDE_PORTDNN_HELPERS_EVENT_HANDLING_H_

#include <CL/sycl.hpp>

namespace sycldnn {
namespace helpers {

// Helper to convert a vector of events into a single event that is dependent on
// all of the input events
cl::sycl::event multi_event_to_one(const std::vector<cl::sycl::event>& events,
                                   cl::sycl::queue& q) {
  return q.submit([&](sycl::handler& cgh) {
    cgh.depends_on(events);
    cgh.host_task([=]() {});
  });
}

}  // namespace helpers
}  // namespace sycldnn

#endif  // INCLUDE_PORTDNN_HELPERS_EVENT_HANDLING_H_
