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
#ifndef PORTDNN_BENCH_FIXTURE_ADD_COMPUTECPP_INFO_H_
#define PORTDNN_BENCH_FIXTURE_ADD_COMPUTECPP_INFO_H_

#include <CL/sycl.hpp>

#include "string_reporter.h"

extern bool const computecpp_available;

extern char const* const computecpp_version;
extern char const* const computecpp_edition;

namespace sycldnn {
namespace bench {
namespace computecpp_info {

/**
 * Add ComputeCpp meta-data (if available) to the benchmark label. The
 * version of compute++ is tied to the version of ComputeCpp, so the associated
 * meta-data of compute++ will be the same.
 *
 * portDNN benchmarks will include these attributes only if ComputeCpp info is
 * available. Benchmarks from other libraries such as MKL-DNN will never include
 * them.
 *
 * \param [out] reporter The benchmark string reporter to add the info to.
 */
inline void add_computecpp_version(StringReporter& reporter) {
  if (computecpp_available) {
    reporter.add_to_label("@computecpp-version", computecpp_version);
    reporter.add_to_label("@computecpp-edition", computecpp_edition);
  }
}

}  // namespace computecpp_info
}  // namespace bench
}  // namespace sycldnn

#endif  // PORTDNN_BENCH_FIXTURE_ADD_COMPUTECPP_INFO_H_
