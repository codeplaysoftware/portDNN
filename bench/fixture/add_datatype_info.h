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
#ifndef PORTDNN_BENCH_FIXTURE_ADD_DATATYPE_INFO_H_
#define PORTDNN_BENCH_FIXTURE_ADD_DATATYPE_INFO_H_

#ifdef SNN_ENABLE_HALF
#include <CL/sycl.hpp>
#endif  // SNN_ENABLE_HALF

#include "string_reporter.h"

namespace sycldnn {
namespace bench {
namespace datatype_info {

/**
 * Add the datatype used to the benchmark label.
 *
 * \param [out] reporter The benchmark string reporter to add the info to.
 */
template <typename DataType>
inline void add_datatype_info(StringReporter& reporter);

template <>
inline void add_datatype_info<float>(StringReporter& reporter) {
  reporter.add_to_label("@datatype", "float");
}

template <>
inline void add_datatype_info<double>(StringReporter& reporter) {
  reporter.add_to_label("@datatype", "double");
}

#ifdef SNN_ENABLE_HALF
template <>
inline void add_datatype_info<cl::sycl::half>(StringReporter& reporter) {
  reporter.add_to_label("@datatype", "sycl::half");
}
#endif  // SNN_ENABLE_HALF

}  // namespace datatype_info
}  // namespace bench
}  // namespace sycldnn

#endif  // PORTDNN_BENCH_FIXTURE_ADD_DATATYPE_INFO_H_
