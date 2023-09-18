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
#ifndef PORTDNN_INCLUDE_BACKEND_CRTP_BACKEND_H_
#define PORTDNN_INCLUDE_BACKEND_CRTP_BACKEND_H_

/**
 * \file
 * Contains the implementation of \ref sycldnn::backend::CRTPBackend, a helper
 * class for the curiously recurring template pattern as used in the various
 * providers which make up the portDNN backends.
 *
 * See for \ref sycldnn::backend::EigenMatmulProvider for an example of its use.
 */

namespace sycldnn {
namespace backend {

/**
 * A helper class for curiously recurring templated backend providers. Provides
 * an underlying_backend method to access the derived type backend in the CRTP
 * base class.
 */
template <typename Backend, template <typename> class Provider>
struct CRTPBackend {
 protected:
  /**
   * Get a const reference to the underlying derived backend type used in the
   * CRTP base class.
   *
   * The templated type deduction requires that this method is called through
   * `this` in the base class:
   * \code
   *     auto backend = this->underlying_backend();
   * \endcode
   * the method is not visible, and will cause a compile error, if called as:
   * \code
   *     auto backend = underlying_backend();
   * \endcode
   *
   * \return A const reference to the underlying backend
   */
  Backend const& underlying_backend() const {
    return static_cast<Backend const&>(*this);
  }

  /**
   * Get a reference to the underlying derived backend type used in the CRTP
   * base class.
   *
   * The templated type deduction requires that this method is called through
   * `this` in the base class:
   * \code
   *     auto backend = this->underlying_backend();
   * \endcode
   * the method is not visible, and will cause a compile error, if called as:
   * \code
   *     auto backend = underlying_backend();
   * \endcode
   *
   * \return A reference to the underlying backend
   */
  Backend& underlying_backend() { return static_cast<Backend&>(*this); }

 private:
  // Making the default constructor private, and the Provider<Backend> a friend
  // ensures that this class is used in the correct way for CRTP:
  //
  //     struct Provider : CRTPBackend<Backend, Provider> {...}
  //
  // and will cause a compile time error if the inheritance instead looks like:
  //
  //     struct Provider : CRTPBackend<Backend, OtherClass> {...}
  CRTPBackend() = default;
  friend Provider<Backend>;
};

}  // namespace backend
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_BACKEND_CRTP_BACKEND_H_
