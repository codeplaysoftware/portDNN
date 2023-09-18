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

#ifndef PORTDNN_INCLUDE_SCATTER_ND_OPERATORS_H_
#define PORTDNN_INCLUDE_SCATTER_ND_OPERATORS_H_

/**
 * \file
 * Contains the declarations of the \ref sycldnn::scatter_nd::Assign
 * and \ref sycldnn::scatter_nd::Add and \ref sycldnn::scatter_nd::Sub
 * and \ref sycldnn::scatter_nd::Mul and \ref sycldnn::scatter_nd::Div
 * tag types.
 */

namespace sycldnn {
namespace scatter_nd {

/**
 * ScatterND operator that assigns the given value to the specified entry.
 */
struct Assign {
  /**
   * Forward declaration of apply method
   * \tparam U          Device pointer type
   * \tparam DataType   Pointer data type
   * \tparam IndexType  Index data type
   * \param  ptr        Device pointer
   * \param  offset     Device pointer offset
   * \param  val        Update value
   * \return Returns nothing
   */
  template <typename U, typename DataType, typename IndexType>
  static void apply(U& ptr, IndexType offset, DataType val);
};

/**
 * ScatterND operator that performs and in-place addition with the given value
 * and the specifid entry.
 */
struct Add {
  /**
   * Forward declaration of apply method
   * \tparam U          Device pointer type
   * \tparam DataType   Pointer data type
   * \tparam IndexType  Index data type
   * \param  ptr        Device pointer
   * \param  offset     Device pointer offset
   * \param  val        Update value
   * \return Returns nothing
   */
  template <typename U, typename DataType, typename IndexType>
  static void apply(U& ptr, IndexType offset, DataType val);
};

/**
 * ScatterND operator that performs and in-place subtraction with the given
 * value and the specifid entry.
 */
struct Sub {
  /**
   * Forward declaration of apply method
   * \tparam U          Device pointer type
   * \tparam DataType   Pointer data type
   * \tparam IndexType  Index data type
   * \param  ptr        Device pointer
   * \param  offset     Device pointer offset
   * \param  val        Update value
   * \return Returns nothing
   */
  template <typename U, typename DataType, typename IndexType>
  static void apply(U& ptr, IndexType offset, DataType val);
};

/**
 * ScatterND operator that performs and in-place multiplication with the given
 * value and the specifid entry.
 */
struct Mul {
  /**
   * Forward declaration of apply method
   * \tparam U          Device pointer type
   * \tparam DataType   Pointer data type
   * \tparam IndexType  Index data type
   * \param  ptr        Device pointer
   * \param  offset     Device pointer offset
   * \param  val        Update value
   * \return Returns nothing
   */
  template <typename U, typename DataType, typename IndexType>
  static void apply(U& ptr, IndexType offset, DataType val);
};

/**
 * ScatterND operator that performs and in-place division with the given value
 * and the specifid entry.
 */
struct Div {
  /**
   * Forward declaration of apply method
   * \tparam U          Device pointer type
   * \tparam DataType   Pointer data type
   * \tparam IndexType  Index data type
   * \param  ptr        Device pointer
   * \param  offset     Device pointer offset
   * \param  val        Update value
   * \return Returns nothing
   */
  template <typename U, typename DataType, typename IndexType>
  static void apply(U& ptr, IndexType offset, DataType val);
};

}  // namespace scatter_nd
}  // namespace sycldnn
#endif  // PORTDNN_INCLUDE_SCATTER_ND_OPERATORS_H_
