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
#ifndef SYCLDNN_SRC_HELPERS_REGISTER_TILE_H_
#define SYCLDNN_SRC_HELPERS_REGISTER_TILE_H_

namespace sycldnn {
namespace helpers {
/**
 * 3D tile of size X x Y x Z of data type T.
 */
template <typename T, int X, int Y, int Z>
struct RegisterTile3D {
  SNN_ALWAYS_INLINE T& data(int x, int y, int z) { return data_[x][y][z]; }
  SNN_ALWAYS_INLINE T const& data(int x, int y, int z) const {
    return data_[x][y][z];
  }

 private:
  T data_[X][Y][Z];
};
/**
 * 2D tile of size X x Y of data type T.
 */
template <typename T, int X, int Y>
struct RegisterTile2D {
  SNN_ALWAYS_INLINE T& data(int x, int y) { return data_[x][y]; }
  SNN_ALWAYS_INLINE T const& data(int x, int y) const { return data_[x][y]; }

 private:
  T data_[X][Y];
};
/**
 * 1D tile of size X of data type T.
 */
template <typename T, int X>
struct RegisterTile1D {
  SNN_ALWAYS_INLINE T& data(int x) { return data_[x]; }
  SNN_ALWAYS_INLINE T const& data(int x) const { return data_[x]; }

 private:
  T data_[X];
};
}  // namespace helpers
}  // namespace sycldnn
#endif  // SYCLDNN_SRC_HELPERS_REGISTER_TILE_H_
