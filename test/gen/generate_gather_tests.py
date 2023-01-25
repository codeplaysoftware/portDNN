#!/usr/bin/env python
#
# Copyright Codeplay Software Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use these files except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# Automatically generate the softmax test cases using TensorFlow to provide
# the expected values.

from __future__ import print_function

from ntpath import join
import os
from collections import namedtuple

import tensorflow as tf
import numpy as np

import helpers

INCLUDES = r"""
#include <gtest/gtest.h>

#include <vector>

#include "test/gather/gather_fixture.h"
#include "test/types/cartesian_product.h"
#include "test/types/kernel_data_types.h"
#include "test/types/test_backend_types.h"
#include "test/types/to_gtest_types.h"

"""

DATA_TYPES = r"""
using DataTypeList = sycldnn::types::KernelDataTypes;
using Backends = sycldnn::types::DefaultBackendTypes;

using TypeBackendPairs =
    sycldnn::types::CartesianProduct<DataTypeList, Backends>::type;

using namespace sycldnn;
using GTestTypePairs = sycldnn::types::ToGTestTypes<TypeBackendPairs>::type;

using IndexDataType = int32_t;  // or int64_t"""

TYPED_TEST_SUITE_DECL_TPL = r"""
template <typename Pair>
using {test_case} = GatherFixture<Pair, IndexDataType>;
TYPED_TEST_SUITE({test_case}, GTestTypePairs);
"""
TEST_CASE_TPL = r"Gather{n_dimensions}D"
TEST_NAME_TPL = r"G{n_dimensions}D_Axis_"

DIMENSION_LIST = [1, 2, 3]
SHAPES_LIST = [5, 4, 3]


def get_gather_result(max_val, in_shape, axis, indices):
    """
    Compute Gather.

    Will create input tensors of the required size filled with values [1, 2,
    3...max_val] and use this as the input to tf.gather along with the given
    axis and list of indices.

    Returns the computed values in a numpy array.
    """
    total_size = np.prod(in_shape)
    in_vals = helpers.get_tensor_data(total_size, max_val)

    in_tensor = tf.constant(in_vals, shape=in_shape, dtype=np.float64)
    return tf.gather(in_tensor, indices, axis=axis)


def get_test_lines(in_shape, axis, indices, indices_shape):
    """
    Create a list of strings corresponding to the lines in a single test case.

    Uses TensorFlow to compute the expected results for the given parameters,
    and provides the code to call the test fixture to run the test.
    """
    output, max_input_val = helpers.get_result_and_size(
        get_gather_result, in_shape=in_shape, axis=axis, indices=indices)

    in_shape_str = list(map(str, in_shape))

    indices_shape_str = list(map(str, indices_shape))
    test_case = TEST_CASE_TPL.format(n_dimensions=len(in_shape))

    axis_str = (str(axis) if axis >= 0 else 'Neg' + str(-axis)) + '_'

    test_name = TEST_NAME_TPL.format(n_dimensions=len(
        in_shape)) + axis_str + "Inp" + "x".join(in_shape_str) + "_Ind" + "x".join(indices_shape_str)

    test_lines = [
        "TYPED_TEST({}, {}) {{".format(test_case, test_name),
        "  using DataType = typename TestFixture::DataType;",
        "  const std::vector<DataType> exp_out = {};".format(
            helpers.format_tensor(output)
        ),
        "  const std::vector<IndexDataType> indices = {};".format(
            helpers.format_tensor(indices)),
        "  gather::GatherParams params;",
        "  params.axis = {};".format(axis),
        "  params.indices_dims = {{{dims_str}}};".format(
            dims_str=",".join(indices_shape_str)
        ),
        "  params.input_dims = {{{dims_str}}};".format(
            dims_str=",".join(in_shape_str)
        ),
        "  this->test_gather(exp_out, params, indices);",
        "}", ]

    return test_lines


def test_cases(n_dimensions):
    """
    Given a Gather input dimension (within DIMENSION_LIST), yield :
        a) The input shape : extracted from SHAPES_LIST given dimension
        b) All valid axis values
        c) Per axis, sample indices values (single value, vector, matrix)
        d) Shape of indices (1, 1D or 2D Shape)
    """
    np.random.seed(123)
    in_shape = SHAPES_LIST[0:n_dimensions]
    for axis in range(-n_dimensions, n_dimensions):
        pos_axis = axis
        pos_axis = n_dimensions + axis if axis < 0 else axis
        max_indice = in_shape[pos_axis]
        for indices_length in [1, max_indice, 2 * max_indice]:
            indices = np.random.choice(
                range(max_indice), indices_length, True)
            indices_shape = [indices_length] if (indices_length <= max_indice) else [
                indices_length // 2, 2]
            yield in_shape, axis, indices, indices_shape


def gather_test_case(n_dimensions):
    """
    Create a list of strings corresponding to separate lines in the full test
    case. The output contains headers, includes, setup and all the tests for
    the test case.
    """
    scriptname = os.path.basename(__file__)
    test_case = TEST_CASE_TPL.format(n_dimensions=n_dimensions)
    output = [
        helpers.get_license(),
        helpers.get_dont_modify_comment(scriptname=scriptname),
        INCLUDES,
        DATA_TYPES,
        TYPED_TEST_SUITE_DECL_TPL.format(test_case=test_case),
    ]
    for in_shape, axis, indices, indices_shape in test_cases(n_dimensions):
        output.extend(get_test_lines(in_shape, axis, indices, indices_shape))
    return output


FILENAME_TPL = "gather/gather_{n_dimensions}d.cc"


def get_test_case_filename(n_dimensions):
    "Get filename for test case."
    return FILENAME_TPL.format(n_dimensions=n_dimensions)


def generate_gather_tests():
    np.set_printoptions(suppress=True, threshold=1000000, linewidth=1000000)
    test_dir = helpers.get_test_directory()
    os.chdir(test_dir)
    for dimension in DIMENSION_LIST:
        filename = get_test_case_filename(dimension)
        output = gather_test_case(dimension)
        with open(filename, 'w') as f:
            f.write('\n'.join(output))
        print("File '{}' written".format(filename))


if __name__ == "__main__":
    generate_gather_tests()
