#!python
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
# Automatically generate the transpose test cases using TensorFlow to provide
# the expected values.

from __future__ import print_function

import itertools
import os

import tensorflow as tf
import numpy as np

import helpers

INCLUDES = r"""
#include <gtest/gtest.h>
#include <vector>

#include "test/transpose/transpose_fixture.h"
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

using GTestTypePairs = sycldnn::types::ToGTestTypes<TypeBackendPairs>::type;
"""
TYPED_TEST_SUITE_DECL_TPL = r"""
template <typename Pair>
using {test_case} = TransposeFixture<Pair>;
TYPED_TEST_SUITE({test_case}, GTestTypePairs);"""
TEST_CASE_TPL = r"Transpose{n_dimensions}D"
TEST_NAME_TPL = r"T{n_dimensions}D_"

DIMENSION_LIST = [2, 3, 4]


def get_input_sizes():
    """
    Want to test with sizes that are:
        a) Divisible by 4
        b) Divisible by 2 but not 4
        c) Not Divisible by 2
    """
    return [2, 3, 4]


def get_transpose_result(max_val, in_shape, permutation):
    """
    Compute transpose.

    Will create input tensors of the required size filled with values 1, 2,
    3... and use this as the input to tf.transpose along with the given permutation.

    Returns the computed values in a numpy array.
    """
    total_size = np.prod(in_shape)
    in_vals = helpers.get_tensor_data(total_size, max_val)

    in_tensor = tf.constant(in_vals, shape=in_shape, dtype=np.float64)
    return tf.transpose(in_tensor, permutation)


def get_test_lines(in_shape, permutation):
    """
    Create a list of strings corresponding to the lines in a single test case.

    Uses TensorFlow to compute the expected results for the given parameters,
    and provides the code to call the test fixture to run the test.
    """
    output, max_input_val = helpers.get_result_and_size(
        get_transpose_result, in_shape=in_shape, permutation=permutation)
    in_shape_str = list(map(str, in_shape))
    perm_str = list(map(str, permutation))
    test_case = TEST_CASE_TPL.format(n_dimensions=len(in_shape))
    test_name = TEST_NAME_TPL.format(n_dimensions=len(in_shape)) + "x".join(
        in_shape_str) + "_" + "x".join(perm_str)
    test_lines = [
        "TYPED_TEST({}, {}) {{".format(test_case, test_name),
        "  using DataType = typename TestFixture::DataType;",
        "  const std::vector<DataType> exp_out = {};".format(
            helpers.format_tensor(output)),
        "  const std::vector<int> sizes = {{ {size_str} }};".format(
            size_str=",".join(in_shape_str)),
        "  const std::vector<int> perm = {{ {perm_str} }};".format(
            perm_str=",".join(perm_str)),
        "  const DataType max_input_val = {:.1f};".format(max_input_val),
        "  this->run(exp_out, sizes, perm, max_input_val, 0, 0);",
        "}",
    ]
    return test_lines


def test_cases(n_dimensions):
    in_sizes = get_input_sizes()
    for in_shape in itertools.product(in_sizes, repeat=n_dimensions):
        for permutation in itertools.permutations(
                [i for i in range(n_dimensions)]):
            yield in_shape, permutation


def transpose_test_case(n_dimensions):
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
    for in_shape, permutation in test_cases(n_dimensions):
        output.extend(get_test_lines(in_shape, permutation))
    return output


FILENAME_TPL = "transpose/transpose_{n_dimensions}d.cc"


def get_test_case_filename(n_dimensions):
    "Get filename for test case."
    return FILENAME_TPL.format(n_dimensions=n_dimensions)


def generate_transpose_tests():
    np.set_printoptions(suppress=True, threshold=1000000, linewidth=1000000)
    test_dir = helpers.get_test_directory()
    os.chdir(test_dir)
    for dimension in DIMENSION_LIST:
        filename = get_test_case_filename(dimension)
        output = transpose_test_case(dimension)
        with open(filename, 'w') as f:
            f.write('\n'.join(output))
        print("File '{}' written".format(filename))


if __name__ == "__main__":
    generate_transpose_tests()
