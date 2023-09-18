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
# Automatically generate the binaryop test cases using TensorFlow to provide
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

#include "portdnn/binaryop/operators.h"
#include "test/binaryop/fixture.h"
#include "test/types/cartesian_product.h"
#include "test/types/kernel_data_types.h"
#include "test/types/test_backend_types.h"
#include "test/types/to_gtest_types.h"
"""
DATA_TYPES = r"""
using DataTypeList = sycldnn::types::KernelDataTypes;
using Backends = sycldnn::types::AllBackendTypes;

using TypeBackendPairs =
    sycldnn::types::CartesianProduct<DataTypeList, Backends>::type;

using GTestTypePair = sycldnn::types::ToGTestTypes<TypeBackendPairs>::type;
"""
TYPED_TEST_SUITE_DECL_TPL = r"""
template <typename Pair>
using {test_case} = BinaryOpFixture<Pair, sycldnn::binaryop::{op}>;

TYPED_TEST_SUITE({test_case}, GTestTypePair);
"""

TEST_CASE_TPL = r"Binary{op}"

TENSORFLOW_OPS_MAP = {
    "Add": tf.math.add,
    "Sub": tf.math.subtract,
    "Mul": tf.math.multiply,
    "Div": tf.math.divide,
}


def get_binaryop_result(max_val, lhs_dims, rhs_dims, op):
    """
    Compute binary op.

    Returns the computed values in a numpy array.
    """
    lhs_vals = helpers.get_tensor_data(np.prod(lhs_dims), max_val)
    rhs_vals = helpers.get_tensor_data(np.prod(rhs_dims), max_val)
    lhs_tensor = tf.constant(
        lhs_vals,
        shape=lhs_dims,
        dtype=np.float64)
    rhs_tensor = tf.constant(
        rhs_vals,
        shape=rhs_dims,
        dtype=np.float64)
    return op(lhs_tensor, rhs_tensor)


def get_test_lines(test_case, lhs_dims, rhs_dims, op):
    """
    Create a list of strings corresponding to the lines in a single test case.

    Uses TensorFlow to compute the expected results for the given parameters,
    and provides the code to call the test fixture to run the test.
    """
    output, max_input_val = helpers.get_result_and_size(get_binaryop_result,
                                                        lhs_dims=lhs_dims,
                                                        rhs_dims=rhs_dims,
                                                        op=op)
    lhs_name_str = "_".join(str(x) for x in lhs_dims)
    rhs_name_str = "_".join(str(x) for x in rhs_dims)
    test_name = "lhs_{}_rhs_{}".format(lhs_name_str, rhs_name_str)
    out_str = helpers.format_tensor(output)
    lhs_dims_str = ", ".join(str(x) for x in lhs_dims)
    rhs_dims_str = ", ".join(str(x) for x in rhs_dims)
    test_lines = [
        "TYPED_TEST({}, {}) {{".format(test_case, test_name),
        "  using DataType = typename TestFixture::DataType;",
        "  const std::vector<DataType> exp_out = {};".format(out_str),
        "  sycldnn::binaryop::BinaryParams params;",
        "  params.lhs_dims = {{{}}};".format(lhs_dims_str),
        "  params.rhs_dims = {{{}}};".format(rhs_dims_str),
        "  const DataType max_input_val = {:.1f};".format(max_input_val),
        "  this->run(exp_out, params, max_input_val);",
        "}",
    ]
    return test_lines


def get_test_cases(op):
    """
    Create a list of strings corresponding to separate lines in the full test
    case. The output contains headers, includes, setup and all the tests for
    the test case.
    """
    scriptname = os.path.basename(__file__)
    test_case = TEST_CASE_TPL.format(op=op)
    output = [
        helpers.get_license(),
        helpers.get_dont_modify_comment(scriptname=scriptname),
        INCLUDES,
        DATA_TYPES,
        TYPED_TEST_SUITE_DECL_TPL.format(test_case=test_case, op=op),
    ]

    output.extend(
        get_test_lines(
            test_case,
            [1],
            [1],
            TENSORFLOW_OPS_MAP[op]))
    output.extend(
        get_test_lines(
            test_case,
            [1],
            [12],
            TENSORFLOW_OPS_MAP[op]))
    output.extend(
        get_test_lines(
            test_case,
            [12],
            [1],
            TENSORFLOW_OPS_MAP[op]))
    output.extend(
        get_test_lines(
            test_case,
            [1, 3],
            [1],
            TENSORFLOW_OPS_MAP[op]))
    output.extend(
        get_test_lines(
            test_case,
            [2, 3, 4, 5],
            [1],
            TENSORFLOW_OPS_MAP[op]))
    output.extend(
        get_test_lines(
            test_case,
            [2, 3, 4, 5],
            [5],
            TENSORFLOW_OPS_MAP[op]))
    output.extend(
        get_test_lines(
            test_case,
            [4, 5],
            [2, 3, 4, 5],
            TENSORFLOW_OPS_MAP[op]))
    output.extend(
        get_test_lines(
            test_case,
            [1, 4, 5],
            [2, 3, 1, 1],
            TENSORFLOW_OPS_MAP[op]))
    output.extend(
        get_test_lines(
            test_case,
            [3, 4, 5],
            [2, 1, 1, 1],
            TENSORFLOW_OPS_MAP[op]))
    output.extend(
        get_test_lines(
            test_case,
            [10, 1, 64],
            [10, 3, 64],
            TENSORFLOW_OPS_MAP[op]))
    output.extend(
        get_test_lines(
            test_case,
            [10, 3, 64],
            [10, 1, 64],
            TENSORFLOW_OPS_MAP[op]))
    output.extend(
        get_test_lines(
            test_case,
            [3, 1, 8],
            [2, 1, 7, 1],
            TENSORFLOW_OPS_MAP[op]))
    output.extend(
        get_test_lines(
            test_case,
            [2, 1, 1],
            [2, 3, 1],
            TENSORFLOW_OPS_MAP[op]))
    return output


FILENAME_TPL = "binaryop/binaryop_{op}.cc"


def get_test_case_filename(op):
    "Get filename for test case."
    return FILENAME_TPL.format(op=helpers.to_lower_case_str(op))


def generate_binaryop_tests():
    np.set_printoptions(suppress=True, threshold=1000000, linewidth=1000000)
    test_dir = helpers.get_test_directory()
    os.chdir(test_dir)
    for op in TENSORFLOW_OPS_MAP.keys():
        filename = get_test_case_filename(op)
        output = get_test_cases(op)
        with open(filename, 'w') as f:
            f.write('\n'.join(output))
        print("File '{}' written".format(filename))


if __name__ == "__main__":
    generate_binaryop_tests()
