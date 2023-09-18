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
# Automatically generate the reduce test cases using TensorFlow to provide
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

#include "portdnn/reduce/operators.h"
#include "test/reduce/fixture.h"
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
using {test_case} = ReduceFixture<Pair, sycldnn::reduce::{op}>;
TYPED_TEST_SUITE({test_case}, GTestTypePair);"""


TEST_CASE_TPL = r"Reduce{op}"
TEST_NAME_TPL = r"Batch{batch}Outer{outer}Inner{inner}"

TENSORFLOW_OPS_MAP = {
    "Add": tf.math.reduce_sum,
    "Mean": tf.math.reduce_mean,
    "Max": tf.math.reduce_max,
    "Min": tf.math.reduce_min,
}

# Test a few operators with all the sizes.
# Other operators are tested on few sizes to avoid redundancy.
TEST_ALL_SIZES = {
    "Add",
    "Mean",
}


def get_batch_sizes(op):
    if op in TEST_ALL_SIZES:
        return [1, 3]
    else:
        return [1, 2]


def get_input_sizes(op):
    if op in TEST_ALL_SIZES:
        return [1, 6, 8, 33, 512, 1037]
    else:
        return [1, 11]


def get_reduce_result(max_val, batch, outer, inner, op):
    """
    Compute reduction.

    Will create input tensors of the required size filled with values 1, 2,
    3...

    Returns the computed values in a numpy array.
    """
    in_vals = helpers.get_tensor_data(batch * outer * inner, max_val)
    in_tensor = tf.constant(
        in_vals,
        shape=[
            batch,
            outer,
            inner],
        dtype=np.float64)
    return op(in_tensor, axis=1)


def get_test_lines(test_case, batch, outer, inner, op):
    """
    Create a list of strings corresponding to the lines in a single test case.

    Uses TensorFlow to compute the expected results for the given parameters,
    and provides the code to call the test fixture to run the test.
    """
    output, max_input_val = helpers.get_result_and_size(get_reduce_result,
                                                        batch=batch,
                                                        outer=outer,
                                                        inner=inner,
                                                        op=op)
    test_name = TEST_NAME_TPL.format(batch=batch, outer=outer, inner=inner)
    test_lines = [
        "TYPED_TEST({}, {}) {{".format(test_case, test_name),
        "  using DataType = typename TestFixture::DataType;",
        "  const std::vector<DataType> exp_out = {};".format(
            helpers.format_tensor(output)),
        "  const int batches = {};".format(batch),
        "  const int outer = {};".format(outer),
        "  const int inner = {};".format(inner),
        "  const DataType max_input_val = {:.1f};".format(max_input_val),
        "  this->run(exp_out, batches, outer, inner, max_input_val);",
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
    batch_sizes = get_batch_sizes(op)
    in_sizes = get_input_sizes(op)
    for batch, outer, inner in itertools.product(
            batch_sizes, in_sizes, in_sizes):
        output.extend(
            get_test_lines(
                test_case,
                batch,
                outer,
                inner,
                TENSORFLOW_OPS_MAP[op]))
    return output


FILENAME_TPL = "reduce/reduce_{op}.cc"


def get_test_case_filename(op):
    "Get filename for test case."
    return FILENAME_TPL.format(op=helpers.to_lower_case_str(op))


def generate_reduce_tests():
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
    generate_reduce_tests()
