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
# Automatically generate the convolution test cases using TensorFlow to provide
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

#include "test/matmul/fixture.h"
#include "test/types/cartesian_product.h"
#include "test/types/kernel_data_types.h"
#include "test/types/test_backend_types.h"
#include "test/types/to_gtest_types.h"
"""
DATA_TYPES = r"""
using DataTypeList = sycldnn::types::KernelDataTypes;
using BackendTypeList = sycldnn::types::DefaultBackendTypes;
using TypePairList =
    sycldnn::types::CartesianProduct<DataTypeList, BackendTypeList>::type;
using GTestTypeList = sycldnn::types::ToGTestTypes<TypePairList>::type;
"""
TYPED_TEST_SUITE_DECL_TPL = r"""
template <typename DataType>
using {test_case} = MatmulFixture<DataType, {trans_lhs}, {trans_rhs}>;
TYPED_TEST_SUITE({test_case}, GTestTypeList);"""
TEST_CASE_TPL = r"MatmulBatch{batch}Beta{beta}{trans_lhs}{trans_rhs}"
TEST_NAME_TPL = r"M{m}xK{k}xN{n}"

BOOL_LIST = [True, False]
BATCH_LIST = [1, 3]
BETA_LIST = [0, 1]


def get_input_sizes():
    """
    Want to test with sizes that are:
        a) Divisible by 4
        b) Divisible by 2 but not 4
        c) Not Divisible by 2
    """
    return [14, 15, 16]


def get_shape(batch, rows, cols, transpose):
    "Get the shape of the matrix with given rows and columns."
    if transpose:
        return [batch, cols, rows]
    else:
        return [batch, rows, cols]


def get_matmul_result(max_val, batch, m, k, n, beta, trans_lhs, trans_rhs):
    """
    Compute matrix multiplication.

    Will create input tensors of the required size filled with values 1, 2,
    3... and use these to compute the multiplication.

    Returns the computed values in a numpy array.
    """
    lhs_vals = helpers.get_tensor_data(batch * m * k, max_val)
    rhs_vals = helpers.get_tensor_data(batch * k * n, max_val)
    out_vals = helpers.get_tensor_data(batch * m * n, max_val)

    lhs_shape = get_shape(batch, m, k, trans_lhs)
    rhs_shape = get_shape(batch, k, n, trans_rhs)
    out_shape = get_shape(batch, m, n, False)

    lhs_tensor = tf.constant(lhs_vals, shape=lhs_shape, dtype=np.float64)
    rhs_tensor = tf.constant(rhs_vals, shape=rhs_shape, dtype=np.float64)
    initial_out = tf.constant(out_vals, shape=out_shape, dtype=np.float64)
    return beta * initial_out + tf.matmul(lhs_tensor, rhs_tensor,
                                          trans_lhs, trans_rhs)


def get_test_lines(batch, m, k, n, beta, trans_lhs, trans_rhs):
    """
    Create a list of strings corresponding to the lines in a single test case.

    Uses TensorFlow to compute the expected results for the given parameters,
    and provides the code to call the test fixture to run the test.
    """
    output, max_input_val = helpers.get_result_and_size(get_matmul_result,
                                                        batch=batch,
                                                        m=m,
                                                        k=k,
                                                        n=n,
                                                        beta=beta,
                                                        trans_lhs=trans_lhs,
                                                        trans_rhs=trans_rhs)
    test_case = TEST_CASE_TPL.format(batch=batch,
                                     beta=beta,
                                     trans_lhs=trans_lhs,
                                     trans_rhs=trans_rhs)
    test_name = TEST_NAME_TPL.format(m=m, k=k, n=n)
    test_lines = [
        "TYPED_TEST({}, {}) {{".format(test_case, test_name),
        "  using DataType = typename TestFixture::DataType;",
        "  const std::vector<DataType> exp_out = {};".format(
            helpers.format_tensor(output)),
        "  const int batches = {};".format(batch),
        "  const int m = {};".format(m),
        "  const int k = {};".format(k),
        "  const int n = {};".format(n),
        "  const auto beta = static_cast<DataType>({});".format(beta),
        "  const DataType max_input_val = {:.1f};".format(max_input_val),
        "  this->run(exp_out, batches, m, k, n, beta, 0, 0, 0, max_input_val);",
        "}",
    ]
    return test_lines


def test_case_for_transposes(batch, beta, trans_lhs, trans_rhs):
    """
    Create a list of strings corresponding to separate lines in the full test
    case. The output contains headers, includes, setup and all the tests for
    the test case.
    """
    scriptname = os.path.basename(__file__)
    test_case = TEST_CASE_TPL.format(batch=batch,
                                     beta=beta,
                                     trans_lhs=trans_lhs,
                                     trans_rhs=trans_rhs)
    output = [
        helpers.get_license(),
        helpers.get_dont_modify_comment(scriptname=scriptname),
        INCLUDES,
        DATA_TYPES,
        TYPED_TEST_SUITE_DECL_TPL.format(
            test_case=test_case,
            trans_lhs=helpers.to_lower_case_str(trans_lhs),
            trans_rhs=helpers.to_lower_case_str(trans_rhs)),
    ]
    in_sizes = get_input_sizes()
    for m, k, n in itertools.product(in_sizes, in_sizes, in_sizes):
        output.extend(
            get_test_lines(batch, m, k, n, beta, trans_lhs, trans_rhs))
    return output


FILENAME_TPL = "matmul/matmul_batch{batch}_beta{beta}_{trans_lhs}_{trans_rhs}.cc"


def get_test_case_filename(batch, beta, trans_lhs, trans_rhs):
    "Get filename for test case."
    return FILENAME_TPL.format(batch=batch,
                               beta=beta,
                               trans_lhs=helpers.to_lower_case_str(trans_lhs),
                               trans_rhs=helpers.to_lower_case_str(trans_rhs))


def generate_matmul_tests():
    np.set_printoptions(suppress=True, threshold=1000000, linewidth=1000000)
    test_dir = helpers.get_test_directory()
    os.chdir(test_dir)
    for batch, beta, trans_lhs, trans_rhs in itertools.product(
            BATCH_LIST, BETA_LIST, BOOL_LIST, BOOL_LIST):
        filename = get_test_case_filename(batch, beta, trans_lhs, trans_rhs)
        output = test_case_for_transposes(batch, beta, trans_lhs, trans_rhs)
        with open(filename, 'w') as f:
            f.write('\n'.join(output))
        print("File '{}' written".format(filename))


if __name__ == "__main__":
    generate_matmul_tests()
