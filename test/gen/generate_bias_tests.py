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
# Automatically generate the bias test cases using TensorFlow to provide
# the expected values.

from __future__ import print_function

import itertools
import os
from collections import namedtuple

import tensorflow as tf
import numpy as np

import helpers

BATCHES = [1, 2, 4]
CHANNELS = [1, 2, 4]
IN_SIZES = [1, 2, 4]
TEST_TYPES = ["BIAS"]

INCLUDES = r"""
#include <gtest/gtest.h>

#include "test/types/cartesian_product.h"
#include "test/types/kernel_data_types.h"
#include "test/types/test_backend_types.h"
#include "test/types/to_gtest_types.h"

#include "test/bias/bias_fixture.h"

#include <array>
#include <vector>"""
DATA_TYPES = r"""
using namespace sycldnn; // NOLINT(google-build-using-namespace)
using DataTypeList = sycldnn::types::KernelDataTypes;
using Backends = sycldnn::types::DefaultBackendTypes;

using SNNTypePairs =
    sycldnn::types::CartesianProduct<DataTypeList, Backends>::type;
using GTestTypePairs = sycldnn::types::ToGTestTypes<SNNTypePairs>::type;"""
TYPED_TEST_SUITE_DECL_TPL = r"""
template <typename Pair>
using {test_case} =
        BiasFixture<typename Pair::FirstType, typename Pair::SecondType>;
TYPED_TEST_SUITE({test_case}, GTestTypePairs);"""

TestCaseParams = namedtuple(
    'TestCaseParams',
    ['test_type', 'param_gen'])
TestParams = namedtuple('TestParams', ['in_shape'])


def get_bias_results(max_val, input_shape):
    """
    Compute bias-add.

    Will create an input tensor of the required size filled with values 1, 2,
    3... and use these to compute the bias-add.
    Returns the computed values in a numpy array.
    """
    total_inp_size = np.product(input_shape)

    input_vals = helpers.get_tensor_data(total_inp_size, max_val)
    bias_vals = helpers.get_tensor_data(input_shape[3], max_val)

    inp_tensor = tf.constant(input_vals,
                             shape=input_shape,
                             dtype=np.float64)
    bias_tensor = tf.constant(bias_vals,
                              shape=(input_shape[3],),
                              dtype=np.float64)
    return tf.nn.bias_add(inp_tensor,
                          bias_tensor,
                          data_format="NHWC")


def get_result_function(test_case):
    """
    Get the function which will compute the expected values for the given test case.
    """
    return get_bias_results


TEST_CASE_TPL = "{test_type}"
TEST_NAME_TPL = "{in_s[0]}x{in_s[1]}x{in_s[2]}x{in_s[3]}"
IN_SHAPE_INIT_TPL = "{{{{ {0[0]}, {0[1]}, {0[2]}, {0[3]} }}}}"


def get_result_and_size(test_case, test_params):
    """
    Get the result of the bias-add and max input value.

    Ensures that the resulting values are less than the REQUIRED_MAX, and if
    not will adjust the maximum value to allow in the input tensors.
    """
    return helpers.get_result_and_size(get_result_function(test_case),
                                       input_shape=test_params.in_shape)


def get_test_lines(test_case, test_params):
    """
    Create a list of strings corresponding to the lines in a single test case.

    Uses TensorFlow to compute the expected results for the given parameters,
    and provides the code to call the test fixture to run the test.
    """
    output, max_input_val = get_result_and_size(test_case, test_params)
    camel_case_type = helpers.to_camel_case(test_case.test_type)
    test_case_name = TEST_CASE_TPL.format(test_type=camel_case_type)
    test_name = TEST_NAME_TPL.format(in_s=test_params.in_shape)
    in_shape_init = IN_SHAPE_INIT_TPL.format(test_params.in_shape)
    test_lines = [
        "TYPED_TEST({}, {}) {{".format(test_case_name, test_name),
        "  using DataType = typename TestFixture::DataType;",
        "  const std::vector<DataType> exp_out = {};".format(
            helpers.format_tensor(output)),
        "  const std::array<int, 4> in_shape = {};".format(in_shape_init),
        "  const auto params = getBiasParams(in_shape);",
        "  const DataType max_input_val = {:.1f};".format(max_input_val),
        "  this->test_bias(exp_out, params, max_input_val);",
        "}",
    ]
    return test_lines


def test_params_for_test_case(test_case):
    "Test params generator for all different tests in a given test case."
    for in_shape in itertools.product(BATCHES, IN_SIZES, IN_SIZES, CHANNELS):
        yield TestParams(in_shape=in_shape)


def output_for_test_case(test_case):
    """
    Create a list of strings corresponding to separate lines in the full test
    case. The output contains headers, includes, setup and all the tests for
    the test case.
    """
    camel_case_type = helpers.to_camel_case(test_case.test_type)
    test_case_name = TEST_CASE_TPL.format(test_type=camel_case_type)
    output = [
        TYPED_TEST_SUITE_DECL_TPL.format(
            test_case=test_case_name)
    ]

    for test_params in test_case.param_gen(test_case):
        output.extend(get_test_lines(test_case, test_params))
    output.append("\n")
    return output


def get_initial_boilerplate():
    """ Get the boilerplate for the top of the test file. """
    scriptname = os.path.basename(__file__)
    return [
        helpers.get_license(),
        helpers.get_dont_modify_comment(scriptname=scriptname),
        INCLUDES,
        DATA_TYPES,
    ]


FILENAME_TPL = "bias/test_bias.cc"


def get_test_case_filename(test_case):
    "Get filename for test case."
    return FILENAME_TPL.format(test_type=test_case.test_type)


def test_cases(param_gen):
    "Test case generator giving all possible test cases."
    for test_type in TEST_TYPES:
        yield TestCaseParams(test_type=test_type, param_gen=param_gen)


def generate_bias_tests():
    np.set_printoptions(suppress=True, threshold=1000000, linewidth=1000000)
    test_dir = helpers.get_test_directory()
    os.chdir(test_dir)
    for test_case in test_cases(test_params_for_test_case):
        filename = get_test_case_filename(test_case)
        output = get_initial_boilerplate()
        output.extend(output_for_test_case(test_case))
        with open(filename, 'w') as f:
            f.write('\n'.join(output))
        print("File '{}' written".format(filename))


if __name__ == "__main__":
    generate_bias_tests()
