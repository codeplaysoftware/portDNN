# Auto generated convolution tests

The `conv2d_test_gen.py` python script generates a large number of distinct test
cases to cover a number of corner cases and different configurations of 2d
convolutions.

### Generating the tests

Run the script using python (both 2 and 3 are supported) in the `test/conv2d`
directory to generate the test cases.

```
python conv2d_test_gen.py
```

### Dependencies

The script uses TensorFlow to generate the expected output values, as such it
requires the following:

 * TensorFlow
 * Numpy

### Changing the test parameters

The main parameters in the generation of the tests are the window and stride
sizes. The window stride pairs used in the test generation are defined in the
`WINDOW_LIST` and `STRIDE_LIST` lists defined at the start of the script.

```python
WINDOW_LIST = [1, 1, 3, 3, 5, 5]
STRIDE_LIST = [1, 2, 1, 2, 1, 2]
```

Any changes here should be mirrored in the `CMakeLists.txt` file, where the
generated tests are added to the build system.

```
set(_windows 1 1 3 3 5 5)
set(_strides 1 2 1 2 1 2)
```

### Changing the test code

The code which will be included in the generated tests is provided as a number
of strings in the python script. If the generated code needs to be changed then
these strings should be changed.

For example if a new selector type is introduced and should be added to the
generated tests then it should be appended to the selector list:

```cpp
using Selectors = sycldnn::types::TypeList<sycldnn::conv2d::DirectSelector,
                                           sycldnn::conv2d::TiledSelector>;
```

This is defined in the `DATA_TYPES` string, and the required include directive
should be added to the `INCLUDES` string.

