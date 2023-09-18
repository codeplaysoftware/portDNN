# Auto generated convolution tests

The python scripts in `test/gen` generate a large number of distinct test
cases to cover a number of corner cases and different configurations of the
different operations provided by portDNN.

### Generating the tests

We provide a simple script to generate all tests using `python` (both 2 and 3 are
supported) in the `test/gen` directory.

```
python test/gen/generate_all.py
```

### Dependencies

The script uses TensorFlow to generate the expected output values, as such it
requires the following:

 * TensorFlow 2.x
 * Numpy

### Changing the test parameters

The tests generators generate different test cases based on a number of
parameters. These can be changed in the scripts to generate different tests, but
any changes to the filenames and generated files must also be reflected in
changes to the build system, otherwise new tests will not be compiled or run.

#### Conv2D tests

The main parameters in the generation of the conv2d tests are the window and stride
sizes. The window stride pairs used in the test generation are defined in the
`WINDOW_LIST` and `STRIDE_LIST` lists defined at the start of the
`generate_conv2d_tests.py` script.

```python
WINDOW_LIST = [1, 1, 3, 3, 5, 5]
STRIDE_LIST = [1, 2, 1, 2, 1, 2]
```

Any changes here should be mirrored in the corresponding `CMakeLists.txt` file,
where the generated tests are added to the build system.

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
