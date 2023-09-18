# VGG and ResNet Convolutional Neural Network Samples for Image Classification

The [VGG16 network][vgg16-paper] and [ResNet50 network][resnet50-paper] are well-known 
convolutional neural networks that outperformed the conventional machine learning 
methods for image classification on the ImageNet dataset in the ILSVRC competition
held in the years 2014 and 2015 respectively. These sample codes implement said networks in 
pure SYCL code, for inference only, showing a larger example of using the portDNN API. 
Some rough instructions for how they might be used are provided.

## Obtaining the model weights and classes

There is a preprocessing script that unpacks the weights from the file into
some application-specific data files (the purpose of the sample is to show
SYCL code, not HDF5). The Python helper requires [h5py][hdf5-python] library.

```bash
mkdir data && pushd data
```
# For VGG16
```bash
wget --no-verbose https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5
python ${PORT_DNN}/samples/networks/vgg/h5toBin.py vgg16_weights_tf_dim_ordering_tf_kernels.h5
popd
```
# For ResNet50
```bash
wget --no-verbose https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels.h5
python ${PORT_DNN}/samples/networks/resnet50/h5toBin.py resnet50_weights_tf_dim_ordering_tf_kernels.h5
popd
```

## Preparing an image

Similarly, any image can be traced against the weights, but the sample does not
have sophisticated image reading code in it. Instead it expects that the image
will be provided as a normalised 224 x 224 x 3 flattened array, so a Python
script is available to perform this transformation for any image and format
supported by the [PIL][py-img-lib].

```bash
python ${PORT_DNN}/samples/networks/img2bin.py my-favourite-pet.jpg
```

## Testing on an image

The portDNN samples are built in the default CMake configuration. The samples
are built by the targets `vgg` and `resnet50` respectively. The samples first must
be passed the directory where the binary weights files are stored and the second 
argument should be the preprocessed picture that should be classified. The expected
output is of a classification index and a series of times in nanoseconds that corresond
to the total time to run the network on an input, not including data transfer time.

# For VGG16
```bash
${PORT_DNN_BUILD_DIR}/samples/networks/vgg/vgg data/ my-favourite-pet.jpg.bin
```
# For ResNet50
```bash
${PORT_DNN_BUILD_DIR}/samples/networks/resnet50/resnet50 data/ my-favourite-pet.jpg.bin
```

## Classifying Images

If you have the tool [`jq`][jq-cite] available, you can obtain better output
from the sample. You will also need the [ImageNet classes][classes].

Something like the following will parse the output from the sample and display
the final classification, obtained from the classes data.

# For VGG16
```bash
class_id=`{PORT_DNN_BUILD_DIR}/samples/networks/vgg/vgg data my-favourite-pet.jpg.bin | \
grep 'classed' | sed -r 's/[^0-9]*([0-9]+).*/\"\1\"/'`
cat imagenet_class_index.json | jq ".$class_id"
```
# For ResNet50
```bash
class_id=`{PORT_DNN_BUILD_DIR}/samples/networks/resnet50/resnet50 data my-favourite-pet.jpg.bin | \
grep 'classed' | sed -r 's/[^0-9]*([0-9]+).*/\"\1\"/'`
cat imagenet_class_index.json | jq ".$class_id"
```

[vgg16-paper]: https://arxiv.org/pdf/1409.1556.pdf
[resnet50-paper]: https://arxiv.org/pdf/1512.03385.pdf
[hdf5-python]: https://www.h5py.org/
[py-img-lib]: https://pillow.readthedocs.io/en/stable/
[jq-cite]: https://stedolan.github.io/jq/
[classes]: https://github.com/raghakot/keras-vis/blob/master/resources/imagenet_class_index.json
