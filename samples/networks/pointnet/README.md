# PointNet Convolutional Neural Network Sample for 3D Pointcloud Classification

[PointNet][pointnet-paper] is a convolutional neural network architecture for applications concerning 3D recognition such as object classification and part segmentation. These sample codes implement a variant of PointNet for 3D object classification in pure SYCL code, for inference only, showing a larger example of using the SYCL-DNN compat layer. Some rough instructions for how it might be used are provided.

## Obtaining the model weights and classes and preparing an input pointcloud

A preprocessing script is provided which unpacks the weights from a pretrained pytorch model. The script also prepares an input pointcloud for testing inference. The pointcloud is made from 3D scans taken from the [ModelNet10][modelnet] dataset. The script requires an installation of [PyTorch][pytorch]. First download the pretrained PointNet weights from https://github.com/codeplaysoftware/Pretrained-Models and move the pth file into the `samples/pointnet/` directory.

```bash
cd sycl-dnn/samples/networks/pointnet
wget http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip 
unzip ModelNet10.zip
rm ModelNet10.zip
python3 prepareData.py ModelNet10/ pointnet_model.pth
```

The weights will be saved to `data/` and the input pointcloud will be saved as `input_cloud.bin`

## Testing on a pointcloud

The SYCL-DNN samples are built in the default CMake configuration. The sample
is built by the target `pointnet`. The samples must first
be passed the directory where the binary weights files are stored and the second 
argument should be the preprocessed pointcloud that should be classified. The expected
output is of a classification index and a series of times in nanoseconds that corresond
to the total time to run the network on an input, not including data transfer time.

```bash
pointnet ${SYCL_DNN_DIR}/samples/networks/pointnet/data/ ${SYCL_DNN_DIR}/samples/networks/pointnet/pointnet/data/input_cloud.bin

```

[pointnet-paper]: https://arxiv.org/pdf/1612.00593.pdf
[pytorch]: https://pytorch.org/
[modelnet]: https://modelnet.cs.princeton.edu/
