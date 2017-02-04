# Package install

This repository is related to the install of some package related to
deep-learning

## Prerequisite

Install a miniconda on your session by following the help [there](
https://github.com/MickeyMouseScienceReadingGroup/deep-learning)

Edit your `.bashrc` to add the following lines:

```
# added for Cuda 8.0
export PATH="/usr/local/cuda-8.0/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH"

# define environment variable for Theano
export CUDA_ROOT="/usr/local/cuda-8.0"
export THEANO_FLAGS=blas.ldflags="-L/etc/alternatives -lblas",device=gpu

# define home for tensor flow
export CUDA_HOME="/usr/local/cuda-8.0"

# deactivate one of the GPU
export CUDA_VISIBLE_DEVICES="GPU-c213bb11-596c-604d-4636-b48995a5a125"
```

## TensorFlow

You can find the full documentation [there](
https://www.tensorflow.org/get_started/os_setup)

However, you can follow the simplify documentation:

* Create a new environment with conda and activate the environment:

```
conda create -n tensorflow python=2.7
source activate tensorflow

```

* Export the URL of to get the Python wheel (CUDA 8.0/cuDNN 5.0):

```
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp27-none-linux_x86_64.whl

```

* Install Tensorflow through pip:

```
pip install --ignore-installed --upgrade $TF_BINARY_URL
```

## Theano

In the same environment install Theano using pip:

```
pip install Theano
```

## Keras

In the same environment install Keras using pip:

```
pip install keras
```

## Switch backend in Keras:

You can edit `~.keras/keras.json` where you can modify the backend:

``` json
    "image_dim_ordering": "th",
    "backend": "theano"
```

or

``` json
    "image_dim_ordering": "tf",
    "backend": "tensorflow"
```
