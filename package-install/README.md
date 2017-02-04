# Package install

This repository is related to the install of some package related to
deep-learning

## Keras using Theano or TensorFlow

### Prerequisite

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

### TensorFlow (Google)

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

### Theano (Bengio)

In the same environment install Theano using pip:

```
pip install Theano
```

### Keras (Google - Chollet)

In the same environment install Keras using pip:

```
pip install keras
```

#### Switch backend in Keras:

You can edit `~.keras/keras.json` where you can modify the backend:

``` json
{
    "image_dim_ordering": "th",
    "backend": "theano"
}
```

or

``` json
{
    "image_dim_ordering": "tf",
    "backend": "tensorflow"
}
```

#### Test your install

You can test your install with the following script with Theano as backend.

``` python
# 3. Import libraries and modules
import numpy as np
np.random.seed(123)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

# 4. Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 5. Preprocess input data
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# 6. Preprocess class labels
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

# 7. Define model architecture
model = Sequential()

model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1,28,28)))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# 8. Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 9. Fit model on training data
model.fit(X_train, Y_train,
          batch_size=32, nb_epoch=10, verbose=1)

# 10. Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=0)
```

Execute the script with `ipython`


## Pytorch (Facebook)

* Create a new environment with conda and activate the environment:

```
conda create -n pytorch python=2.7
source activate pytorch

```

* Install Pytorch using conda:

```
conda install pytorch torchvision cuda80 -c soumith
```

### Tutorial

A list of tutorial is available [there](https://github.com/pytorch/tutorials)
