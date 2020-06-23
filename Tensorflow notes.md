# Tensorflow notes

**Contents**

[Convolutional Neural Networks (CNN), Conv2D, Pooling, Flatten, Dense Layers ](#Convolutional-Neural-Networks-(CNN))

[Tensors in TF](#Tensors)

[GradientTape, Reverse Mode Automatic Differentiation](#GradientTape)

[Numpy notes](#Numpy-notes)

## Convolutional Neural Networks (CNN) 

Easy and shallow article: https://towardsdatascience.com/the-most-intuitive-and-easiest-guide-for-convolutional-neural-network-3607be47480

Very deep and pervasive article: https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215

### Conv2D

Convolutional layers do a convolution of the input data with the filters. In detail, they do a cross-correlation, because the filters are not reversed before the operation. But since the weights of the filters are learned, it will in the end work like a convolution, so we say "convolution". 

**Doc:** https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D

**Parameters:**

* **filters**: [32, 64, 128] number of different filters that shall be applied to the image (also called kernel) 
* **kernel size**: [(3, 3)] tupel, which determines how big the different filters are
* **strides**: [1 or (1,1)] tuple/list of 2 integers, specifying the step size of the moving convolution window.
* **input_shape**: [e.g. (28, 28, 1)] defines the input shape of the image, (x-axis, y-axis, channels). But attention: the feeded numpy-Array or Tensor has to have shape (n, x, y, c) because n determines how many images have to be processed! This is only used in the first layer of the network.
* **activation**: ['relu']: the activation function
* **dilation_rate**: [1] an integer or tuple/list of 2 integers, specifying the dilation rate to use for dilated convolution, see animation for clarification
* **padding**: ['same'] adds padding to the image. 'valid' means no padding, 'same' means that there is padding (with zeros) such that the output will have same dimensions as input (if you use strides=1). 
* (**data_format**: default: channels_last )

**Tensorflow:**

```python
self.conv2d_1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), input_shape=(28, 28, 1),activation='relu', padding='same')
```

**Learnable variables:**

* kernel weights of each filter

**Input shape**:

4D tensor with shape: `(samples, channels, rows, cols)` if data_format='channels_first' or 4D tensor with shape:`(samples, rows, cols, channels)` if data_format='channels_last'.

**Output shape**:

4D tensor with shape: `(samples, filters, new_rows, new_cols)` if data_format='channels_first' or 4D tensor with shape: `(samples, new_rows, new_cols, filters)` if data_format='channels_last'. `rows` and `cols` values might have changed due to padding.

**Convolution:**

![pooling](Tensorflow%20notes.assets/pooling.gif)

**1**: Convolution with kernel (3,3) , stride (1,1) and padding 'valid'

**2**: Convolution with kernel (4,4), stride (1,1) . The padding is done with an extra preceding ZeroPadding2D-Layer with padding=2. You can use it as follows:

**Doc**: https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding2D

```python
keras.layers.ZeroPadding2D(padding=(2), data_format=**None**) # 2 in for width and height
keras.layers.ZeroPadding2D(padding=(2, 2), data_format=**None**) # width, height 
keras.layers.ZeroPadding2D(padding=((2, 2), (2, 2)), data_format=**None**) # top, bottom, left, right
```

**3**: Convolution with kernel (3,3), stride (1,1), and padding 'same'

**4**: Convolution with kernel (3,3), stride (1,1) . Padding=2

<img src="Tensorflow%20notes.assets/dilation.gif" alt="dilation" style="zoom:50%;" />

**1**: Convolution with kernel(3, 3), strides=1 and dilation_rate=1

**Example for 1 dimensional data with 3 filters**: grayscale images

<img src="Tensorflow%20notes.assets/conv2d_1_channel.png" alt="conv2d_1_channel" style="zoom: 75%;" />

**Left**: Input with shape 9x9, **Middle**: Three filters with kernel size (3,3) and stride=1 with no padding, **Right**: Three outputs (because three filters) with shape 7x7

**Notice**: The dimensionality of the output is greater than the input (because every filter has its own output), but the actually learned parameters are only the weights of the filters. 

**Example for 3 dimensional data with 1 filter**: RGB images

In images we have normally three channels. In this example here is **one** 3-dim filter applied. The convolution can be seen like this:

![3dim_1filter](Tensorflow%20notes.assets/3dim_1filter.gif)

In Conv2D we describe this filter with kernel size (3, 3) but in reality this filter is (3, 3, 3). With 3 * 3 * 3 = 27 weights. In detail these three input images can be seen as a stack of 3 individual 2D shaped surfaces stacked over each other. In the following image, they are unstacked.

![3d_image_1filter](Tensorflow%20notes.assets/3d_image_1filter.jpeg)

Here, there is **one** filter. When we look at each single channel, we can see it as 3 images (with one channel) with 3 own kernels. In fact all weights of each single partial kernel can be learned. The three Intermediate outputs then will be summed element-wise. 

### Pooling

Pooling is the process of merging. It's purpose is to reduce data. There are two kinds of pooling:

**Doc**: https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D

* **MaxPooling**: takes the highest value from inputs
* **AveragePooling**: takes the average value from inputs

**Parameters:**

* **pool_size**: [(2,2)] factors to downscale 
* **strides**: [None] (can also int or tuple): step size of steps, None will default to pool_size
* (**data_format**: defines shape of data channels_last:  (batch, height, width, channels) channels_first: (batch, channels, height, width) 
* (**activation_function**: defaults to 'linear')

**Tensorflow:**

```python
self.max_pool_1 = tf.keras.layers.MaxPooling2D()
self.avg_pool_1 = tf.keras.layers.AveragePooling2D()
```

![pooling](Tensorflow%20notes.assets/pooling.png)

### Flatten

Flattening is used at the end of CNNs. It converts the data to a (1,) dim array for inputting it into the next layer. Normally a fully-connected (dense) layer comes after this and has then the activation_function='softmax'.

**Doc**: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten

**Parameters**:

* (**data_format**), no need to use..

**Tensorflow:**

```python
self.flatten_1 = tf.keras.layers.Flatten()
```

![flattening](Tensorflow%20notes.assets/flattening.png)

### BatchNorm

**Tutorial**: https://www.youtube.com/watch?v=dXB-KQYkzNU

**Andrew Ng:** https://www.youtube.com/watch?v=nUUqwaxLnWs

**Andrew Ng about test-time:** https://www.youtube.com/watch?v=5qefnAek8OA

* **Doc**: https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization
* "Normalize the activations of the previous layer at each batch, i.e. applies a transformation that maintains the **mean activation close to 0** and the **activation standard deviation close to 1**."
* Why we normalize input data:
  * **Problem I**: Different input scales (e. g. age [0..100] and distance [0.100000]) or big scales in the same axis, say miles flights between 0 and 100000 can lead to instability or slow down training much
  * **Solution**: normalize in range [0..1]
* What to do, with hidden units? This is where we use batch normalization
  * **Problem II**: one ore some weights becomes higher than the others
  * **Solution**: normalize in range [0..1]
  * Ng: it reduces the covariance shift for the hidden layer with respect to its inputs. If you think of being a hidden layer, your inputs are changing all the time, so it slows down your learning. By reducing this covariance shift, you speed up the leraning for you. 

1. Normalize output from activation function: $$z = \dfrac{x-\mu}{\sqrt{\sigma^2+ \epsilon}}$$ with $$\mu = \dfrac{1}{N}\sum_i x_i$$ being the mean and $$\sigma^2 = \sqrt{\dfrac{1}{N} \sum_i^N (x_i - \mu)^2}$$ being the stanard deviation of the activation. This leads to a mean of zero.
2. $$f_{out}(z) = z \cdot \gamma + \beta$$

Result: $$f_{out} = \dfrac{x-m}{s} \cdot \gamma + \beta$$ with $$m, s, \gamma, \beta$$ being trainable variables using axis= e.g. 1 for **feature axis**

* $$\beta$$ manages, that the mean will be near to 0 and $$\gamma$$ manages, that the standard deviation of the activation is close to 1
* Parallels to dropout: Since batch normalization is only calculating normalization on the mini-batch, there is some noise in the values of z. This is a similar effect as dropout layers. This has a slight regularization effect, because noise in the hidden units force the neural network not to rely to heavy on the output of a single neuron. So the higher the batch size, the less regularization. Since this fact is true, one normally uses Dropout for regularization and not Minibatch. It's only a side-effect.
* Training / Testing: 

**Parameters**

* **beta_initializer:** [0] initial beta weight
* **gamma_initializer:** [1] initial gamma weight
* **epsilon**: [?]: small float added to variance to avoid dividing by zero
* **training**: [true, if training (using mean and variance of batch), false if testing (using mean and variance of its moving average, which has been learned during training)]

**Trainable variables**

* $$m, s, \gamma, \beta$$

### Dropout

add dropout here

has training mode (does dropout) and inference mode (no dropout) for evaluation and inference ;-)

### Dense

Dense layers are the "normal" layers in neuronal networks. They can be seen as matrices where **Dense** implements the operation `output = activation(dot(input, kernel) + bias)`

**Note:** Never displayed in images, but true: a dense layer has a number of kernel units (=weights) and one bias unit (bias weight). 

<img src="Tensorflow%20notes.assets/dense_bias.jpg" alt="dense_bias" style="zoom:25%;" />

As **hidden layer** they normally have a big number of units and activation functions like 'relu', at the end of a CNN as a **output layer** they normally have a number of units that fits to the encoding of the output.

For **one-hot encoding** there is e. g. a number of 5 units and the output can be seen as a vector of some form like $$\vec{v}^T = \{ 1, 0, 0, 0, 0\}$$. The dense output layer with a softmax activation function gives a probability distribution over the 5-way one-hot encoded output. This is used in classifiers.

**Doc**: https://keras.io/layers/core/

**Parameters**:

* **units**: number of neurons in this layer

* **activation** is element-wise activation function 

**Trainable variables:**

* **kernel** is a weight matrix created by the layer

* **bias** is a bias vector created by the layer (only applicable if `use_bias == True` ): Only one bias per Layer!

*Note: If the input to the layer has a rank greater than 2, then it is flattened prior to the initial dot product with **kernel**.*

**Tensorflow:**

```python
self.hidden = tf.keras.layers.Dense(units=40, activation='relu')
self.dense_out = tf.keras.layers.Dense(units=5, activation='softmax')
```

### A complete CNN

![full_cnn](Tensorflow%20notes.assets/full_cnn.jpeg)

* Omniglot dataset has (28,28,1) shaped images.
* They are fed to a **Conv2D** with (5, 5) shaped kernels and 'valid' padding. Since there is no padding, the resulting output is only (24,24, n1) shaped. n1 is the number of filters applied to the input.
* **MaxPooling2D** with (2, 2)
* Then **Conv2D** like first, results in (8, 8, n2) where n2 is number of kernels in first Conv2D times number of kernels in second Conv2D. 
* **MaxPooling2D** with (2, 2) results in (4, 4, n2)
* **Flatten** results in 1* n2 * 4 * 4 
* How to calculate:
  * dim of flattened * num of filters * image_width * image_height y ) n3 = 1 * n2 * 4* 4
* **Output** has dim 10 (e. g. 10 classes)

## Tensors

Tensors are a standard way of describing scalars, vectors, matrices and other analogous structures. Tensors are used a lot in Tensorflow. A numpy-array is the representation of tensors. I think on tensors like a generalization of all the structures that we use in mathematics/ML/big data for representing data.

Visual explanation of mathematics: https://www.youtube.com/watch?v=f5liqUk0ZTw

<img src="Tensorflow%20notes.assets/tensors.jpeg" alt="tensors" style="zoom:50%;" />

| Rank | Math                                                         | Numpy                                                        | dim  |  shape......... |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ---- | --------------: |
| 0    | Scalar: e. g. 1337                                           | int `np.array(1337)`                                         | 1    |           (1, ) |
| 1    | Vector e.g. $$\vec{v} = ( 1, 2, 3)$$                         | 1-dim array: `np.array([1, 2, 3)`                            | 1    |           (3, ) |
| 2    | Matrix e. g. $$M=   \begin{bmatrix} 1 & 2 & 3 \\  4 & 5 & 6 \end{bmatrix} $$ | 2-dim array: `np.array([[1,2,3], [4,5,6]])`                  | 2    |          (3, 3) |
| 3    | Cube or vector of matrices                                   | 3-dim array: `np.array([[[1,2,3], [4,5,6]], [[1,2,3], [4,5,6]], [[1,2,3], [4,5,6]]])` | 3    |       (3, 2, 3) |
| 4    | vector of cubes                                              | `x = np.array([[[[1,2,3], [4,5,6]], [[1,2,3], [4,5,6]], [[1,2,3], [4,5,6]]], [[[1,2,3], [4,5,6]], [[1,2,3], [4,5,6]], [[1,2,3], [4,5,6]]], [[[1,2,3], [4,5,6]], [[1,2,3], [4,5,6]], [[1,2,3], [4,5,6]]]])` | 4    |    (3, 3, 2, 3) |
| 5    | matrix of cubes                                              | ...                                                          | 5    | (3, 3, 3, 2, 3) |

* **Attention**: Numpy-arrays as well as Tensorflow-tensors can be imagined like a scalar, vector, matrix etc. but the data structure behind it is not! So the rank 2 tensor can be seen as a matrix for imagination but the data structure internal is just a buffer of scalars that are indexed by views (in the case of a matrix by 2 views). 

* We can use **Numpy-Arrays** as well as **Tensorflow-Tensors** for feeding neural networks and functions in Tensorflow.

**Common shapes of tensors**

| shape                                                | application                                                  |
| :--------------------------------------------------- | ------------------------------------------------------------ |
| `(batch_size, dim)`                                  | Input for neural net (dim = number of inputs). Here a vector of dim inputs is fed to the neural network |
| `(batch_size, pixel_x, pixel_y, channels)`           | Images                                                       |
| `(batch_size, timestep, pixel_x, pixel_y, channels)` | Video                                                        |

## GradientTape

Differentiation is used a lot in machine learning. In Tensorflow a technique called **Reverse Mode Automatic Differentiation** is used. There are actually three different approaches to solve differentiations. Humans use the variant called **symbolic differentiation** (this is using rules to determine derivative). We then get something like $$F(x) = x^2 \rightarrow f(x) = 2x \rightarrow f'(x) = 2$$. With this, we can describe the whole graph and by applying x-values to the function, we can derive the y-values. The second approach is **numerical differentiation**. The main idea is something like we approximate, the derivative in a point by using something like this: $$f'(x) = \dfrac{f(x + h) - f(x)}{h}$$. If we compute it for many x-values, we can approximate the derivative y-values for many points of f(x). The third approach is **Automatic Differentiation**. In Tensorflow the **Reverse Mode Automatic Differentiation** is used.

### Reverse Mode Automatic Differentiation

**Principal idea**:

* split a complex equation in easy equations by applying the chain rule
* draw a computation graph and connect the elements of the chain
* solve derivations by forward pass calculations with small deltas.
* Notice: Using this method, we need more RAM. We have to store the graph and its forward pass values.

* explained in a symbolic manner

https://stats.stackexchange.com/questions/224140/step-by-step-example-of-reverse-mode-automatic-differentiation

* explained with computation graph

https://www.youtube.com/watch?v=nJyUyKN-XBQ

* **example**: $$J = 3 (a + b \cdot c) $$
* **1**: Split into sub equations and draw computation graph
* $$J = 3v$$
* $$v = a + u$$
* $$u = bc$$

<img src="Tensorflow%20notes.assets/computationgraph.jpeg" style="zoom: 25%;" />

* **2**: Compute forward pass

* **3**: Compute derivations by changing values of the variables

| $$\Delta$$       | symbolic  |
| ---------- | ------------ |
| $$J = 33$$ and $$v = 11 \rightarrow 11.001$$ then $$J = 33.003$$ |  |
| $$\dfrac{\Delta J}{\Delta v} = \dfrac{0.003}{0.001} = 3$$ | $$\dfrac{\delta J}{\delta v} = 3$$ |
| $$\dfrac{\delta J }{\delta a} = ?$$ |  |
| $$a = 5 \rightarrow 5.001$$ and $$v= 11 \rightarrow 11.001$$ |  |
| $$\dfrac{\Delta v}{\Delta a} = \dfrac{0.001}{0.001} = 1$$ |  |
| here is a chain $$ a\rightarrow v \rightarrow J$$ it follows: $$\dfrac{\delta J}{\delta a} =\dfrac{\delta J}{\delta v} \cdot \dfrac{\delta v}{\delta a} = 3 \cdot 1 = 3$$ | $$\dfrac{\delta J}{\delta a} =\dfrac{\delta J}{\delta v} \cdot \dfrac{\delta v}{\delta a}$$ |

### GradientTape in Tensorflow

* https://www.tensorflow.org/tutorials/customization/autodiff

GradientTape is the structure in Tensorflow, that we use for differentiation.

GradientTape records all operations executed inside the context of tf.GradientTape onto a tape. It stores also the results of forward pass and the gradients. It uses therefore the reverse mode automatic differentiation. Gradients can be calculated by using gradienttape.gradients.

* gradientTape.watch(x) is used to make TF see x (which is no trainable variable (e.g. some constants)).
* persistent=True is used if you need to calculate gradients more than one time (for costs of RAM, is not needed often)

**Example for doing several gradients with `persistent=True`**

```python
# Create input Tensor (Matrix with ones)
x = tf.ones((2, 2))
print(x)

# tell GradientTape to watch x 
# Use persistent=True to be able to use gradient method more than one time
with tf.GradientTape(persistent=True) as t:
    t.watch(x)
    y = tf.reduce_sum(x) # = 4
    print(y)
    z = tf.multiply(y, y) # = 16
    print(z)

    # so we have z = y*y with y= x_1 + x_2 + x_3 + x_4 
    # so z = (x_1 + x_2 + x_3 + x_4 )^2
# dz/dx_i
dz_dx = t.gradient(z, x)
for i in [0, 1]:
    for j in [0, 1]:
        print(dz_dx[i][j].numpy())
        assert dz_dx[i][j].numpy() == 8.0

# dz/dy
dz_dy = t.gradient(z, y)
print(dz_dy)
assert dz_dy.numpy() == 8.0

# dy/dx
dy_dx = t.gradient(y, x)
print(dy_dx)
assert dy_dx[0][0].numpy() == 1.0

# IMPORTANT: Delete pointer to tape if persistent=True to give it free to garbage collector
del t
```

**Example for calculating higher order gradient (no `persistent=True`needed)**

```python
x = tf.Variable(1.0)  # Create a Tensorflow variable initialized to 1.0

with tf.GradientTape() as t:
  with tf.GradientTape() as t2:
    y = x * x * x #x^3
  # Compute the gradient inside the 't' context manager
  # which means the gradient computation is differentiable as well.
  dy_dx = t2.gradient(y, x) # y' = 3x^2
d2y_dx2 = t.gradient(dy_dx, x) # y'' = 6x

assert dy_dx.numpy() == 3.0
assert d2y_dx2.numpy() == 6.0
```

## logits

**Doc**: https://developers.google.com/machine-learning/glossary/#logits

The vector of raw (non-normalized) predictions that a classification model generates, which is ordinarily then passed to a normalization function. If the model is solving a [**multi-class classification**](https://developers.google.com/machine-learning/glossary/#multi-class) problem, logits typically become an input to the [softmax function](https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits_v2). The softmax function then generates a vector of (normalized) probabilities with one value for each possible class.

In addition, logits sometimes refer to the element-wise inverse of the [**sigmoid function**](https://developers.google.com/machine-learning/glossary/#sigmoid_function). For more information, see [tf.nn.sigmoid_cross_entropy_with_logits](https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits).

e. g.:

```python
def loss_function(self, pred_y, y):
    return keras_backend.mean(keras.losses.categorical_crossentropy(y, pred_y))

def compute_loss(model, x, y, loss_fn=loss_function):
    logits = model.forward(x)
    cce = loss_fn(y, logits)
    return cce, logits
```

## Tensorboard methods

### Sequential

build model with .Sequential()..

### Functional

```python
inputs = tf.keras.Input(shape=(32,))  # Returns an input placeholder

# A layer instance is callable on a tensor, and returns a tensor.
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(64, activation='relu')(x)
predictions = layers.Dense(10, activation='softmax')(x)
```

### Custom layers

Mischung aus Sequential und Functional

custom layers mit call()...

### Custom training

no model.fit(), model.evaluate(), hier gradientTape...

### Tensorboard

