# Math for machine learning

### Calculating Matrices

<img src="Machine%20learning%20maths.assets/matrix-product-is-defined.jpg" style="zoom:50%;" /> <img src="Machine%20learning%20maths.assets/Matrix_Falk_5.gif" />




### Neuronal nets and matrices

![](Machine%20learning%20maths.assets/Gekritzel-4.jpg)

### Bias in neuronal networks

* Bias is same for all neurons in one layer

* Bias is learnable too

* TF: weights = kernel, bias = bias

<img src="Machine%20learning%20maths.assets/bias.jpeg" style="zoom:25%;" />

## Activation functions

### Sigmoid / Logistic 

* good for
* blablabla

$$ y = \dfrac{1}{1+e^-z} $$

$$\dfrac{\delta y}{\delta z} = y' = y(1-y) =  \dfrac{1}{1+e^-z} \cdot (1 - \dfrac{1}{1+e^-z}) $$

![simoid and derivate](Machine%20learning%20maths.assets/16A3A_rt4YmumHusvTvVTxw.png)

### ReLU (Rectifier linear unit)

* good for
* $$y = max(0, z)$$ meaning z for z greater than 0, otherwise 0
* $$\dfrac{\delta y}{\delta z} = \begin{cases} 0 ~\text{for}~ z \leq 0 \\ 1 ~\text{for}~ z > 0 \end{cases}$$

![relu](Machine%20learning%20maths.assets/1oePAhrm74RNnNEolprmTaQ.png)

### Softmax

<span style="color:red">a_i = label,  -> change to y_j; Maybe use the uncomplicated version from categorical cross entropy below..</span>

* Softmax function takes an N-dimensional vector of real numbers and transforms it into a vector of real numbers in range [0,1] which add up to 1.
* This property of softmax function that it outputs a probability distribution makes it suitable for probabilistic interpretation in classification tasks.
* $$ p_i = \dfrac{e^{a_i}}{\sum_{k=1}^N e^{a_k}} $$
* for float64: upper bound $$10^{308}$$, so make it numerical stable by multiplying numerator and demoniator with constant C: $$ p_i = \dfrac{C \cdot e^{a_i}}{C \cdot\sum_{k=1}^N e^{a_k}} $$ which results in $$ p_i = \dfrac{e^{a_i + \log(C)}}{\sum_{k=1}^N e^{a_k + \log(c)}} $$. C can be choosen free, but normally you use $$\log(C) = -\max(a)$$
* $$\dfrac{\delta p_i}{\delta a_j} = \begin{cases} p_i(1-p_j) ~\text{if}~ i = j \\ -p_j p_i ~~~~~~~~\text{if}~ i \neq j \end{cases}$$
* source: https://deepnotes.io/softmax-crossentropy#cross-entropy-loss

### Others

![](Machine%20learning%20maths.assets/activationfunctions.png)

## Loss functions

### Mean Squared Error (MSE) - L2 Loss

$$t_i = label, y_i = prediction$$ -> change to $$y$$ and $$\hat{y}$$

* t = label/truth/actual; y = predicted
* for **one** sample: $$E = (t_i -y_i)^2$$
* for **many** samples: $$E = \dfrac{1}{n} \sum^n_{i = 1} \dfrac{1}{2}(t_i - y_i)^2$$
* $$\dfrac{\delta E_i}{\delta y_i} = 2*\dfrac{1}{2}(t_i-y_i)*1 = (t_i - y_i)$$
* good for

### Root Mean Squared Error - RMSE

* $$RMSE = \sqrt{MSE}$$

### Mean Absolute Error - MAE - L1 Loss

* MAE = $$(t_i - y_i)^1$$ L1 Loss

### Cross-entropy loss

**aka: Logistic loss, Multinomial Logistic Loss**

Classification problems can be divided into multi-class classification problems and multi-label classification problems. In **multi-class classificiation** one sample is classified to **ONE** class. In **multi-label classification** one sample is classified to **multiple** classes. 

The cross-entropy loss is a group of cross-entropy functions to solve those problems.

![classification](Machine%20learning%20maths.assets/classification.jpg)

The cross entropy is defined as:

* for **many samples**: $$CE = - \dfrac{1}{N} \sum\limits_{k=1}^N \sum\limits_{i=1}^C y_i \log(\hat{y_i}) $$

* for **one sample**: $$CE = - \sum\limits_{i=1}^C y_i \log(\hat{y_i}) $$

<img src="Machine%20learning%20maths.assets/cross_entropy.jpg" alt="cross_entropy" style="zoom:25%;" />

https://www.youtube.com/watch?v=tRsSi_sqXjI

https://gombru.github.io/2018/05/23/cross_entropy_loss/

https://towardsdatascience.com/intuitive-explanation-of-cross-entropy-5d45fc9fd240

#### Binary cross entropy

This is used, when ...

* there are only two labels and you have one single output neuron `0 for C_0 1 for C_1`

* we split a multi-label problem in C binary classification problems. => every output neuron (of the C neurons) will be handled as an individual binary classification problem (is in class C_i/ not in class C_i)

In a binary classification problem, there are only 2 classes! Thus we can simplify to:

for **one** sample: $$CE = - \sum\limits_{i=1}^C y_i \log(\hat{y_i}) = CE = - \sum\limits_{i=1}^2 y_i \log(\hat{y_i}) = - y_1 \log(\hat{y_1}) - (1-y_1) \log (1-\hat{y_1})$$

for **many** samples: $$E(t,y) = - \dfrac{1}{N} \sum\limits_{i = 1}^N \left[ t_i \cdot log(y_i) + (1-t_i) \cdot log(1-y_i) \right]$$

$$\dfrac{\delta E_i}{\delta y_i} = - y_i \dfrac{1}{\hat{y_i}} - (1-y_i) \cdot \dfrac{1}{1-\hat{y_i}} = - \dfrac{y_i}{\hat{y_i}} - \dfrac{1-y_i}{1-\hat{y_i}} = {\dfrac{\hat{y_i} - y_i}{\hat{y_i} (1-\hat{y_i})}}$$

If you **do not use** an activation function like **sigmoid** -> so it is linear, you use:

**Tensorflow**: https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/losses/log_loss

If you add an activation function like **sigmoid**, the $$\hat{y_i}$$ will change to $$sigmoid(\hat{y_i})$$. In Tensorflow you would use **BinaryCrossentropy**

**Tensorflow**: https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy

#### Categorical Cross Entropy Loss

**aka: Softmax loss, softmax activation plus cross entropy loss**

Here a dense output layer with **softmax activation** function is used and we process the output with a **cross entropy loss** function. 

<img src="Machine%20learning%20maths.assets/categorical_cross_entropy.jpg" alt="categorical_cross_entropy" style="zoom:25%;" />

$$CE = - \sum\limits_{i=1}^C y_i \log \dfrac{e^{\hat{y_i}}}{\sum\limits_{j=1}^C e^{\hat{y_j}}} $$

## Learning

### Gradients

* $$grad f = \nabla(x1, .., x_n) = (\dfrac{\delta f}{\delta x_1}, ..., \dfrac{\delta f}{\delta x_n} ) $$
* always directs to the direction of the steepest slope
* usage: negative gradient for minimization of loss function

<img src="Machine%20learning%20maths.assets/grad_1dim.png" style="zoom: 25%;" /><img src="Machine%20learning%20maths.assets/grad_2dim.png" alt="grad_2dim" style="zoom:50%;" />

### Backpropagation <span style="color:red">überarbeiten</span>

<span style="color:red">t_i = label, y_i = prediction -> change to y_i and \hat{y_i}</span>

* Training of neural network by using gradient descent method
* Algorithm: for a single neuron! This can be done with matrices for every layer!

**Step 1:** Forward pass: Calculate output by multiplying input matrix with layers 

* **net function**: weighted sum $$ z = \sum^n_{i=1} x_i \cdot w_i$$

* **activation function**: e. g. simgoid $$y_i(z_i)= \dfrac{1}{1+e^-z_i} $$

* **output function**: usually identity function: $$f_{out}(a) = a$$, this will not be mentioned further, because it is always the same as the activation value

* Process: $$z_i = x_i w_i $$, then $$y_i = sigmoid(z_i)$$ 

**Step 2:** Calculate loss and the gradient for the weight

* **Loss:** $$\mathcal{L}_i = \dfrac{1}{2} \cdot (t_i - y_i)$$ store all losses for plots and testing (and for loss over all samples, which is the goal to minimize)
* **update rule:** $$w_{k+1} = w_k + \Delta w_{k+1}$$ with $$\Delta w = \eta \cdot \dfrac{\delta E}{w}$$
* $$\dfrac{\delta E_i}{\delta w} = \delta_i = \begin{cases} \dfrac{\delta E_i}{\delta y_i} \cdot \dfrac{\delta y_i}{\delta z_i} = \dfrac{\delta E_i}{\delta y_i} \cdot (t_i - y_i) ~\text{for output layer} \\ \dfrac{\delta E_i}{\delta y_i} \cdot \dfrac{\delta y_i}{\delta z_i} \cdot \dfrac{\delta z_i}{\delta w_i} = \dfrac{\delta E_i}{\delta y_i} \cdot \sum_L (\delta_l \cdot w_{l,i}) ~\text{for hidden layer}\end{cases} $$
* for output layer: only derivate of loss fct multiplied with derivate of activation function
* for hidden layer: derivate of loss fct multiplied with the weighted sum of the **next** layer. 
* **gradient**: $$\nabla_{\theta} = x_i \cdot \dfrac{\delta E_i}{\delta w_i}$$

**Step 3:** Backpropagate it

$$\theta = \theta - \eta \cdot \nabla_{\theta}$$ especially: $$w_{k + 1} = w_k - \eta \cdot \nabla_{\theta}$$

* Steps 1 to 3 can be repeated for e epochs or until the average loss is smaller than $$\epsilon$$ or early stopping ($$L_{t-1} - L_{t} < \epsilon$$)

#### Gradient with sigmoid activation function and binary cross-entropy loss function

<span style="color:red">t_i = label, y_i = prediction -> change to y and \hat{y}</span>

* update rule in backpropagation for weights w: $$w_{k+1} = w_k + \Delta w_{k+1}$$ with $$\Delta w = \eta \cdot \dfrac{\delta E}{\delta w}$$
* Calculate gradient for loss function with respect to the weights:
* need to apply chain rule: $$\dfrac{\delta E_i}{\delta w} = \dfrac{\delta E_i}{\delta y_i} \cdot \dfrac{\delta y_i}{\delta z_i} \cdot \dfrac{\delta z_i}{\delta w}$$
* $$E_i$$ = loss = binary cross entropy = $$E(t_i, y_i) = -t_i \cdot log(y_i)-(1-t_i) \cdot log(1-y_i)$$
* $$y_i$$ = activation = sigmoid = $$y_i = \dfrac{1}{1+e^-z_i}$$
* $$z_i$$ = weighted sum = $$x_i \cdot w^T$$ with $$\dfrac{\delta z_i}{\delta w} = x_i$$

* with $$\dfrac{\delta E_i}{\delta y_i} = \dfrac{y_i - t_i}{y_i (1-y_i)}$$; $$\dfrac{\delta y_i}{\delta z_i} = y_i(1-y_i)$$ and $$\dfrac{\delta z_i}{\delta w} = x_i$$

* $$\dfrac{\delta E_i}{\delta w} = \dfrac{\delta E_i}{\delta y_i} \cdot \dfrac{\delta y_i}{\delta z_i} \cdot \dfrac{\delta z_i}{\delta w} = {\dfrac{y_i - t_i}{y_i (1-y_i)}} \cdot y_i(1-y_i) \cdot  x_i = (y_i - t_i) \cdot x_i$$ 
* with $$y_i$$ being the predicted value, $$t_i$$ being label/truth and $$x_i$$ being input value
* See in the web:
* Neural network implemetation - classification: https://peterroelants.github.io/posts/neural-network-implementation-part02/
* Deeper explanation of the math: https://peterroelants.github.io/posts/cross-entropy-logistic/

### Optimizers

* gradient descent is a mathematical method to find the steepest slope. Its negative $$-\nabla$$ form is used to find the steepest negative slope (to find minima). 
* Backpropagation is an evective **algorithm** for applying **Gradient Descent** to a neural network in supervised learning. Therefore gradient descent is used to minimize the loss of a loss function. 
* With Backpropation one can calculate the gradients for the neurons in each layer with the optimizer one optimizes (=adjust) the neurons.
* Common parameters for optimizers are: _learning_rate_, _regularization_ (which gives penalty if a weight dominates by becoming very big) -> overfitting, often SGD (Stochastic Gradient Descent) is used.
* Optimizers change the neural network with respect to the result of the loss function in order to minimize the loss (or error)
* Some popular opzimizers, based on gradient descent are listed below:
* https://algorithmia.com/blog/introduction-to-optimizers

#### Gradient Descent

**parameters: learning rate**

* gradient descent is the grand daddy of all gradient descent based optimizers
* it has variants:
* "_full batched_" gradient descent: take all samples and do a gradient descent (to much cost for large datasets)
* _Stochastic Gradient Descent (SGD)_: only take one sample per gradient descent step
* _mini batch SGD_: Use a batch of independent equally distributed n samples per gradient descent

SGD has trouble navigating ravines, i.e. areas where the surface curves much more steeply in one dimension than in another, which are common around local optima. In these scenarios, SGD oscillates across the slopes of the ravine while only making hesitant progress along the bottom towards the local optimum.

<img src="Machine%20learning%20maths.assets/sgd_without_mom.gif" style="zoom:50%;" /> <img src="Machine%20learning%20maths.assets/sgd_with_mom.gif" style="zoom:50%;" />

<b>Left</b>: without moment, <b>Right</b>: with momentum

Momentum is a method that helps accelerate SGD in the relevant direction and dampens oscillations. It does this by adding a fraction $$\gamma$$ of the update vector of the past time step to the current update vector:

$$v_t = \gamma v_{t-1} + \eta \nabla_{\theta}J(\theta)$$

$$\theta = \theta - v_t$$ 

#### Adagard

**paramters: <span style="color:red">fill in</span>**

* adapts learning rate specifically to individual features
* though, some weights will have different learning rates
* works good for sparse data
* Problem: learning rate tends to get really small over time

#### RMSprop / Adaprop

**paramters: <span style="color:red">fill in</span>**

* special version of Adagard
* Instead of letting all of the gradients accumulate for momentum, it only accumulates gradients in a fixed window
* RMSprop is similar to Adaprop

#### Adam (adaptive moment estimation)

**paramters: <span style="color:red">fill in</span>**

* uses past gradients to calculate current gradients
* utilizes the concept of momentum by adding fractions of previous gradients to the current one

#### Overview

<img src="Machine%20learning%20maths.assets/contours_evaluation_optimizers.gif" style="zoom:50%;" /><img src="Machine%20learning%20maths.assets/saddle_point_evaluation_optimizers.gif" style="zoom:50%;" />

<b>Left</b>: SGD optimization on loss surface contours <b>Right</b>: SGD optimization on saddle point

<span style="color:red">still open to read:</span>**  https://ruder.io/optimizing-gradient-descent/

## Distributions, Probabilities and Likelihoods

### Distributions

**distributions**: Normal, Exponential, Gamma ... (and many more)

#### Normal distribution

![normal_distribution](Machine%20learning%20maths.assets/normal_distribution.jpg)

* The normal distribution gives the probability for a point x with given **mean** (average) $$\mu$$ and **standard deviation** $$\sigma$$. 
* **mean** describes the point, with the highest probability, and the **standard deviation** how strong the points will differ from this mean point (it is the width of the curve)
* for normal distribution:
  * intervall of +/- $$\sigma$$: 68,27 % of all measurements
  * intervall of +/- $$2\sigma$$: 95,45 % of all measurements
  * intervall of +/- $$3\sigma$$: 99,73 % of all measurements
* Distributions are used to describe the probability, where points from experiments will be measured. Therefore a distribution is chosen, that is as similar as possible to the occurence of the measured points.

### Maximum Likelihood

**goal**: Find the optimal way to fit a distribution to the data

#### Recipe to fit a distribution best to the given data

1. Look at the data and choose the right distribution (here **normal distribution** chosen)
2. shift the normal distribution from left to right to find the right **position for $$\mu$$**, let $$\sigma$$ be constant for that

<img src="Machine%20learning%20maths.assets/find_max_likelihood.png" alt="find_max_likelihood" style="zoom:50%;" />

* the maximum of the likelihood is the best position for the given data
* this is the **maximum likelihood estimate for the mean**

3. Change $$\sigma$$ to change the width of the curve. Observe how the likelihood changes. Take the $$\sigma$$ where the likelihood is maximized.

<img src="Machine%20learning%20maths.assets/find_max_likelihood_sigma.png" alt="find_max_likelihood_sigma" style="zoom:50%;" />

Now the distribution fits best to the observed data, by using the maximum likelihood estimations for mean and standard deviation

**Youtube**: https://www.youtube.com/watch?v=XepXtl9YKwc

### Probability vs Likelihood

#### Probability

![dist_probability](Machine%20learning%20maths.assets/dist_probability.png)

**probability** for data is measured the **area under the curve** for a **given mean and standard deviation**

This is used to estimate in which area the new data point will be: $$P(data|distribution)$$

#### Likelihood

![dist_likelihood](Machine%20learning%20maths.assets/dist_likelihood.png)

**likelihood** takes the **point on the curve** for **given data** and we can shift the distribution to the right (by changing $$\mu$$) to better fit to the measurements: $$L(distribution|data)$$

**Youtube**: https://www.youtube.com/watch?v=pYxNSUDSFH4

### Math

Summary:

* Normal distribution: $$P(x|\mu,\sigma) = \dfrac{1}{\sqrt{2 \pi \sigma^2}}e^{\dfrac{-(x-\mu)^2}{2\sigma^2}}$$
* with arithmetic mean $$\bar{x} = \mu = \dfrac{1}{n} \sum_{i=1}^n x_i$$
* and standard deviation $$\sigma = \sqrt{\dfrac{1}{n-1} \sum_{i=1}^n (x_i - \mu )^2}$$
* smaller $$\mu$$ shifts mean to the left, greater $$\mu$$ shifts to the right
* smaller $$\sigma$$ makes Amplitude higher and width smaller, and reverse

#### Maximum Likelihood for n measurements:

$$L(\mu, \sigma|x_1, x_2, ..., x_n) = L(\mu, \sigma|x_1) \cdot ... \cdot L(\mu, \sigma|x_n)  = \prod_1^n L(\mu, \sigma|x_i)$$

$$= \dfrac{1}{\sqrt{2 \pi \sigma^2}}e^{-(x_1-\mu)^2/2\sigma^2} \cdot ... \cdot \dfrac{1}{\sqrt{2 \pi \sigma^2}}e^{-(x_n-\mu)^2/2\sigma^2}$$

Find maximum for $$\mu$$ (position of mean) and $$\sigma$$ (width of curve) by partially derive with respect to $$\mu$$ and $$\sigma$$ and search for slope = 0. 

1. Take log of likelihood function to make derivatives easier. Use for simplification, the following rules:
   1. $$log(a \cdot b) = log(a) + log(b)$$
   2. $$log(a^b) = b \cdot log(a)$$

$$= \ln(\dfrac{1}{\sqrt{2 \pi \sigma^2}}e^{-(x_1-\mu)^2/2\sigma^2} \cdot ... \cdot \dfrac{1}{\sqrt{2 \pi \sigma^2}}e^{-(x_n-\mu)^2/2\sigma^2})$$

See math in Youtube ...

$$\ln[L(\mu, \sigma|x_1 ..., x_n)] = -\dfrac{n}{2} \ln(2\pi) - n \ln(\sigma) - \dfrac{(x_1 - \mu)2}{2\sigma^2} - ... - \dfrac{(x_n - \mu)2}{2\sigma^2}$$

Take partial derivatives:

$$\dfrac{\delta}{\delta \mu} \ln[L(\mu, \sigma|x_1 ..., x_n)]= \dfrac{x_1 - \mu}{\sigma^2} + ... + \dfrac{x_n - \mu}{\sigma^2} = \dfrac{1}{\sigma^2}[(x_1 + ... + x_n) -n\mu]$$

$$\dfrac{\delta}{\delta\sigma} \ln[L(\mu, \sigma|x_1 ..., x_n)]= \dfrac{x_1 - \mu}{\sigma^3} + ... + \dfrac{x_n - \mu}{\sigma^3} = -\dfrac{n}{\sigma} + \dfrac{1}{\sigma^3} [(x_1-\mu)^2 + ... + (x_n-\mu)^2]$$

Now solve the partial derivatives for 0

$$0 = \dfrac{1}{\sigma^2}[(x_1 + ... + x_n) -n\mu] => \mu = \dfrac{(x_1 + .. + x_n)}{n}$$

$$ 0 = -\dfrac{n}{\sigma} + \dfrac{1}{\sigma^3} [(x_1-\mu)^2 + ... + (x_n-\mu)^2] => \sigma = \sqrt{\dfrac{1}{n-1} \sum_{i=1}^n (x_i - \mu)^2}$$

**Youtube**: https://www.youtube.com/watch?v=Dn6b9fCIUpM



## Regularization

* process of adding additional information to prevent overfitting
* is used in loss function
  * Model gets parsimonious

### Norms L1, L2, P-Norm

1-Norm (L1 Norm)

$||w||_1 = |w_1| + |w_2| + ... + |w_N|$



2-Norm (L2 Norm)

$||w||_2 = (w_1^2 + w_2^2 + ... + w_N^2)^\frac{1}{2}$



squared 2-Norm (squred L2 Norm) (see it as complexity of model Google-Developers)

$||w||_2^2 = w_1^2 + w_2^2 + ... + w_N^2$



P-Norm

$||w||_p = (w_1^p + w_2^p + ... + w_N^p)^\frac{1}{p}$

### Regression models

* **Lasso regression** = Regression that uses L1 Norm for regularization
* **Ridge regeression** = Regression that uses L2 Norm for regularization

#### Regression

$\hat{y} = w_1 x_1 + w_2 x_2 + .. + w_N x_N + b$

#### Loss functions

$L = E(y - \hat{y})$

$L = E(y - \hat{y}) + \lambda \sum_{i=1}^N |w_i|$

$L = E(y - \hat{y}) + \lambda \sum_{i=1}^N w_i^2$

$\lambda$ is a chosen hyperparameter

Example for L0, L1, L2 with linear regression, with only one parameter 

$L = (w x +b - y)$

$L = (w x +b - y) + \lambda |w|$

$L = (w x +b - y) + \lambda w^2$

#### Intuition

* Loss function without regularization is not prone to overfitting
* $\lambda$ makes the loss shift away from the ideal weights, so overfitting is reduced 
* $\lambda$ is independent from the model, thus it prevents from overfitting to very less data 
* See this as complexity: L2-Norm adds the complexity of the model

Example: 

$w_1^2 + w_2^2 +  + w_3^2 + w_4^2... + w_5^2 = 1 + 2+ 1+ 25 + 1= 30$ Here only the weight $w_4$ brings the great complexity to the model, but the rest of the weights have less influence

Source for "Complexity view": https://developers.google.com/machine-learning/crash-course/regularization-for-simplicity/l2-regularization



**Deeper reading**: https://towardsdatascience.com/intuitions-on-l1-and-l2-regularisation-235f2db4c261

* L1 penalizes sum of absolute value of weights.

* L1 has a sparse solution

* L1 has multiple solutions

* L1 has built in feature selection

* L1 is robust to outliers

* L1 generates model that are simple and interpretable but cannot learn complex patterns



* L2 regularization penalizes sum of square weights.

* L2 has a non sparse solution

* L2 has one solution

* L2 has no feature selection

* L2 is not robust to outliers

* L2 gives better prediction when output variable is a function of all input features

* L2 regularization is able to learn complex data patterns

List from: https://medium.com/datadriveninvestor/l1-l2-regularization-7f1b4fe948f2