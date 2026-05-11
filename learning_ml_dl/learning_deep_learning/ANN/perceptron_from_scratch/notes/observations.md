# Observations — Single Perceptron from Scratch

## 1. Purpose of this observations file

This file records the detailed observations from running the `perceptron_scratch.py` project.

The goal of the project is not to build a production-level neural network. The goal is to understand, from first principles, how a single perceptron behaves when it is implemented directly using raw PyTorch tensors.

The script demonstrates the complete learning loop:

```text
input features
    ↓
linear combination: z = w · x + b
    ↓
sigmoid activation: y_hat = sigmoid(z)
    ↓
binary cross-entropy loss
    ↓
automatic differentiation using loss.backward()
    ↓
manual gradient descent update
    ↓
gradient zeroing
    ↓
repetition over multiple training steps
```

The project is intentionally simple. It uses only one training sample so that every numerical value can be inspected and understood clearly.

---

## 2. High-level observation from the complete run

The script ran successfully and produced the expected learning behavior.

The main observed behavior is:

```text
z increases
y_hat increases toward the true label 1
loss decreases
gradient magnitude decreases
weights increase
bias increases
```

This is the correct behavior because the true label is:

```text
y_true = 1.0
```

and the initial prediction is already close to 1:

```text
y_hat = 0.967849
```

Since the prediction is correct but not perfect, gradient descent makes the perceptron slightly more confident by increasing the raw score `z`.

---

## 3. Observed input sample

The script uses one input sample:

```python
x = tensor([7.4000, 0.7000, 3.5100])
```

This input has three features.

In the teaching example, these features are interpreted as simplified wine-quality-related quantities:

```text
x[0] = fixed acidity
x[1] = volatile acidity
x[2] = pH
```

The exact physical meaning is not important for the mathematical demonstration. What matters is that the perceptron receives a vector of numerical features.

The shape of the input is:

```text
Shape of x = (3,)
```

This means it is a one-dimensional tensor containing three numbers.

---

## 4. Observed initial weights and bias

The script initializes the weights using:

```python
torch.manual_seed(42)
w = torch.randn(3, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```

The observed initial values are:

```text
Initial weights w = tensor([0.3367, 0.1288, 0.2345])
Initial bias b    = tensor([0.])
```

The weight tensor has shape:

```text
Shape of w = (3,)
```

The bias tensor has shape:

```text
Shape of b = (1,)
```

The important observation is that the number of weights matches the number of input features.

Because the input has 3 features, the perceptron requires 3 weights:

```text
one input feature → one corresponding weight
```

The bias is a separate trainable scalar-like quantity. It shifts the raw score independently of the input features.

---

## 5. Observation about reproducibility

Because the code uses:

```python
torch.manual_seed(42)
```

the same random weights are generated every time the script is run.

This is why the initial weights repeatedly appear as:

```text
tensor([0.3367, 0.1288, 0.2345])
```

This is useful for learning because it makes the output reproducible.

Without setting the random seed, each execution could start from different random weights, giving slightly different numerical values.

---

## 6. Observed linear combination

The perceptron first computes:

```text
z = w · x + b
```

The observed output is:

```text
torch.dot(w, x)             : 3.404638
bias b                      : 0.000000
pre-activation z = w·x + b  : 3.404638
Shape of z                  : (1,)
```

The dot product expands as:

```text
w · x = w1*x1 + w2*x2 + w3*x3
```

Using the observed numbers:

```text
w1 = 0.3367, x1 = 7.4
w2 = 0.1288, x2 = 0.70
w3 = 0.2345, x3 = 3.51
```

Therefore:

```text
z ≈ (0.3367)(7.4) + (0.1288)(0.70) + (0.2345)(3.51) + 0
z ≈ 3.404638
```

This number is called the pre-activation, raw score, or logit.

---

## 7. Interpretation of the observed pre-activation value

The observed pre-activation is:

```text
z = 3.404638
```

This is a positive value.

For a sigmoid-based binary classifier:

```text
large positive z  → prediction close to 1
z = 0             → prediction 0.5
large negative z  → prediction close to 0
```

Therefore, even before training, the randomly initialized perceptron already leans strongly toward class 1 for this particular sample.

This happens because the random weights and the positive input features combine to produce a positive raw score.

---

## 8. Observed sigmoid activation

The script applies the sigmoid function:

```python
y_hat = torch.sigmoid(z)
```

The observed result is:

```text
y_hat = sigmoid(z)          : 0.967849
Interpreted probability     : 96.78% for class 1
Shape of y_hat              : (1,)
```

The sigmoid function is:

```text
sigmoid(z) = 1 / (1 + exp(-z))
```

For the observed value:

```text
z = 3.404638
```

the sigmoid output is:

```text
sigmoid(3.404638) = 0.967849
```

This means the perceptron predicts class 1 with approximately 96.78% confidence.

---

## 9. Observation about the prediction

The prediction is:

```text
y_hat = 0.967849
```

The true label is:

```text
y_true = 1.0
```

Therefore, the model is already correct in terms of classification.

If one uses the usual binary decision threshold:

```text
if y_hat >= 0.5 → class 1
if y_hat <  0.5 → class 0
```

then:

```text
0.967849 >= 0.5
```

so the predicted class is:

```text
predicted class = 1
```

The true class is also:

```text
true class = 1
```

Therefore, the model is correct from the start for this one sample.

However, the prediction is not exactly 1.0, so the binary cross-entropy loss is not zero.

---

## 10. Observed binary cross-entropy loss

The script computes binary cross-entropy loss:

```python
loss = F.binary_cross_entropy(y_hat, y_true)
```

The observed result is:

```text
True label y_true           : 1.0
Binary cross-entropy loss   : 0.032679
Shape of loss               : ()
```

The shape of loss is:

```text
()
```

This means the loss is a scalar tensor.

For binary classification, binary cross-entropy is:

```text
L = -[y log(y_hat) + (1 - y) log(1 - y_hat)]
```

Because the true label is:

```text
y = 1
```

the formula reduces to:

```text
L = -log(y_hat)
```

Using the observed prediction:

```text
L = -log(0.967849)
L ≈ 0.032679
```

This is small because the model is already predicting the correct class with high confidence.

---

## 11. Observation about why the loss is small

The loss is small because:

```text
true label = 1
prediction = 0.967849
```

The prediction is close to the target.

Binary cross-entropy heavily penalizes confident wrong predictions, but it gives small losses to confident correct predictions.

For example, if the target is 1:

```text
prediction close to 1 → small loss
prediction close to 0 → large loss
```

Here the prediction is close to 1, so the loss is small.

---

## 12. Observed CUDA warning

During the backward pass, the following warning appears:

```text
UserWarning: CUDA initialization: The NVIDIA driver on your system is too old ...
```

This warning does not stop the script.

The script continues and completes successfully.

This warning means that PyTorch detected a CUDA/GPU-related mismatch. The installed PyTorch build expects a newer NVIDIA driver than the one available on the system.

For this script, the warning is not important because the tensors are tiny and the computation can run on CPU.

The observation is:

```text
The code is mathematically correct.
The training output is valid.
The warning is environmental, not algorithmic.
```

For larger neural-network training on GPU, this issue should be fixed by either:

```text
1. installing a PyTorch build compatible with the available CUDA/driver version,
2. updating the NVIDIA driver,
3. or forcing CPU execution explicitly.
```

---

## 13. Observed gradients after backward pass

After calling:

```python
loss.backward()
```

the observed gradients are:

```text
Gradient dL/dw : tensor([-0.2379, -0.0225, -0.1128])
Gradient dL/db : tensor([-0.0322])
```

These gradients are stored inside:

```python
w.grad
b.grad
```

The gradients tell us how the loss changes with respect to each trainable parameter.

---

## 14. Mathematical meaning of the observed gradients

For sigmoid activation combined with binary cross-entropy loss, the gradient simplifies to:

```text
dL/dw_i = (y_hat - y) * x_i
dL/db   = (y_hat - y)
```

The observed prediction is:

```text
y_hat = 0.967849
```

The true label is:

```text
y = 1.0
```

Therefore:

```text
y_hat - y = 0.967849 - 1.0
          = -0.032151
```

So:

```text
dL/db = -0.032151
```

This matches the observed:

```text
dL/db = tensor([-0.0322])
```

For each weight:

```text
dL/dw1 = -0.032151 * 7.4  ≈ -0.2379
dL/dw2 = -0.032151 * 0.70 ≈ -0.0225
dL/dw3 = -0.032151 * 3.51 ≈ -0.1128
```

This matches the observed gradient tensor:

```text
tensor([-0.2379, -0.0225, -0.1128])
```

---

## 15. Important observation about gradient signs

All gradients are negative:

```text
dL/dw1 < 0
dL/dw2 < 0
dL/dw3 < 0
dL/db  < 0
```

This means increasing the corresponding parameters will reduce the loss.

Gradient descent updates parameters using:

```text
new parameter = old parameter - learning_rate * gradient
```

When the gradient is negative:

```text
old parameter - learning_rate * negative value
= old parameter + positive value
```

Therefore, the parameters increase.

This is exactly what is observed after the manual update.

---

## 16. Why increasing the weights makes sense here

The true label is:

```text
y = 1
```

The prediction is:

```text
y_hat = 0.967849
```

The prediction is already close to 1 but still slightly lower than 1.

To move the prediction closer to 1, the model must increase `y_hat`.

For a sigmoid classifier:

```text
increasing z increases y_hat
```

And:

```text
z = w · x + b
```

Since the input features are positive, increasing the weights and bias increases `z`.

Therefore, the correct update direction is:

```text
increase weights
increase bias
```

This is exactly what the gradient descent update does.

---

## 17. Observed gradient descent update

The learning rate is:

```text
lr = 0.01
```

The observed values before and after update are:

```text
Weights before update : tensor([0.3367, 0.1288, 0.2345])
Bias before update    : tensor([0.])

Weights after update  : tensor([0.3391, 0.1290, 0.2356])
Bias after update     : tensor([0.0003])

Actual change Delta w : tensor([0.0024, 0.0002, 0.0011])
Actual change Delta b : tensor([0.0003])
```

The weight changes are positive because the gradients were negative.

For the first weight:

```text
w1_new = w1_old - lr * dL/dw1
       = 0.3367 - 0.01 * (-0.2379)
       = 0.3367 + 0.002379
       ≈ 0.3391
```

This matches the output.

---

## 18. Observation about different weight update magnitudes

The first weight changes the most:

```text
Delta w1 ≈ 0.0024
```

The second weight changes the least:

```text
Delta w2 ≈ 0.0002
```

The third weight has an intermediate change:

```text
Delta w3 ≈ 0.0011
```

This happens because:

```text
dL/dw_i = (y_hat - y) * x_i
```

The input values are:

```text
x1 = 7.4
x2 = 0.70
x3 = 3.51
```

The first feature has the largest magnitude, so its corresponding gradient has the largest magnitude.

The second feature has the smallest magnitude, so its corresponding gradient has the smallest magnitude.

This shows an important practical issue in machine learning:

```text
Feature scaling matters.
```

If input features have very different numerical ranges, their weights can receive very different gradient magnitudes.

---

## 19. Observation about feature scaling

Because the input feature values are:

```text
7.4, 0.70, 3.51
```

they are not on the same numerical scale.

The feature with value 7.4 produces a larger gradient contribution than the feature with value 0.70.

This does not necessarily mean the first feature is physically or scientifically more important. It may only mean that it has a larger numerical scale.

In real neural-network training, inputs are often standardized or normalized using transformations such as:

```text
x_scaled = (x - mean) / standard_deviation
```

or scaled to a fixed range.

This helps make optimization more balanced.

---

## 20. Observed gradient zeroing

After the update, the script zeroes the gradients:

```python
w.grad.zero_()
b.grad.zero_()
```

The observed result is:

```text
w.grad : tensor([0., 0., 0.])
b.grad : tensor([0.])
```

This confirms that the gradients were successfully cleared.

This is necessary because PyTorch accumulates gradients by default.

If gradients were not zeroed, the next call to `loss.backward()` would add new gradients to the old ones.

That would make the next update incorrect for this simple gradient descent loop.

---

## 21. Observed training loop over 50 steps

The script then repeats gradient descent for 50 steps on the same sample.

The observed training table is:

```text
Step           z       y_hat        Loss    ||grad_w||
--------------------------------------------------------------
1        3.40464     0.96785     0.03268      0.264283
5        3.49010     0.97040     0.03004      0.243275
10       3.58776     0.97308     0.02728      0.221250
15       3.67686     0.97532     0.02499      0.202854
20       3.75879     0.97722     0.02304      0.187261
25       3.83459     0.97885     0.02138      0.173880
30       3.90513     0.98026     0.01994      0.162272
35       3.97107     0.98150     0.01868      0.152107
40       4.03298     0.98259     0.01757      0.143135
45       4.09132     0.98356     0.01658      0.135157
50       4.14647     0.98443     0.01570      0.128018
```

This table is one of the most important outputs of the project.

It shows the evolution of the model over repeated gradient descent steps.

---

## 22. Observation: z increases during training

The raw score increases from:

```text
z = 3.40464 at step 1
```

to:

```text
z = 4.14647 at step 50
```

This is expected because the model needs to push the sigmoid output closer to 1.

Since:

```text
sigmoid(z) increases as z increases
```

the model increases `z` by increasing the weights and bias.

---

## 23. Observation: prediction moves toward the target

The prediction increases from:

```text
y_hat = 0.96785 at step 1
```

to:

```text
y_hat = 0.98443 at step 50
```

The target is:

```text
y_true = 1.0
```

Therefore, the model is moving in the correct direction.

The prediction does not jump suddenly to 1.0 because:

```text
1. the learning rate is small,
2. sigmoid saturates near 1,
3. the gradient becomes smaller as the prediction approaches the target.
```

---

## 24. Observation: loss decreases during training

The loss decreases from:

```text
Loss = 0.03268 at step 1
```

to:

```text
Loss = 0.01570 at step 50
```

This confirms that gradient descent is minimizing the loss.

The decrease is monotonic in this simple one-sample case.

In more realistic training, the loss may fluctuate because mini-batches and datasets introduce stochasticity.

---

## 25. Observation: gradient norm decreases during training

The gradient norm decreases from:

```text
||grad_w|| = 0.264283 at step 1
```

to:

```text
||grad_w|| = 0.128018 at step 50
```

This is expected.

The gradient is proportional to:

```text
y_hat - y
```

At step 1:

```text
|y_hat - y| = |0.96785 - 1| = 0.03215
```

At step 50:

```text
|y_hat - y| = |0.98443 - 1| = 0.01557
```

The prediction error becomes smaller, so the gradient signal becomes smaller.

---

## 26. Interpretation of decreasing gradient norm

The decreasing gradient norm means the model is receiving a weaker correction signal as it gets closer to the correct target.

This is desirable.

When the model is wrong, gradients should be large enough to correct it.

When the model is already close to correct, gradients should become smaller.

This is a central feature of smooth optimization.

---

## 27. Observed final state after 50 steps

The final printed state is:

```text
Final prediction y_hat       : 0.984426
Target y_true                : 1.0
Final BCE loss               : 0.015696
Final weights w2             : tensor([0.4179, 0.1365, 0.2730])
Final bias b2                : tensor([0.0110])
```

The final prediction is closer to the target than the initial prediction.

The final loss is lower than the initial loss.

The final weights and bias are larger than the initial values.

---

## 28. Observation about final weights

The initial weights were approximately:

```text
w_initial = [0.3367, 0.1288, 0.2345]
```

The final weights are:

```text
w_final = [0.4179, 0.1365, 0.2730]
```

All weights increased.

The first weight increased the most because the first input feature had the largest magnitude.

This confirms the gradient formula:

```text
dL/dw_i = (y_hat - y) * x_i
```

---

## 29. Observation about final bias

The bias increased from:

```text
b_initial = 0.0
```

to:

```text
b_final = 0.0110
```

The bias increases because increasing the bias increases `z`, which increases the sigmoid output.

The bias acts as a trainable offset.

Even if all features were zero, the bias could still shift the prediction.

---

## 30. Observation from the convergence plot

The file:

```text
01_perceptron_convergence.png
```

contains two panels:

```text
1. Loss decreasing toward 0
2. Prediction converging to 1.0
```

The left panel shows the binary cross-entropy loss decreasing smoothly.

The right panel shows the predicted probability increasing toward the target value 1.

This plot visually confirms the numerical trend printed in the training table.

---

## 31. Observation: the loss curve is smooth

The loss curve is smooth because training is performed on one fixed sample.

There is no data shuffling, no random mini-batch selection, and no stochastic variation.

In real neural-network training, the loss curve is often noisier because every mini-batch may give a slightly different gradient direction.

Here the gradient direction is consistent because the same sample is used repeatedly.

---

## 32. Observation: the prediction curve is nearly flat near 1

The prediction curve changes only slightly:

```text
0.96785 → 0.98443
```

This may look small visually, but it is correct.

The reason is sigmoid saturation.

When `z` is already large and positive, sigmoid is already close to 1 and its slope is small.

Therefore, even a noticeable increase in `z` produces only a small increase in `y_hat`.

---

## 33. Observation from sigmoid saturation

The sigmoid function has the following rough behavior:

```text
z = 0  → sigmoid(z) = 0.5
z = 1  → sigmoid(z) ≈ 0.731
z = 2  → sigmoid(z) ≈ 0.881
z = 3  → sigmoid(z) ≈ 0.953
z = 4  → sigmoid(z) ≈ 0.982
z = 5  → sigmoid(z) ≈ 0.993
```

The model operates roughly in the region:

```text
z = 3.4 to 4.1
```

This is already a saturated positive region.

That is why the prediction approaches 1 slowly.

---

## 34. Observation from the activation function plot

The file:

```text
01_activation_functions.png
```

contains four activation-function panels:

```text
1. Sigmoid
2. Tanh
3. ReLU
4. Leaky ReLU
```

Each panel shows:

```text
solid curve  = activation value f(z)
dotted curve = approximate derivative f'(z)
```

The purpose of this figure is to show not only the output of each activation function, but also how gradients behave through that activation.

This is important because neural networks learn through gradients.

---

## 35. Observation about sigmoid activation

The sigmoid plot shows:

```text
for large negative z → output near 0
for z = 0            → output 0.5
for large positive z → output near 1
```

The derivative is largest near:

```text
z = 0
```

and becomes small when:

```text
z << 0 or z >> 0
```

This means sigmoid suffers from vanishing gradients in saturated regions.

The perceptron in this project uses sigmoid as the output activation, which is appropriate for binary classification.

However, using sigmoid inside many hidden layers can slow learning because of vanishing gradients.

---

## 36. Observation about tanh activation

The tanh plot shows an S-shaped curve similar to sigmoid, but its range is:

```text
-1 to 1
```

Unlike sigmoid, tanh is zero-centered:

```text
tanh(0) = 0
```

This can be useful in hidden layers because activations can be positive or negative.

However, tanh also saturates for large positive or negative `z`, so it can also suffer from vanishing gradients.

---

## 37. Observation about ReLU activation

The ReLU plot shows:

```text
ReLU(z) = 0 for z < 0
ReLU(z) = z for z > 0
```

The derivative is:

```text
0 for z < 0
1 for z > 0
```

The positive side is useful because the gradient does not vanish for large positive values.

However, the negative side has zero gradient.

This produces the well-known dying ReLU problem:

```text
if a neuron remains in z < 0, it outputs 0 and receives no gradient.
```

---

## 38. Observation about Leaky ReLU activation

Leaky ReLU modifies ReLU by allowing a small nonzero slope for negative `z`.

In the script:

```python
F.leaky_relu(z_t, negative_slope=0.1)
```

This means:

```text
LeakyReLU(z) = 0.1z for z < 0
LeakyReLU(z) = z    for z > 0
```

Its derivative is:

```text
0.1 for z < 0
1.0 for z > 0
```

This avoids a completely dead negative region.

Therefore, Leaky ReLU can continue learning even when a neuron is in the negative region.

---

## 39. Observation about output activation versus hidden-layer activation

The project uses sigmoid for the perceptron output.

This is correct for binary classification because sigmoid produces a probability-like value between 0 and 1.

However, for hidden layers in deeper networks, ReLU-like activations are often preferred because they avoid the strong saturation problem of sigmoid and tanh on the positive side.

A practical rule is:

```text
binary classification output layer → sigmoid
hidden layers                      → ReLU, Leaky ReLU, GELU, etc.
multiclass output layer            → softmax
```

---

## 40. Observation about manual gradient descent

The script manually performs the update:

```python
with torch.no_grad():
    w -= lr * w.grad
    b -= lr * b.grad
```

This is equivalent in spirit to what an optimizer such as stochastic gradient descent would do internally.

In real PyTorch projects, this manual code is usually replaced by:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

The scratch implementation is useful because it makes the update rule visible.

---

## 41. Observation about `torch.no_grad()`

The update is performed inside:

```python
with torch.no_grad():
```

This is necessary because parameter updates are not part of the differentiable forward model.

The model should differentiate through:

```text
z = w · x + b
y_hat = sigmoid(z)
loss = BCE(y_hat, y)
```

But it should not build a gradient graph through the parameter update itself.

Therefore, using `torch.no_grad()` during manual parameter updates is correct.

---

## 42. Observation about gradient accumulation

The project correctly zeroes gradients after each update.

This is essential because PyTorch accumulates gradients.

If the script did not call:

```python
w.grad.zero_()
b.grad.zero_()
```

then gradients from previous steps would remain and be added to the next gradients.

That would make the training behavior different from standard gradient descent.

The observed zero gradients confirm that the code is handling this correctly.

---

## 43. Observation about the learning rate

The learning rate is:

```text
lr = 0.01
```

This is small enough to give stable training.

The loss decreases smoothly and does not oscillate or diverge.

If the learning rate were much larger, possible outcomes could include:

```text
1. faster initial progress,
2. overshooting the minimum,
3. unstable oscillation,
4. divergence of the loss.
```

If the learning rate were much smaller, the model would learn more slowly.

---

## 44. Observation about training on one sample

The project trains on only one input-output pair.

This is useful for educational purposes because it allows direct inspection of every number.

However, this does not demonstrate generalization.

The perceptron is only learning to make this single sample more confidently class 1.

It is not learning a robust decision boundary from many examples.

For real machine learning, the next step would be to train on a dataset containing many samples and evaluate performance on unseen test data.

---

## 45. Observation about model capacity

A single perceptron with sigmoid output is a linear classifier.

Its decision boundary is defined by:

```text
w · x + b = 0
```

For three input features, this boundary is a plane in three-dimensional feature space.

For more input features, it becomes a hyperplane.

This model can only solve linearly separable classification problems.

It cannot model highly nonlinear relationships unless nonlinear features are manually engineered or hidden layers are added.

---

## 46. Observation about relation to multilayer perceptrons

This project demonstrates the core unit of a multilayer perceptron.

A multilayer perceptron is built by stacking many such operations:

```text
linear transformation
activation
linear transformation
activation
...
output layer
```

The same core ideas remain:

```text
forward pass
loss computation
backward pass
parameter update
```

Therefore, understanding this single perceptron is essential before moving to deeper networks.

---

## 47. Observation about autograd

PyTorch automatically computes gradients because the parameters were created with:

```python
requires_grad=True
```

This tells PyTorch to track operations involving those tensors.

When `loss.backward()` is called, PyTorch traverses the computation graph backward and applies the chain rule.

The important observation is that the manually expected gradients match the PyTorch-computed gradients.

This confirms that autograd is doing exactly what the mathematical derivation predicts.

---

## 48. Observation about tensor shapes

The tensor shapes are consistent:

```text
x      shape = (3,)
w      shape = (3,)
b      shape = (1,)
z      shape = (1,)
y_hat  shape = (1,)
loss   shape = ()
```

This is a good simple example of tensor-shape discipline.

The dot product between two tensors of shape `(3,)` produces a scalar-like value. Adding `b` with shape `(1,)` gives a tensor of shape `(1,)`.

The binary cross-entropy loss reduces the prediction and target into a scalar loss.

---

## 49. Observation about the difference between scalar tensor shapes

The script shows:

```text
Shape of z     = (1,)
Shape of loss  = ()
```

This distinction is useful.

A tensor with shape `(1,)` is a one-element vector.

A tensor with shape `()` is a scalar tensor.

Both can contain one numerical value, but they are not exactly the same shape.

This matters in larger PyTorch programs where shape mismatches can cause errors.

---

## 50. Observation about prediction confidence

The prediction is already very confident at initialization:

```text
96.78% for class 1
```

This happens by chance because random weights produced a large positive `z`.

If the random seed were different, the initial prediction could be very different.

For example, the model could initially predict something closer to:

```text
0.5
```

or even close to:

```text
0
```

In that case, the initial loss and gradients would be different.

---

## 51. Observation about what would happen if the label were 0

If the true label were:

```text
y_true = 0
```

but the initial prediction remained:

```text
y_hat = 0.967849
```

then the model would be confidently wrong.

The binary cross-entropy loss would be large.

The gradient would be:

```text
y_hat - y = 0.967849 - 0 = 0.967849
```

This is positive.

Gradient descent would then decrease the weights and bias.

That would reduce `z`, reduce `y_hat`, and move the prediction toward 0.

This shows that the sign of the gradient automatically determines the correction direction.

---

## 52. Observation about the key gradient formula

The most important formula demonstrated by the output is:

```text
dL/dw = (y_hat - y) x
dL/db = (y_hat - y)
```

This formula explains:

```text
1. why gradients are negative in this run,
2. why weights increase,
3. why the first weight changes most,
4. why gradients become smaller over training,
5. why the loss decreases.
```

This formula is the central mathematical observation of the project.

---

## 53. Observation about why the first feature dominates the update

Because:

```text
x1 = 7.4
x2 = 0.70
x3 = 3.51
```

and:

```text
dL/dw_i = (y_hat - y) x_i
```

the first feature produces the largest absolute gradient.

This means the first weight changes the most.

This is not necessarily a sign that feature 1 is most meaningful.

It is partly a consequence of the input scale.

This is why normalization is important in real models.

---

## 54. Observation about using BCE with sigmoid

The script uses:

```python
y_hat = torch.sigmoid(z)
loss = F.binary_cross_entropy(y_hat, y_true)
```

This is pedagogically clear because it explicitly shows the sigmoid output.

In production PyTorch binary classification, one often uses:

```python
F.binary_cross_entropy_with_logits(z, y_true)
```

This function combines sigmoid and binary cross-entropy in a more numerically stable way.

For teaching, the explicit sigmoid is better because it helps visualize the probability.

---

## 55. Observation about the saved output files

The script saves two figures:

```text
01_activation_functions.png
01_perceptron_convergence.png
```

The first figure helps understand activation functions and their gradients.

The second figure helps understand learning behavior over repeated gradient descent steps.

Both files are useful supporting outputs for the project.

---

## 56. Observation about the activation-function gradient labels

In the activation plots, the dotted curves are labeled as derivatives.

The labels shown in the legend appear as:

```text
f'(z)
```

The derivative plots help connect activation functions to backpropagation.

Backpropagation multiplies gradients through each layer. If an activation derivative is very small, the gradient passed backward also becomes very small.

This is why activation-function choice matters in deep learning.

---

## 57. Observation about numerical gradient in activation plots

The script computes the derivative curves numerically using:

```python
np.gradient(values, dz)
```

This is not the symbolic derivative.

It is a finite-difference approximation.

For visualization, this is acceptable.

For actual backpropagation, PyTorch computes gradients analytically/automatically through autograd operations, not through this plotted numerical approximation.

---

## 58. Observation about why ReLU plot extends beyond y-axis

In the activation plot, ReLU and Leaky ReLU can produce values larger than 1 when `z > 1`.

The y-axis range is limited to:

```text
-1.5 to 1.5
```

Therefore, the positive part of ReLU and Leaky ReLU may extend beyond the displayed axis range.

This is not an error.

It simply reflects that ReLU is unbounded above.

Unlike sigmoid and tanh, ReLU does not squash outputs into a fixed interval.

---

## 59. Observation about scientific analogy

The perceptron can be interpreted similarly to a simple physical model:

```text
features       → measured observables
weights        → coupling strengths or importance coefficients
bias           → baseline offset
z              → effective field or score
activation     → nonlinear response function
loss           → mismatch between model output and desired target
gradient       → direction of correction
learning rate  → step size in parameter space
```

In this analogy, training adjusts the coupling strengths so that the model output agrees better with the observed target.

This is conceptually similar to fitting model parameters in computational physics.

---

## 60. Observation about why this project is useful for soft-matter/physics ML

For applying neural networks to soft-matter physics, this project teaches the most basic mechanism behind supervised learning.

In a physics setting:

```text
x could be structural descriptors, densities, order parameters, contact maps, RDF features, or simulation observables.
y could be phase label, folding state, crystallinity, aggregation state, or a measured property.
```

The perceptron is the simplest learnable mapping from descriptors to output.

More advanced neural networks are built by adding more layers, more nonlinearities, and more parameters.

But the core learning rule is the same.

---

## 61. Main conclusions from the run

The main conclusions are:

```text
1. The perceptron starts with random weights.
2. The initial random weights produce z = 3.404638.
3. Sigmoid converts this into y_hat = 0.967849.
4. The true label is y = 1, so the prediction is already correct.
5. Binary cross-entropy loss is small: 0.032679.
6. Backpropagation produces negative gradients.
7. Negative gradients cause gradient descent to increase the weights and bias.
8. Increasing weights and bias increases z.
9. Increasing z increases y_hat.
10. y_hat moves closer to 1.
11. The loss decreases.
12. The gradient norm decreases as the prediction error decreases.
13. The training curves confirm the printed values.
14. The activation plots show why activation choice affects gradient flow.
```

---

## 62. Most important conceptual takeaway

The most important learning mechanism shown by this project is:

```text
The model does not know directly how to change its weights.
The loss tells the model how wrong it is.
Backpropagation calculates how each parameter contributed to that wrongness.
Gradient descent changes each parameter in the direction that reduces the loss.
```

For this particular run:

```text
prediction was slightly below the target 1
therefore gradients were negative
therefore weights increased
therefore z increased
therefore sigmoid output increased
therefore loss decreased
```

This is the complete learning loop.

---

## 63. What should be studied next

After understanding this project, the natural next steps are:

```text
1. Train the perceptron on many samples instead of one sample.
2. Convert the manual tensors into an nn.Module.
3. Use torch.optim.SGD instead of manual updates.
4. Add a train/test split.
5. Plot accuracy along with loss.
6. Standardize input features.
7. Add hidden layers to build a multilayer perceptron.
8. Compare sigmoid, tanh, ReLU, and Leaky ReLU in hidden layers.
9. Study overfitting and generalization.
10. Apply the same ideas to a real scientific dataset.
```

---

## 64. Final summary

This project successfully demonstrates a single perceptron's full forward and backward learning process.

The observed outputs are internally consistent with the mathematics of sigmoid binary classification.

The numerical results show that the model learns in the correct direction:

```text
z:       3.40464  → 4.14647
y_hat:   0.96785  → 0.98443
loss:    0.03268  → 0.01570
grad_w:  0.26428  → 0.12802
```

These trends confirm that gradient descent is working correctly.

The project is a strong first step toward understanding deeper neural networks because it exposes the fundamental operations that remain present in all neural-network training:

```text
linear transformation
nonlinear activation
loss computation
automatic differentiation
parameter update
gradient clearing
training loop
visual diagnostics
```
