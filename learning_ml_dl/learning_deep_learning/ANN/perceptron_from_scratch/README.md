# Perceptron From Scratch in PyTorch

> **Project type:** Deep learning fundamentals / educational PyTorch implementation  
> **Main script:** `perceptron_scratch.py`  
> **Core topic:** A single perceptron implemented using raw PyTorch tensors, manual gradient descent, and explicit autograd inspection.

---

## 1. Project overview

This project demonstrates the smallest useful building block of a neural network: a **single perceptron**.

A perceptron takes an input vector, multiplies each input feature by a learnable weight, adds a learnable bias, and then passes the result through an activation function. In this project, the activation function is the **sigmoid function**, which converts the raw output into a probability between 0 and 1.

The script is intentionally written in a low-level, explicit, and highly documented style. It avoids high-level PyTorch abstractions such as `torch.nn.Module`, `torch.nn.Linear`, and optimizers like `torch.optim.SGD` so that the internal mechanism of learning is fully visible.

The goal is not to write the shortest possible PyTorch code. The goal is to understand what PyTorch is doing internally when a neural network learns.

---

## 2. What this project teaches

This project teaches the full learning cycle of a neural-network unit:

```text
input data
   ↓
linear combination: z = w · x + b
   ↓
sigmoid activation: y_hat = sigmoid(z)
   ↓
binary cross-entropy loss
   ↓
backpropagation with loss.backward()
   ↓
gradient descent parameter update
   ↓
gradient zeroing
   ↓
repeat training steps
```

By running the script, you see:

1. How an input vector is represented as a PyTorch tensor.
2. How weights and bias are initialized.
3. Why `requires_grad=True` is necessary for learnable parameters.
4. How the perceptron computes the raw score `z`.
5. How sigmoid converts `z` into a probability.
6. How binary cross-entropy measures prediction error.
7. How `loss.backward()` computes gradients automatically.
8. How to manually update weights using gradient descent.
9. Why gradients must be cleared after each update.
10. Why loss decreases during training.
11. Why prediction moves toward the true label.
12. How common activation functions behave and why their gradients matter.

---

## 3. Repository / project structure

A typical project directory looks like this:

```text
ANN/
│
├── perceptron_scratch_v1.py
│   └── Original perceptron implementation.
│
├── perceptron_scratch_v1_very_detailed.py
│   └── Very heavily inline-documented educational version.
│
├── 01_activation_functions.png
│   └── Figure showing sigmoid, tanh, ReLU, and Leaky ReLU with their gradients.
│
├── 01_perceptron_convergence.png
│   └── Figure showing loss decrease and prediction convergence during training.
│
└── README.md
    └── This detailed explanation file.
```

The two most important files are:

```text
perceptron_scratch_v1_very_detailed.py
README.md
```

The Python script performs the computation. This README explains the purpose, mathematics, output, and interpretation of that computation.

---

## 4. Requirements

The script uses only a small number of Python libraries:

```text
python >= 3.8
pytorch
graphical matplotlib backend or non-interactive backend
numpy
matplotlib
```

The code should work with Python 3.10, which matches the command used in the example run:

```bash
python3.10 perceptron_scratch_v1.py
```

or for the heavily documented version:

```bash
python3.10 perceptron_scratch_v1_very_detailed.py
```

---

## 5. Installing dependencies

If you are using a Python virtual environment, activate it first.

Example:

```bash
python3.10 -m venv venv
source venv/bin/activate
```

Then install the required packages:

```bash
pip install torch numpy matplotlib
```

If you are on a machine without a working GPU setup, the CPU version of PyTorch is enough for this project because the tensors are extremely small.

This script does not need a GPU.

---

## 6. How to run the project

From the project directory, run:

```bash
python3.10 perceptron_scratch_v1_very_detailed.py
```

or, if you want to run the original file:

```bash
python3.10 perceptron_scratch_v1.py
```

After running, the script prints detailed terminal output and saves two images:

```text
01_activation_functions.png
01_perceptron_convergence.png
```

---

## 7. Main mathematical model

The perceptron uses the following equations.

### 7.1 Input vector

The input is one sample with three features:

```python
x = torch.tensor([7.4, 0.70, 3.51])
```

Mathematically:

```text
x = [x1, x2, x3]
```

In the example:

```text
x1 = 7.4
x2 = 0.70
x3 = 3.51
```

These numbers are treated as input features. In a wine-quality example, they could represent chemical measurements. In a physics analogy, they could represent physical descriptors such as density, temperature, and concentration.

The perceptron does not know the physical meaning of these numbers. It only sees a numerical vector.

---

### 7.2 Weight vector

The perceptron has one weight per input feature:

```python
w = torch.randn(3, requires_grad=True)
```

Mathematically:

```text
w = [w1, w2, w3]
```

Each weight controls how strongly the corresponding feature contributes to the final prediction.

If a weight is large and positive, increasing that feature increases the raw score `z`.

If a weight is large and negative, increasing that feature decreases the raw score `z`.

If a weight is close to zero, that feature has little effect on `z`.

---

### 7.3 Bias

The bias is initialized as:

```python
b = torch.zeros(1, requires_grad=True)
```

The bias is a trainable offset. It shifts the raw score up or down independently of the input features.

Mathematically, the bias is useful because it lets the model move the decision boundary without changing the slopes associated with the input features.

---

### 7.4 Linear combination

The perceptron first computes:

```text
z = w · x + b
```

Expanded:

```text
z = w1*x1 + w2*x2 + w3*x3 + b
```

In the code:

```python
z = torch.dot(w, x) + b
```

This raw value `z` is also called:

```text
pre-activation
logit
raw score
linear response
```

It is called pre-activation because it is calculated before applying the nonlinear activation function.

---

### 7.5 Sigmoid activation

The raw score `z` can be any real number. For binary classification, we want a probability-like number between 0 and 1.

So the script applies the sigmoid function:

```text
sigmoid(z) = 1 / (1 + exp(-z))
```

In the code:

```python
y_hat = torch.sigmoid(z)
```

The output `y_hat` is interpreted as:

```text
the predicted probability that the input belongs to class 1
```

For example:

```text
y_hat = 0.967849
```

means:

```text
the model predicts 96.7849% probability for class 1
```

---

### 7.6 Binary cross-entropy loss

The true label is:

```python
y_true = torch.tensor([1.0])
```

The binary cross-entropy loss is:

```text
L = -[y log(y_hat) + (1-y) log(1-y_hat)]
```

If the true label is `1`, this simplifies to:

```text
L = -log(y_hat)
```

If the true label is `0`, this simplifies to:

```text
L = -log(1 - y_hat)
```

In the code:

```python
loss = F.binary_cross_entropy(y_hat, y_true)
```

Binary cross-entropy is small when the predicted probability is close to the correct label and large when the prediction is confidently wrong.

---

## 8. Why `requires_grad=True` is important

The weights and bias are created as:

```python
w = torch.randn(3, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```

The argument:

```python
requires_grad=True
```

tells PyTorch to track operations involving these tensors.

This allows PyTorch to construct a computational graph:

```text
w, b, x
   ↓
z = w · x + b
   ↓
y_hat = sigmoid(z)
   ↓
loss = BCE(y_hat, y_true)
```

When the script calls:

```python
loss.backward()
```

PyTorch travels backward through this graph and computes:

```text
dL/dw
dL/db
```

These gradients are stored in:

```python
w.grad
b.grad
```

Without `requires_grad=True`, `w.grad` and `b.grad` would not be computed.

---

## 9. Interpreting the first forward pass

The output begins with something like:

```text
Input x (3 features)  : tensor([7.4000, 0.7000, 3.5100])

Initial weights w     : tensor([0.3367, 0.1288, 0.2345])
Initial bias b        : tensor([0.])
```

These are the starting model parameters.

Then the script prints:

```text
torch.dot(w, x)             : 3.404638
bias b                      : 0.000000
pre-activation z = w·x + b  : 3.404638
```

This means:

```text
z = 3.404638
```

Because this number is positive and relatively large, the sigmoid output will be close to 1.

The script then prints:

```text
y_hat = sigmoid(z)          : 0.967849
Interpreted probability     : 96.78% for class 1
```

So the model initially predicts class 1 with high confidence.

The true label is:

```text
y_true = 1.0
```

So the initial prediction is already correct.

---

## 10. Interpreting the first loss value

The printed loss is:

```text
Binary cross-entropy loss   : 0.032679
```

This is small because the model predicted:

```text
y_hat = 0.967849
```

while the true label is:

```text
y_true = 1.0
```

For `y_true = 1`, the loss is:

```text
L = -log(y_hat)
```

So:

```text
L = -log(0.967849) ≈ 0.032679
```

A loss close to zero means the prediction is close to correct.

---

## 11. Backpropagation and gradients

After computing the loss, the script calls:

```python
loss.backward()
```

This computes gradients.

The output is:

```text
Gradient dL/dw : tensor([-0.2379, -0.0225, -0.1128])
Gradient dL/db : tensor([-0.0322])
```

These gradients tell the model how the loss would change if each parameter changed slightly.

For sigmoid plus binary cross-entropy, the gradient has a simple form:

```text
dL/dz = y_hat - y_true
```

Since:

```text
y_hat = 0.967849
y_true = 1.0
```

we get:

```text
dL/dz = 0.967849 - 1.0 = -0.032151
```

Because:

```text
z = w · x + b
```

the gradients are:

```text
dL/dw = (y_hat - y_true) * x
dL/db = y_hat - y_true
```

Therefore:

```text
dL/dw1 = -0.032151 * 7.4  ≈ -0.2379
dL/dw2 = -0.032151 * 0.70 ≈ -0.0225
dL/dw3 = -0.032151 * 3.51 ≈ -0.1128
dL/db  = -0.032151
```

This matches the PyTorch output.

---

## 12. Why the gradients are negative

The gradients are negative because the prediction is slightly below the target.

The model predicted:

```text
y_hat = 0.967849
```

The target is:

```text
y_true = 1.0
```

So the model needs to increase the prediction.

To increase the sigmoid output, the model must increase `z`.

Because:

```text
z = w · x + b
```

and all features in `x` are positive, increasing the weights and bias increases `z`.

Gradient descent updates parameters as:

```text
parameter_new = parameter_old - learning_rate * gradient
```

If the gradient is negative, subtracting the negative gradient increases the parameter.

Example:

```text
w_new = w_old - 0.01 * (-0.2379)
w_new = w_old + 0.002379
```

So a negative gradient causes the weight to increase.

That is exactly what should happen here.

---

## 13. Manual gradient descent update

The learning rate is:

```text
lr = 0.01
```

The update rule is:

```text
w_new = w_old - lr * dL/dw
b_new = b_old - lr * dL/db
```

The output shows:

```text
Weights before update : tensor([0.3367, 0.1288, 0.2345])
Weights after update  : tensor([0.3391, 0.1290, 0.2356])
```

The weights increased because the gradients were negative.

The output also shows:

```text
Actual change Delta w : tensor([0.0024, 0.0002, 0.0011])
Actual change Delta b : tensor([0.0003])
```

This confirms that the model took a small step in the direction that reduces loss.

---

## 14. Why `torch.no_grad()` is used

The parameter update is written inside:

```python
with torch.no_grad():
    w -= lr * w.grad
    b -= lr * b.grad
```

This is important.

The forward pass and loss computation should be tracked by autograd.

But the parameter update itself should not become part of the computational graph.

The update is an optimization operation, not a differentiable part of the model.

So `torch.no_grad()` tells PyTorch:

```text
Do not track these operations for gradient computation.
```

This is exactly what PyTorch optimizers do internally.

---

## 15. Why gradients must be zeroed

After the update, the script calls:

```python
w.grad.zero_()
b.grad.zero_()
```

PyTorch accumulates gradients by default.

This means if you call:

```python
loss.backward()
```

multiple times, new gradients are added to old gradients.

That behavior is useful in some advanced training methods, but for basic gradient descent, we want each update to use only the current gradient.

So the standard training pattern is:

```text
forward pass
loss computation
backward pass
parameter update
zero gradients
```

In high-level PyTorch code, this usually looks like:

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

In this script, the same logic is implemented manually.

---

## 16. Understanding the 50-step training loop

The script then reinitializes the weights and trains the perceptron for 50 gradient descent steps on the same sample.

The printed table looks like:

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

This table shows successful learning.

The trends are:

```text
z increases
prediction y_hat increases
loss decreases
gradient norm decreases
```

These are exactly the expected trends for a correctly implemented learning loop on this one sample.

---

## 17. Why `z` increases during training

At step 1:

```text
z = 3.40464
```

At step 50:

```text
z = 4.14647
```

Because the true label is 1, the model tries to push the sigmoid output closer to 1.

Since sigmoid increases when `z` increases, gradient descent pushes `z` upward.

---

## 18. Why `y_hat` increases slowly

At step 1:

```text
y_hat = 0.96785
```

At step 50:

```text
y_hat = 0.98443
```

The prediction increases, but slowly.

This happens because sigmoid saturates near 1.

For example:

```text
z = 0  -> sigmoid(z) = 0.500
z = 1  -> sigmoid(z) = 0.731
z = 2  -> sigmoid(z) = 0.881
z = 3  -> sigmoid(z) = 0.953
z = 4  -> sigmoid(z) = 0.982
z = 5  -> sigmoid(z) = 0.993
```

The model starts at `z ≈ 3.4`, already in the high-confidence region of sigmoid.

So even though `z` increases noticeably, the probability changes only slightly.

This is one reason sigmoid can cause slow learning in deep hidden layers.

---

## 19. Why the loss decreases

For this example, the true label is 1, so:

```text
loss = -log(y_hat)
```

As `y_hat` gets closer to 1, `-log(y_hat)` gets closer to 0.

That is why the loss decreases from:

```text
0.03268
```

to:

```text
0.01570
```

The model is becoming more confident in the correct answer.

---

## 20. Why the gradient norm decreases

The weight-gradient formula is:

```text
dL/dw = (y_hat - y_true) * x
```

Since `x` is fixed, the gradient magnitude mainly depends on:

```text
|y_hat - y_true|
```

At step 1:

```text
|0.96785 - 1.0| = 0.03215
```

At step 50:

```text
|0.98443 - 1.0| = 0.01557
```

The error becomes smaller, so the gradient becomes smaller.

This is why the gradient norm decreases from:

```text
0.264283
```

to:

```text
0.128018
```

A smaller gradient means the model is closer to the correct prediction for this sample.

---

## 21. Final training result

The script prints:

```text
Final prediction y_hat : 0.984426
Target y_true          : 1.0
Final BCE loss         : 0.015696
Final weights w2       : tensor([0.4179, 0.1365, 0.2730])
Final bias b2          : tensor([0.0110])
```

This means that after 50 gradient descent steps, the model predicts class 1 with about:

```text
98.44% confidence
```

The loss is now smaller than at the beginning.

This confirms that the learning loop is working.

---

## 22. Generated figure: perceptron convergence

The script saves:

```text
01_perceptron_convergence.png
```

This figure contains two panels.

### 22.1 Left panel: loss curve

The left panel shows binary cross-entropy loss versus gradient descent step.

Expected behavior:

```text
loss should decrease
```

In this project, the loss decreases smoothly because:

1. There is only one training sample.
2. The learning rate is small.
3. The initial prediction is already close to correct.
4. There is no stochastic mini-batch noise.

In real neural-network training, the loss curve is often noisier because training uses many samples and mini-batches.

---

### 22.2 Right panel: prediction curve

The right panel shows predicted probability `y_hat` versus gradient descent step.

The target line is:

```text
y = 1
```

Expected behavior:

```text
y_hat should move upward toward 1
```

The prediction moves from approximately:

```text
0.96785
```

to:

```text
0.98443
```

This confirms that the perceptron is becoming more confident in the correct class.

---

## 23. Generated figure: activation functions

The script saves:

```text
01_activation_functions.png
```

This figure shows four activation functions and their approximate gradients:

1. Sigmoid
2. Tanh
3. ReLU
4. Leaky ReLU

In each panel:

```text
solid line  = activation function f(z)
dotted line = approximate derivative f'(z)
```

The derivative matters because backpropagation depends on gradients.

If a derivative is very small, learning can be slow.

If a derivative is zero, the neuron may stop learning in that region.

---

## 24. Sigmoid activation

Sigmoid is:

```text
sigmoid(z) = 1 / (1 + exp(-z))
```

Output range:

```text
0 < sigmoid(z) < 1
```

Important values:

```text
z = 0       -> sigmoid(z) = 0.5
z >> 0      -> sigmoid(z) approaches 1
z << 0      -> sigmoid(z) approaches 0
```

Derivative:

```text
sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))
```

The derivative is largest at `z = 0` and becomes small for large positive or negative `z`.

This is called saturation.

In this project, sigmoid is used for the final binary output because it gives a probability.

---

## 25. Tanh activation

Tanh is:

```text
tanh(z)
```

Output range:

```text
-1 < tanh(z) < 1
```

Important values:

```text
z = 0       -> tanh(z) = 0
z >> 0      -> tanh(z) approaches 1
z << 0      -> tanh(z) approaches -1
```

Derivative:

```text
tanh'(z) = 1 - tanh(z)^2
```

Tanh is zero-centered, which can make it more convenient than sigmoid in some hidden layers.

However, tanh also saturates for large positive or negative `z`, so it can also suffer from vanishing gradients.

---

## 26. ReLU activation

ReLU is:

```text
ReLU(z) = max(0, z)
```

So:

```text
if z < 0, ReLU(z) = 0
if z > 0, ReLU(z) = z
```

Derivative:

```text
if z < 0, ReLU'(z) = 0
if z > 0, ReLU'(z) = 1
```

ReLU is very common in hidden layers because it does not saturate for positive `z`.

However, ReLU has a problem for negative `z`: the gradient is zero.

If a neuron stays in the negative region for all inputs, it may stop learning. This is known as the dying ReLU problem.

---

## 27. Leaky ReLU activation

Leaky ReLU is:

```text
LeakyReLU(z) = z       if z > 0
LeakyReLU(z) = 0.1 z   if z < 0
```

In the script, the negative slope is:

```python
negative_slope=0.1
```

Derivative:

```text
if z > 0, derivative = 1
if z < 0, derivative = 0.1
```

Leaky ReLU reduces the dying ReLU problem because the gradient is not exactly zero for negative `z`.

---

## 28. Why sigmoid is used here but not usually in hidden layers

Sigmoid is useful at the output of a binary classifier because it gives a probability.

But sigmoid is often avoided in hidden layers because it saturates.

When sigmoid saturates, its gradient becomes very small.

Very small gradients can make deep networks learn slowly.

Modern neural networks often use:

```text
ReLU
Leaky ReLU
GELU
SiLU / Swish
```

in hidden layers, while still using sigmoid for binary classification output.

---

## 29. Important PyTorch concepts demonstrated

### 29.1 Tensor

A tensor is PyTorch's main data structure.

Example:

```python
x = torch.tensor([7.4, 0.70, 3.51])
```

This is a one-dimensional tensor of shape `(3,)`.

---

### 29.2 Autograd

Autograd is PyTorch's automatic differentiation system.

It records operations involving tensors with `requires_grad=True` and computes derivatives when `.backward()` is called.

---

### 29.3 `.backward()`

The command:

```python
loss.backward()
```

computes gradients of the loss with respect to all tensors involved in the graph that have `requires_grad=True`.

In this script, those tensors are:

```text
w
b
```

---

### 29.4 `.grad`

After calling `.backward()`, gradients are stored in:

```python
w.grad
b.grad
```

These are used to update the parameters.

---

### 29.5 `torch.no_grad()`

This disables gradient tracking temporarily.

It is used during manual parameter updates because we do not want the update operation itself to be tracked by autograd.

---

### 29.6 `.zero_()`

The underscore means the operation is in-place.

So:

```python
w.grad.zero_()
```

modifies the existing gradient tensor and fills it with zeros.

---

### 29.7 `.detach()`

The command:

```python
w.detach()
```

returns a tensor with the same values but detached from the computational graph.

This is useful for printing, logging, or converting tensors to NumPy arrays without carrying autograd history.

---

### 29.8 `.item()`

The command:

```python
loss.item()
```

converts a one-element tensor into a normal Python number.

This is useful for printing and storing values in lists for plotting.

---

## 30. Common warning: old NVIDIA driver

You may see a warning like:

```text
CUDA initialization: The NVIDIA driver on your system is too old...
```

This is not caused by a mistake in the perceptron code.

It means your installed PyTorch build is trying to check CUDA/GPU support, but the system NVIDIA driver is too old for that CUDA version.

For this project, this warning is harmless because the tensors are tiny and CPU execution is enough.

Possible solutions:

1. Ignore the warning for this CPU-only educational script.
2. Install a CPU-only PyTorch build.
3. Install a PyTorch CUDA version compatible with your driver.
4. Update the NVIDIA driver if you actually need GPU training.

For this perceptron example, no GPU is required.

---

## 31. Why training on one sample is useful

Training on one sample is not useful for real prediction.

However, it is extremely useful for understanding neural-network mechanics.

With one sample:

1. The forward pass is easy to inspect.
2. The gradient formulas are easy to verify.
3. The loss curve is smooth.
4. The prediction movement is easy to understand.
5. There is no confusion from batches, shuffling, datasets, or optimizers.

This project is therefore a conceptual bridge between pure mathematical equations and real PyTorch neural-network training.

---

## 32. Important limitation of this project

This script does not demonstrate generalization.

Generalization means the model performs well on new unseen data.

Here, the model sees only one sample:

```text
x = [7.4, 0.70, 3.51]
y = 1
```

The model is only learning to push this one sample's prediction toward 1.

A real binary classifier would require:

```text
many input samples
many labels
train/validation/test split
batch training
performance metrics
```

This project is about understanding the mechanism, not building a production classifier.

---

## 33. How this connects to a full neural network

A full neural network repeats the same basic idea many times.

A single perceptron computes:

```text
z = w · x + b
y_hat = activation(z)
```

A hidden layer computes many such neurons at once:

```text
z = XW + b
h = activation(z)
```

A multilayer perceptron stacks several layers:

```text
input
  ↓
linear layer
  ↓
activation
  ↓
linear layer
  ↓
activation
  ↓
output layer
```

The same core operations still apply:

```text
forward pass
loss computation
backward pass
parameter update
```

This project teaches those operations in their simplest possible form.

---

## 34. Suggested experiments

After understanding this script, try modifying one thing at a time.

### Experiment 1: Change the true label to 0

Change:

```python
y_true = torch.tensor([1.0])
```

to:

```python
y_true = torch.tensor([0.0])
```

Expected result:

```text
The model should decrease z.
The sigmoid output should move toward 0.
The weights should move in the opposite direction.
```

This is one of the best ways to understand gradient signs.

---

### Experiment 2: Change the learning rate

Try:

```python
lr = 0.001
lr = 0.01
lr = 0.1
lr = 1.0
```

Expected behavior:

```text
small lr  -> slow learning
moderate lr -> stable learning
large lr -> possibly unstable or too aggressive
```

---

### Experiment 3: Use different initial weights

Remove or change:

```python
torch.manual_seed(42)
```

Expected result:

```text
Different random initial weights will produce different initial predictions.
```

This helps demonstrate why initialization matters.

---

### Experiment 4: Use a negative input feature

Change one component of `x` to a negative value.

Expected result:

```text
The sign of that feature will affect the sign of the corresponding weight gradient.
```

This helps you understand the formula:

```text
dL/dw = (y_hat - y_true) * x
```

---

### Experiment 5: Train for more steps

Change:

```python
for step in range(1, 51):
```

to:

```python
for step in range(1, 501):
```

Expected result:

```text
The prediction should move even closer to 1.
The loss should get even smaller.
The gradient norm should continue to decrease.
```

---

### Experiment 6: Replace manual gradient descent with `torch.optim.SGD`

Once the manual version is clear, you can replace:

```python
with torch.no_grad():
    w -= lr * w.grad
    b -= lr * b.grad
```

with a PyTorch optimizer.

This helps you see that optimizers automate the update step but do not change the fundamental mathematics.

---

## 35. Common mistakes and how to identify them

### Mistake 1: Forgetting `requires_grad=True`

Symptom:

```text
w.grad is None
b.grad is None
```

Cause:

```text
PyTorch did not track gradients for w and b.
```

Fix:

```python
w = torch.randn(3, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```

---

### Mistake 2: Forgetting to zero gradients

Symptom:

```text
updates become too large or incorrect
loss behaves strangely
```

Cause:

```text
PyTorch accumulated gradients from previous backward passes.
```

Fix:

```python
w.grad.zero_()
b.grad.zero_()
```

---

### Mistake 3: Updating parameters without `torch.no_grad()`

Symptom:

```text
PyTorch may complain about in-place operations on leaf tensors.
```

Cause:

```text
The update operation is being tracked by autograd.
```

Fix:

```python
with torch.no_grad():
    w -= lr * w.grad
    b -= lr * b.grad
```

---

### Mistake 4: Shape mismatch in loss function

Symptom:

```text
ValueError or warning about target/input shape mismatch
```

Cause:

```text
y_hat and y_true do not have compatible shapes.
```

Fix:

Make sure both have shape `(1,)` for this example:

```python
y_hat.shape == y_true.shape
```

---

### Mistake 5: Using sigmoid incorrectly with BCE variants

This script uses:

```python
y_hat = torch.sigmoid(z)
loss = F.binary_cross_entropy(y_hat, y_true)
```

This is correct.

A more numerically stable alternative in larger projects is:

```python
loss = F.binary_cross_entropy_with_logits(z, y_true)
```

If using `binary_cross_entropy_with_logits`, do not apply sigmoid before the loss, because that function internally combines sigmoid and BCE.

---

## 36. Conceptual summary

The perceptron learns by repeatedly doing this:

```text
1. Compute prediction.
2. Compare prediction with true label.
3. Calculate loss.
4. Compute gradients.
5. Move weights opposite to the gradient.
6. Clear old gradients.
7. Repeat.
```

For this specific project:

```text
true label = 1
initial prediction ≈ 0.96785
final prediction ≈ 0.98443
initial loss ≈ 0.03268
final loss ≈ 0.01570
```

So the model successfully moves the prediction closer to the true label and reduces the loss.

---

## 37. Most important formulas

### Perceptron equation

```text
z = w · x + b
```

### Sigmoid activation

```text
y_hat = 1 / (1 + exp(-z))
```

### Binary cross-entropy

```text
L = -[y log(y_hat) + (1-y) log(1-y_hat)]
```

### Gradient for sigmoid plus BCE

```text
dL/dz = y_hat - y
```

### Weight gradient

```text
dL/dw = (y_hat - y) x
```

### Bias gradient

```text
dL/db = y_hat - y
```

### Gradient descent update

```text
w_new = w_old - lr * dL/dw
b_new = b_old - lr * dL/db
```

These formulas explain nearly every numerical value printed by the script.

---

## 38. Learning checklist

After studying this project, you should be able to answer the following questions:

```text
[ ] What is a perceptron?
[ ] What does the weight vector do?
[ ] What does the bias do?
[ ] What is z = w · x + b?
[ ] Why is z called a pre-activation?
[ ] What does sigmoid do?
[ ] Why is sigmoid useful for binary classification?
[ ] What is binary cross-entropy?
[ ] Why is the loss small when y_hat is close to y_true?
[ ] What does loss.backward() compute?
[ ] Where are gradients stored in PyTorch?
[ ] Why do we update parameters opposite to the gradient?
[ ] Why do we use torch.no_grad() during manual updates?
[ ] Why must gradients be zeroed after each update?
[ ] Why does the loss decrease during training?
[ ] Why does the gradient norm decrease here?
[ ] Why does sigmoid saturate near 0 and 1?
[ ] Why is ReLU common in hidden layers?
[ ] What is the dying ReLU problem?
[ ] Why does Leaky ReLU help?
```

If you can answer these questions, you understand the core mechanics of a neural network training loop.

---

## 39. Recommended next steps

After this project, the natural progression is:

```text
1. Train a perceptron on multiple samples.
2. Convert the manual weights into nn.Linear.
3. Use torch.optim.SGD instead of manual updates.
4. Build a two-layer neural network.
5. Study backpropagation through hidden layers.
6. Train on a real dataset.
7. Add train/validation/test splits.
8. Study overfitting and regularization.
9. Study mini-batch gradient descent.
10. Move toward multilayer perceptrons for scientific datasets.
```

For your longer-term goal of applying deep learning to soft matter physics, this project is the correct starting point because it makes the fundamental optimization mechanism transparent.

---

## 40. Final takeaway

This project shows that neural-network learning is not magic.

A model makes a prediction, computes how wrong it is, calculates gradients, and updates parameters in the direction that reduces the error.

The entire process is summarized by:

```text
prediction -> loss -> gradient -> update -> improved prediction
```

This is the foundation of perceptrons, multilayer perceptrons, convolutional networks, transformers, and almost every modern deep learning model.

