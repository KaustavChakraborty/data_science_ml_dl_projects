# PyTorch MLP Foundations on the Wine Quality Dataset

This repository is a carefully structured, beginner-to-intermediate PyTorch learning project that explains the essential building blocks of tabular deep learning using the UCI Red Wine Quality dataset. It is designed for someone who wants to understand not only *what* the code does, but also *why* each step is necessary and *how* the mathematical ideas map onto PyTorch code.

The repository contains four connected scripts:

```text
dataset_loading_v1.py
mlp_architecture_v1.py
forward_pass_traced_v1.py
loss_functions_v1.py
```

These four scripts together form the conceptual path:

```text
raw tabular data
    ↓
preprocessing and PyTorch Dataset/DataLoader
    ↓
MLP architecture using nn.Module
    ↓
forward propagation traced layer by layer
    ↓
loss functions and training objectives
```

The project does **not** yet implement a full training loop. Instead, it focuses on the foundations required before training: data preparation, model construction, forward propagation, activation behavior, initialization, and loss-function selection.

---

# Table of Contents

1. [Project Goal](#1-project-goal)  
2. [Recommended Repository Name](#2-recommended-repository-name)  
3. [Repository Structure](#3-repository-structure)  
4. [Installation](#4-installation)  
5. [How to Run](#5-how-to-run)  
6. [Dataset Background](#6-dataset-background)  
7. [Script 1: `dataset_loading_v1.py`](#7-script-1-dataset_loading_v1py)  
8. [Script 2: `mlp_architecture_v1.py`](#8-script-2-mlp_architecture_v1py)  
9. [Script 3: `forward_pass_traced_v1.py`](#9-script-3-forward_pass_traced_v1py)  
10. [Script 4: `loss_functions_v1.py`](#10-script-4-loss_functions_v1py)  
11. [Expected Output Figures](#11-expected-output-figures)  
12. [Conceptual Flow Across the Four Scripts](#12-conceptual-flow-across-the-four-scripts)  
13. [Mathematical Summary](#13-mathematical-summary)  
14. [Important Practical Lessons](#14-important-practical-lessons)  
15. [Suggested Next Script: Training Loop](#15-suggested-next-script-training-loop)  
16. [Common Mistakes This Project Helps Avoid](#16-common-mistakes-this-project-helps-avoid)  
17. [Who This Project Is For](#17-who-this-project-is-for)  
18. [Final Summary](#18-final-summary)  

---

# 1. Project Goal

The goal of this project is to build a complete conceptual foundation for neural-network-based binary classification using PyTorch.

The project uses the Red Wine Quality dataset as a compact, real-world tabular dataset. The task is converted into binary classification:

```text
ordinary wine = 0
good wine     = 1
```

The project answers the following questions step by step:

## Data questions

- How is a tabular machine-learning dataset loaded?
- What do rows and columns represent?
- How do we inspect feature ranges, feature distributions, and class balance?
- Why do we separate features and labels?
- Why do we normalize features?
- How do we convert NumPy/Pandas data into PyTorch tensors?
- How does a custom PyTorch `Dataset` work?
- What is a `DataLoader`, and why do we use mini-batches?

## Model architecture questions

- What is an MLP?
- How do we define a model using `nn.Module`?
- What is the difference between `__init__()` and `forward()`?
- What does `nn.Linear` store internally?
- How do PyTorch weight matrices correspond to mathematical notation?
- How do we count trainable parameters?
- Why does more parameter count not automatically mean better performance?
- What is the role of dropout and batch normalization?

## Forward propagation questions

- What happens to one sample as it passes through each layer?
- What is the difference between pre-activation `z` and post-activation `a`?
- What does ReLU do?
- Why are around 50% of ReLU outputs often zero at initialization?
- How does batch processing work?
- How can PyTorch hooks inspect intermediate activations?
- How does weight initialization affect activation flow through deep networks?

## Loss-function questions

- What is a loss function?
- What is Binary Cross-Entropy?
- Why is BCE preferred for binary classification?
- Why is MSE not ideal for classification?
- What is `BCEWithLogitsLoss`?
- Why should the model output raw logits when using `BCEWithLogitsLoss`?
- When are MSE and MAE appropriate?

---

# 2. Recommended Repository Name

The recommended GitHub directory/repository name is:

```text
pytorch-mlp-foundations-wine-quality
```

This name is descriptive because:

- `pytorch` indicates the deep-learning framework,
- `mlp` indicates the model family,
- `foundations` indicates that the project is educational and concept-focused,
- `wine-quality` indicates the dataset/application.

Other possible names:

```text
pytorch-tabular-deep-learning-foundations
mlp-from-data-to-loss
wine-quality-mlp-learning-pipeline
deep-learning-basics-pytorch-wine
tabular-mlp-pytorch-wine-quality
```

---

# 3. Repository Structure

A clean recommended structure is:

```text
pytorch-mlp-foundations-wine-quality/
│
├── README.md
│
├── dataset_loading_v1.py
├── mlp_architecture_v1.py
├── forward_pass_traced_v1.py
├── loss_functions_v1.py
│
├── outputs/
│   ├── 02_feature_distributions.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_weight_init.png
│   ├── 04_activation_flow.png
│   ├── 04_pre_post_activation.png
│   └── 05_loss_functions.png
│
└── requirements.txt
```

A minimal `requirements.txt` can be:

```text
torch
numpy
pandas
matplotlib
seaborn
scikit-learn
```

The `outputs/` directory is optional, but recommended. The current scripts save figures in the working directory. You can either keep that behavior or modify each `plt.savefig(...)` call to save inside `outputs/`.

Example:

```python
plt.savefig("outputs/03_weight_init.png", dpi=150, bbox_inches="tight")
```

---

# 4. Installation

## 4.1 Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/pytorch-mlp-foundations-wine-quality.git
cd pytorch-mlp-foundations-wine-quality
```

## 4.2 Create a Python virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

## 4.3 Install dependencies

Install packages directly:

```bash
pip install torch numpy pandas matplotlib seaborn scikit-learn
```

Or install using `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

# 5. How to Run

Run the scripts in this order:

```bash
python dataset_loading_v1.py
python mlp_architecture_v1.py
python forward_pass_traced_v1.py
python loss_functions_v1.py
```

The order matters conceptually:

```text
dataset_loading_v1.py
    prepares and explains the data

mlp_architecture_v1.py
    builds and inspects neural-network architectures

forward_pass_traced_v1.py
    traces how data moves through the network

loss_functions_v1.py
    explains how prediction error is measured
```

---

# 6. Dataset Background

The first script uses the UCI Red Wine Quality dataset.

The dataset contains:

```text
1599 wine samples
11 physicochemical input features
1 quality score target
```

The 11 input features are:

```text
fixed acidity
volatile acidity
citric acid
residual sugar
chlorides
free sulfur dioxide
total sulfur dioxide
density
pH
sulphates
alcohol
```

The original target column is:

```text
quality
```

The quality score in this dataset mainly ranges from 3 to 8.

The project converts the original score into binary labels:

```python
label = 1 if quality >= 7 else 0
```

So:

```text
label = 0 → ordinary wine
label = 1 → good wine
```

This turns the problem into binary classification.

---

# 7. Script 1: `dataset_loading_v1.py`

## 7.1 Purpose

This script demonstrates a complete tabular-data preparation pipeline for PyTorch.

It teaches how to go from a raw CSV dataset to PyTorch-ready tensors, custom datasets, and mini-batch loaders.

The script covers:

1. loading the dataset,
2. inspecting rows, columns, shape, and memory usage,
3. printing statistical summaries,
4. examining quality-score distribution,
5. creating binary labels,
6. checking class imbalance,
7. separating features and labels,
8. normalizing features,
9. converting arrays to PyTorch tensors,
10. defining a custom `Dataset`,
11. splitting into train/validation/test subsets,
12. creating `DataLoader` objects,
13. inspecting one mini-batch,
14. plotting feature distributions,
15. plotting a correlation heatmap.

---

## 7.2 Loading the Dataset

The script first attempts to download the dataset from the UCI repository:

```python
url = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)

df = pd.read_csv(url, sep=";")
```

If internet access is unavailable, it attempts to load a local file:

```python
df = pd.read_csv("winequality-red.csv", sep=";")
```

This makes the script robust because it can work both online and offline, provided the CSV file is available locally.

---

## 7.3 Dataset Shape and Columns

The dataset has:

```text
1599 samples
12 columns
```

The 12 columns include:

```text
11 input features
1 original quality score
```

The script prints:

```python
df.shape
df.columns.tolist()
df.head(3)
```

This is important because before building any model, one should verify:

- number of samples,
- number of columns,
- column names,
- whether values look reasonable,
- whether the CSV was parsed correctly.

---

## 7.4 Statistical Summary

The script uses:

```python
df.describe()
```

to print count, mean, standard deviation, minimum, quartiles, and maximum for each numerical column.

This helps identify:

- feature ranges,
- feature scales,
- possible outliers,
- whether features are on different numerical scales.

For example, some features have values near 1, while sulfur dioxide features can have much larger ranges. This matters because neural networks are sensitive to input scale.

---

## 7.5 Quality-Score Distribution

The script computes:

```python
vc = df["quality"].value_counts().sort_index()
```

and prints the number of samples for each quality score.

This reveals that most wines are in the middle-quality range, especially quality 5 and quality 6. Very few samples have very low or very high scores.

This is important because the binary classification problem is not based on a balanced set of ordinary and good wines.

---

## 7.6 Binary Label Creation

The original quality score is converted into a binary target using:

```python
QUALITY_THRESHOLD = 7
df["label"] = (df["quality"] >= QUALITY_THRESHOLD).astype(int)
```

Meaning:

```text
quality >= 7 → good wine     → label 1
quality <  7 → ordinary wine → label 0
```

This is a domain choice. Quality 7 and above are treated as good/premium wines.

---

## 7.7 Class Imbalance

After creating labels, the script computes:

```python
n_good = df["label"].sum()
n_bad = len(df) - n_good
ratio = n_good / len(df)
```

Typical result:

```text
Class 0 ordinary: 1382 samples
Class 1 good    : 217 samples
```

So the positive class is only about:

```text
13.6%
```

This is a major outcome.

A naive classifier that always predicts class 0 would get:

```text
86.4% accuracy
```

but it would detect zero good wines.

Therefore, accuracy alone is misleading. Future training should evaluate:

```text
precision
recall
F1-score
confusion matrix
ROC-AUC
PR-AUC
```

For imbalanced binary classification, recall and precision for the positive class are especially important.

---

## 7.8 Feature/Label Separation

The script separates input features and target labels:

```python
feature_names = [c for c in df.columns if c not in ["quality", "label"]]

X = df[feature_names].values
y = df["label"].values
```

Result:

```text
X shape = (1599, 11)
y shape = (1599,)
```

This means:

- `X` contains all input features,
- `y` contains the binary labels.

Each row of `X` corresponds to one wine sample.

---

## 7.9 Feature Normalization

The script applies standard normalization:

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

The transformation is:

```text
x_scaled = (x - mean) / standard_deviation
```

After this transformation, each feature has approximately:

```text
mean = 0
standard deviation = 1
```

This matters because raw features may have very different scales.

For example:

```text
density is near 1.0
alcohol is around 8–15
total sulfur dioxide can be much larger
```

Without normalization, the optimizer may behave poorly because features with larger numerical scales can dominate gradient updates.

### Important note about data leakage

The script prints the key lesson:

```text
Always split BEFORE fitting the scaler.
```

For production-quality ML, the ideal sequence is:

```text
1. split raw data into train/validation/test
2. fit StandardScaler only on training data
3. transform validation and test using the training scaler
```

This prevents validation/test information from leaking into training preprocessing.

---

## 7.10 PyTorch Tensor Conversion

The normalized feature matrix and label vector are converted to tensors:

```python
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)
```

The shapes are:

```text
X_tensor shape = (1599, 11)
y_tensor shape = (1599,)
```

The data type `float32` is standard for neural-network training.

For binary classification with `BCEWithLogitsLoss`, labels should be floating-point values, not integer class indices.

---

## 7.11 Custom Dataset Class

The script defines:

```python
class WineDataset(Dataset):
    ...
```

The class implements:

```python
__len__()
```

and:

```python
__getitem__()
```

These are required by PyTorch.

### `__len__`

Returns the number of samples:

```python
def __len__(self):
    return len(self.X)
```

### `__getitem__`

Returns one feature-label pair:

```python
def __getitem__(self, idx):
    return self.X[idx], self.y[idx]
```

This allows PyTorch to call:

```python
dataset[i]
```

and receive:

```text
X_i, y_i
```

where `X_i` is one 11-dimensional feature vector and `y_i` is its binary label.

---

## 7.12 Train/Validation/Test Split

The dataset is split into:

```text
70% training
15% validation
15% test
```

The script computes:

```python
n_train = int(0.70 * n_total)
n_val = int(0.15 * n_total)
n_test = n_total - n_train - n_val
```

Typical split:

```text
Train: 1119 samples
Val  : 239 samples
Test : 241 samples
```

The split is done using:

```python
random_split(
    dataset,
    [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(42)
)
```

The fixed seed makes the split reproducible.

### What each split is used for

Training set:

```text
Used to update model weights.
```

Validation set:

```text
Used during model development to tune architecture, learning rate, regularization, and threshold.
```

Test set:

```text
Used only once at the end to estimate final generalization performance.
```

---

## 7.13 DataLoaders

The script creates:

```python
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=64, shuffle=False)
test_loader  = DataLoader(test_set, batch_size=64, shuffle=False)
```

A `DataLoader` wraps a dataset and returns mini-batches.

For training:

```text
shuffle=True
```

because the model should see samples in a different order each epoch.

For validation/test:

```text
shuffle=False
```

because evaluation does not require random ordering.

### Batch shape

A training batch has:

```text
X_batch shape = (32, 11)
y_batch shape = (32,)
```

This means:

```text
32 wine samples
11 features per sample
32 binary labels
```

---

## 7.14 Feature Distribution Plot

The script saves:

```text
02_feature_distributions.png
```

This figure compares feature distributions between:

```text
ordinary wines
good wines
```

Important observations usually include:

- good wines tend to have higher alcohol,
- good wines tend to have lower volatile acidity,
- good wines tend to have somewhat higher sulphates,
- good wines tend to have somewhat higher citric acid,
- many features overlap strongly between classes.

This tells us the task is learnable but not trivial.

There is no single perfect threshold on one feature that separates good and ordinary wines.

---

## 7.15 Correlation Heatmap

The script saves:

```text
02_correlation_heatmap.png
```

This figure shows pairwise correlations between features and the binary label.

Important observations:

- alcohol usually has the strongest positive correlation with the good-wine label,
- volatile acidity has a negative correlation with the good-wine label,
- sulphates and citric acid show moderate positive relationships,
- fixed acidity and pH are strongly negatively correlated,
- free sulfur dioxide and total sulfur dioxide are positively correlated,
- alcohol and density are negatively correlated.

This gives intuition about which features may be useful for the model.

---

## 7.16 Main Takeaways from `dataset_loading_v1.py`

After this script, you should understand:

```text
how to load a tabular CSV dataset
how to inspect basic dataset properties
how to create binary labels
why class imbalance matters
why normalization matters
how to convert data to PyTorch tensors
how to write a custom Dataset
how to create DataLoaders
how to inspect mini-batch shapes
how to visualize feature distributions
how to interpret simple correlations
```

---

# 8. Script 2: `mlp_architecture_v1.py`

## 8.1 Purpose

This script explains how to build MLPs in PyTorch.

It answers:

```text
How do we define a neural network that maps 11 input features to one binary output?
```

The script covers:

1. `nn.Module`,
2. `nn.Linear`,
3. ReLU and Sigmoid activations,
4. hardcoded MLP architecture,
5. configurable MLP architecture,
6. `nn.Sequential`,
7. trainable parameter counting,
8. per-layer parameter breakdown,
9. weight initialization,
10. dummy forward-pass shape checks.

---

## 8.2 What is an MLP?

An MLP, or multilayer perceptron, is a neural network made of fully connected layers.

A typical MLP layer computes:

```text
z = W x + b
a = activation(z)
```

where:

- `x` is the input to the layer,
- `W` is the weight matrix,
- `b` is the bias vector,
- `z` is the pre-activation,
- `a` is the post-activation.

For this project, the model input has 11 features and the output is one probability/logit for binary classification.

---

## 8.3 `SimpleMLP`

The script defines:

```python
class SimpleMLP(nn.Module):
    ...
```

The architecture is:

```text
11 → 64 → 32 → 1
```

with:

```text
ReLU after hidden layers
Sigmoid after output layer
```

In code:

```python
self.layer1 = nn.Linear(11, 64)
self.layer2 = nn.Linear(64, 32)
self.layer3 = nn.Linear(32, 1)

self.relu = nn.ReLU()
self.sigmoid = nn.Sigmoid()
```

The forward pass is:

```python
x = self.layer1(x)
x = self.relu(x)

x = self.layer2(x)
x = self.relu(x)

x = self.layer3(x)
x = self.sigmoid(x)
```

The output lies in:

```text
(0, 1)
```

and can be interpreted as:

```text
P(good wine)
```

---

## 8.4 Why Inherit from `nn.Module`?

PyTorch models inherit from:

```python
nn.Module
```

This gives the model:

```text
automatic parameter registration
model.parameters()
model.train()
model.eval()
model.to(device)
model.state_dict()
model.load_state_dict()
```

Without inheriting from `nn.Module`, PyTorch would not know how to collect weights and biases for training.

---

## 8.5 `__init__()` vs `forward()`

In PyTorch model classes:

```python
__init__()
```

defines the layers.

```python
forward()
```

defines how data flows through the layers.

Example:

```python
def __init__(self):
    self.layer1 = nn.Linear(11, 64)

def forward(self, x):
    x = self.layer1(x)
    return x
```

The important distinction is:

```text
__init__ creates reusable layer objects
forward applies those layers to data
```

---

## 8.6 PyTorch `nn.Linear` Shape Convention

For:

```python
nn.Linear(in_features, out_features)
```

PyTorch stores the weight matrix as:

```text
(out_features, in_features)
```

Therefore:

```python
nn.Linear(11, 64)
```

has:

```text
weight shape = (64, 11)
bias shape   = (64,)
```

For a batch:

```text
X shape = (batch_size, 11)
```

the output is:

```text
X @ W.T + b
```

and has shape:

```text
(batch_size, 64)
```

---

## 8.7 `DeepMLP`: Configurable Architecture

The script also defines:

```python
class DeepMLP(nn.Module):
    ...
```

This class allows the architecture to be created from a list of hidden sizes.

Example:

```python
DeepMLP(11, [128, 64, 32])
```

creates:

```text
11 → 128 → 64 → 32 → 1
```

This is useful because in real machine learning, you often experiment with different model widths and depths.

---

## 8.8 Architectures Compared

The script creates several models:

```text
tiny    : 11 → 16 → 1
small   : 11 → 64 → 32 → 1
medium  : 11 → 128 → 64 → 32 → 1
large   : 11 → 256 → 128 → 64 → 32 → 1
deep    : 11 → 128 → 128 → 64 → 64 → 32 → 16 → 1
dropout : medium architecture with dropout
batchnorm: medium architecture with batch normalization
```

This shows how parameter counts change with architecture.

---

## 8.9 Parameter Counting

For one linear layer:

```python
nn.Linear(m, n)
```

the number of trainable parameters is:

```text
weights = n × m
biases  = n
total   = n × m + n
```

Example:

```python
nn.Linear(11, 128)
```

has:

```text
weights = 128 × 11 = 1408
biases  = 128
total   = 1536
```

The script counts trainable parameters using:

```python
sum(p.numel() for p in model.parameters() if p.requires_grad)
```

This is useful because model size affects:

```text
memory cost
training time
overfitting risk
generalization
```

---

## 8.10 Interpreting Parameter Counts

A model with more parameters is more expressive, but not automatically better.

For a small dataset with 1599 samples, very large networks may overfit.

A reasonable starting point is:

```text
11 → 64 → 32 → 1
```

or:

```text
11 → 128 → 64 → 32 → 1
```

The very large or deep models should only be used if smaller models clearly underfit.

---

## 8.11 Dropout

Dropout randomly sets some activations to zero during training.

Example:

```python
nn.Dropout(p=0.3)
```

means each activation has a 30% probability of being temporarily removed during training.

Dropout:

```text
does not add trainable parameters
acts as regularization
helps reduce overfitting
is active during model.train()
is disabled during model.eval()
```

---

## 8.12 Batch Normalization

BatchNorm normalizes activations within a batch.

For each feature, BatchNorm learns:

```text
gamma: scale parameter
beta : shift parameter
```

Therefore, BatchNorm adds trainable parameters.

BatchNorm may help:

```text
stabilize training
reduce sensitivity to initialization
allow larger learning rates
```

---

## 8.13 Weight Initialization

The script compares first-layer weight distributions and saves:

```text
03_weight_init.png
```

The two initialization types discussed are:

```text
Kaiming initialization
Xavier initialization
```

### Kaiming Initialization

Kaiming initialization is usually preferred for ReLU networks.

It roughly preserves activation variance through layers by accounting for the fact that ReLU zeros out many negative values.

### Xavier Initialization

Xavier initialization is often used for sigmoid/tanh-style activations.

It considers both input and output dimensions.

### Why initialization matters

Bad initialization can cause:

```text
vanishing activations
exploding activations
poor gradient flow
slow training
unstable training
```

---

## 8.14 Dummy Shape Verification

The script sends fake batches through the model:

```python
x_dummy = torch.randn(bs, 11)
out = model_simple(x_dummy)
```

for batch sizes:

```text
1
8
32
```

Expected output shapes:

```text
(1, 1)
(8, 1)
(32, 1)
```

This verifies that:

```text
the model accepts 11 input features
the batch dimension is flexible
the output shape is correct for binary classification
```

Before training any model, this kind of shape check is extremely useful.

---

## 8.15 Main Takeaways from `mlp_architecture_v1.py`

After this script, you should understand:

```text
how to define an MLP with nn.Module
why super().__init__() is necessary
how __init__ and forward differ
how nn.Linear stores weights and biases
how ReLU and Sigmoid are used
how to build configurable architectures
how nn.Sequential chains layers
how to count trainable parameters
why more parameters does not always mean better performance
what dropout and batchnorm do
why initialization matters
how to verify input/output shapes before training
```

---

# 9. Script 3: `forward_pass_traced_v1.py`

## 9.1 Purpose

This script traces forward propagation through an MLP.

It answers:

```text
What happens to a sample as it passes through the network layer by layer?
```

The script covers:

1. manual layer-by-layer tracing,
2. single-sample forward pass,
3. batch forward pass,
4. pre-activation vs post-activation,
5. ReLU active/inactive outputs,
6. forward hooks,
7. activation statistics,
8. initialization effects,
9. visualization of activation flow,
10. visualization of pre-ReLU vs post-ReLU distributions.

---

## 9.2 Model Used

The model is:

```text
Linear(11 → 64)
ReLU
Linear(64 → 32)
ReLU
Linear(32 → 1)
Sigmoid
```

The mathematical flow is:

```text
z1 = W1 x + b1
a1 = ReLU(z1)

z2 = W2 a1 + b2
a2 = ReLU(z2)

z3 = W3 a2 + b3
ŷ  = sigmoid(z3)
```

---

## 9.3 Single-Sample Trace

The script creates one fake normalized input:

```python
x_single = torch.randn(11)
```

This has shape:

```text
(11,)
```

This represents one wine sample with 11 normalized features.

The script then manually computes:

```python
z1 = L1(x_single)
a1 = torch.relu(z1)

z2 = L2(a1)
a2 = torch.relu(z2)

z3 = L3(a2)
yhat = torch.sigmoid(z3)
```

This makes the mathematical forward pass visible.

---

## 9.4 Shape Flow for One Sample

The shape flow is:

```text
x_single : (11,)
z1       : (64,)
a1       : (64,)
z2       : (32,)
a2       : (32,)
z3       : (1,)
yhat     : (1,)
```

This verifies that one sample is transformed from 11 raw input features to one predicted probability.

---

## 9.5 Pre-Activation and Post-Activation

For each hidden layer:

```text
z = linear output before activation
a = activation output after nonlinearity
```

For ReLU:

```text
a = ReLU(z) = max(0, z)
```

So:

```text
z < 0 → a = 0
z > 0 → a = z
```

This means ReLU introduces nonlinearity and sparsity.

---

## 9.6 Active ReLU Outputs

The script counts:

```python
(a1 > 0).sum()
```

This gives the number of active ReLU outputs.

If roughly 50% of activations are positive at initialization, this is normal for centered random pre-activations.

Important distinction:

```text
zero activation for one sample or one batch is normal
```

but:

```text
a permanently dead ReLU neuron is inactive for nearly all inputs
```

The script uses the phrase "dead neurons" in one plot title, but the more precise term for this educational context is:

```text
zero activations for this batch
```

---

## 9.7 Batch Forward Pass

The script creates:

```python
x_batch = torch.randn(32, 11)
```

This represents:

```text
32 samples
11 features per sample
```

The output is:

```text
(32, 1)
```

which means:

```text
one predicted probability per sample
```

Batch processing is important because real training uses mini-batches, not individual samples.

---

## 9.8 Forward Hooks

The script uses forward hooks:

```python
layer.register_forward_hook(...)
```

A forward hook is automatically called when a layer performs a forward pass.

The hook receives:

```text
module
input
output
```

This allows inspection of intermediate outputs without modifying the model architecture.

The script records:

```text
shape
minimum
maximum
mean
standard deviation
fraction positive
```

This is useful for detecting:

```text
wrong tensor shapes
exploding activations
vanishing activations
ReLU inactivity
unexpected output ranges
```

The script removes hooks after use:

```python
h.remove()
```

This is important because hooks persist unless explicitly removed.

---

## 9.9 Initialization and Activation Flow

The script builds an 8-layer ReLU network and compares activation standard deviation across layers for:

```text
Kaiming initialization
Xavier initialization
Too large initialization
Zero initialization
```

It saves:

```text
04_activation_flow.png
```

### Kaiming initialization

Kaiming initialization is designed for ReLU networks.

It keeps activation scale relatively stable through depth.

### Xavier initialization

Xavier may gradually reduce activation scale in deeper ReLU networks.

This can lead to vanishing activations.

### Too-large initialization

Weights initialized with very large standard deviation cause activation magnitudes to explode rapidly with depth.

This can lead to:

```text
huge activations
huge gradients
NaN loss
unstable training
```

### Zero initialization

Zero weights cause all neurons to compute identical outputs.

This destroys symmetry breaking and prevents neurons from learning diverse features.

---

## 9.10 Pre-ReLU vs Post-ReLU Distribution

The script saves:

```text
04_pre_post_activation.png
```

This figure compares:

```text
z1 before ReLU
a1 after ReLU
```

Before ReLU, the distribution is usually approximately centered around zero.

After ReLU:

```text
negative values become exactly zero
positive values remain unchanged
```

So the post-ReLU histogram has:

```text
a large spike at zero
a positive tail
```

This visually explains ReLU behavior.

---

## 9.11 Main Takeaways from `forward_pass_traced_v1.py`

After this script, you should understand:

```text
how one input sample moves through an MLP
how a batch moves through an MLP
what pre-activation z means
what post-activation a means
how ReLU changes activations
why about half of ReLU outputs may be zero at initialization
how to use hooks to inspect internal tensors
why hooks must be removed after use
how initialization affects activation flow
why too-large or zero initialization is harmful
how to diagnose activation distributions
```

---

# 10. Script 4: `loss_functions_v1.py`

## 10.1 Purpose

This script explains loss functions for classification and regression.

It answers:

```text
How do we measure how wrong a model prediction is?
```

The script covers:

1. Binary Cross-Entropy,
2. Mean Squared Error,
3. BCE vs MSE for binary classification,
4. manual BCE calculation,
5. PyTorch `BCELoss`,
6. loss landscape visualization,
7. gradient comparison,
8. `BCEWithLogitsLoss`,
9. MSE/MAE for regression,
10. loss-function selection guide.

---

## 10.2 Binary Cross-Entropy

For binary classification:

```text
y ∈ {0, 1}
ŷ ∈ (0, 1)
```

Binary Cross-Entropy is:

```text
L = -[y log(ŷ) + (1-y) log(1-ŷ)]
```

For a positive sample:

```text
y = 1
L = -log(ŷ)
```

For a negative sample:

```text
y = 0
L = -log(1-ŷ)
```

This means:

```text
correct confident prediction   → small loss
uncertain prediction           → moderate loss
confident wrong prediction     → large loss
```

---

## 10.3 Manual BCE Demonstration

The script manually evaluates BCE for six cases:

```text
y=1, ŷ=0.9 → correct and confident
y=1, ŷ=0.5 → uncertain
y=1, ŷ=0.1 → wrong and confident

y=0, ŷ=0.1 → correct and confident
y=0, ŷ=0.5 → uncertain
y=0, ŷ=0.9 → wrong and confident
```

Expected values:

```text
correct confident: 0.1054
uncertain        : 0.6931
wrong confident  : 2.3026
```

This comes from:

```text
-log(0.9) = 0.1054
-log(0.5) = 0.6931
-log(0.1) = 2.3026
```

---

## 10.4 PyTorch `BCELoss`

The script checks:

```python
nn.BCELoss(reduction="none")
```

against the manual BCE formula.

The match confirms that PyTorch computes the same mathematical expression:

```text
L = -[y log(ŷ) + (1-y) log(1-ŷ)]
```

`reduction="none"` returns one loss value per sample.

During training, the mean loss is usually used.

---

## 10.5 MSE Comparison

Mean Squared Error is:

```text
L = (ŷ - y)^2
```

For binary classification, MSE is usually weaker than BCE.

For example:

```text
y = 1
ŷ = 0.1
```

BCE:

```text
2.3026
```

MSE:

```text
0.8100
```

BCE penalizes the confident wrong prediction more strongly.

MSE is bounded between 0 and 1 for binary probability outputs, while BCE can grow without bound as the prediction becomes confidently wrong.

---

## 10.6 Loss Landscape Plot

The script saves:

```text
05_loss_functions.png
```

This figure has four panels.

### Panel 1: Loss for true positive `y=1`

Shows:

```text
BCE = -log(ŷ)
MSE = (1-ŷ)^2
```

As:

```text
ŷ → 0
```

BCE goes to infinity, while MSE stays bounded.

### Panel 2: Gradient magnitude for `y=1`

For BCE:

```text
dL/dŷ = -1/ŷ
```

So when `ŷ` is near 0, the gradient magnitude is very large.

For MSE:

```text
dL/dŷ = -2(1-ŷ)
```

This is bounded.

### Panel 3: Both binary classes

For `y=1`, BCE explodes when `ŷ → 0`.

For `y=0`, BCE explodes when `ŷ → 1`.

This shows that BCE strongly penalizes confident mistakes for both classes.

### Panel 4: Gradient with respect to logit `z`

This is the most important panel.

A neural network often outputs a raw logit:

```text
z
```

Then:

```text
ŷ = sigmoid(z)
```

For BCE with sigmoid:

```text
dL/dz = ŷ - y
```

For `y=1`, if the model is badly wrong and `ŷ ≈ 0`, then:

```text
dL/dz ≈ -1
```

This is a strong gradient.

For MSE with sigmoid, the gradient contains:

```text
ŷ(1-ŷ)
```

When sigmoid saturates near 0 or 1, this term becomes tiny.

Therefore:

```text
MSE + sigmoid can suffer from vanishing gradients.
```

---

## 10.7 `BCEWithLogitsLoss`

The script compares:

```python
nn.BCELoss()(torch.sigmoid(z), y)
```

with:

```python
nn.BCEWithLogitsLoss()(z, y)
```

They produce nearly identical values.

However, `BCEWithLogitsLoss` is preferred because it combines sigmoid and BCE internally in a numerically stable way.

Correct setup for binary classification:

```python
model output = raw logit
loss = nn.BCEWithLogitsLoss()
```

Do **not** put Sigmoid at the end of the model if using `BCEWithLogitsLoss`.

During inference or evaluation:

```python
probs = torch.sigmoid(logits)
```

---

## 10.8 MSE and MAE for Regression

The script also explains that MSE is correct for regression.

Regression means predicting continuous values.

Examples:

```text
actual wine quality score
viscosity
storage modulus G'
radius of gyration Rg
diffusion coefficient
energy
```

For regression:

```python
nn.MSELoss()
```

or:

```python
nn.L1Loss()
```

is appropriate.

The model output should usually be linear:

```python
nn.Linear(hidden_dim, 1)
```

with no sigmoid activation.

---

## 10.9 Loss Function Selection Guide

Summary:

| Task | Recommended loss | Output activation |
|---|---|---|
| Binary classification | `nn.BCEWithLogitsLoss()` | None, raw logit |
| Binary classification alternative | `nn.BCELoss()` | `nn.Sigmoid()` |
| Multi-class classification | `nn.CrossEntropyLoss()` | None, raw logits |
| Regression | `nn.MSELoss()` / `nn.L1Loss()` | None, linear output |

Important rule:

```text
The final activation and loss function must be chosen together.
```

Wrong:

```python
model ends with Sigmoid()
criterion = nn.BCEWithLogitsLoss()
```

Correct:

```python
model outputs raw logits
criterion = nn.BCEWithLogitsLoss()
```

---

## 10.10 Main Takeaways from `loss_functions_v1.py`

After this script, you should understand:

```text
what a loss function is
why loss functions define the training objective
how BCE is computed manually
how PyTorch BCELoss matches the manual formula
why BCE is better than MSE for binary classification
how loss gradients affect training
why MSE + sigmoid can have vanishing gradients
why BCEWithLogitsLoss is preferred
when MSE and MAE are appropriate
how to choose loss functions for different ML tasks
```

---

# 11. Expected Output Figures

Running the scripts produces the following figures:

```text
02_feature_distributions.png
02_correlation_heatmap.png
03_weight_init.png
04_activation_flow.png
04_pre_post_activation.png
05_loss_functions.png
```

## 11.1 `02_feature_distributions.png`

Shows distributions of each wine feature for ordinary and good wines.

Purpose:

```text
Understand which features separate the classes visually.
```

## 11.2 `02_correlation_heatmap.png`

Shows feature-feature correlations and feature-label correlations.

Purpose:

```text
Understand linear relationships between variables.
```

## 11.3 `03_weight_init.png`

Shows initial weight distributions for Kaiming/default-like and Xavier initialization.

Purpose:

```text
Understand how different initialization schemes choose different weight scales.
```

## 11.4 `04_activation_flow.png`

Shows activation standard deviation across layers for different initialization schemes.

Purpose:

```text
Understand vanishing and exploding activations.
```

## 11.5 `04_pre_post_activation.png`

Shows first-layer pre-ReLU and post-ReLU activation distributions.

Purpose:

```text
Understand how ReLU converts negative values into zeros.
```

## 11.6 `05_loss_functions.png`

Shows BCE vs MSE loss curves and gradient behavior.

Purpose:

```text
Understand why BCE is preferred for binary classification.
```

---

# 12. Conceptual Flow Across the Four Scripts

The four scripts are designed to be read and run in sequence.

```text
dataset_loading_v1.py
```

teaches:

```text
How do we prepare a real dataset for PyTorch?
```

```text
mlp_architecture_v1.py
```

teaches:

```text
How do we define a neural network architecture?
```

```text
forward_pass_traced_v1.py
```

teaches:

```text
How does data move through the network?
```

```text
loss_functions_v1.py
```

teaches:

```text
How do we measure prediction error?
```

Together:

```text
Data → Model → Forward Pass → Loss
```

This is the foundation of training.

A future training script would add:

```text
Loss → Backpropagation → Optimizer update → Validation → Metrics
```

---

# 13. Mathematical Summary

## 13.1 Linear Layer

For one sample:

```text
z = W x + b
```

For a batch:

```text
Z = X Wᵀ + b
```

PyTorch stores:

```text
W shape = (out_features, in_features)
```

---

## 13.2 ReLU

```text
ReLU(z) = max(0, z)
```

Meaning:

```text
z < 0 → 0
z > 0 → z
```

---

## 13.3 Sigmoid

```text
sigmoid(z) = 1 / (1 + exp(-z))
```

Maps a raw logit to a probability:

```text
z → ŷ ∈ (0, 1)
```

---

## 13.4 Binary Cross-Entropy

```text
L = -[y log(ŷ) + (1-y) log(1-ŷ)]
```

For `y=1`:

```text
L = -log(ŷ)
```

For `y=0`:

```text
L = -log(1-ŷ)
```

---

## 13.5 MSE

```text
L = (ŷ - y)^2
```

Used mainly for regression.

---

## 13.6 BCE with Logits

Recommended binary-classification setup:

```python
criterion = nn.BCEWithLogitsLoss()
```

Model output should be raw logits:

```python
logits = model(x)
```

Convert to probabilities only when needed:

```python
probs = torch.sigmoid(logits)
```

---

# 14. Important Practical Lessons

## 14.1 Normalize Features

Neural networks train better when input features are on comparable scales.

Use:

```python
StandardScaler()
```

or another appropriate normalization method.

---

## 14.2 Avoid Data Leakage

Fit preprocessing tools such as scalers only on the training set.

Correct:

```text
split first
fit scaler on training set
transform validation/test with training scaler
```

Incorrect:

```text
fit scaler on full dataset
then split
```

---

## 14.3 Do Not Trust Accuracy Alone for Imbalanced Data

The wine dataset is imbalanced.

A model can get high accuracy by predicting only the majority class.

Use:

```text
precision
recall
F1-score
PR-AUC
ROC-AUC
confusion matrix
```

---

## 14.4 Use `BCEWithLogitsLoss` for Binary Classification

Preferred setup:

```python
model = nn.Sequential(
    nn.Linear(...),
    nn.ReLU(),
    nn.Linear(..., 1)
)

criterion = nn.BCEWithLogitsLoss()
```

No sigmoid inside the model during training.

---

## 14.5 Handle Class Imbalance

For the wine dataset:

```text
ordinary wines = 1382
good wines     = 217
```

A useful positive-class weight is:

```text
pos_weight = 1382 / 217 ≈ 6.37
```

Future training can use:

```python
pos_weight = torch.tensor([6.37])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

This makes mistakes on good wines more costly.

---

## 14.6 Start with a Small Model

For only 1599 samples, do not start with a huge network.

Good starting architecture:

```text
11 → 64 → 32 → 1
```

or:

```text
11 → 128 → 64 → 32 → 1
```

Large and deep models should be justified by validation performance.

---

## 14.7 Verify Shapes Before Training

Always test a dummy batch:

```python
x_dummy = torch.randn(32, 11)
out = model(x_dummy)
print(out.shape)
```

Expected:

```text
(32, 1)
```

This prevents many training-loop errors.

---

## 14.8 Use Hooks for Debugging

Forward hooks are useful for inspecting:

```text
hidden layer outputs
activation means and standard deviations
dead/inactive ReLU behavior
exploding/vanishing activations
```

Remember to remove hooks after use.

---

# 15. Suggested Next Script: Training Loop

The next logical script would be:

```text
06_training_loop.py
```

It should combine all four previous scripts and add actual training.

Suggested components:

```text
load/preprocess data
create DataLoaders
define MLP without final Sigmoid
define BCEWithLogitsLoss with pos_weight
define optimizer
run training loop
run validation loop
track loss
compute precision/recall/F1
plot loss curves
plot confusion matrix
evaluate on test set
save trained model
```

Suggested model:

```text
Input: 11 features
Hidden 1: 64 neurons + ReLU
Hidden 2: 32 neurons + ReLU
Output: 1 raw logit
```

Suggested loss:

```python
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([6.37]))
```

Suggested optimizer:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

Suggested evaluation:

```python
logits = model(X)
probs = torch.sigmoid(logits)
preds = (probs > threshold).float()
```

For imbalanced data, tune the threshold using validation precision-recall behavior.

---

# 16. Common Mistakes This Project Helps Avoid

## Mistake 1: Training before understanding data

This project first inspects the dataset, class balance, feature ranges, and distributions.

## Mistake 2: Ignoring class imbalance

The dataset is strongly imbalanced. Accuracy alone can be misleading.

## Mistake 3: Not normalizing features

Raw tabular features can have very different scales. Normalization is essential for stable neural-network training.

## Mistake 4: Confusing logits and probabilities

A raw model output is a logit. After sigmoid, it becomes a probability.

## Mistake 5: Using Sigmoid with `BCEWithLogitsLoss`

`BCEWithLogitsLoss` already applies sigmoid internally.

## Mistake 6: Using MSE for binary classification

MSE can produce weak gradients when used with sigmoid for classification.

## Mistake 7: Assuming more parameters means better model

Large models can overfit small datasets.

## Mistake 8: Not checking tensor shapes

Shape errors are common in PyTorch. Dummy forward passes prevent many bugs.

## Mistake 9: Forgetting to remove hooks

Hooks persist unless removed and can cause confusing behavior.

## Mistake 10: Calling `forward()` directly

Use:

```python
output = model(x)
```

not:

```python
output = model.forward(x)
```

---

# 17. Who This Project Is For

This project is useful for:

```text
students learning PyTorch
researchers moving from scientific computing to machine learning
soft matter physicists learning neural networks
biophysics researchers learning tabular ML
materials scientists working with property-prediction datasets
anyone learning neural networks from first principles
```

Although the example dataset is wine quality, the same workflow applies to scientific problems such as:

```text
phase classification
glass transition prediction
material property prediction
protein conformational-state classification
viscosity prediction
diffusion coefficient prediction
radius of gyration prediction
storage modulus prediction
free-energy regression
```

---

# 18. Final Summary

This repository is a foundation-level PyTorch project that explains the deep-learning workflow from data to loss.

The four scripts teach:

```text
dataset_loading_v1.py
    data loading, preprocessing, normalization, Dataset, DataLoader

mlp_architecture_v1.py
    nn.Module, Linear layers, MLP architectures, parameters, initialization

forward_pass_traced_v1.py
    forward propagation, activations, hooks, initialization effects

loss_functions_v1.py
    BCE, MSE, BCEWithLogitsLoss, regression losses, gradient behavior
```

The central message is:

```text
A neural network is not a black box.
It is a chain of tensor transformations, nonlinear activations,
trainable parameters, and a loss-defined optimization objective.
```

Once these four components are clear, the next step is straightforward:

```text
write the training loop
compute gradients
update weights
validate performance
evaluate carefully
```

This project is therefore the conceptual foundation for building reliable PyTorch models on tabular scientific datasets.
