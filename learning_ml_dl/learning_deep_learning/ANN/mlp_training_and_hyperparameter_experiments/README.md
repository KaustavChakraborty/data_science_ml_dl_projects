# PyTorch MLP Training Lab

A compact, educational deep-learning project that demonstrates how to build a complete neural-network training pipeline in PyTorch and how to study the effect of important hyperparameters through controlled experiments.

This project contains two Python scripts:

```text
training_loop.py   # complete training, evaluation, saving, loading, inference pipeline
experiments.py     # systematic hyperparameter experiments
```

The project uses the UCI Red Wine Quality dataset and converts the original wine-quality score into a binary classification task:

```text
quality >= 7  -> good wine      -> label 1
quality < 7   -> ordinary wine  -> label 0
```

The main purpose is not only to obtain a prediction model, but to understand how a neural network learns, how to diagnose learning behavior, and how choices such as network depth, width, learning rate, activation function, and batch size affect training.

---

## Recommended Directory Name

Recommended directory name:

```text
pytorch-mlp-training-lab
```

Why this name is suitable:

- `pytorch` clearly identifies the deep-learning framework.
- `mlp` describes the model family: multilayer perceptron.
- `training-lab` communicates that this is an educational experiment-based project.
- The name is general enough to reuse later for other tabular scientific datasets, not only wine-quality data.

Suggested placement inside a larger deep-learning repository:

```text
deep_learning/
└── pytorch-mlp-training-lab/
    ├── README.md
    ├── training_loop_v1.py
    └── experiments_v1.py
```

Other possible names:

```text
wine-quality-mlp-lab
mlp-training-and-hyperparameter-experiments
tabular-deep-learning-pytorch
deep-learning-training-pipeline
neural-network-training-lab
```

---

## Project Motivation

Most beginner deep-learning examples show only the final code needed to train a model. This project is different. It is designed to expose the full workflow step by step:

```text
raw data
-> train/validation/test split
-> train-only preprocessing
-> PyTorch tensor conversion
-> DataLoader construction
-> model definition
-> loss function
-> optimizer
-> forward pass
-> backward pass
-> parameter update
-> validation
-> learning-curve plotting
-> final test evaluation
-> confusion matrix
-> model saving/loading
-> inference on a new sample
```

The second script then turns the project into a laboratory. It changes one hyperparameter at a time and observes the effect on validation loss and validation accuracy. This is important because deep learning is not only about writing a model; it is about understanding how optimization, generalization, and model capacity interact.

---

## What This Project Teaches

By studying this project, you will learn:

- How to load a real tabular dataset.
- How to convert a quality score into a binary classification label.
- Why data splitting must happen before preprocessing statistics are calculated.
- Why the scaler must be fitted only on the training set.
- How to avoid data leakage.
- How to convert NumPy arrays into PyTorch tensors.
- How to use `TensorDataset` and `DataLoader`.
- How to define a multilayer perceptron using `torch.nn.Module`.
- How to use binary cross-entropy loss.
- How the Adam optimizer is used in practice.
- Why `optimizer.zero_grad()` is required.
- What `loss.backward()` computes.
- What `optimizer.step()` updates.
- Why `model.train()` and `model.eval()` are different.
- How to track loss and accuracy across epochs.
- How to identify overfitting from learning curves.
- Why validation loss and validation accuracy must be interpreted together.
- Why accuracy can be misleading for imbalanced datasets.
- How to read a classification report.
- How to interpret a confusion matrix.
- How to save and reload a model using `state_dict`.
- How to run inference on a new sample using the same scaler as training.
- How depth, width, learning rate, activation function, and batch size affect training.

---

## Dataset

The project uses the red wine quality dataset from the UCI Machine Learning Repository.

The scripts try to load the dataset directly from:

```text
https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
```

If loading from the URL fails, the scripts try to load a local file:

```text
winequality-red.csv
```

Therefore, the project supports both online and offline execution.

For offline execution, place the dataset in the project directory:

```text
pytorch-mlp-training-lab/
├── winequality-red.csv
├── training_loop_v1.py
└── experiments_v1.py
```

---

## Input Features

The dataset contains physicochemical measurements of red wine samples. The model uses 11 numerical features:

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

Each wine sample is represented as an 11-dimensional vector:

```text
x = [x1, x2, ..., x11]
```

The neural network maps this input vector to a probability-like output:

```text
P(good wine)
```

---

## Target Definition

The original dataset has a `quality` column. The scripts convert this into a binary label:

```python
df["label"] = (df["quality"] >= 7).astype(int)
```

Meaning:

```text
quality 3, 4, 5, 6  -> ordinary wine -> label 0
quality 7, 8        -> good wine     -> label 1
```

The prediction rule is:

```text
if P(good wine) >= 0.5:
    predict good wine
else:
    predict ordinary wine
```

---

## Important Note About Class Imbalance

The dataset is imbalanced. Most samples are ordinary wines, and only a smaller fraction are good wines.

This means that accuracy alone can be misleading. A model that predicts almost everything as ordinary wine can still achieve a high accuracy if ordinary wines dominate the dataset.

Therefore, the project encourages looking at:

```text
precision
recall
F1-score
confusion matrix
validation loss
minority-class performance
```

For this project, the minority class is:

```text
good wine
```

The most important question is not only:

```text
How many total samples did the model classify correctly?
```

but also:

```text
Out of all truly good wines, how many did the model actually detect?
```

That question is answered by the recall of the `good wine` class.

---

## Project Structure

Recommended structure:

```text
pytorch-mlp-training-lab/
│
├── README.md
├── training_loop_v1.py
├── experiments_v1.py
│
├── winequality-red.csv              # optional, only needed for offline use
│
├── outputs/                         # optional, recommended for generated plots
│   ├── 06_learning_curves.png
│   ├── 06_confusion_matrix.png
│   ├── 07A_depth.png
│   ├── 07B_width.png
│   ├── 07C_learning_rate.png
│   ├── 07D_activation.png
│   └── 07E_batch_size.png
│
└── models/                          # optional, recommended for saved models
    └── wine_mlp.pth
```

The current scripts save plots and model files in the working directory. Later, you can modify the paths so all generated plots go into `outputs/` and all trained weights go into `models/`.

---

## File Descriptions

### `training_loop_v1.py`

This script demonstrates the complete training pipeline.

It performs the following tasks:

1. Sets random seeds for reproducibility.
2. Selects CPU or GPU automatically.
3. Loads the red wine dataset.
4. Creates a binary label from the wine-quality score.
5. Splits the data into train, validation, and test sets.
6. Fits `StandardScaler` only on the training set.
7. Transforms train, validation, and test features.
8. Converts arrays into PyTorch tensors.
9. Creates `DataLoader` objects.
10. Defines a small MLP model.
11. Defines binary cross-entropy loss.
12. Defines the Adam optimizer.
13. Trains the model for multiple epochs.
14. Evaluates the model on validation data after each epoch.
15. Records training and validation history.
16. Plots learning curves.
17. Evaluates the final model on the test set.
18. Prints a classification report.
19. Plots a confusion matrix.
20. Saves the trained model weights.
21. Reloads the model weights.
22. Verifies that original and reloaded predictions match.
23. Runs inference on a new raw wine sample.

This script is the best starting point for understanding the full anatomy of a supervised deep-learning workflow.

---

### `experiments_v1.py`

This script performs controlled hyperparameter experiments.

It keeps the dataset, split, scaling procedure, and general training setup fixed, then changes one hyperparameter at a time.

The experiments are:

```text
Experiment A: Network depth
Experiment B: Network width
Experiment C: Learning rate
Experiment D: Activation function
Experiment E: Batch size
```

For each experiment, the script plots:

```text
validation loss vs epoch
validation accuracy vs epoch
```

This helps diagnose:

```text
underfitting
overfitting
unstable optimization
slow convergence
vanishing-gradient behavior
batch-size effects
model-capacity effects
```

---

## Installation

Create a virtual environment:

```bash
python3.10 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install torch pandas numpy matplotlib scikit-learn seaborn
```

A minimal `requirements.txt` can be created as:

```text
torch
pandas
numpy
matplotlib
scikit-learn
seaborn
```

Install from `requirements.txt` using:

```bash
pip install -r requirements.txt
```

---

## How to Run

### Run the complete training pipeline

```bash
python training_loop_v1.py
```

Expected outputs:

```text
06_learning_curves.png
06_confusion_matrix.png
wine_mlp.pth
```

The script will print:

```text
training loss
validation loss
training accuracy
validation accuracy
test loss
test accuracy
classification report
model save/load verification
new-sample prediction
```

---

### Run the hyperparameter experiments

```bash
python experiments_v1.py
```

Expected outputs:

```text
07A_depth.png
07B_width.png
07C_learning_rate.png
07D_activation.png
07E_batch_size.png
```

This script may take a few minutes depending on your machine.

---

## Understanding `training_loop_v1.py`

### Reproducibility

The script sets seeds:

```python
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
```

This helps make results more repeatable. Neural-network training depends on random initialization and shuffled mini-batches, so fixing seeds is important when comparing experiments.

---

### Device Selection

The script uses:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

If a compatible GPU is available, PyTorch will use it. Otherwise, training runs on CPU. For this dataset, CPU training is perfectly acceptable because the dataset is small.

---

### Data Leakage Prevention

The correct preprocessing order is:

```text
1. split data into train/validation/test
2. fit scaler only on training data
3. transform train/validation/test using the training scaler
```

This project follows that correct order.

The wrong approach would be to fit the scaler on the whole dataset before splitting. That would leak validation and test statistics into the training process.

---

### Model Architecture

The default model is:

```text
11 -> 64 -> ReLU -> 32 -> ReLU -> 1 -> Sigmoid
```

The input has 11 features. The final sigmoid produces a value between 0 and 1:

```text
P(good wine)
```

---

### Loss Function

The script uses binary cross-entropy:

```python
criterion = nn.BCELoss()
```

The mathematical form is:

```text
L = -[y log(p) + (1 - y) log(1 - p)]
```

where:

```text
y = true label, 0 or 1
p = predicted probability of class 1
```

If the true label is 1 and the model predicts a very low probability, the loss is large. If the true label is 0 and the model predicts a very high probability, the loss is also large.

---

### Optimizer

The script uses Adam:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

Adam is commonly used because it adapts the update size for each parameter. The learning rate `0.001` is a standard default for Adam.

---

### Core Training Loop

The core batch-level training loop is:

```python
predictions = model(X_batch).squeeze()
loss = criterion(predictions, y_batch)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

Meaning:

```text
forward pass      -> compute predictions
loss calculation  -> measure prediction error
zero gradients    -> clear old gradients
backward pass     -> compute new gradients
optimizer step    -> update model parameters
```

This is the central mechanism of neural-network learning.

---

### Why `model.train()` and `model.eval()` Matter

`model.train()` tells PyTorch the model is in training mode. This matters for layers such as dropout and batch normalization.

`model.eval()` tells PyTorch the model is in evaluation mode. During validation or testing, the model should not behave as if it is still training.

Even though the current MLP does not use dropout or batch normalization, using these modes is still a correct habit and makes the code reusable for future models.

---

### Learning Curves

The script plots loss and accuracy over epochs.

Interpretation:

```text
training loss decreases, validation loss decreases
-> model is learning useful general patterns

training loss decreases, validation loss increases
-> overfitting

both losses stay high
-> underfitting

loss oscillates strongly
-> learning rate may be too high
```

Learning curves are one of the most important diagnostic tools in deep learning.

---

### Classification Report

The classification report includes:

```text
precision
recall
F1-score
support
```

For `good wine`:

```text
precision = when the model predicts good wine, how often is it correct?
recall    = out of all truly good wines, how many did the model find?
F1-score  = balance between precision and recall
```

In imbalanced datasets, class-wise recall and F1-score are often more meaningful than total accuracy.

---

### Confusion Matrix

The confusion matrix shows:

```text
actual ordinary predicted ordinary
actual ordinary predicted good
actual good predicted ordinary
actual good predicted good
```

This is especially useful for seeing whether the model is missing the minority class.

---

### Model Saving and Reloading

The script saves model weights using:

```python
torch.save(model.state_dict(), "wine_mlp.pth")
```

It reloads them using:

```python
model_loaded = WineMLP().to(device)
model_loaded.load_state_dict(torch.load("wine_mlp.pth", map_location=device))
model_loaded.eval()
```

This is the recommended PyTorch approach because saving only `state_dict` is more portable than saving the entire model object.

---

### Inference on a New Sample

The script demonstrates inference on a new raw sample.

Important rule:

```text
Use the same scaler fitted on the training data.
```

Correct:

```python
new_sample_scaled = scaler.transform(new_sample_raw)
```

Incorrect:

```python
new_scaler = StandardScaler()
new_sample_scaled = new_scaler.fit_transform(new_sample_raw)
```

A new scaler would destroy consistency between training and inference.

---

## Understanding `experiments_v1.py`

The experiment script changes one variable at a time. This is important because controlled experimentation allows you to attribute changes in behavior to the variable being tested.

---

### Experiment A: Network Depth

Question:

```text
Does adding more hidden layers improve performance?
```

Configurations:

```text
[64]
[64, 32]
[64, 64, 32]
[64, 64, 32, 16]
[64, 64, 32, 32, 16, 8]
```

Main lesson:

```text
Deeper is not always better.
```

For small tabular datasets, shallow networks often perform as well as deeper networks. Deeper models may overfit because they have more capacity than the dataset can support.

---

### Experiment B: Network Width

Question:

```text
How does the number of neurons per layer affect performance?
```

Configurations:

```text
[8, 8]
[32, 32]
[64, 64]
[128, 128]
[512, 512]
```

Main lesson:

```text
Too narrow -> underfitting
Moderate width -> good balance
Too wide -> possible overfitting or overconfidence
```

Width increases representational capacity, but large models need enough data and regularization.

---

### Experiment C: Learning Rate

Question:

```text
How does optimizer step size affect convergence?
```

Tested values:

```text
1.0
0.1
0.01
0.001
0.0001
```

Main lesson:

```text
learning rate too high -> unstable training or failure
learning rate too low  -> slow training
reasonable learning rate -> stable convergence
```

For Adam, `0.001` is often a safe default.

---

### Experiment D: Activation Function

Question:

```text
How does the activation function affect learning?
```

Tested functions:

```text
ReLU
LeakyReLU
Tanh
Sigmoid
ELU
```

Main lesson:

```text
ReLU, LeakyReLU, and ELU are usually strong choices.
Sigmoid hidden layers often train more slowly due to saturation and vanishing gradients.
```

Sigmoid is acceptable at the output layer for binary classification, but it is usually not preferred in hidden layers.

---

### Experiment E: Batch Size

Question:

```text
How does batch size affect gradient estimation and training behavior?
```

Tested values:

```text
8
32
128
512
```

Main lesson:

```text
small batch -> noisy gradients, many updates, sometimes better exploration
medium batch -> practical trade-off
large batch -> smoother gradients, fewer updates
very large batch -> may plateau or generalize worse
```

Batch size changes both training speed and optimization behavior.

---

## How to Interpret the Generated Plots

### If training loss and validation loss both decrease

The model is learning patterns that generalize.

### If training loss decreases but validation loss increases

The model is overfitting.

### If both losses remain high

The model is underfitting.

### If validation loss oscillates strongly

The learning rate may be too high, or the batch size may be too small.

### If validation accuracy is high but validation loss increases

The model may be becoming overconfident. Accuracy only checks whether predictions cross the 0.5 threshold. Binary cross-entropy also cares about confidence.

---

## Typical Observations

When running these scripts, you may observe:

```text
1. A shallow MLP can perform well on this dataset.
2. Very narrow networks tend to underfit.
3. Very wide networks may achieve high accuracy but show unstable or increasing validation loss.
4. Very high learning rates can collapse performance near the majority-class baseline.
5. Very low learning rates learn slowly.
6. Sigmoid hidden layers usually perform worse than ReLU-like activations.
7. Medium batch sizes often provide a good practical trade-off.
8. Accuracy alone is not sufficient for imbalanced classification.
```

---

## Current Limitations

This project is intentionally simple and educational. Current limitations include:

```text
no early stopping
no class weighting
no threshold tuning
no ROC-AUC or PR-AUC
no cross-validation
no dropout
no weight decay comparison
no command-line interface
no config file
no automatic output directory creation
```

These are useful directions for future improvement.

---

## Suggested Future Improvements

### 1. Add Early Stopping

Save the model with the best validation loss instead of the final epoch.

```python
best_val_loss = float("inf")

if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save(model.state_dict(), "best_wine_mlp.pth")
```

---

### 2. Use `BCEWithLogitsLoss`

A more numerically stable approach is:

```text
remove final Sigmoid from the model
use nn.BCEWithLogitsLoss()
```

This combines sigmoid and binary cross-entropy internally.

---

### 3. Handle Class Imbalance

Possible methods:

```text
class weighting
oversampling the minority class
undersampling the majority class
threshold tuning
precision-recall analysis
```

---

### 4. Add More Metrics

Useful metrics for imbalanced binary classification:

```text
balanced accuracy
macro F1-score
ROC-AUC
PR-AUC
minority-class recall
minority-class precision
```

---

### 5. Add Command-Line Arguments

Example future usage:

```bash
python training_loop_v1.py --epochs 100 --batch-size 64 --lr 0.001
```

This would make the project more flexible.

---

### 6. Organize Outputs

Recommended future output structure:

```text
outputs/figures/
outputs/models/
outputs/logs/
```

This keeps the project directory clean.

---

## Suggested Git Commands

From your deep-learning repository:

```bash
mkdir -p pytorch-mlp-training-lab
cp training_loop_v1.py experiments_v1.py pytorch-mlp-training-lab/
cp README.md pytorch-mlp-training-lab/

git add pytorch-mlp-training-lab/
git commit -m "Add PyTorch MLP training loop and hyperparameter experiment lab"
git push
```

---

## Suggested GitHub Description

Short description:

```text
A PyTorch-based educational lab for learning MLP training pipelines and systematic hyperparameter experiments on tabular data.
```

Long description:

```text
This project demonstrates a complete neural-network training workflow using PyTorch, including train/validation/test splitting, leakage-free preprocessing, MLP training, validation monitoring, learning-curve plotting, test-set evaluation, confusion-matrix analysis, model saving/loading, inference, and controlled experiments on depth, width, learning rate, activation function, and batch size.
```

---

## Final Takeaway

This project teaches a practical deep-learning lesson:

```text
A neural network should not be judged only by final accuracy.
It should be understood using learning curves, validation loss,
class-wise metrics, confusion matrices, and hyperparameter sensitivity.
```

`training_loop_v1.py` teaches the complete supervised-learning pipeline.

`experiments_v1.py` teaches how to think experimentally about neural-network design.

Together, the two scripts form a strong foundation for building larger deep-learning workflows for scientific, engineering, and tabular datasets.
