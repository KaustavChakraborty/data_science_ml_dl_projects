# Logistic Regression from Scratch in Python

A fully manual implementation of **binary logistic regression** using only **NumPy** for the core learning algorithm, along with **Matplotlib** for visualization and a few utilities from **scikit-learn** for dataset generation, preprocessing, train-test splitting, and evaluation.

This project is designed as a **learning-focused, transparent implementation** of logistic regression. Instead of using `sklearn.linear_model.LogisticRegression`, the code explicitly builds the model from first principles:

- sigmoid function
- linear score computation
- binary cross-entropy loss
- gradient computation
- gradient descent optimization
- regularization (L1, L2, ElasticNet)
- mini-batch / full-batch / SGD training
- probability prediction
- evaluation and visualization

---

## Overview

For a binary classification problem, logistic regression models the probability of class 1 as

\[
P(y=1 \mid x) = \sigma(z)
\]

where

\[
z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p
\]

and the sigmoid function is

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

The model is trained by minimizing **binary cross-entropy loss**, optionally with regularization:

- **None**: no regularization
- **L2**: shrinks coefficients smoothly
- **L1**: encourages sparsity
- **ElasticNet**: combines L1 and L2

---

## Main Features

- Binary logistic regression implemented from scratch
- Supports:
  - full-batch gradient descent
  - mini-batch gradient descent
  - stochastic gradient descent (SGD)
- Regularization options:
  - no regularization
  - L1
  - L2
  - ElasticNet
- Tracks loss during training
- Predicts both:
  - class probabilities
  - class labels
- Includes three experiments:
  1. Basic training and decision boundary visualization
  2. Regularization sweep over multiple `C` and learning-rate values
  3. Batch-size comparison for convergence behavior
- Automatically skips 2D decision-boundary plotting when the dataset has more than 2 features

---

## File

- `logistic_regression_from_scratch_v3.py`

---

## Dependencies

Install the required Python packages:

```bash
pip install numpy matplotlib scikit-learn


## Project Structure

```text
logistic_regression_from_scratch/
├── README.md
├── requirements.txt
├── src/
│   └── logistic_regression_from_scratch.py
├── figures/
└── notes/
    └── observations.md
