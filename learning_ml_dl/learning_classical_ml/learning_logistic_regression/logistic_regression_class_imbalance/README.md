# Logistic Regression for Imbalanced Classification

This project demonstrates how to handle **class imbalance** in binary classification using **logistic regression**. It compares several standard strategies and shows why metrics such as **F1 score**, **Matthews Correlation Coefficient (MCC)**, and **AUC-PR** are much more informative than plain accuracy in rare-event problems.

The code is designed as an educational and practical example for problems such as:

- rare phase detection in soft matter
- anomaly detection
- rare material failure prediction
- uncommon crystallographic phase classification
- any binary classification task where one class is much less frequent than the other

---

## Overview

In imbalanced classification, one class dominates the dataset. A model can therefore achieve apparently high accuracy simply by predicting the majority class most of the time, while completely failing to detect the minority class.

This script explores that issue using a synthetic dataset with approximately:

- **90% majority class (class 0)**
- **10% minority class (class 1)**

It then compares the following strategies:

1. **Baseline logistic regression**
   - no imbalance handling
2. **Class-weighted logistic regression**
   - uses `class_weight='balanced'`
3. **SMOTE oversampling**
   - balances the training set using synthetic minority examples
4. **Threshold optimization**
   - tunes the probability threshold to maximize F1 score

The script also visualizes:

- comparison of methods using F1, MCC, and AUC-PR
- precision, recall, and F1 as functions of classification threshold
- the precision-recall curve

---

## Learning Goals

This project helps illustrate the following key ideas:

- why **accuracy can be misleading** for imbalanced datasets
- how **class weighting** changes the optimization objective
- how **SMOTE** balances the training data
- why **precision**, **recall**, **F1**, **MCC**, and **AUC-PR** are better metrics for rare-event detection
- why **threshold tuning** can improve final classification performance without changing the model itself
- how ranking quality and hard classification performance are related but not identical

---

## Project Structure

A typical project layout may look like this:

```text
.
├── logistic_regression_class_imbalance.py
├── imbalance_comparison.png
├── threshold_optimization.png
└── README.md
