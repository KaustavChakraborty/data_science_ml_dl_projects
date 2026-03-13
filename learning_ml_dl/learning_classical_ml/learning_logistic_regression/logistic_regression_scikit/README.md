# Logistic Regression in scikit-learn — End-to-End Binary Classification Project

A fully documented, experiment-driven machine learning project that demonstrates how to build, evaluate, tune, diagnose, and extend **Logistic Regression** models using **scikit-learn**.

This project is centered on a single Python script that walks through a complete classification workflow, from safe preprocessing and baseline modeling to hyperparameter tuning, nonlinear feature engineering, learning curves, and probability diagnostics. It is designed as both a **learning resource** and a **practical reference implementation**.

---

## Overview

Logistic Regression is one of the most important baseline models in machine learning. Although conceptually simple, it remains extremely useful because it is:

- interpretable  
- probabilistic  
- fast to train  
- easy to regularize  
- surprisingly strong on many tabular datasets  

This project shows how to use Logistic Regression properly in real workflows rather than as a one-line toy model. It includes:

- safe preprocessing with `Pipeline`
- evaluation with many classification metrics
- confusion matrix visualization
- coefficient-based interpretation
- hyperparameter tuning with `GridSearchCV`
- cross-validation-based model selection
- nonlinear decision boundaries via polynomial features
- learning curves for bias-variance diagnosis
- ROC, PR, and calibration curve comparisons

The core implementation is in `logistic_regression_scikit_v1.py`.

---

## Project Goals

The main goals of this project are:

1. Demonstrate a complete, production-aware binary classification workflow with scikit-learn.
2. Explain Logistic Regression in a mathematically meaningful but practical way.
3. Show how preprocessing and regularization affect performance.
4. Compare simple linear decision boundaries with nonlinear polynomial feature expansions.
5. Teach model diagnosis using learning curves, ROC curves, PR curves, and calibration plots.
6. Provide a readable, well-commented script suitable for self-study, teaching, or portfolio use.

---

## What the Script Covers

The script is organized as a sequence of experiments.

### Experiment 1 — Basic Logistic Regression on Real Data
Uses the breast cancer dataset from scikit-learn and builds a standard Logistic Regression pipeline:

- loads the dataset
- splits into training and test sets
- scales features with `StandardScaler`
- fits Logistic Regression with L2 regularization
- evaluates train/test accuracy and multiple additional metrics
- plots a confusion matrix
- visualizes the most influential coefficients

### Experiment 2 — Hyperparameter Tuning with Grid Search
Explores how penalty choice and regularization strength affect model performance:

- tunes `penalty`, `C`, `solver`, and `l1_ratio`
- uses `GridSearchCV`
- performs stratified 5-fold cross-validation
- optimizes by **ROC-AUC**
- compares train vs validation AUC trends across values of `C`

### Experiment 3 — Nonlinear Data with Polynomial Features
Demonstrates that Logistic Regression is linear only in the feature space it sees:

- creates a synthetic two-moons dataset
- fits Logistic Regression with polynomial features of degree 1, 2, and 3
- shows how nonlinear feature engineering bends the decision boundary
- compares test accuracy across feature degrees

### Experiment 4 — Learning Curves
Shows how to diagnose model behavior as training set size increases:

- compares a strongly regularized model (`C=0.001`) with a more flexible model (`C=1.0`)
- plots training and cross-validation accuracy vs training size
- helps identify underfitting and good-fit behavior

### Experiment 5 — ROC, PR, and Calibration Curve Comparison
Compares multiple Logistic Regression variants with different regularization settings:

- strong vs moderate vs weak L2 regularization
- L1-sparse Logistic Regression
- ROC curves
- Precision-Recall curves
- calibration/reliability diagrams

---

## Mathematical Background

Logistic Regression models the probability of class 1 as:

\[
P(y=1 \mid x) = \sigma(z)
\]

where

\[
z = w^T x + b
\]

and the sigmoid function is

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

The model outputs a probability between 0 and 1. A threshold, usually 0.5, is then used to convert the probability into a class label.

### Important Interpretation Point
This project uses the scikit-learn breast cancer dataset, where:

- `0 = malignant`
- `1 = benign`

That means:

- `predict_proba(X)[:, 1]` is the predicted probability of **benign**, not malignant
- positive coefficients push predictions toward **benign**
- negative coefficients push predictions toward **malignant**

This is essential when interpreting coefficients, confusion matrices, and performance metrics.

---

## Project Structure

A minimal structure for this project is:

```text
.
├── logistic_regression_scikit_v1.py
├── README.md
├── cm_cancer_basic.png
├── coef_cancer.png
├── grid_search_cv.png
├── polynomial_boundaries.png
├── learning_curves.png
└── roc_pr_calibration.png
```

### Main Script
- `logistic_regression_scikit_v1.py` — the complete, highly documented implementation.

### Output Figures Generated by the Script
- `cm_cancer_basic.png` — confusion matrix for the baseline breast cancer model
- `coef_cancer.png` — top 15 standardized Logistic Regression coefficients
- `grid_search_cv.png` — cross-validation AUC vs regularization strength
- `polynomial_boundaries.png` — decision boundaries for polynomial Logistic Regression on the moons dataset
- `learning_curves.png` — underfit vs good-fit learning curves
- `roc_pr_calibration.png` — multi-model probability diagnostics

---

## Key Features and Design Choices

### 1. Safe preprocessing with `Pipeline`
The project uses `Pipeline` to chain preprocessing and modeling steps together.

This is important because feature scaling must be learned **only from the training data**. Fitting a scaler on the full dataset before the train/test split would cause **data leakage**.

Using a pipeline ensures the correct order:

1. fit the scaler on the training set  
2. transform the training set  
3. train the classifier  
4. transform test or validation data using the already fitted scaler  
5. generate predictions  

### 2. Standardization
`StandardScaler` is used because Logistic Regression is sensitive to feature scales, especially under regularization. Standardization makes optimization more stable and coefficients more comparable.

### 3. Regularization
The project demonstrates how regularization affects bias and variance.

- **small `C`** → strong regularization → higher bias → possible underfitting
- **large `C`** → weak regularization → lower bias but increased risk of overfitting

The script explores:

- **L2 regularization**
- **L1 regularization**
- **Elastic Net regularization**

### 4. Rich evaluation beyond accuracy
The project does not rely on a single metric. It reports:

- Accuracy
- Precision
- Recall
- F1 score
- Matthews Correlation Coefficient (MCC)
- ROC-AUC
- Average Precision / PR-AUC
- Log Loss
- Brier Score

This is important because different metrics answer different questions:

- accuracy measures overall correctness
- precision measures reliability of positive predictions
- recall measures ability to recover positives
- ROC-AUC evaluates ranking quality across thresholds
- PR-AUC is often more informative for imbalanced data
- log loss and Brier score assess probability quality

### 5. Interpretability through coefficients
Because the baseline model uses standardized inputs, the learned coefficients can be compared more meaningfully. Large absolute coefficients indicate features that strongly influence the model’s log-odds.

### 6. Diagnostic visualizations
The project includes several plots that make model behavior easier to understand visually.

---

## Installation

### Requirements
This project uses Python 3 and the following main libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

### Install with pip

```bash
pip install numpy pandas matplotlib scikit-learn
```

### Recommended environment
A clean virtual environment is recommended:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
pip install numpy pandas matplotlib scikit-learn
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install numpy pandas matplotlib scikit-learn
```

---

## How to Run

Run the script directly:

```bash
python logistic_regression_scikit_v1.py
```

The script will:

1. print experiment summaries and metric tables to the terminal
2. train several Logistic Regression variants
3. generate and save output figures in the current working directory

---

## Expected Outputs

When you run the script, you should expect:

- printed evaluation metrics for baseline and tuned models
- best hyperparameters from grid search
- reported AUC values for compared models
- several saved diagnostic plots

Because the random seed is fixed (`SEED = 42`), the synthetic-data generation and most data splitting behavior are reproducible.

---

## Detailed Walkthrough of the Experiments

### Experiment 1: Baseline Model on Breast Cancer Data

This experiment builds a standard supervised learning workflow:

- load the dataset with `load_breast_cancer()`
- split into train and test sets with stratification
- preprocess with `StandardScaler`
- fit `LogisticRegression`
- evaluate on held-out test data

#### Why this experiment matters
It establishes a strong and interpretable baseline on a real dataset using best practices.

#### Metrics reported
The helper function `full_evaluation_report()` computes:

- Train Accuracy
- Test Accuracy
- Precision
- Recall
- F1
- MCC
- AUC-ROC
- AUC-PR
- Log Loss
- Brier Score

It also prints a class-wise `classification_report`.

#### Confusion matrix
The confusion matrix plot helps you see the types of mistakes made by the classifier. Since the dataset labels are ordered as malignant then benign, the display labels are set accordingly.

#### Coefficient plot
The coefficient bar chart identifies the strongest positive and negative drivers of the benign class probability.

---

### Experiment 2: Hyperparameter Tuning

This section uses `GridSearchCV` to search over multiple regularization families and strengths.

#### Parameter search space
The script includes:

- L2 with `lbfgs`
- L1 with `liblinear`
- Elastic Net with `saga`

#### Why use ROC-AUC for tuning?
ROC-AUC is threshold-independent and measures how well the model ranks positive examples above negative ones. It is often a more robust model-selection objective than raw accuracy.

#### Cross-validation strategy
`StratifiedKFold` with 5 folds is used to preserve class balance in every split.

#### Output interpretation
The script prints:

- best hyperparameter combination
- best mean cross-validation ROC-AUC
- held-out test performance of the best model

It also plots train and validation AUC as a function of `C` for the L2 models, giving a compact bias-variance view.

---

### Experiment 3: Polynomial Logistic Regression on Nonlinear Data

The two-moons dataset is intentionally nonlinear, so standard linear Logistic Regression struggles to separate the classes with a straight boundary.

#### Feature expansion
The script uses `PolynomialFeatures` with degrees 1, 2, and 3.

For two input variables:

- degree 1 gives linear features only
- degree 2 adds quadratic terms like `x1^2`, `x2^2`, and `x1*x2`
- degree 3 adds cubic interaction terms as well

#### Why this matters
Logistic Regression is linear in the transformed feature space, not necessarily in the original input space. By changing the feature representation, the same model can learn curved decision boundaries.

#### Output interpretation
The script prints:

- the number of generated features
- test accuracy for each polynomial degree

It also saves a figure showing:

- predicted probability surfaces
- the 0.5 decision boundary
- overlaid test samples

This is especially useful for understanding underfitting vs increased representation power.

---

### Experiment 4: Learning Curves

Learning curves show how performance changes as the amount of training data increases.

#### What is plotted
For each training size, the script computes:

- training accuracy
- cross-validation accuracy

with mean and standard deviation across folds.

#### Models compared
- `C=0.001` — strongly regularized, likely to underfit
- `C=1.0` — more flexible, generally a better fit

#### What to look for
- **both curves low and close** → underfitting / high bias
- **large train-validation gap** → overfitting / high variance
- **validation curve still rising** → more data may still help

This experiment is very useful for diagnosing whether the issue is model capacity, regularization, or data quantity.

---

### Experiment 5: ROC, PR, and Calibration Curves

This experiment compares multiple fitted Logistic Regression variants on the same scaled breast cancer data.

#### Compared models
- strong L2 regularization
- default/moderate L2 regularization
- weak L2 regularization
- L1 sparse model

#### Diagnostics included

**ROC Curves**  
Shows the tradeoff between true positive rate and false positive rate over thresholds.

**Precision-Recall Curves**  
More informative when the positive class is rare or when precision matters.

**Calibration Curves**  
Checks whether predicted probabilities correspond to actual observed frequencies.

For example, if the model predicts 0.8 for many samples, then roughly 80% of those samples should truly belong to class 1 if the model is well calibrated.

---

## Main Helper Functions

### `full_evaluation_report(model, X_train, X_test, y_train, y_test, name="Model")`
A reusable helper for computing a broad set of metrics and printing a classification report.

### `plot_model_diagnostics(models_dict, X_test, y_test, title="Model Diagnostics")`
Generates a 3-panel diagnostic figure for:

- ROC curves
- Precision-Recall curves
- Calibration curves

These helpers make the script cleaner and easier to extend.

---

## Reproducibility

The project defines:

```python
SEED = 42
```

This seed is used in:

- train/test splitting
- synthetic data generation
- fold shuffling where applicable
- some model components

This improves reproducibility across runs.

---

## Educational Value of the Project

This project is especially useful for:

- students learning classification with scikit-learn
- researchers building a first strong baseline model
- instructors teaching regularization and evaluation
- practitioners who want a clean Logistic Regression reference pipeline
- anyone wanting a portfolio project with both code and explanatory structure

It is more than a minimal example because it teaches not only how to train a model, but also how to think about:

- leakage prevention
- metric choice
- bias-variance tradeoff
- feature engineering
- calibration
- interpretability

---

## Limitations

This project is intentionally focused on Logistic Regression and binary classification. It does not include:

- multiclass classification workflows
- missing-value imputation pipelines
- categorical feature encoding
- threshold optimization for precision/recall tradeoffs
- model persistence with `joblib`
- deployment or API serving

These can be added as future extensions.

---

## Possible Future Improvements

Some natural next steps for extending the project are:

1. Add threshold tuning based on precision-recall tradeoffs.  
2. Save and reload trained models with `joblib`.  
3. Add feature selection or recursive elimination.  
4. Include class imbalance strategies such as `class_weight='balanced'`.  
5. Add calibrated probabilities using `CalibratedClassifierCV` explicitly in a comparison section.  
6. Turn the script into a notebook with section-by-section explanations.  
7. Export metrics and grid-search results to CSV.  
8. Add support for user-provided tabular datasets.  
9. Extend to multiclass classification.  
10. Add unit tests for helper functions.  

---

## Example Use Cases

This project can serve as a template for:

- medical diagnosis baselines
- credit risk prediction
- churn classification
- spam filtering
- any binary tabular classification problem where interpretability and probability estimates matter

---

## Notes on the Breast Cancer Dataset

The breast cancer dataset is a common benchmark included in scikit-learn. It contains real-valued features computed from digitized images of breast mass cell nuclei.

In this project, it is used to demonstrate:

- train/test splitting with stratification
- baseline medical classification modeling
- coefficient interpretation on scaled tabular features
- threshold-independent diagnostics like ROC-AUC

---

## Why This Project Stands Out

Many Logistic Regression examples stop after fitting a model and printing accuracy. This project goes much further by showing:

- how to build the model safely
- how to compare regularization choices
- how to tune hyperparameters correctly with CV
- how to diagnose generalization with learning curves
- how to extend linear models using feature engineering
- how to evaluate probability quality, not just class predictions

That makes it a strong educational and portfolio-grade project.

---

## License

Add a license of your choice if you intend to publish or distribute this project. A common option is the MIT License.

Example:

```text
MIT License
```

---

## Acknowledgment

Built with:

- Python
- NumPy
- pandas
- Matplotlib
- scikit-learn

---

## Quick Summary

This is a complete scikit-learn Logistic Regression project that teaches:

- correct preprocessing
- baseline modeling
- model evaluation
- interpretability
- hyperparameter tuning
- nonlinear feature engineering
- bias-variance diagnosis
- probability diagnostics

It is suitable for study, teaching, research baselines, and portfolio presentation.
