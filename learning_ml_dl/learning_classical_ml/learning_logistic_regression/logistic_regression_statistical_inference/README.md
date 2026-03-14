# Logistic Regression for Statistical Inference with `statsmodels`

This project demonstrates how to use **logistic regression for statistical inference** rather than only prediction. While machine learning libraries such as `scikit-learn` are excellent for building predictive models, scientific research often requires much more than classification accuracy. In many research settings, the key questions are:

- Which predictors are statistically significant?
- How large is the effect of each predictor?
- What is the uncertainty in those effects?
- How do the predictors change the odds of the outcome?
- Does a larger model significantly improve fit over a simpler one?
- Are coefficient estimates reliable, or are they distorted by multicollinearity?

This script answers those questions using **`statsmodels`** on a subset of the Breast Cancer Wisconsin dataset.

---

## Project Goals

The aim of this project is to provide a compact but scientifically meaningful example of logistic regression inference, including:

- fitting a binary logistic regression model with **maximum likelihood estimation**
- obtaining a full coefficient table with:
  - coefficient estimates
  - standard errors
  - Wald z-statistics
  - p-values
  - confidence intervals
- converting coefficients into **odds ratios**
- visualizing odds ratios using a **forest plot**
- comparing nested models using a **Likelihood Ratio Test (LRT)**
- detecting multicollinearity using **Variance Inflation Factor (VIF)**

This makes the script especially useful for:

- scientific data analysis
- thesis work
- publication-style reporting
- biomedical statistics exercises
- learning how inference differs from prediction

---

## Why `statsmodels` Instead of Only `scikit-learn`?

`scikit-learn` is designed mainly for prediction and machine learning workflows. It is excellent for:

- train/test evaluation
- cross-validation
- pipelines
- preprocessing
- model selection for predictive performance

However, `scikit-learn` does **not** natively focus on formal inference output such as:

- p-values
- coefficient confidence intervals
- likelihood-based statistical tests
- publication-style regression tables

For these tasks, `statsmodels` is more appropriate.

In this project:

- `scikit-learn` is used only for:
  - loading the dataset
  - scaling the features
- `statsmodels` is used for:
  - fitting the logistic regression model
  - extracting inferential statistics
  - performing likelihood-based model comparison

---

## Dataset

The script uses the **Breast Cancer Wisconsin** dataset from `scikit-learn`.

### Characteristics

- binary classification dataset
- 569 observations
- real-valued numeric predictors
- medically relevant structure
- widely used for statistical and machine learning demonstrations

For clarity, the script uses only the **first 8 features** rather than the full feature set. This makes the regression table and plots easier to interpret while still preserving meaningful structure.

The selected predictors are:

1. mean radius
2. mean texture
3. mean perimeter
4. mean area
5. mean smoothness
6. mean compactness
7. mean concavity
8. mean concave points

---

## What the Script Does

The script proceeds in several stages.

### 1. Load the data
The breast cancer dataset is loaded from `sklearn.datasets`.

### 2. Select the first 8 predictors
Only the first 8 features are used to keep the analysis concise and interpretable.

### 3. Standardize the predictors
The predictors are transformed using `StandardScaler`.

Why standardization is useful here:

- it improves numerical stability of optimization
- it places all predictors on comparable scales
- coefficient magnitudes become easier to compare
- odds ratios correspond to a **1 standard deviation change** in each feature

### 4. Add an intercept term
Unlike `scikit-learn`, `statsmodels` does not automatically include an intercept in the design matrix. Therefore, a constant column is explicitly added using `sm.add_constant()`.

### 5. Fit logistic regression using `statsmodels.Logit`
The model is estimated by **maximum likelihood**, using the Newton-Raphson algorithm.

### 6. Print the full statistical summary
The script prints a regression summary including:

- log-likelihood
- pseudo-\(R^2\)
- AIC and BIC
- coefficient estimates
- standard errors
- z-statistics
- p-values
- confidence intervals

### 7. Build a coefficient summary table
A custom `pandas` DataFrame is created containing:

- feature names
- coefficients
- standard errors
- z-statistics
- p-values
- coefficient confidence intervals
- odds ratios
- odds-ratio confidence intervals

### 8. Generate a forest plot
A publication-style **forest plot** is generated for odds ratios and their 95% confidence intervals.

Features are color-coded by significance:

- **red** → significant at \(p < 0.05\)
- **gray** → not significant

### 9. Perform a likelihood ratio test
The script compares:

- a **full model** with all 8 predictors
- a **reduced model** with only the first 4 predictors

This determines whether the added predictors significantly improve model fit.

### 10. Compute variance inflation factors
The script calculates **VIF** values for each predictor to diagnose multicollinearity.

---

## Statistical Concepts Covered

This project is useful because it demonstrates several important statistical ideas in one place.

### Logistic Regression

Binary logistic regression models the probability of class membership through the logit transform:

\[
\log \left( \frac{p}{1-p} \right)
=
\beta_0 + \beta_1 x_1 + \cdots + \beta_k x_k
\]

where:

- \(p\) is the probability of the positive class
- \(\beta_0\) is the intercept
- \(\beta_j\) are regression coefficients

### Coefficients

Each coefficient represents the change in **log-odds** associated with a one-unit increase in the predictor, holding other predictors fixed.

Because the features are standardized here, each coefficient corresponds to a **1 standard deviation increase** in the feature.

### Wald z-statistics and p-values

For each coefficient, the script reports a Wald test:

- null hypothesis: \(\beta_j = 0\)
- test statistic: coefficient / standard error

This provides a p-value for whether a predictor appears statistically significant in the multivariable model.

### Confidence Intervals

Confidence intervals quantify uncertainty in the coefficient estimates.

- if the coefficient CI includes 0, the effect may not be statistically significant
- if the odds-ratio CI includes 1, the odds effect may not be statistically significant

### Odds Ratios

The odds ratio is:

\[
\text{OR} = e^{\beta}
\]

Interpretation:

- **OR > 1** → increasing the predictor increases the odds of the positive class
- **OR < 1** → increasing the predictor decreases the odds of the positive class
- **OR = 1** → no effect

### Likelihood Ratio Test (LRT)

The LRT compares nested models:

\[
2\left[\ell(\text{full}) - \ell(\text{reduced})\right]
\sim \chi^2
\]

It asks whether the additional predictors in the larger model improve fit significantly.

### Variance Inflation Factor (VIF)

VIF quantifies multicollinearity.

Typical interpretation:

- **VIF < 5** → generally acceptable
- **VIF > 10** → severe multicollinearity

Large VIF values indicate that a predictor is highly linearly explained by the others, which can destabilize coefficients.

---

## Outputs

The script produces three main kinds of output.

### 1. Full statistical regression summary
Printed in the terminal via `result.summary2()`.

This includes:

- model fit measures
- regression table
- inferential statistics for each coefficient

### 2. Custom coefficient summary table
Printed in a clean tabular format with:

- coefficient estimates
- standard errors
- z-statistics
- p-values
- confidence intervals
- odds ratios

### 3. Forest plot
Saved as:

```text
forest_plot_or.png
