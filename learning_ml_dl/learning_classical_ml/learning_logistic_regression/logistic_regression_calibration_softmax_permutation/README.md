# Logistic Regression Advanced Topics

A detailed educational project exploring **advanced concepts in logistic regression** using scikit-learn, standard benchmark datasets, and visualization-heavy analysis. This project is designed not only to run models, but to explain **how logistic regression behaves in practice** when probability quality, multiclass structure, regularization, and model interpretability are studied carefully.

This repository is especially useful for:

- students learning statistical machine learning,
- candidates preparing for machine learning interviews,
- practitioners who want intuition beyond basic binary classification,
- anyone who wants to understand how logistic regression behaves under different modeling choices.

The project goes beyond the usual “fit model and report accuracy” workflow. It studies four important themes:

1. **Probability calibration** — whether predicted probabilities can be trusted.
2. **Multinomial logistic regression** — how softmax-based multiclass learning differs from one-vs-rest.
3. **Regularization paths** — how coefficients evolve as regularization strength changes.
4. **Permutation feature importance** — which features actually matter for predictive performance on held-out data.

---

## 1. Project overview

The code demonstrates that logistic regression is much richer than a simple linear classifier. In many beginner workflows, logistic regression is introduced as a model that produces class labels and a decision boundary. In practice, however, it is also a:

- **probabilistic model** that outputs class probabilities,
- **multiclass model** through the softmax formulation,
- **regularized model** whose coefficients depend strongly on penalty choice,
- **interpretable model** that allows both coefficient-based and performance-based feature analysis.

The central idea of this project is that different evaluation questions require different tools:

- If you care about **ranking**, metrics like AUC matter.
- If you care about **trustworthy probabilities**, calibration metrics matter.
- If you care about **coefficient shrinkage and sparsity**, regularization paths matter.
- If you care about **feature importance in terms of predictive performance**, permutation importance matters.

This project is therefore a compact but conceptually deep study of logistic regression from several angles.

---

## 2. What the script covers

The main script explores the following topics.

### Topic 1: Probability calibration
The script compares four classifiers on the breast cancer dataset:

- Logistic Regression with `C=1.0`
- Logistic Regression with very strong regularization (`C=0.001`)
- The same strongly regularized model calibrated with **Platt scaling**
- The same strongly regularized model calibrated with **isotonic regression**

The goal is to answer the question:

> If the model predicts 0.70, does that really mean the event happens about 70% of the time?

This section evaluates models with:

- **Brier score**
- **log-loss**
- **ROC-AUC**
- calibration curves (reliability diagrams)
- histograms of predicted probabilities

This topic is important because a model can classify well but still produce poor probabilities.

---

### Topic 2: Multinomial logistic regression (softmax)
Using the Iris dataset, the project compares:

- **multinomial logistic regression** (`multi_class='multinomial'`)
- **one-vs-rest logistic regression** (`multi_class='ovr'`)

It then:

- reports test accuracy,
- prints the softmax coefficient matrix,
- visualizes class probability surfaces for each class.

This section is important because many first-time learners assume multiclass logistic regression is always just a stack of binary classifiers. The project shows that the true multinomial formulation can behave very differently and often better.

---

### Topic 3: Regularization path
Using the first eight features of the breast cancer dataset, the script studies how coefficients change across a wide range of `C` values for:

- **L2 regularization**
- **L1 regularization**

This section demonstrates:

- smooth shrinkage under L2,
- exact zeros under L1,
- variable entry/exit patterns,
- effects of collinearity among features.

It is one of the best visual ways to understand why L1 is associated with feature selection while L2 is associated with coefficient stabilization.

---

### Topic 4: Permutation feature importance
The script fits a logistic regression model on the full breast cancer dataset and estimates **permutation importance** using AUC as the scoring metric.

This section answers a different question from coefficients:

> If I break one feature by shuffling it, how much does held-out model performance deteriorate?

That makes it a more direct, performance-based importance analysis.

---

## 3. Datasets used

The script relies entirely on standard datasets from scikit-learn.

### 3.1 Breast Cancer Wisconsin dataset
Used for:

- probability calibration,
- regularization path,
- permutation feature importance.

This is a binary classification dataset where the model predicts whether a tumor is malignant or benign. The dataset contains multiple geometric and texture-related measurements of cell nuclei.

Important note for interpretation:

- In the scikit-learn target encoding, **0 = malignant** and **1 = benign**.

This matters when interpreting the sign of coefficients.

### 3.2 Iris dataset
Used for:

- multinomial logistic regression,
- coefficient interpretation in multiclass setting,
- softmax probability surface visualization.

It contains three flower classes:

- setosa,
- versicolor,
- virginica.

The features are:

- sepal length,
- sepal width,
- petal length,
- petal width.

The Iris dataset is especially suitable for teaching multiclass logistic regression because the classes are mutually exclusive and partially separable with meaningful geometry.

---

## 4. Core concepts behind the project

### 4.1 Logistic regression as a probabilistic classifier
In binary classification, logistic regression models the probability of the positive class through the logistic sigmoid:

\[
P(y=1 \mid x) = \frac{1}{1 + e^{-(\beta^T x + b)}}
\]

The output is a probability, not just a class score. That means the quality of the probability itself matters.

---

### 4.2 Calibration
A calibrated model is one for which predicted probabilities match empirical frequencies.

For example, among all samples assigned probability 0.8, about 80% should actually belong to the positive class.

A model may have:

- high accuracy,
- high AUC,
- strong discrimination,

and still be poorly calibrated.

That is why this project separately studies calibration.

---

### 4.3 Multinomial logistic regression
For multiclass problems with classes \(k = 1, \dots, K\), multinomial logistic regression uses the softmax function:

\[
P(y=k \mid x) = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}
\]

where:

\[
z_k = \beta_k^T x + b_k
\]

Each class has its own linear score, and the scores compete through the softmax normalization.

This is different from one-vs-rest, where separate binary classifiers are trained independently.

---

### 4.4 Regularization
Regularization penalizes large coefficients to reduce overfitting.

In scikit-learn logistic regression, `C` is the inverse of regularization strength:

\[
C = \frac{1}{\lambda}
\]

So:

- small `C` means **strong regularization**,
- large `C` means **weak regularization**.

Two common penalties are used here:

#### L2 penalty
- shrinks coefficients smoothly,
- rarely sets coefficients exactly to zero,
- often works well when many features contribute small or moderate signal.

#### L1 penalty
- encourages sparsity,
- can set coefficients exactly to zero,
- can act like embedded feature selection.

---

### 4.5 Permutation importance
Permutation importance measures the drop in predictive performance when a feature is randomly shuffled.

If shuffling a feature causes a strong performance drop, the model depended heavily on that feature.

This is often more informative than raw coefficient magnitude, especially when features are highly correlated.

---

## 5. Why feature scaling is used throughout

The project applies `StandardScaler` before fitting logistic regression models.

This is important for several reasons:

1. Logistic regression with regularization is scale-sensitive.
2. Coefficient magnitudes become more comparable after scaling.
3. Optimization becomes more stable and efficient.
4. Regularization paths are much easier to interpret when predictors are standardized.

Without scaling, a feature measured on a large numerical scale could dominate coefficient size for purely numerical reasons.

---

## 6. Files produced by the script

Running the script generates four image files.

### `calibration.png`
Contains:

- reliability diagrams (calibration curves),
- histogram of predicted probabilities.

Use this figure to study:

- probability quality,
- underconfidence vs overconfidence,
- confidence sharpness,
- effects of post-hoc calibration.

---

### `multinomial_softmax.png`
Contains class-wise softmax probability surfaces for the Iris dataset.

Use this figure to study:

- how multinomial logistic regression distributes probability across classes,
- how one class competes with the others,
- how class geometry appears in a 2D slice of a 4D feature space.

---

### `regularization_path.png`
Contains side-by-side regularization paths for:

- L2 logistic regression,
- L1 logistic regression.

Use this figure to study:

- coefficient shrinkage,
- sparsity,
- collinearity effects,
- how features enter the model under L1.

---

### `permutation_importance.png`
Contains boxplots of permutation importance values for the top 15 features.

Use this figure to study:

- held-out predictive importance,
- variability across repeated shuffles,
- redundancy among features,
- why coefficient size is not the full importance story.

---

## 7. Expected high-level observations

A first-time viewer of the project should expect the following broad conclusions.

### 7.1 Calibration results
The baseline logistic regression with `C=1.0` tends to have the best overall probability quality in this project.

The strongly regularized logistic regression with `C=0.001` still separates classes fairly well but produces much poorer probabilities.

This creates one of the most important lessons in the project:

> A model can have high AUC and still have poor calibration.

Platt scaling and isotonic regression improve the strongly regularized model, but they do not necessarily outperform the already well-tuned baseline model.

---

### 7.2 Multinomial results
The multinomial softmax model performs much better than one-vs-rest on the given split in this project.

The coefficient matrix usually makes intuitive sense:

- **setosa** is associated with smaller petals,
- **virginica** is associated with larger petal features,
- **versicolor** acts as an intermediate class.

The contour plots also show how some classes dominate in certain regions of the chosen feature slice.

---

### 7.3 Regularization results
The regularization path reveals that:

- L2 spreads weight smoothly across correlated variables,
- L1 creates sparse solutions and exact zeros,
- highly correlated size-related features compete with one another,
- coefficient signs can become counterintuitive when interpreted conditionally.

This is a good reminder that coefficients in multivariate logistic regression are not the same as simple one-variable trends.

---

### 7.4 Permutation importance results
The permutation importance plot usually shows that no single feature completely dominates performance. Instead, predictive power is shared across groups of correlated variables.

This suggests redundancy in the dataset, which is exactly what one would expect in a medical measurement dataset with multiple related geometric descriptors.

---

## 8. How to run the project

### 8.1 Basic execution
Run the script with Python:

```bash
python logistic_regression_advanced_topics_v1.py
```

If your environment uses a specific interpreter version, for example:

```bash
python3.8 logistic_regression_advanced_topics_v1.py
```

---

### 8.2 What will appear in the terminal
The script prints section headers and summary metrics for:

- Topic 1: probability calibration,
- Topic 2: multinomial logistic regression,
- Topic 3: regularization path,
- Topic 4: permutation feature importance.

The calibration section prints Brier score, log-loss, and AUC. The multinomial section prints test accuracy and the coefficient matrix.

---

### 8.3 What will appear graphically
The script displays figures using `matplotlib` and also saves them to disk. Depending on your environment:

- figures may pop up interactively,
- or they may simply be saved as PNG files.

If you are running on a remote machine without GUI forwarding, the saved files are the main outputs to inspect.

---

## 9. Suggested project structure

A clean repository structure could look like this:

```text
project/
│
├── logistic_regression_advanced_topics_v1.py
├── README.md
├── observations.md
├── requirements.txt
├── calibration.png
├── multinomial_softmax.png
├── regularization_path.png
└── permutation_importance.png
```

This keeps code, documentation, dependencies, and results together in an easy-to-review format.

---

## 10. Detailed interpretation guide for first-time readers

This section is written for someone opening the project for the first time and wanting to understand why each part matters.

### 10.1 Why start with calibration?
Many projects report only accuracy. That is often insufficient.

Suppose two models both classify correctly most of the time:

- Model A predicts probabilities like 0.51 and 0.55.
- Model B predicts probabilities like 0.95 and 0.02.

These models may have similar classification accuracy but very different confidence behavior.

In applications such as medicine, risk modeling, and decision support, the **quality of the probability** matters just as much as the final label.

That is why the project begins with calibration rather than only classification accuracy.

---

### 10.2 Why compare multinomial softmax with one-vs-rest?
A first-time learner may assume all multiclass logistic regression is the same.

It is not.

In a multinomial softmax model, all classes are learned together and their probabilities sum to one through a coupled normalization. In one-vs-rest, each class is modeled independently against all others.

The difference becomes important when classes overlap or compete asymmetrically. The Iris example provides a concrete demonstration of this.

---

### 10.3 Why study regularization paths instead of just one fitted model?
When you fit a single logistic regression model, you only see one final coefficient vector. That hides a lot of the story.

A regularization path shows:

- which features become active early,
- which features stay suppressed,
- whether coefficients are stable,
- whether correlated predictors compete for influence,
- how sensitive the model is to penalty strength.

This makes regularization much more intuitive than reading about it abstractly.

---

### 10.4 Why use permutation importance in addition to coefficients?
Coefficient size is not the same as feature importance.

A large coefficient may arise because:

- the feature is genuinely important,
- the feature is compensating for correlated neighbors,
- the regularization scheme favors it,
- the model is encoding a conditional effect rather than a marginal trend.

Permutation importance instead asks:

> If I destroy this feature’s information, does the model’s actual predictive performance suffer?

That is often a much more practical definition of importance.

---

## 11. Common beginner misunderstandings this project helps correct

### Misunderstanding 1: High AUC means good probabilities
Not necessarily.

AUC measures ranking quality, not calibration. A model can rank examples very well and still assign poor probability values.

---

### Misunderstanding 2: Logistic regression coefficients are always easy to interpret
Only partly true.

They are interpretable, but in multivariate settings with correlated predictors, coefficients are **conditional effects**, not isolated marginal truths.

---

### Misunderstanding 3: L1 and L2 just “shrink coefficients a bit differently”
They differ much more meaningfully than that.

- L2 keeps all variables in play and spreads weights.
- L1 can set variables exactly to zero and build sparse models.

That difference has major implications for feature selection and model stability.

---

### Misunderstanding 4: The biggest coefficient is the most important feature
Not necessarily.

Large coefficients can occur due to scale, correlation, or regularization effects. Performance-based importance may tell a different story.

---

### Misunderstanding 5: A 2D plot of a classifier always shows the full decision structure
No.

The multinomial contour plots are 2D slices through a 4D standardized feature space. Features not shown are fixed at zero in standardized units. This is a useful visualization, but not the full geometry of the model.

---

## 12. Strengths of the project

This project has several strengths as a learning and portfolio piece.

### Conceptual depth
It covers multiple advanced but fundamental topics within one classical model family.

### Strong visualization
Each section includes a plot that connects mathematics to model behavior.

### Educational value
The project is suitable for teaching, interview preparation, and self-study.

### Reproducibility
It uses benchmark datasets and a fixed random seed.

### Interpretability focus
The work emphasizes understanding model behavior, not just obtaining a score.

---

## 13. Limitations and caveats

A good README should also make the project’s limits clear.

### 13.1 Results depend on train/test split
The exact numerical results can vary with the split, even though the random seed fixes one reproducible run.

### 13.2 Reliability diagrams can look noisy on modest sample sizes
Calibration curves use binning and can be visually jagged even when the underlying model is reasonable.

### 13.3 Coefficients are sensitive to collinearity
Several breast-cancer features are strongly correlated, so coefficient signs and magnitudes should not be overinterpreted as simple domain truths.

### 13.4 Permutation importance is conditional on the fitted model
A feature may appear less important not because it is intrinsically weak, but because related features already capture much of the same information.

### 13.5 The multinomial probability surfaces are slices, not the full 4D decision manifold
They are pedagogically helpful, but they do not fully represent the whole feature space.

---

## 14. Who this project is for

This project is especially suitable for:

- machine learning beginners moving beyond introductory classification,
- students wanting deeper understanding of logistic regression,
- candidates preparing for data science or ML interviews,
- researchers who want a compact demonstration of model interpretation tools,
- anyone building intuition about the relationship between coefficients, probabilities, and performance.

---

## 15. Suggested extensions

If you want to expand the project further, some natural next steps are:

### Modeling extensions
- add Elastic Net regularization,
- compare logistic regression with SVM, random forest, and gradient boosting,
- add class imbalance experiments,
- study threshold tuning and precision-recall tradeoffs.

### Probability-focused extensions
- add calibration error metrics such as ECE,
- compare calibration behavior across model families,
- study calibration under class imbalance.

### Interpretation extensions
- compare permutation importance with SHAP values,
- add partial dependence or ICE plots,
- compare coefficient-based and model-agnostic explanations.

### Statistical extensions
- compute confidence intervals through bootstrap,
- perform repeated train/test splits or cross-validation,
- study coefficient stability under resampling.

---

## 16. How to present this project in an interview

A strong concise summary would be:

> This project studies advanced logistic regression behavior from four angles: probability calibration, multiclass softmax modeling, regularization paths, and permutation-based feature importance. The key lesson is that good classification performance is not the same as good probability estimation, and coefficient magnitude is not the same as practical feature importance. The project also shows how L1 and L2 regularization behave differently in the presence of correlated predictors and why multinomial softmax can outperform one-vs-rest in true multiclass settings.

That summary immediately signals that you understand not only how to run logistic regression, but how to analyze it critically.

---

## 17. Related documentation in this project

This README explains the project structure, goals, theory, and usage.

For result-by-result interpretation, see:

- `observations.md`

That file discusses the outputs in greater detail and helps connect the figures and printed metrics to statistical meaning.

---

## 18. Final takeaway

This project shows that logistic regression remains one of the best models for learning core machine learning ideas because it is simple enough to interpret and rich enough to expose important concepts:

- probabilities versus decisions,
- discrimination versus calibration,
- binary versus multiclass modeling,
- shrinkage versus sparsity,
- coefficients versus predictive importance.

For a first-time reader, the project is a guided tour through these ideas. For a more advanced reader, it is a compact and well-structured demonstration that model evaluation should always depend on the question being asked.
