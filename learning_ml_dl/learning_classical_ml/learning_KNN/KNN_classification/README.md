# KNN Classification Project

## 1. Project overview

This project is a **comprehensive, concept-driven implementation of K-Nearest Neighbors (KNN) classification using scikit-learn**, designed not merely to train a classifier, but to **teach, diagnose, visualize, and evaluate** how KNN behaves across multiple datasets and settings.

Unlike a minimal machine-learning example that fits one model and prints one accuracy value, this project studies KNN from several angles:

- how the choice of **number of neighbors** changes the decision boundary
- why **feature scaling** is essential for distance-based learning
- how to use **cross-validation** to choose a good value of \(K\)
- how to build a **safe preprocessing + model pipeline**
- how to perform **hyperparameter optimization** using `GridSearchCV`
- how to interpret model performance using:
  - accuracy
  - confusion matrix
  - classification report
  - ROC curve
  - ROC-AUC
  - learning curves
- how different **distance metrics** affect KNN performance

The project is therefore both:

1. a **working KNN classification pipeline**, and  
2. a **teaching / experimentation framework** for understanding KNN deeply.

---

## 2. Core motivation

KNN is one of the simplest and most intuitive supervised learning algorithms, but it is also one of the easiest to misunderstand.

At first glance, KNN appears simple:

> To classify a new point, find the nearest training points and let them vote.

However, in practice, KNN depends strongly on:

- the choice of \(K\)
- the scaling of features
- the geometry of the data
- the distance metric
- the amount of training data
- class overlap and noise

This project exists to make those effects visible and measurable.

The guiding idea of the project is:

> **Do not treat KNN as a black box. Visualize it, tune it, diagnose it, and interpret it.**

---

## 3. What this project demonstrates

This project includes several complementary components:

### 3.1 Decision-boundary visualization
A nonlinear 2D dataset is used to show how the classifier’s decision regions change when \(K\) changes.

This reveals the **bias-variance tradeoff** directly in feature space.

### 3.2 Validation curve for \(K\)
Training accuracy and cross-validation accuracy are plotted as functions of \(K\).

This provides a systematic way to identify the range of \(K\) values that generalize best.

### 3.3 Full breast-cancer classification pipeline
A real tabular dataset is used to demonstrate a realistic end-to-end classification workflow:

- train/test split
- scaling
- pipeline construction
- grid search
- ROC-AUC-based model selection
- test evaluation

### 3.4 Learning curve
The effect of training set size on performance is analyzed.

This helps answer questions such as:

- Is the model overfitting?
- Is the model underfitting?
- Would more data likely help?

### 3.5 Distance-metric comparison
The project compares Euclidean, Manhattan, Minkowski, Chebyshev, and Cosine distance metrics on a multiclass dataset.

This demonstrates that **KNN performance depends strongly on the geometry induced by the distance metric**.

---

## 4. Project goals

The project is built with the following goals in mind:

1. **Teach KNN from first principles through experiments**
2. **Demonstrate best practices** for distance-based classification
3. **Avoid data leakage** through proper use of pipelines
4. **Use visualization to connect geometry and performance**
5. **Show that model selection must be validation-driven**
6. **Provide a reusable template** for future KNN experiments

---

## 5. Main script functionality

The main script performs the following high-level tasks:

1. Generate and visualize the **effect of \(K\)** on decision boundaries using the moons dataset
2. Compute a **validation curve** showing train and CV accuracy as \(K\) varies
3. Run a **full KNN model-selection pipeline** on the breast-cancer dataset
4. Plot a **learning curve** to inspect data-efficiency and bias-variance behavior
5. Compare several **distance metrics** on the wine dataset

---

## 6. Why KNN is a good model to study this way

KNN is especially suitable for this kind of project because it is a **local, instance-based, non-parametric** algorithm.

That means:

- it stores the training data rather than learning an explicit formula
- prediction depends directly on **neighborhood structure**
- the effect of model complexity is highly intuitive
- its success or failure is tightly linked to:
  - scaling
  - distance definition
  - local class density
  - noise and outliers

So KNN is ideal for a project where the goal is not only prediction, but also **understanding geometric learning behavior**.

---

## 7. Theoretical background

## 7.1 What KNN does

For a new query point \(x^\*\), KNN:

1. computes the distance from \(x^\*\) to every training point
2. selects the \(K\) nearest neighbors
3. predicts the class using either:
   - **uniform voting**, or
   - **distance-weighted voting**

For classification, the prediction rule is:

\[
\hat{y}(x^\*) = \arg\max_c \sum_{i \in N_K(x^\*)} \mathbf{1}(y_i = c)
\]

For distance-weighted KNN, closer neighbors count more heavily.

---

## 7.2 Why scaling is mandatory

KNN relies entirely on distance.

If one feature has a much larger numerical range than another, it dominates the distance computation.

For example:

- feature A ranges from 0 to 1
- feature B ranges from 0 to 10,000

Then raw Euclidean distance is governed mainly by feature B, even if feature A is equally important.

That is why this project uses `StandardScaler`.

Standardization transforms each feature as:

\[
z = \frac{x - \mu}{\sigma}
\]

This gives all features comparable scale and makes distance computation meaningful.

---

## 7.3 Bias-variance tradeoff in KNN

The number of neighbors \(K\) acts like a smoothing parameter.

### Small \(K\)
- very flexible
- sensitive to noise
- can memorize local irregularities
- low bias, high variance
- prone to overfitting

### Large \(K\)
- smoother decision rule
- more robust to local noise
- may blur true class boundaries
- higher bias, lower variance
- prone to underfitting

This project visualizes this phenomenon explicitly.

---

## 7.4 Why cross-validation matters

Training accuracy alone is misleading in KNN.

For example, with \(K=1\), each training point is its own nearest neighbor, so training accuracy can be extremely high or even perfect.

But such a model may generalize poorly.

Cross-validation estimates performance on unseen data more reliably by repeatedly splitting the data into train/validation folds.

That is why this project uses **5-fold stratified cross-validation**.

---

## 7.5 Why pipelines matter

This project uses:

```python
Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])
```

This is crucial because preprocessing must be fitted **only on the training data** inside each fold.

If scaling were done before cross-validation on the full dataset, the validation fold would leak information into preprocessing statistics.

That would produce overly optimistic results.

So the pipeline is not just stylistic cleanliness; it is necessary for **correct evaluation**.

---

## 8. Datasets used in the project

The project uses multiple datasets, each chosen for a specific reason.

### 8.1 Moons dataset
Used for:
- decision-boundary visualization
- validation curve for \(K\)

Why this dataset?
- 2D
- nonlinear
- easy to visualize
- ideal for showing how KNN adapts to curved geometry

### 8.2 Breast-cancer dataset
Used for:
- full classification pipeline
- hyperparameter tuning
- ROC analysis
- learning curves

Why this dataset?
- real biomedical classification problem
- binary labels: malignant vs benign
- multiple numerical features
- features live on different scales, so scaling matters
- suitable for ROC-AUC evaluation

### 8.3 Wine dataset
Used for:
- distance metric comparison

Why this dataset?
- multiclass problem
- continuous tabular features
- nontrivial feature correlations
- useful for examining metric sensitivity

### 8.4 Optional overview datasets
The broader project may also include synthetic visual overview panels such as:
- linear binary data
- circles dataset
- multi-class blobs
- physics-inspired colloidal phase diagram
- regression-style polymer-viscosity plot

These help place KNN in a broader conceptual machine-learning context, even though the main classification workflow focuses on the first three datasets listed above.

---

## 9. Project workflow

The overall workflow can be thought of as five stages:

### Stage 1: Understand geometry
Use 2D nonlinear data to visualize what KNN is doing.

### Stage 2: Tune \(K\)
Use validation curves to select the neighborhood size that gives the best generalization.

### Stage 3: Build a robust pipeline
Move to a real dataset and combine scaling + KNN in a leak-free pipeline.

### Stage 4: Evaluate comprehensively
Use multiple metrics, not just accuracy.

### Stage 5: Diagnose model behavior
Use learning curves and metric comparisons to understand data efficiency and geometric assumptions.

---

## 10. File outputs produced by the script

The script generates several output figures:

### `knn_K_effect.png`
Shows decision boundaries for different values of \(K\) on the moons dataset.

### `knn_validation_curve.png`
Shows training and CV accuracy as functions of \(K\).

### `knn_evaluation.png`
Contains:
- confusion matrix
- ROC curve

for the best breast-cancer KNN model.

### `knn_learning_curve.png`
Shows training and CV accuracy versus training set size.

### `datasets_overview.png`
Provides an overview of different datasets used for KNN/SVM-style experiments.

---

## 11. Detailed explanation of each experiment

## 11.1 Experiment 1: Effect of \(K\) on the decision boundary

### Purpose
To visualize how the neighborhood size changes model flexibility.

### Dataset
`make_moons(n_samples=300, noise=0.2)`

### Method
- scale the 2D features
- fit KNN for different values of \(K\)
- predict class on a dense 2D grid
- color the full feature plane by predicted class
- overlay the training samples

### What this reveals
- \(K=1\) produces a jagged, highly local boundary
- moderate \(K\) values produce a smoother boundary that still respects the nonlinear shape
- very large \(K\) values oversmooth the class geometry

### Core insight
This plot is a visual explanation of the **bias-variance tradeoff**.

---

## 11.2 Experiment 2: Validation curve for \(K\)

### Purpose
To move from visual intuition to quantitative model selection.

### Dataset
`make_moons(n_samples=600, noise=0.2)`

### Method
- build pipeline: scaler + KNN
- vary \(K\) over a specified range
- compute:
  - training accuracy
  - cross-validation accuracy
- plot mean and standard deviation over folds

### What to look for
- if training accuracy is much higher than CV accuracy at low \(K\), the model is overfitting
- if both curves are low at large \(K\), the model is underfitting
- the best \(K\) is usually near the peak of the CV curve

### Insight
This experiment translates geometric behavior into generalization performance.

---

## 11.3 Experiment 3: Full breast-cancer classification pipeline

### Purpose
To demonstrate a realistic KNN workflow on a real dataset.

### Dataset
`load_breast_cancer()`

### Steps
1. load features and labels
2. stratified train/test split
3. build pipeline:
   - `StandardScaler`
   - `KNeighborsClassifier`
4. define hyperparameter grid
5. run `GridSearchCV`
6. select best model using ROC-AUC
7. evaluate on held-out test set
8. plot confusion matrix and ROC curve

### Hyperparameters tuned
- `n_neighbors`
- `metric`
- `weights`
- `p` for Minkowski distance

### Why ROC-AUC was used
Accuracy can be informative, but ROC-AUC is often more robust for binary classification because it evaluates ranking quality across thresholds, not only one thresholded decision boundary.

---

## 11.4 Experiment 4: Learning curve

### Purpose
To understand how model performance evolves with increasing training set size.

### What it diagnoses
- whether the model is underfitting
- whether the model is overfitting
- whether adding more data is likely to help

### Interpretation framework
- large train/CV gap: variance issue
- both low: bias issue
- CV still rising strongly: more data likely useful
- train and CV converging at high performance: model is learning stably

---

## 11.5 Experiment 5: Distance metric comparison

### Purpose
To examine how different notions of distance affect KNN.

### Metrics compared
- Euclidean
- Manhattan
- Minkowski with \(p=3\)
- Chebyshev
- Cosine

### Why this matters
KNN is entirely defined by neighborhood structure, and neighborhood structure is entirely defined by the distance metric.

Different metrics imply different geometric assumptions.

---

## 12. Results summary from the project run

The following summary is based on the observed outputs from the project.

## 12.1 Best \(K\) on the moons validation curve
The best cross-validation performance on the moons dataset was observed around:

- **Best K = 21**
- **CV accuracy ≈ 0.9683**

### Interpretation
This indicates that a moderately sized neighborhood provides the best tradeoff between:
- fitting nonlinear structure
- resisting local noise
- maintaining stable generalization

Very small \(K\) values were too sensitive; very large values oversmoothed the boundary.

---

## 12.2 Best breast-cancer pipeline configuration
Grid search selected:

- **metric = euclidean**
- **n_neighbors = 11**
- **weights = distance**
- **p = 2**
- **Best CV AUC ≈ 0.9925**

### Interpretation
This is a very strong result.

It suggests:
- the dataset is well-structured for local neighborhood classification
- Euclidean geometry works well once features are standardized
- distance-weighted voting improves performance, meaning closer neighbors contain more reliable local information than farther neighbors within the same neighborhood

---

## 12.3 Test-set performance on breast-cancer dataset
Observed test metrics:

- **Accuracy ≈ 0.9737**
- **ROC AUC ≈ 0.9917**

### Interpretation
This means the model generalizes extremely well to unseen examples.

High AUC indicates that the classifier ranks malignant vs benign cases very effectively across thresholds.

---

## 12.4 Confusion matrix interpretation
Observed confusion matrix:

- malignant predicted malignant: 39
- malignant predicted benign: 3
- benign predicted malignant: 0
- benign predicted benign: 72

### Interpretation
This means:
- the classifier correctly identified all benign samples in the test set
- only 3 malignant cases were predicted as benign
- no benign case was incorrectly labeled malignant

This asymmetry is important.

In a medical setting, false negatives on malignant cases are typically more serious than false positives. So while the overall model is excellent, the few missed malignant cases deserve more attention than the perfect benign recall might initially suggest.

---

## 12.5 Classification report interpretation

### Malignant class
- precision = 1.00
- recall = 0.93
- f1-score = 0.96

This means:
- when the model predicts malignant, it is extremely reliable
- however, it misses a small fraction of actual malignant cases

### Benign class
- precision = 0.96
- recall = 1.00
- f1-score = 0.98

This means:
- all benign cases were successfully recovered
- a very small fraction of benign predictions may include ambiguity indirectly through the malignant miss pattern

### Global
- weighted and macro averages are both very strong
- class balance is not perfectly equal, but not so skewed as to make accuracy meaningless

---

## 12.6 ROC curve interpretation
The ROC curve lies close to the top-left corner, and the AUC is approximately 0.992.

### What that means
The model has excellent discrimination ability.

If one randomly selects:
- one malignant case
- one benign case

the model will assign a higher malignant-related decision score to the malignant case with very high probability.

This indicates strong ranking performance independent of any one hard threshold.

---

## 12.7 Learning curve interpretation
The learning curve shows:

- high training accuracy
- cross-validation accuracy rising with training size
- a relatively small gap between training and validation curves at larger sample sizes
- convergence at a high level of accuracy

### Meaning
This suggests:
- the model is not suffering severe overfitting
- the model is learning stable structure rather than memorizing noise
- more data helps early on, but gains become smaller as the curve begins to plateau

### Practical insight
The model is already in a good regime. More data may still help slightly, but the learning curve suggests diminishing returns beyond the current sample size.

---

## 12.8 Distance metric comparison interpretation

Observed results:

- Euclidean: excellent
- Manhattan: slightly better CV mean than Euclidean
- Cosine: also strong
- Minkowski \(p=3\): noticeably worse
- Chebyshev: clearly worse

### Insight
This shows that the geometry of this dataset is more compatible with:
- Euclidean neighborhoods
- Manhattan neighborhoods
- Cosine-based similarity neighborhoods

than with:
- Chebyshev’s max-coordinate distance
- the specific \(p=3\) Minkowski geometry tried here

This is exactly the kind of project outcome that illustrates why the distance metric should not be treated as a trivial default.

---

## 13. Why the project is methodologically sound

This project follows several important best practices.

### 13.1 Feature scaling before KNN
Essential for all distance-based methods.

### 13.2 Stratified splitting
Maintains class balance in train/test and CV folds.

### 13.3 Pipeline-based preprocessing
Prevents data leakage.

### 13.4 Cross-validation for model selection
Avoids choosing hyperparameters from test-set performance.

### 13.5 Separate test-set evaluation
Ensures an honest final estimate of generalization.

### 13.6 Multiple evaluation metrics
Prevents overreliance on a single number like accuracy.

---

## 14. Strengths of the project

### 14.1 Educational clarity
The project explains KNN through both pictures and metrics.

### 14.2 Multiple datasets
It avoids drawing all conclusions from only one dataset.

### 14.3 Geometry-aware analysis
Decision boundaries and distance metrics are central, which is exactly appropriate for KNN.

### 14.4 Proper model-selection workflow
Grid search and validation curves are used correctly.

### 14.5 Strong diagnostic depth
Learning curves and confusion matrices help interpret not just how good the model is, but why it behaves the way it does.

---

## 15. Limitations of the current project

Despite being strong and well-designed, the project still has natural limitations.

### 15.1 No precision-recall curve yet
For medical classification, precision-recall analysis can be especially informative when the positive class is clinically important.

### 15.2 No calibration analysis
KNN probability estimates may not always be well-calibrated.

### 15.3 No explicit class-imbalance experiments
The breast-cancer dataset is not extremely imbalanced, so the project does not stress-test KNN under severe imbalance.

### 15.4 No runtime / scalability benchmarking
KNN can be computationally expensive at inference time because it stores and searches all training points.

### 15.5 No dimensionality-reduction study
KNN often benefits from PCA or feature selection in higher-dimensional spaces.

### 15.6 No sensitivity analysis over noise level
For the moons data, only one representative noise level was used.

---

## 16. Possible project extensions

A number of meaningful extensions could make the project even stronger.

### 16.1 Add precision-recall analysis
Especially useful for medical datasets.

### 16.2 Add PCA + KNN experiments
Test whether dimensionality reduction improves stability or accuracy.

### 16.3 Add feature selection
Investigate whether performance changes when only the most informative features are kept.

### 16.4 Compare against other classifiers
For example:
- logistic regression
- SVM
- decision tree
- random forest

### 16.5 Benchmark inference time
Important because KNN’s prediction cost grows with dataset size.

### 16.6 Explore weighted vs uniform voting more deeply
For some datasets, distance weighting can matter substantially.

### 16.7 Add class-imbalance experiments
Use resampling or weighted evaluation to study KNN robustness.

### 16.8 Explore neighborhood search acceleration
For example:
- KD-tree
- Ball-tree
- brute-force comparisons

### 16.9 Add uncertainty / confidence diagnostics
Investigate how class probability estimates change with local neighborhood purity.

---

## 17. How to run the project

Assuming your main file is named something like:

```bash
python3.10 KNN_classification_v2.py
```

or:

```bash
python3.10 KNN_classification.py
```

the script will run the full sequence of experiments and save the plots to the current working directory.

### Expected console flow
Typical output includes:
- section headers for each experiment
- best \(K\) from validation curve
- grid-search summary
- best hyperparameters
- test accuracy and ROC-AUC
- classification report
- metric-comparison table

---

## 18. Dependencies

A typical environment for this project includes:

- Python 3.x
- NumPy
- Matplotlib
- scikit-learn

Install with:

```bash
pip install numpy matplotlib scikit-learn
```

---

## 19. Suggested project structure

A clean project structure may look like this:

```text
KNN_project/
├── KNN_classification.py
├── KNN_classification_v2.py
├── README.md
├── observations.md
├── knn_K_effect.png
├── knn_validation_curve.png
├── knn_evaluation.png
├── knn_learning_curve.png
└── datasets_overview.png
```

---

## 20. How to read the project outputs

### 20.1 If you are a beginner
Start with:
1. decision boundaries
2. validation curve
3. full evaluation figure

This gives the best intuition.

### 20.2 If you are tuning the model
Focus on:
1. validation curve
2. grid-search results
3. confusion matrix
4. ROC-AUC

### 20.3 If you are diagnosing generalization
Focus on:
1. train vs CV curves
2. learning curve
3. stability across distance metrics

---

## 21. Main conceptual takeaways

This project demonstrates several major lessons about KNN.

### Lesson 1
KNN is simple to define but highly sensitive to modeling choices.

### Lesson 2
Feature scaling is non-negotiable for distance-based classification.

### Lesson 3
The number of neighbors \(K\) is a smoothing parameter that controls the bias-variance tradeoff.

### Lesson 4
Decision-boundary visualization is one of the clearest ways to understand KNN behavior.

### Lesson 5
Cross-validation is essential for choosing \(K\) and other hyperparameters honestly.

### Lesson 6
Pipelines are necessary to avoid leakage.

### Lesson 7
Different distance metrics encode different geometric assumptions and can materially affect performance.

### Lesson 8
Learning curves help distinguish:
- overfitting
- underfitting
- data limitations

---

## 22. Practical conclusions from this project

From the observed results, the main practical conclusions are:

1. **Moderate values of \(K\)** work best on the noisy moons dataset.
2. For the breast-cancer dataset, a **distance-weighted Euclidean KNN** performs extremely well.
3. The final tuned model achieves **very strong discrimination performance**.
4. The project’s learning curve suggests **stable generalization** rather than severe overfitting.
5. Distance metric choice matters; **not all metrics are equally suitable**.

---

## 23. Scientific / methodological conclusion

From a machine-learning methodology standpoint, this project successfully demonstrates that:

- KNN can be a strong baseline and, in some cases, a highly competitive model
- its effectiveness depends strongly on geometric compatibility between the data and the chosen distance measure
- proper preprocessing, validation, and visualization are essential to using KNN responsibly
- model interpretation is strongest when geometric intuition and statistical evaluation agree

---

## 24. Final summary

This project is not just a KNN classifier.

It is a **carefully structured study of KNN behavior** across:

- synthetic and real datasets
- geometric visualization
- hyperparameter tuning
- probabilistic evaluation
- learning dynamics
- metric sensitivity

As a result, it serves equally well as:

- a **learning resource**
- an **experimental notebook in script form**
- a **template for future classification studies**
- a **reference implementation of best practices for KNN with scikit-learn**

If your goal is to understand not only how to run KNN, but how to **reason about its behavior**, this project provides a strong and comprehensive foundation.
