# =============================================================================
# SCIKIT-LEARN LOGISTIC REGRESSION — FULL PIPELINE (EXTREMELY DOCUMENTED)
# =============================================================================
#
# WHAT THIS SCRIPT DOES
# ---------------------
# This script demonstrates a complete, practical workflow for binary
# classification using Logistic Regression in scikit-learn.
#
# It covers:
#   1. Data loading and train/test splitting
#   2. Safe preprocessing with Pipeline (prevents data leakage)
#   3. Basic Logistic Regression fitting
#   4. Comprehensive evaluation using many classification metrics
#   5. Confusion matrix visualization
#   6. Interpreting model coefficients as feature importance
#   7. Hyperparameter tuning with GridSearchCV
#   8. Cross-validation to estimate generalization performance
#   9. Nonlinear modeling through PolynomialFeatures
#  10. Learning curves to diagnose underfitting/overfitting
#  11. ROC, PR, and calibration curves for model comparison
#
# MATHEMATICAL IDEA OF LOGISTIC REGRESSION
# ----------------------------------------
# Logistic Regression models the probability of class 1 using:
#
#     p(y=1 | x) = sigmoid(z)
#
# where
#
#     z = w^T x + b
#
# and
#
#     sigmoid(z) = 1 / (1 + exp(-z))
#
# It is called "regression" historically, but it is used for classification.
# The model outputs probabilities, then applies a threshold (usually 0.5)
# to convert probabilities into class labels.
#
# IMPORTANT NOTE ABOUT THE BREAST CANCER DATASET
# ----------------------------------------------
# In sklearn.datasets.load_breast_cancer():
#
#     target 0 = malignant
#     target 1 = benign
#
# So in this dataset:
#
#     predict_proba(X)[:, 1]
#
# means "predicted probability of BENIGN", not malignant.
#
# That is extremely important when interpreting:
#   - coefficients
#   - class probabilities
#   - confusion matrix display labels
#
# =============================================================================


# =============================================================================
# 1. IMPORTS
# =============================================================================

# Numerical array computations
import numpy as np

# Table-like data handling (used mainly for organizing CV results)
import pandas as pd

# Plotting library
import matplotlib.pyplot as plt

# GridSpec lets us arrange multiple subplots in a flexible layout
import matplotlib.gridspec as gridspec

# Main Logistic Regression estimator from scikit-learn
from sklearn.linear_model import LogisticRegression

# StandardScaler:
#   rescales each feature to mean 0 and standard deviation 1
# PolynomialFeatures:
#   expands original features into polynomial combinations
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Pipeline chains preprocessing + model together in a single object
# This is crucial for avoiding leakage in CV and train/test workflows
from sklearn.pipeline import Pipeline

# Tools for splitting data, cross-validation, hyperparameter tuning,
# and learning-curve generation
from sklearn.model_selection import (
    train_test_split,      # split into training and testing sets
    cross_val_score,       # simple CV scoring (imported but not directly used here)
    StratifiedKFold,       # CV splitter preserving class proportions
    GridSearchCV,          # exhaustive hyperparameter search
    learning_curve         # computes train/validation scores vs training size
)

# Metrics for classification performance
from sklearn.metrics import (
    accuracy_score,              # fraction of correct predictions
    precision_score,             # TP / (TP + FP)
    recall_score,                # TP / (TP + FN)
    f1_score,                    # harmonic mean of precision and recall
    roc_auc_score,               # area under ROC curve
    average_precision_score,     # area summary of Precision-Recall curve
    log_loss,                    # cross-entropy loss
    confusion_matrix,            # raw confusion matrix counts
    classification_report,       # precision/recall/F1 per class
    roc_curve,                   # FPR/TPR points
    precision_recall_curve,      # precision/recall points across thresholds
    ConfusionMatrixDisplay,      # confusion matrix plotting helper
    matthews_corrcoef,           # MCC, useful especially under imbalance
    brier_score_loss             # calibration/probability-quality metric
)

# Datasets:
#   - breast cancer (real dataset)
#   - synthetic classification dataset (imported, not used)
#   - moons dataset (nonlinearly separable)
from sklearn.datasets import load_breast_cancer, make_classification, make_moons

# Calibration plotting helper and probability calibration wrapper
from sklearn.calibration import CalibrationDisplay, CalibratedClassifierCV

# Used to suppress warnings for cleaner notebook/script output
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# 2. GLOBAL RANDOM SEED
# =============================================================================
# Using a fixed seed makes train/test split and synthetic data generation
# reproducible. If someone reruns this script, they get the same random outcome.
SEED = 42


# =============================================================================
# 3. HELPER FUNCTION: FULL EVALUATION REPORT
# =============================================================================
def full_evaluation_report(model, X_train, X_test, y_train, y_test, name="Model"):
    """
    Evaluate a fitted classification model on train and test data.

    PARAMETERS
    ----------
    model : fitted sklearn-like estimator
        Must implement:
            - predict(X)
            - predict_proba(X)
    X_train, X_test : array-like
        Feature matrices for training and test sets.
    y_train, y_test : array-like
        True labels for training and test sets.
    name : str
        A label for printing the report.

    RETURNS
    -------
    metrics : dict
        Dictionary containing many evaluation metrics.

    WHY THIS FUNCTION EXISTS
    ------------------------
    Instead of evaluating the model in scattered places, this helper centralizes
    all standard metrics in one place.

    WHAT IT COMPUTES
    ----------------
    1. Predicted labels on test set:
         y_pred = model.predict(X_test)

       These are hard 0/1 class predictions using the model's default threshold
       (usually 0.5 for LogisticRegression).

    2. Predicted probabilities on test set:
         y_prob = model.predict_proba(X_test)[:, 1]

       This gives the predicted probability of class 1.
       IMPORTANT: for the breast cancer dataset, class 1 = benign.

    3. Predicted labels on training set:
         y_pred_train = model.predict(X_train)

       This is used to compare training vs test accuracy. A large gap may hint
       at overfitting.
    """

    # -------------------------------------------------------------------------
    # Hard class predictions on test set
    # Example: [1, 0, 1, 1, 0, ...]
    # -------------------------------------------------------------------------
    y_pred = model.predict(X_test)

    # -------------------------------------------------------------------------
    # Predicted probabilities for class 1 on test set
    # Example: [0.98, 0.12, 0.73, 0.61, ...]
    # In breast cancer dataset, this is P(benign | x)
    # -------------------------------------------------------------------------
    y_prob = model.predict_proba(X_test)[:, 1]

    # -------------------------------------------------------------------------
    # Hard class predictions on training set
    # Used to compare training performance with test performance
    # -------------------------------------------------------------------------
    y_pred_train = model.predict(X_train)

    # -------------------------------------------------------------------------
    # METRICS DICTIONARY
    # Each metric captures a different aspect of performance.
    # -------------------------------------------------------------------------
    metrics = {
        'Name': name,

        # Training accuracy:
        # proportion of training samples correctly classified
        'Train Acc': accuracy_score(y_train, y_pred_train),

        # Test accuracy:
        # proportion of test samples correctly classified
        'Test Acc': accuracy_score(y_test, y_pred),

        # Precision:
        # among samples predicted as class 1, how many truly were class 1?
        #   Precision = TP / (TP + FP)
        'Precision': precision_score(y_test, y_pred, zero_division=0),

        # Recall:
        # among all true class 1 samples, how many did the model recover?
        #   Recall = TP / (TP + FN)
        'Recall': recall_score(y_test, y_pred, zero_division=0),

        # F1 score:
        # harmonic mean of precision and recall
        # useful when both false positives and false negatives matter
        'F1': f1_score(y_test, y_pred, zero_division=0),

        # Matthews Correlation Coefficient:
        # a balanced metric using all four confusion-matrix entries
        # ranges from -1 (total disagreement) to +1 (perfect prediction)
        'MCC': matthews_corrcoef(y_test, y_pred),

        # AUC-ROC:
        # threshold-independent ranking metric
        # probability that a random positive is ranked above a random negative
        'AUC-ROC': roc_auc_score(y_test, y_prob),

        # AUC-PR / Average Precision:
        # especially useful in imbalanced problems
        # summarizes precision-recall behavior across thresholds
        'AUC-PR': average_precision_score(y_test, y_prob),

        # Log Loss / Cross-Entropy:
        # penalizes incorrect and overconfident probability predictions
        'Log Loss': log_loss(y_test, y_prob),

        # Brier Score:
        # mean squared error between predicted probabilities and true labels
        # smaller is better; assesses calibration + probabilistic quality
        'Brier Score': brier_score_loss(y_test, y_prob),
    }

    # -------------------------------------------------------------------------
    # Nicely formatted printing
    # -------------------------------------------------------------------------
    print(f"\n{'=' * 55}")
    print(f"  {name}")
    print(f"{'=' * 55}")

    # Print all numeric metrics except the model name
    for k, v in metrics.items():
        if k != 'Name':
            print(f"  {k:<20s}: {v:.4f}")

    print()

    # classification_report gives per-class precision/recall/F1/support
    # target_names here are generic because this helper is reusable
    print(classification_report(
        y_test, y_pred,
        target_names=['Class 0', 'Class 1']
    ))

    return metrics


# =============================================================================
# 4. HELPER FUNCTION: PLOT ROC + PR + CALIBRATION CURVES
# =============================================================================
def plot_model_diagnostics(models_dict, X_test, y_test, title="Model Diagnostics"):
    """
    Plot three diagnostic views for one or more fitted models:

      1. ROC curves
      2. Precision-Recall curves
      3. Calibration curves (reliability diagrams)

    PARAMETERS
    ----------
    models_dict : dict
        Dictionary of the form:
            {
                'model name 1': fitted_model_1,
                'model name 2': fitted_model_2,
                ...
            }

    X_test, y_test : array-like
        Test data to evaluate each model.

    title : str
        Figure title.

    RETURNS
    -------
    fig : matplotlib Figure

    WHY THIS FUNCTION EXISTS
    ------------------------
    Different models may have:
      - similar accuracy
      - different ranking performance
      - different calibration quality

    This function compares all those aspects visually in one figure.
    """

    # Create a wide figure with 3 side-by-side panels
    fig = plt.figure(figsize=(18, 5))

    # GridSpec lets us explicitly allocate 1 row, 3 columns
    gs = gridspec.GridSpec(1, 3, figure=fig)

    # Create axes for:
    #   left  -> ROC
    #   middle-> PR
    #   right -> Calibration
    ax_roc = fig.add_subplot(gs[0])
    ax_pr  = fig.add_subplot(gs[1])
    ax_cal = fig.add_subplot(gs[2])

    # Create a set of colors for the different models
    colors = plt.cm.Set1(np.linspace(0, 0.8, len(models_dict)))

    # -------------------------------------------------------------------------
    # Loop over each fitted model and draw all three diagnostics
    # -------------------------------------------------------------------------
    for (name, model), color in zip(models_dict.items(), colors):

        # Probability of class 1 for each test point
        y_prob = model.predict_proba(X_test)[:, 1]

        # ==============================
        # ROC CURVE
        # ==============================
        # roc_curve returns threshold-varying:
        #   FPR = False Positive Rate
        #   TPR = True Positive Rate (Recall)
        fpr, tpr, _ = roc_curve(y_test, y_prob)

        # Scalar summary of ROC
        auc = roc_auc_score(y_test, y_prob)

        # Plot ROC line
        ax_roc.plot(
            fpr, tpr,
            color=color, lw=2,
            label=f"{name} (AUC={auc:.3f})"
        )

        # ==============================
        # PRECISION-RECALL CURVE
        # ==============================
        # precision_recall_curve returns precision and recall over thresholds
        prec, rec, _ = precision_recall_curve(y_test, y_prob)

        # Average precision is a scalar summary
        ap = average_precision_score(y_test, y_prob)

        ax_pr.plot(
            rec, prec,
            color=color, lw=2,
            label=f"{name} (AP={ap:.3f})"
        )

        # ==============================
        # CALIBRATION CURVE
        # ==============================
        # A calibration curve checks whether predicted probabilities
        # correspond to actual observed frequencies.
        #
        # Example:
        # If the model says "0.8 probability" for many cases,
        # then about 80% of those cases should truly belong to class 1.
        CalibrationDisplay.from_predictions(
            y_test, y_prob,
            n_bins=10,   # bin predicted probabilities into 10 groups
            ax=ax_cal,
            name=name,
            color=color
        )

    # -------------------------------------------------------------------------
    # ROC baseline: random classifier
    # A random classifier lies on the diagonal from (0,0) to (1,1)
    # -------------------------------------------------------------------------
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC=0.5)')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curves')
    ax_roc.legend(fontsize=8)
    ax_roc.grid(alpha=0.3)

    # -------------------------------------------------------------------------
    # PR baseline:
    # For random ranking, average precision is roughly the positive-class rate
    # in the data.
    # -------------------------------------------------------------------------
    base_rate = y_test.mean()
    ax_pr.axhline(
        base_rate,
        color='k', linestyle='--', lw=1,
        label=f'Random (AP={base_rate:.2f})'
    )
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title('Precision-Recall Curves')
    ax_pr.legend(fontsize=8)
    ax_pr.grid(alpha=0.3)

    # Calibration axis title
    ax_cal.set_title('Calibration Curves\n(Reliability Diagram)')

    # Global title for the entire figure
    fig.suptitle(title, fontsize=13, fontweight='bold')

    # Adjust spacing
    plt.tight_layout()

    return fig


# =============================================================================
# 5. EXPERIMENT 1: BASIC LOGISTIC REGRESSION ON REAL DATA
# =============================================================================
print("=" * 60)
print("EXPERIMENT 1: Basic sklearn LogisticRegression")
print("=" * 60)

# -------------------------------------------------------------------------
# Load the breast cancer dataset
# -------------------------------------------------------------------------
data = load_breast_cancer()

# Feature matrix:
#   shape = (n_samples, n_features)
# Each row = one patient/sample
# Each column = one measured feature
X = data.data

# Target vector:
#   0 = malignant
#   1 = benign
y = data.target

# -------------------------------------------------------------------------
# Split into train and test sets
# -------------------------------------------------------------------------
# test_size=0.2 means:
#   80% training data
#   20% test data
#
# stratify=y preserves the malignant/benign ratio in both train and test.
# This is very important in classification.
#
# random_state=SEED ensures reproducibility.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

# -------------------------------------------------------------------------
# PRODUCTION-STYLE PIPELINE
# -------------------------------------------------------------------------
# Why Pipeline?
# Because scaling must be learned ONLY from training data.
# If you scale before splitting, you leak information from test set into train.
#
# The pipeline ensures the following happens correctly:
#   fit(X_train, y_train):
#       scaler.fit(X_train)
#       X_train_scaled = scaler.transform(X_train)
#       clf.fit(X_train_scaled, y_train)
#
#   predict(X_test):
#       X_test_scaled = scaler.transform(X_test)
#       clf.predict(X_test_scaled)
#
# This is the safe and correct way.
pipeline = Pipeline([
    ('scaler', StandardScaler()),        # z-score standardization
    ('clf',   LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='lbfgs',      # L-BFGS quasi-Newton solver
        max_iter=5000,       
        tol=1e-4,            # explicit tolerance
        random_state=SEED,
        class_weight=None    # 'balanced' for imbalanced datasets
    ))
])

# -------------------------------------------------------------------------
# Fit the pipeline
# -------------------------------------------------------------------------
pipeline.fit(X_train, y_train)

# -------------------------------------------------------------------------
# Evaluate the fitted model
# -------------------------------------------------------------------------
metrics_basic = full_evaluation_report(
    pipeline,
    X_train, X_test,
    y_train, y_test,
    name="LR (L2, C=1.0)"
)


# -------------------------------------------------------------------------
# CONFUSION MATRIX
# -------------------------------------------------------------------------
# Confusion matrix layout for binary classification:
#
#                 Pred 0      Pred 1
# True 0          TN-like?*   FP-like?*
# True 1          FN-like?*   TP-like?*
#
# More precisely, which entry is called TP depends on which class you regard
# as the "positive" class. In sklearn's binary metrics here, class 1 is the
# positive class by default.
#
# For this dataset:
#   class 0 = malignant
#   class 1 = benign
#
# So display labels must follow that order.
y_pred = pipeline.predict(X_test)

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=['Malignant', 'Benign'],  # correct order: 0 then 1
    colorbar=False,
    ax=ax,
    cmap='Blues'
)
ax.set_title("Confusion Matrix — Breast Cancer (L2, C=1.0)")
plt.tight_layout()
plt.savefig("cm_cancer_basic.png", dpi=150, bbox_inches='tight')
# plt.show()

# -------------------------------------------------------------------------
# FEATURE IMPORTANCE / COEFFICIENT INTERPRETATION
# -------------------------------------------------------------------------
# For logistic regression:
#
#   log( p / (1-p) ) = w^T x + b
#
# where p = P(class 1 | x)
#
# Since we scaled the inputs, each coefficient is in standardized units.
# That makes coefficients more comparable across features.
#
# Large |coefficient| means the feature strongly affects the log-odds.
#
# IMPORTANT:
# Here class 1 = benign.
# So:
#   positive coefficient  -> pushes prediction toward BENIGN
#   negative coefficient  -> pushes prediction toward MALIGNANT
coef = pipeline.named_steps['clf'].coef_[0]
names = data.feature_names

# Sort by absolute magnitude so the strongest features appear first
sorted_idx = np.argsort(np.abs(coef))[::-1][:15]   # top 15 strongest features

fig, ax = plt.subplots(figsize=(10, 5))

# Color code:
#   red  -> positive coefficient -> increases P(benign)
#   blue -> negative coefficient -> increases P(malignant)
colors_bar = ['#F44336' if c > 0 else '#2196F3' for c in coef[sorted_idx]]

ax.barh(
    range(15),
    coef[sorted_idx],
    color=colors_bar,
    alpha=0.8
)

ax.set_yticks(range(15))
ax.set_yticklabels([names[i] for i in sorted_idx], fontsize=9)

# Vertical line at zero helps interpret sign
ax.axvline(0, color='black', lw=0.8)

ax.set_xlabel("Coefficient Value (standardized feature units)")
ax.set_title(
    "Top 15 Logistic Regression Coefficients\n"
    "(Red = increases P(benign), Blue = increases P(malignant))"
)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig("coef_cancer.png", dpi=150, bbox_inches='tight')
# plt.show()


# =============================================================================
# 6. EXPERIMENT 2: HYPERPARAMETER TUNING WITH GRID SEARCH + CROSS-VALIDATION
# =============================================================================
print("\n" + "=" * 60)
print("EXPERIMENT 2: Hyperparameter Tuning (GridSearchCV)")
print("=" * 60)

# -------------------------------------------------------------------------
# PARAMETER GRID
# -------------------------------------------------------------------------
# We search across different forms of regularization:
#
#   L2:
#       shrinks coefficients smoothly toward zero
#
#   L1:
#       encourages sparsity, can set some coefficients exactly to zero
#
#   Elastic Net:
#       combination of L1 and L2
#
# We also vary C:
#   smaller C -> stronger regularization
#   larger C  -> weaker regularization
#
# Each penalty has solver constraints:
#   - lbfgs supports L2
#   - liblinear supports L1/L2 for smaller-scale problems
#   - saga supports elastic net and large-scale sparse settings
param_grid = [
    {
        'clf__penalty': ['l2'],
        'clf__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        'clf__solver': ['lbfgs']
    },
    {
        'clf__penalty': ['l1'],
        'clf__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        'clf__solver': ['liblinear']
    },
    {
        'clf__penalty': ['elasticnet'],
        'clf__C': [0.01, 0.1, 1.0, 10.0],
        'clf__solver': ['saga'],
        'clf__l1_ratio': [0.1, 0.5, 0.9]  # only relevant for elastic net
    }
]

# -------------------------------------------------------------------------
# STRATIFIED 5-FOLD CROSS-VALIDATION
# -------------------------------------------------------------------------
# In each fold:
#   - 4/5 of the training set is used to fit
#   - 1/5 is used to validate
#
# StratifiedKFold preserves class ratio in every fold.
# shuffle=True randomizes the fold assignments.
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)

# -------------------------------------------------------------------------
# GRID SEARCH OBJECT
# -------------------------------------------------------------------------
# GridSearchCV:
#   - tries every hyperparameter combination
#   - evaluates each one via cross-validation
#   - selects the best according to the scoring metric
#
# We use scoring='roc_auc' because:
#   - it is threshold-independent
#   - it evaluates ranking quality
#   - it is often better than raw accuracy for model selection
grid_search = GridSearchCV(
    estimator=pipeline,         # base pipeline to tune
    param_grid=param_grid,      # all parameter combinations to try
    cv=cv,                      # cross-validation scheme
    scoring='roc_auc',          # objective to maximize
    n_jobs=4,                  # CPU cores
    verbose=1,                  # print progress
    return_train_score=True     # also store train-fold scores for diagnostics
)

# Run the full hyperparameter search
grid_search.fit(X_train, y_train)

# Best hyperparameter combination found across all CV folds
print(f"\nBest parameters: {grid_search.best_params_}")

# Mean CV score for that best parameter set
print(f"Best CV AUC-ROC: {grid_search.best_score_:.4f}")

# Retrieve the fully refit best model
# GridSearchCV automatically refits the best configuration on the full training set
best_model = grid_search.best_estimator_

# Evaluate best model on held-out test data
metrics_tuned = full_evaluation_report(
    best_model,
    X_train, X_test,
    y_train, y_test,
    name="LR Tuned (Best)"
)

# -------------------------------------------------------------------------
# VISUALIZE CV RESULTS FOR L2 MODELS
# -------------------------------------------------------------------------
# cv_results_ is a dictionary of arrays holding all search outcomes.
# We convert it to a DataFrame for easier filtering/plotting.
# Plot CV results: how AUC varies with C for L2
cv_results = pd.DataFrame(grid_search.cv_results_)
l2_results = cv_results[cv_results['param_clf__penalty'] == 'l2'].copy()
l2_results = l2_results.sort_values('param_clf__C')

# Explicit numeric conversion to avoid matplotlib / pandas object-dtype issues
x_c = l2_results['param_clf__C'].astype(float).to_numpy()
y_test_mean = l2_results['mean_test_score'].astype(float).to_numpy()
y_test_std = l2_results['std_test_score'].astype(float).to_numpy()
y_train_mean = l2_results['mean_train_score'].astype(float).to_numpy()

fig, ax = plt.subplots(figsize=(8, 4))

ax.semilogx(
    x_c,
    y_test_mean,
    'o-', color='#2196F3', lw=2, label='CV AUC (Test)'
)

ax.semilogx(
    x_c,
    y_train_mean,
    's--', color='#F44336', lw=2, label='CV AUC (Train)'
)

ax.fill_between(
    x_c,
    y_test_mean - y_test_std,
    y_test_mean + y_test_std,
    alpha=0.2, color='#2196F3'
)

ax.set_xlabel("C (Inverse Regularization Strength)")
ax.set_ylabel("AUC-ROC")
ax.set_title("Bias-Variance Tradeoff: Effect of Regularization Strength\n"
             "(Small C = high regularization = high bias; Large C = overfitting)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("grid_search_cv.png", dpi=150, bbox_inches='tight')
# plt.show()
# plt.show()


# =============================================================================
# 7. EXPERIMENT 3: NONLINEAR DATA + POLYNOMIAL FEATURES
# =============================================================================
print("\n" + "=" * 60)
print("EXPERIMENT 3: Non-linear Data (Moons) + Polynomial Features")
print("=" * 60)

# -------------------------------------------------------------------------
# Create a synthetic "two moons" dataset
# -------------------------------------------------------------------------
# This dataset is intentionally nonlinear.
# Plain linear logistic regression cannot separate it very well using only x1, x2.
X_m, y_m = make_moons(
    n_samples=800,
    noise=0.25,        # add overlap/noise to make the problem realistic
    random_state=SEED
)

# Train/test split for the moons dataset
Xtr_m, Xte_m, ytr_m, yte_m = train_test_split(
    X_m, y_m,
    test_size=0.25,
    random_state=SEED,
    stratify=y_m
)

# -------------------------------------------------------------------------
# Train three models with polynomial expansion:
#   degree 1 -> linear features only
#   degree 2 -> includes x1^2, x2^2, x1*x2
#   degree 3 -> includes cubic interactions too
# -------------------------------------------------------------------------
models_poly = {}

for degree in [1, 2, 3]:

    # Pipeline:
    #   1. expand features
    #   2. scale them
    #   3. fit logistic regression
    pipe = Pipeline([
        (
            'poly',
            PolynomialFeatures(
                degree=degree,
                include_bias=False  # don't add constant 1 column; clf handles intercept
            )
        ),
        (
            'scaler',
            StandardScaler()
        ),
        ('clf',    LogisticRegression(C=1.0, solver='lbfgs', max_iter=5000,
                              tol=1e-4, random_state=SEED))
    ])

    # Fit current polynomial logistic regression model
    pipe.fit(Xtr_m, ytr_m)

    # Store for later plotting
    models_poly[f"Degree {degree}"] = pipe

    # Accuracy on test set
    acc = accuracy_score(yte_m, pipe.predict(Xte_m))

    # Number of generated features after polynomial expansion
    n_features = pipe.named_steps['poly'].n_output_features_

    print(f"  Degree {degree}: {n_features} features, Test Accuracy = {acc:.4f}")

# -------------------------------------------------------------------------
# Decision boundary visualization
# -------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# Define the rectangular plotting region
x1_min, x1_max = X_m[:, 0].min() - 0.3, X_m[:, 0].max() + 0.3
x2_min, x2_max = X_m[:, 1].min() - 0.3, X_m[:, 1].max() + 0.3

# Create a dense grid of points over the 2D feature space
xx1, xx2 = np.meshgrid(
    np.linspace(x1_min, x1_max, 300),
    np.linspace(x2_min, x2_max, 300)
)

# Convert meshgrid into N x 2 coordinate list
grid = np.c_[xx1.ravel(), xx2.ravel()]

# Plot each model's predicted probability surface
for ax, (name, model) in zip(axes, models_poly.items()):

    # Predict class-1 probabilities over the entire grid
    Z = model.predict_proba(grid)[:, 1].reshape(xx1.shape)

    # Filled contour plot of probabilities
    cf = ax.contourf(
        xx1, xx2, Z,
        levels=50,
        cmap='RdBu_r',
        alpha=0.6,
        vmin=0, vmax=1
    )

    # The 0.5 contour is the decision boundary
    ax.contour(
        xx1, xx2, Z,
        levels=[0.5],
        colors='black',
        linewidths=2
    )

    # Overlay the test points
    for cls, color in zip([0, 1], ['#2196F3', '#F44336']):
        mask = (yte_m == cls)
        ax.scatter(
            Xte_m[mask, 0],
            Xte_m[mask, 1],
            c=color,
            edgecolors='white',
            lw=0.5,
            s=40,
            alpha=0.8
        )

    ax.set_title(f"Logistic Regression — {name}")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")

# Shared colorbar
plt.colorbar(cf, ax=axes[-1], label='P(y=1 | x)')

fig.suptitle(
    "Effect of Polynomial Feature Engineering on Decision Boundary",
    fontsize=12, fontweight='bold'
)

plt.tight_layout()
plt.savefig("polynomial_boundaries.png", dpi=150, bbox_inches='tight')
# plt.show()


# =============================================================================
# 8. EXPERIMENT 4: LEARNING CURVES
# =============================================================================
print("\n" + "=" * 60)
print("EXPERIMENT 4: Learning Curves")
print("=" * 60)

# Learning curves show:
#   - training score as training size increases
#   - validation score as training size increases
#
# This helps diagnose:
#   - underfitting (both low)
#   - overfitting (large gap)
#   - data hunger (validation keeps improving with more data)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Compare:
#   1. strong regularization -> likely underfit
#   2. moderate regularization -> better fit
for ax, (pipe_name, pipe) in zip(axes, [
    ("Underfit (C=0.001)",
 Pipeline([('sc', StandardScaler()),
           ('clf', LogisticRegression(C=0.001, max_iter=5000, tol=1e-4))])),
    ("Good Fit (C=1.0)",
    Pipeline([('sc', StandardScaler()),
            ('clf', LogisticRegression(C=1.0, max_iter=5000, tol=1e-4))])),
]):

    # learning_curve trains the model on progressively larger fractions
    # of the training data and evaluates train/CV performance each time.
    train_sizes, train_scores, val_scores = learning_curve(
        estimator=pipe,
        X=X,
        y=y,
        train_sizes=np.linspace(0.1, 1.0, 15),  # 15 sizes from 10% to 100%
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        scoring='accuracy',
        n_jobs=-1
    )

    # Mean and std across CV folds for each training size
    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    # Plot training performance
    ax.plot(
        train_sizes, train_mean,
        'o-', color='#F44336', lw=2,
        label='Train'
    )
    ax.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.15, color='#F44336'
    )

    # Plot validation performance
    ax.plot(
        train_sizes, val_mean,
        'o-', color='#2196F3', lw=2,
        label='CV Validation'
    )
    ax.fill_between(
        train_sizes,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.15, color='#2196F3'
    )

    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Learning Curve — {pipe_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.85, 1.01)

plt.tight_layout()
plt.savefig("learning_curves.png", dpi=150, bbox_inches='tight')
# plt.show()


# =============================================================================
# 9. EXPERIMENT 5: MULTI-MODEL COMPARISON WITH ROC / PR / CALIBRATION
# =============================================================================
print("\n" + "=" * 60)
print("EXPERIMENT 5: ROC, PR, Calibration Curves — Model Comparison")
print("=" * 60)

# -------------------------------------------------------------------------
# Manually scale the breast cancer data for this comparison section
# -------------------------------------------------------------------------
# Here we scale explicitly outside a pipeline because we are simply creating
# a few fitted models for comparison on the same transformed data.
#
# In real production workflows, Pipeline is usually preferred.
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# -------------------------------------------------------------------------
# Fit several logistic regression variants
# -------------------------------------------------------------------------
# Idea:
#   compare how regularization strength / penalty type changes performance
models_compare = {
    'L2 C=0.01 (strong reg)':  LogisticRegression(C=0.01,  penalty='l2', solver='lbfgs', max_iter=5000, tol=1e-4).fit(X_train_sc, y_train),
    'L2 C=1.0  (default)':     LogisticRegression(C=1.0,   penalty='l2', solver='lbfgs', max_iter=5000, tol=1e-4).fit(X_train_sc, y_train),
    'L2 C=100  (weak reg)':    LogisticRegression(C=100.0, penalty='l2', solver='lbfgs', max_iter=5000, tol=1e-4).fit(X_train_sc, y_train),
    'L1 C=1.0  (sparse)':      LogisticRegression(C=1.0,   penalty='l1', solver='liblinear', max_iter=5000, tol=1e-4).fit(X_train_sc, y_train),
}

# Print AUC for each model
for name, model in models_compare.items():
    auc = roc_auc_score(y_test, model.predict_proba(X_test_sc)[:, 1])
    print(f"  {name:<35s}: AUC-ROC = {auc:.4f}")

# Create the comparison figure
fig = plot_model_diagnostics(
    models_compare,
    X_test_sc, y_test,
    title="Breast Cancer: ROC, PR, and Calibration by Regularization Strength"
)

plt.savefig("roc_pr_calibration.png", dpi=150, bbox_inches='tight')
# plt.show()