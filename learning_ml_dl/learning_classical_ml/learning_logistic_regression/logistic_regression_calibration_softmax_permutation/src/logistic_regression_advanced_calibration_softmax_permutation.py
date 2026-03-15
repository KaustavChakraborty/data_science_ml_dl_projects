# ============================================================
# CODE: ADVANCED TOPICS
# ============================================================
# This script demonstrates several interview-level and practical
# extensions of logistic regression beyond the basic binary case.
#
# Topics covered:
#   1. Probability Calibration
#   2. Multinomial (Softmax) Logistic Regression
#   3. Regularization path (how coefficients evolve with C)
#   4. Feature importance via permutation
#   5. Interview-level edge cases
# ============================================================


# ------------------------------------------------------------
# Import core numerical and data handling libraries
# ------------------------------------------------------------
import numpy as np                 # numerical arrays, vectorized operations
import pandas as pd                # tabular display of coefficients/results
import matplotlib.pyplot as plt    # plotting


# ------------------------------------------------------------
# Import Logistic Regression model from scikit-learn
# ------------------------------------------------------------
from sklearn.linear_model import LogisticRegression


# ------------------------------------------------------------
# Import preprocessing tools
# ------------------------------------------------------------
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------
# Import calibration utilities
# ------------------------------------------------------------
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
# CalibratedClassifierCV:
#   Wraps an estimator and calibrates predicted probabilities using either:
#     - 'sigmoid'  -> Platt scaling
#     - 'isotonic' -> isotonic regression
#
# CalibrationDisplay:
#   Used to plot reliability diagrams / calibration curves.


# ------------------------------------------------------------
# Import train/test split and cross-validation helper
# ------------------------------------------------------------
from sklearn.model_selection import train_test_split, cross_val_score


# ------------------------------------------------------------
# Import evaluation metrics
# ------------------------------------------------------------
from sklearn.metrics import (
    roc_auc_score,     # quality of ranking/class separation
    brier_score_loss,  # calibration-sensitive probability error metric
    log_loss,          # penalizes bad probability estimates strongly
    accuracy_score     # fraction of correctly classified samples
)


# ------------------------------------------------------------
# Import built-in datasets
# ------------------------------------------------------------
from sklearn.datasets import load_iris, load_breast_cancer, make_classification
# load_iris:
#   Classic 3-class dataset for multinomial logistic regression demo.
#
# load_breast_cancer:
#   Binary classification dataset used here for calibration,
#   regularization path, and permutation importance.


# ------------------------------------------------------------
# Import model inspection utility
# ------------------------------------------------------------
from sklearn.inspection import permutation_importance
# Permutation importance:
#   Measures how much predictive performance drops if one feature
#   is randomly shuffled. Large drop = model relied strongly on it.

# ------------------------------------------------------------
# Silence warnings for cleaner teaching/demo output
# ------------------------------------------------------------
import warnings
warnings.filterwarnings('ignore')


# ------------------------------------------------------------
# Fixed random seed for reproducibility
# ------------------------------------------------------------
SEED = 42


# ═══════════════════════════════════════════════════════════════════════════════
# TOPIC 1: Calibration — Does P(ŷ=0.7) really mean 70% probability?
# ═══════════════════════════════════════════════════════════════════════════════
#
# Main idea:
#   A classifier may rank examples well (high AUC) and still output poorly
#   calibrated probabilities.
#
# Example:
#   If a model says "0.7 probability" for many samples, then among those
#   samples roughly 70% should truly belong to class 1 if the model is
#   well calibrated.
#
# Logistic regression is often reasonably calibrated compared with many
# other classifiers, but regularization and model misspecification can
# still affect probability quality.
# ═══════════════════════════════════════════════════════════════════════════════
print("="*60)
print("TOPIC 1: PROBABILITY CALIBRATION")
print("="*60)


# ------------------------------------------------------------
# Load binary classification dataset
# ------------------------------------------------------------
data  = load_breast_cancer()
X, y  = data.data, data.target
# X: feature matrix of shape (n_samples, n_features)
# y: binary target vector (0/1)


# ------------------------------------------------------------
# Split into train and test sets
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,        # 25% test set, 75% training set
    random_state=SEED,     # reproducible split
    stratify=y             # preserve class ratio in both splits
)


# ------------------------------------------------------------
# Standardize features
# ------------------------------------------------------------
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_train)
X_te_sc = scaler.transform(X_test)


# ------------------------------------------------------------
# Define several models to compare calibration behavior
# ------------------------------------------------------------
models_cal = {
    'LR (well-calibrated)':
        LogisticRegression(
            C=1.0,
            solver='lbfgs',
            max_iter=1000,
            random_state=SEED
        ),

    'LR (overly confident, small C)':
        LogisticRegression(
            C=0.001,
            solver='lbfgs',
            max_iter=1000,
            random_state=SEED
        ),

    'LR + Platt scaling':
        CalibratedClassifierCV(
            LogisticRegression(
                C=0.001,
                solver='lbfgs',
                max_iter=1000,
                random_state=SEED
            ),
            method='sigmoid',  # Platt scaling = fit sigmoid on top of raw scores
            cv=5               # internal 5-fold CV for calibration
        ),

    'LR + Isotonic regression':
        CalibratedClassifierCV(
            LogisticRegression(
                C=0.001,
                solver='lbfgs',
                max_iter=1000,
                random_state=SEED
            ),
            method='isotonic', # non-parametric monotone calibration mapping
            cv=5               # internal 5-fold CV for calibration
        ),
}

# Interpretation of the model choices:
#   - Standard LR at C=1.0 acts as a reference model.
#   - Very small C means stronger regularization, which can distort score
#     magnitudes and sometimes probability confidence behavior.
#   - Platt scaling learns a sigmoid correction.
#   - Isotonic regression learns a flexible monotonic correction.


# ------------------------------------------------------------
# Fit each calibration model
# ------------------------------------------------------------
for name, model in models_cal.items():
    model.fit(X_tr_sc, y_train)


# ------------------------------------------------------------
# Evaluate probability quality and ranking quality
# ------------------------------------------------------------
print("\nCalibration Metrics (lower Brier = better probability estimates):")
for name, model in models_cal.items():
    y_prob = model.predict_proba(X_te_sc)[:, 1]
    # predict_proba returns probabilities for both classes [P(y=0), P(y=1)]
    # [:, 1] selects probability of the positive class.

    bs     = brier_score_loss(y_test, y_prob)
    # Brier score = mean squared error of predicted probabilities.
    # Lower is better. Sensitive to calibration and refinement.

    ll     = log_loss(y_test, y_prob)
    # Log loss heavily penalizes confident wrong predictions.

    auc    = roc_auc_score(y_test, y_prob)
    # AUC evaluates ranking/separation ability, not calibration directly.

    print(f"  {name:<45s}: Brier={bs:.4f}, LogLoss={ll:.4f}, AUC={auc:.4f}")


# ------------------------------------------------------------
# Plot calibration curves and predicted-probability histograms
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Two panels:
#   axes[0] -> reliability diagram / calibration curve
#   axes[1] -> histogram of predicted probabilities


# ------------------------------------------------------------
# Reliability diagrams
# ------------------------------------------------------------
for name, model in models_cal.items():
    CalibrationDisplay.from_estimator(
        model,
        X_te_sc,
        y_test,
        n_bins=10,   # group predictions into 10 probability bins
        ax=axes[0],
        name=name
    )

axes[0].set_title("Calibration Curves (Reliability Diagrams)\n"
                  "Perfect calibration = diagonal line")
axes[0].grid(True, alpha=0.3)

# Interpretation:
#   - Perfectly calibrated model would lie on the diagonal y=x.
#   - Curves below diagonal often indicate overconfidence.
#   - Curves above diagonal often indicate underconfidence.


# ------------------------------------------------------------
# Histogram of predicted probabilities
# ------------------------------------------------------------
for (name, model), color in zip(
        models_cal.items(),
        ['#1F77B4','#FF7F0E','#2CA02C','#D62728']):
    y_prob = model.predict_proba(X_te_sc)[:, 1]
    axes[1].hist(
        y_prob,
        bins=30,
        alpha=0.5,
        color=color,
        label=name,
        density=True
    )

axes[1].set_xlabel("Predicted Probability P(y=1|x)")
axes[1].set_ylabel("Density")
axes[1].set_title("Distribution of Predicted Probabilities\n"
                  "(Overconfident = bimodal near 0 and 1)")
axes[1].legend(fontsize=7)

# Interpretation:
#   - Very sharp mass near 0 and 1 can indicate high confidence predictions.
#   - Whether that confidence is justified is answered by calibration curves
#     and metrics like Brier score / log loss.


# ------------------------------------------------------------
# Final layout, save, display
# ------------------------------------------------------------
plt.tight_layout()
plt.savefig("calibration.png", dpi=150, bbox_inches='tight')
# plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# TOPIC 2: Multinomial Logistic Regression (Softmax) — Iris 3-class
# ═══════════════════════════════════════════════════════════════════════════════
#
# Main idea:
#   Binary logistic regression handles 2 classes.
#   For K > 2 classes, there are two common strategies:
#
#   1. One-vs-Rest (OvR):
#      Train K separate binary classifiers.
#
#   2. Multinomial / Softmax logistic regression:
#      Learn all class scores jointly and convert them into probabilities
#      using the softmax function.
#
# Here we compare both on the Iris 3-class dataset.
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TOPIC 2: MULTINOMIAL LOGISTIC REGRESSION (SOFTMAX)")
print("="*60)


# ------------------------------------------------------------
# Load Iris data
# ------------------------------------------------------------
iris = load_iris()
X_i, y_i = iris.data, iris.target
# X_i has 4 features:
#   sepal length, sepal width, petal length, petal width
# y_i has 3 classes:
#   setosa, versicolor, virginica


# ------------------------------------------------------------
# Train/test split
# ------------------------------------------------------------
X_tr_i, X_te_i, y_tr_i, y_te_i = train_test_split(
    X_i, y_i,
    test_size=0.25,
    random_state=SEED,
    stratify=y_i
)


# ------------------------------------------------------------
# Standardize features
# ------------------------------------------------------------
sc_i   = StandardScaler()
Xtr_sc = sc_i.fit_transform(X_tr_i)
Xte_sc = sc_i.transform(X_te_i)


# ------------------------------------------------------------
# Compare multinomial softmax vs one-vs-rest
# ------------------------------------------------------------
# multi_class='multinomial' -> true softmax model
# multi_class='ovr'         -> trains one binary classifier per class
for mc in ['multinomial', 'ovr']:
    clf = LogisticRegression(
        multi_class=mc,
        solver='lbfgs' if mc == 'multinomial' else 'liblinear',
        # lbfgs supports multinomial optimization directly
        # liblinear is commonly used for OvR binary problems
        C=1.0,
        max_iter=1000,
        random_state=SEED
    )

    clf.fit(Xtr_sc, y_tr_i)
    acc = accuracy_score(y_te_i, clf.predict(Xte_sc))
    print(f"  {mc:<15s}: Test Accuracy = {acc:.4f}")


# ------------------------------------------------------------
# Fit a multinomial logistic regression model explicitly
# ------------------------------------------------------------
clf_mn = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    C=1.0,
    max_iter=1000,
    random_state=SEED
)
clf_mn.fit(Xtr_sc, y_tr_i)


# ------------------------------------------------------------
# Examine learned coefficient structure
# ------------------------------------------------------------
print(f"\nCoefficient matrix shape: {clf_mn.coef_.shape}  (K * p = 3 * 4)")
print("Each row = one class's weight vector \beta_k")

coef_df_iris = pd.DataFrame(
    clf_mn.coef_,
    columns=iris.feature_names,
    index=[f'Class {k}: {iris.target_names[k]}' for k in range(3)]
)

print(coef_df_iris.round(4))

# In multinomial logistic regression:
#   - coef_ has one row per class.
#   - Each row tells how strongly each feature contributes to that class score.
#   - Final class probabilities are produced via softmax over class scores.


# ------------------------------------------------------------
# Visualize softmax probabilities over a 2D slice of feature space
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# Define bounds using first two standardized test features
x_min, x_max = Xte_sc[:,0].min()-0.5, Xte_sc[:,0].max()+0.5
y_min, y_max = Xte_sc[:,1].min()-0.5, Xte_sc[:,1].max()+0.5

# Create a dense grid over the first two features
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

# Build full 4D input vectors from the 2D grid:
#   - feature 0 = xx
#   - feature 1 = yy
#   - feature 2 = 0
#   - feature 3 = 0
#
# This means we are visualizing a 2D slice through the full 4D feature space,
# fixing the last two standardized features at zero.
grid_full = np.column_stack([
    xx.ravel(),
    yy.ravel(),
    np.zeros(xx.size),
    np.zeros(xx.size)
])

# Predict softmax probabilities over this entire grid
proba_grid = clf_mn.predict_proba(grid_full)


# ------------------------------------------------------------
# Plot one probability surface per class
# ------------------------------------------------------------
for k, (ax, class_name) in enumerate(zip(axes, iris.target_names)):
    Z = proba_grid[:, k].reshape(xx.shape)
    # Z is the predicted probability of class k across the grid.

    cf = ax.contourf(xx, yy, Z, levels=20, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(cf, ax=ax, label=f'P(class={class_name})')

    colors_k = ['#2196F3','#4CAF50','#F44336']
    for ci, color in zip(range(3), colors_k):
        mask = (y_te_i == ci)
        ax.scatter(
            Xte_sc[mask,0],
            Xte_sc[mask,1],
            c=color,
            edgecolors='white',
            lw=0.5,
            s=50,
            alpha=0.8,
            label=iris.target_names[ci]
        )

    ax.set_title(f"P(y = {class_name} | x)\nSoftmax output")
    ax.set_xlabel("Feature 1 (standardized)")
    ax.set_ylabel("Feature 2 (standardized)")

    if k == 0:
        ax.legend(fontsize=8)

# Interpretation:
#   - Each panel shows the model's probability assigned to one class.
#   - Because probabilities come from softmax, they compete with one another;
#     raising one class's probability lowers others.
#   - The scatter points help relate probability regions to real test samples.


# ------------------------------------------------------------
# Final layout, save, display
# ------------------------------------------------------------
plt.suptitle("Multinomial Logistic Regression: Softmax Probabilities per Class",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("multinomial_softmax.png", dpi=150, bbox_inches='tight')
# plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# TOPIC 3: Regularization Path — How Coefficients Change with λ
# ═══════════════════════════════════════════════════════════════════════════════
#
# Main idea:
#   Logistic regression coefficients depend on regularization strength.
#
# Recall:
#   In scikit-learn, C = 1 / λ
#
# Therefore:
#   - small C  -> strong regularization
#   - large C  -> weak regularization
#
# This section sweeps across many C values and records how each coefficient
# changes for:
#   - L2 regularization (Ridge-like shrinkage)
#   - L1 regularization (Lasso-like sparsity / exact zeros)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TOPIC 3: REGULARIZATION PATH")
print("="*60)
print("How each β_j shrinks as we increase regularization strength (decrease C)")


# ------------------------------------------------------------
# Use only first 8 breast cancer features for a cleaner path plot
# ------------------------------------------------------------
X_p, y_p = load_breast_cancer().data[:, :8], load_breast_cancer().target
feature_names_path = list(load_breast_cancer().feature_names[:8])
# Restricting to 8 features keeps the plot readable while still showing
# coefficient shrinkage behavior clearly.


# ------------------------------------------------------------
# Standardize features before regularized regression
# ------------------------------------------------------------
scaler_p = StandardScaler()
X_p_sc   = scaler_p.fit_transform(X_p)


# ------------------------------------------------------------
# Define a range of inverse regularization strengths
# ------------------------------------------------------------
C_values = np.logspace(-4, 3, 80)
# This creates 80 logarithmically spaced values from 10^-4 to 10^3.
# That gives a wide sweep from very strong to very weak regularization.

coefs_l2 = []
coefs_l1 = []
# These lists will store the learned coefficient vectors for each C.


# ------------------------------------------------------------
# Fit L2 and L1 logistic regression at each C
# ------------------------------------------------------------
for C in C_values:
    m_l2 = LogisticRegression(
        C=C,
        penalty='l2',
        solver='lbfgs',
        max_iter=2000
    ).fit(X_p_sc, y_p)

    m_l1 = LogisticRegression(
        C=C,
        penalty='l1',
        solver='liblinear',
        max_iter=2000
    ).fit(X_p_sc, y_p)

    coefs_l2.append(m_l2.coef_[0])
    coefs_l1.append(m_l1.coef_[0])

# Why different solvers?
#   - lbfgs supports L2 efficiently
#   - liblinear supports L1 for logistic regression in scikit-learn


# ------------------------------------------------------------
# Convert coefficient lists to arrays
# ------------------------------------------------------------
coefs_l2 = np.array(coefs_l2)   # shape: (80, 8)
coefs_l1 = np.array(coefs_l1)
# Each row corresponds to one C value.
# Each column corresponds to one feature coefficient.


# ------------------------------------------------------------
# Plot coefficient paths for L2 and L1
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
colors_path = plt.cm.tab10(np.linspace(0, 1, 8))
# Use distinct colors for the 8 feature trajectories.


# ------------------------------------------------------------
# Loop through regularization types and plot trajectories
# ------------------------------------------------------------
for k, (name, coefs) in enumerate([
    ('L2 (Ridge)', coefs_l2),
    ('L1 (Lasso)', coefs_l1)
]):
    ax = axes[k]

    for j, (feat, color) in enumerate(zip(feature_names_path, colors_path)):
        ax.semilogx(
            C_values,
            coefs[:, j],
            color=color,
            lw=2,
            label=feat
        )

    ax.axvline(
        1.0,
        color='black',
        lw=1,
        linestyle=':',
        label='C=1 (default)'
    )

    ax.set_xlabel("C = 1/λ (Inverse Regularization Strength)\n← Stronger regularization | Weaker regularization →")
    ax.set_ylabel("Coefficient Value βⱼ")
    ax.set_title(f"Regularization Path: {name}\n"
                 f"({'Smooth shrinkage' if k==0 else 'Exact zeros — feature selection'})")
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', lw=0.5)

# Interpretation:
#   L2:
#     - coefficients shrink smoothly toward zero
#     - rarely become exactly zero
#
#   L1:
#     - coefficients can become exactly zero
#     - acts like embedded feature selection
#     - path often has "kinks" where variables enter/leave the model


# ------------------------------------------------------------
# Final layout, save, display
# ------------------------------------------------------------
plt.suptitle("Regularization Paths: How Coefficients Evolve with Regularization Strength",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("regularization_path.png", dpi=150, bbox_inches='tight')
# plt.show()


# ------------------------------------------------------------
# Print summary notes for interpretation
# ------------------------------------------------------------
print("\nNote:")
print("  L2: all coefficients shrink continuously, never exactly 0")
print("  L1: coefficients hit zero at different C values (kink points)")
print("  At small C, only the most predictive features survive in L1")



# ═══════════════════════════════════════════════════════════════════════════════
# TOPIC 4: Permutation Feature Importance
# ═══════════════════════════════════════════════════════════════════════════════
#
# Main idea:
#   Raw coefficient magnitude is not always a reliable measure of feature
#   importance, especially when:
#     - features are correlated
#     - scales differ
#     - regularization redistributes weight among correlated predictors
#
# Permutation importance estimates importance more directly:
#   1. Measure model performance on test data.
#   2. Shuffle one feature column randomly.
#   3. Recompute performance.
#   4. If performance drops a lot, that feature mattered.
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TOPIC 4: PERMUTATION FEATURE IMPORTANCE")
print("="*60)
print("More reliable than coefficient magnitude (especially with correlated features)")


# ------------------------------------------------------------
# Load data again for a fresh train/test split
# ------------------------------------------------------------
X_bc, y_bc = load_breast_cancer().data, load_breast_cancer().target

X_tr_bc, X_te_bc, y_tr_bc, y_te_bc = train_test_split(
    X_bc, y_bc,
    test_size=0.25,
    random_state=SEED,
    stratify=y_bc
)


# ------------------------------------------------------------
# Standardize training and test sets
# ------------------------------------------------------------
sc_bc = StandardScaler()

clf_bc = LogisticRegression(
    C=1.0,
    solver='lbfgs',
    max_iter=1000,
    random_state=SEED
)

clf_bc.fit(sc_bc.fit_transform(X_tr_bc), y_tr_bc)
# Fit scaler on training data, transform training data, then fit logistic model.


# ------------------------------------------------------------
# Compute permutation importance on the test set
# ------------------------------------------------------------
perm_imp = permutation_importance(
    clf_bc,
    sc_bc.transform(X_te_bc),  # test features transformed using training scaler
    y_te_bc,
    n_repeats=30,              # repeat shuffling 30 times for stability
    random_state=SEED,
    scoring='roc_auc'          # importance measured as decrease in AUC
)

# Output fields of perm_imp include:
#   importances_mean : mean performance drop across repeats
#   importances_std  : variability across repeats
#   importances      : raw per-repeat importance values


# ------------------------------------------------------------
# Sort features by mean importance and keep top 15
# ------------------------------------------------------------
sort_idx = perm_imp.importances_mean.argsort()[::-1][:15]
feat_names = load_breast_cancer().feature_names
# argsort()[::-1] sorts descending, so most important features come first.


# ------------------------------------------------------------
# Plot boxplots of importance distributions
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

ax.boxplot(
    perm_imp.importances[sort_idx].T,
    vert=False,
    labels=[feat_names[i] for i in sort_idx]
)

ax.axvline(0, color='red', lw=1, linestyle='--')
ax.set_xlabel("Decrease in AUC-ROC when feature is permuted\n(Higher = more important)")
ax.set_title("Permutation Feature Importance (Top 15)\n"
             "Breast Cancer Classification")
ax.grid(True, alpha=0.3, axis='x')

# Interpretation:
#   - Larger positive values mean shuffling that feature hurts model performance,
#     so the model depended on it.
#   - If importance is near zero, the feature contributes little unique signal.
#   - Negative values can occasionally appear due to randomness/noise, meaning
#     shuffling accidentally improved performance in some repeats.


# ------------------------------------------------------------
# Final layout, save, display
# ------------------------------------------------------------
plt.tight_layout()
plt.savefig("permutation_importance.png", dpi=150, bbox_inches='tight')
# plt.show()
