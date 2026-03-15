# ============================================================
# CODE: HANDLING CLASS IMBALANCE
# ============================================================
# This script demonstrates multiple standard strategies for dealing
# with imbalanced binary classification problems.
#
# Imbalanced classification means one class is much more common
# than the other. For example:
#   - rare phase detection in soft matter
#   - anomaly detection
#   - rare material failure prediction
#   - uncommon crystallographic phase classification
#
# In such problems, a model can achieve high accuracy simply by
# predicting the majority class most of the time, so accuracy alone
# becomes misleading.
#
# Methods covered in this script:
#   1. Baseline (no imbalance handling)
#   2. class_weight='balanced'
#   3. SMOTE oversampling
#   4. Undersampling (mentioned in header, though not explicitly used below)
#   5. SMOTE + Undersampling combined (mentioned in header, though not explicitly used below)
#   6. Threshold tuning
#   7. Visualization of performance under different strategies
#
# The overall workflow is:
#   - generate a synthetic imbalanced dataset
#   - split into train and test sets
#   - standardize features
#   - train several logistic regression variants
#   - evaluate them using metrics suitable for imbalance
#   - visualize metric comparisons and threshold tradeoffs
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# LogisticRegression is the classification model used throughout.
from sklearn.linear_model import LogisticRegression

# StandardScaler standardizes each feature to zero mean and unit variance.
# This is helpful for logistic regression because optimization becomes
# more stable and coefficient scales become comparable.
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    precision_recall_curve, confusion_matrix, classification_report,
    matthews_corrcoef, ConfusionMatrixDisplay
)

# make_classification creates a synthetic classification dataset.
# It is very useful for controlled demonstrations of ML behavior.
from sklearn.datasets import make_classification

# Standard sklearn Pipeline is imported but not used below.
from sklearn.pipeline import Pipeline

from sklearn.metrics import f1_score, precision_score, recall_score

# The following block tries to import tools from imbalanced-learn.
# If the package is missing, the code will still run for baseline
# and class_weight methods, but SMOTE-based methods will be skipped.
try:
    # SMOTE: synthetic minority oversampling
    from imblearn.over_sampling import SMOTE

    # RandomUnderSampler: random downsampling of majority class
    from imblearn.under_sampling import RandomUnderSampler

    # SMOTETomek: combine oversampling with Tomek link cleanup
    from imblearn.combine import SMOTETomek

    # ImbPipeline: pipeline compatible with imbalance-resampling workflows
    from imblearn.pipeline import Pipeline as ImbPipeline

    HAS_IMBALANCED = True
except ImportError:
    print("Install imbalanced-learn: pip install imbalanced-learn")
    HAS_IMBALANCED = False

# Reproducibility seed.
# Using a fixed random seed ensures the data split, synthetic data,
# SMOTE sampling, and logistic regression randomness are repeatable.
SEED = 42


# ─────────────────────────────────────────────────────────────
# Generate a challenging imbalanced dataset
# ─────────────────────────────────────────────────────────────
# make_classification creates a binary dataset with:
#   - 2000 total samples
#   - 10 total features
#   - 5 informative features (carry real signal)
#   - 2 redundant features (linear combinations of informative ones)
#   - class weights [0.90, 0.10], meaning:
#         class 0 ≈ 90%
#         class 1 ≈ 10%
#   - flip_y=0.01 adds 1% label noise
#
# This simulates a realistic imbalanced classification setting where
# the minority class is rare and the data is not perfectly clean.
X, y = make_classification(
    n_samples=2000, n_features=10, n_informative=5, n_redundant=2,
    weights=[0.90, 0.10],   # 90% class 0, 10% class 1
    flip_y=0.01, random_state=SEED
)

# Show the raw class counts.
# np.bincount(y) returns [count_class_0, count_class_1].
print(f"Class distribution: {np.bincount(y)}")

# y.mean() gives the fraction of class-1 labels because labels are 0/1.
print(f"Minority fraction: {y.mean():.3f}")

# Split into training and testing sets.
# test_size=0.25 means:
#   - 75% training data
#   - 25% testing data
#
# stratify=y is very important in imbalanced classification.
# It preserves approximately the same class ratio in both train and test sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=SEED, stratify=y
)

# Create a feature scaler.
scaler = StandardScaler()

# Fit the scaler ONLY on training data and transform the training features.
# This prevents data leakage from the test set into preprocessing.
X_train_sc = scaler.fit_transform(X_train)

# Apply the same learned scaling transformation to the test set.
X_test_sc  = scaler.transform(X_test)


# ─────────────────────────────────────────────────────────────
# Evaluation function for imbalanced problems
# ─────────────────────────────────────────────────────────────
def evaluate_imbalanced(model, X_tr, X_te, y_tr, y_te,
                         method_name, threshold=0.5):
    """
    Reports metrics appropriate for imbalanced classification.
    Does NOT primarily use accuracy (misleading for imbalanced data).

    Parameters
    ----------
    model : fitted classifier
        Must support predict_proba().
    X_tr : array-like
        Training features (passed in for interface consistency, though not used here).
    X_te : array-like
        Test features on which predictions are made.
    y_tr : array-like
        Training labels (passed in for interface consistency, though not used here).
    y_te : array-like
        True test labels.
    method_name : str
        Label describing the strategy being evaluated.
    threshold : float, default=0.5
        Probability cutoff used to convert predicted probabilities
        into class labels.

    Returns
    -------
    results : dict
        Dictionary containing all computed metrics.
    """

    # Predict probability of the positive class (class 1).
    # For binary classifiers, predict_proba(X) returns shape (n_samples, 2):
    #   column 0 -> P(class 0)
    #   column 1 -> P(class 1)
    y_prob = model.predict_proba(X_te)[:, 1]

    # Convert probabilities into hard class predictions using the chosen threshold.
    # Default threshold in many classifiers is 0.5, but in imbalanced settings,
    # tuning the threshold can substantially improve recall/F1.
    y_pred = (y_prob >= threshold).astype(int)

    # Compute confusion matrix entries.
    # For binary classification, confusion_matrix(...).ravel() gives:
    #   tn = true negatives
    #   fp = false positives
    #   fn = false negatives
    #   tp = true positives
    tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()

    # Build a dictionary of evaluation metrics.
    results = {
        'Method':    method_name,
        'Threshold': threshold,

        # Accuracy = (TP + TN) / total
        # This can look artificially high in imbalanced problems because
        # the majority class dominates the count.
        'Accuracy':  (tp+tn)/(tp+tn+fp+fn),

        # Precision = TP / (TP + FP)
        # Of all predicted positives, how many were actually positive?
        'Precision': tp/(tp+fp) if (tp+fp)>0 else 0,

        # Recall = TP / (TP + FN)
        # Of all actual positives, how many were recovered?
        'Recall':    tp/(tp+fn) if (tp+fn)>0 else 0,

        # F1 = harmonic mean of precision and recall
        # Useful when balancing both false positives and false negatives matters.
        'F1':        f1_score(y_te, y_pred, zero_division=0),

        # MCC = Matthews Correlation Coefficient
        # A very strong single-number metric for imbalanced binary classification.
        # It accounts for TP, TN, FP, FN in a balanced way.
        'MCC':       matthews_corrcoef(y_te, y_pred),

        # ROC-AUC measures ranking quality across thresholds.
        # Can still look optimistic in strongly imbalanced datasets.
        'AUC-ROC':   roc_auc_score(y_te, y_prob),

        # PR-AUC / Average Precision is often more informative for imbalanced problems.
        # It emphasizes performance on the positive class.
        'AUC-PR':    average_precision_score(y_te, y_prob),
    }

    # Print a readable summary of the results.
    print(f"\n{method_name} (threshold={threshold:.2f}):")
    print(f"  Accuracy:  {results['Accuracy']:.4f}  ← misleading!")
    print(f"  Precision: {results['Precision']:.4f}")
    print(f"  Recall:    {results['Recall']:.4f}")
    print(f"  F1:        {results['F1']:.4f}")
    print(f"  MCC:       {results['MCC']:.4f}  ← best single metric")
    print(f"  AUC-ROC:   {results['AUC-ROC']:.4f}")
    print(f"  AUC-PR:    {results['AUC-PR']:.4f}  ← best for imbalanced")

    return results


# ── Strategy 1: Baseline (no handling) ─────────────────────────────────────────
# This is the reference model.
# No class reweighting, no resampling, no threshold tuning.
# It tells us how plain logistic regression behaves on the raw imbalanced data.
print("=" * 60)
print("STRATEGY 1: Baseline — No Imbalance Handling")
print("=" * 60)

# Standard logistic regression:
#   C=1.0            -> inverse regularization strength
#   solver='lbfgs'   -> optimization algorithm
#   max_iter=1000    -> enough iterations to ensure convergence
#   random_state     -> reproducibility
model_base = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=SEED)

# Fit on the scaled training data.
model_base.fit(X_train_sc, y_train)

# Evaluate on the scaled test data using default threshold 0.5.
res1 = evaluate_imbalanced(model_base, X_train_sc, X_test_sc,
                             y_train, y_test, "Baseline")


# ── Strategy 2: class_weight='balanced' ─────────────────────────────────────────
# This strategy modifies the logistic regression loss so that mistakes
# on minority-class samples are penalized more heavily.
#
# sklearn computes class weights roughly as:
#   w_i = n_samples / (n_classes * n_samples_in_class_i)
#
# Therefore the minority class receives a larger weight, forcing the
# optimizer to pay more attention to it.
print("\n" + "=" * 60)
print("STRATEGY 2: class_weight='balanced'")
print("="*60)

model_cw = LogisticRegression(
    C=1.0, solver='lbfgs', max_iter=1000,
    class_weight='balanced',    # Key modification for imbalance-aware training
    random_state=SEED
)

# Fit the weighted model.
model_cw.fit(X_train_sc, y_train)

# Evaluate the weighted model using threshold 0.5.
res2 = evaluate_imbalanced(model_cw, X_train_sc, X_test_sc,
                            y_train, y_test, "class_weight=balanced")



# ── Strategy 3: SMOTE Oversampling ─────────────────────────────────────────────
# This block runs only if imbalanced-learn is installed.
#
# SMOTE = Synthetic Minority Over-sampling Technique
#
# Instead of merely duplicating minority samples, SMOTE creates new
# synthetic minority points by interpolating between neighboring
# minority-class examples.
#
# Conceptually:
#   1. Take a minority sample x_i
#   2. Find one of its minority nearest neighbors x_j
#   3. Generate:
#         x_new = x_i + δ (x_j - x_i),   where δ ~ Uniform(0,1)
#
# This helps the classifier see a larger and smoother minority region.
if HAS_IMBALANCED:
    print("\n" + "=" * 60)
    print("STRATEGY 3: SMOTE Oversampling")
    print("=" * 60)

    smote = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=SEED)

    # fit_resample() learns the minority neighborhood structure from the
    # training set and returns a resampled balanced training dataset.
    X_train_smote, y_train_smote = smote.fit_resample(X_train_sc, y_train)

    # Show class counts after SMOTE.
    # Typically this balances the minority class up to the majority level.
    print(f"  After SMOTE: {np.bincount(y_train_smote)}")

    # Train logistic regression on the resampled dataset.
    model_smote = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=SEED)
    model_smote.fit(X_train_smote, y_train_smote)

    # Important:
    # evaluation is still done on the original untouched test set.
    # Resampling should only ever be applied to training data.
    res3 = evaluate_imbalanced(model_smote, X_train_sc, X_test_sc,
                                y_train, y_test, "SMOTE Oversampling")


# ── Strategy 4: Threshold Optimization ─────────────────────────────────────────
# Instead of retraining a new model, this strategy keeps the
# class_weight='balanced' model and changes only the decision threshold.
#
# Default binary decision rule:
#     predict class 1 if P(class 1) >= 0.5
#
# But in imbalanced problems, 0.5 is often not the best threshold.
# Lowering the threshold usually increases recall.
# Raising the threshold usually increases precision.
#
# Here, the code searches for the threshold that maximizes F1 score.
print("\n" + "=" * 60)
print("STRATEGY 4: Threshold Optimization on class_weight model")
print("=" * 60)

# Predicted probabilities from the class_weight-balanced model.
y_prob_cw = model_cw.predict_proba(X_test_sc)[:, 1]

# Compute the precision-recall curve over many thresholds.
# Returns:
#   precisions : precision values
#   recalls    : recall values
#   thresholds : thresholds used to obtain those values
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob_cw)

# Compute F1 score at each threshold using:
#   F1 = 2PR / (P + R)
# Small epsilon avoids division by zero.
f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)

# Identify the threshold index that gives the largest F1.
best_thresh_idx = np.argmax(f1_scores)

# precision_recall_curve returns one fewer threshold than precision/recall points,
# so this conditional safely handles the edge case.
best_threshold  = thresholds[best_thresh_idx] if best_thresh_idx < len(thresholds) else 0.5

print(f"  Optimal threshold: {best_threshold:.4f}")
print(f"  Max F1 at threshold: {f1_scores[best_thresh_idx]:.4f}")

# Re-evaluate the same class_weight model, but now using the optimized threshold.
res4 = evaluate_imbalanced(model_cw, X_train_sc, X_test_sc,
                            y_train, y_test, "Optimal Threshold",
                            threshold=best_threshold)


# ── Comprehensive Comparison Plot ────────────────────────────────────────────
# Collect all result dictionaries into one list for easy plotting.
all_results = [res1, res2, res4]
if HAS_IMBALANCED:
    all_results.append(res3)

# Create a figure with 3 side-by-side subplots.
# Each subplot compares one metric across methods.
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# Extract method names for x-axis labels.
methods = [r['Method'] for r in all_results]

# Metrics chosen for comparison:
#   F1     -> balance of precision/recall
#   MCC    -> strong balanced scalar metric
#   AUC-PR -> especially informative for imbalanced classification
metric_pairs = [
    ('F1', 'F1 Score'),
    ('MCC', 'Matthews Correlation Coefficient'),
    ('AUC-PR', 'AUC-PR (Best for Imbalanced Data)'),
]

# Predefined colors for the bars.
# Only the first len(methods) colors are used.
colors = ['#EF5350','#42A5F5','#66BB6A','#FFA726'][:len(methods)]

# Loop through each subplot and plot one metric.
for ax, (metric_key, metric_label) in zip(axes, metric_pairs):
    # Gather the metric values across strategies.
    values = [r[metric_key] for r in all_results]

    # Create bar chart.
    bars = ax.bar(range(len(methods)), values, color=colors, alpha=0.85,
                   edgecolor='white', linewidth=1.5)

    # Set x-tick positions and labels.
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=25, ha='right', fontsize=9)

    # Label axes and title.
    ax.set_ylabel(metric_label)
    ax.set_title(metric_label)

    # Force the y-axis to run from 0 to a bit above 1 for clarity.
    ax.set_ylim(0, 1.05)

    # Add horizontal gridlines for easier visual comparison.
    ax.grid(True, alpha=0.3, axis='y')

    # Add numerical value labels above each bar.
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Add a global title across the whole figure.
fig.suptitle("Class Imbalance Handling Strategies — Comparison\n"
             "(90% Class 0, 10% Class 1)", fontsize=12, fontweight='bold')

# Adjust subplot spacing to reduce overlap.
plt.tight_layout()

# Save the figure to disk.
plt.savefig("imbalance_comparison.png", dpi=150, bbox_inches='tight')

# Show the figure.
# plt.show()


# ── Threshold sweep visualization ────────────────────────────────────────────
# This section visualizes how the classification threshold affects:
#   - F1
#   - Precision
#   - Recall
#
# It also plots the Precision-Recall curve directly.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# F1 vs Threshold
# Generate a dense grid of thresholds from 0.01 to 0.99.
thresholds_plot = np.linspace(0.01, 0.99, 200)

# For each threshold t:
#   - classify probability >= t as positive
#   - compute F1
#
# NOTE:
# precision_score and recall_score are used below exactly as in the
# original code, but they were not imported at the top.
f1_vals   = [f1_score(y_test, (y_prob_cw >= t).astype(int), zero_division=0) for t in thresholds_plot]
prec_vals = [precision_score(y_test, (y_prob_cw >= t).astype(int), zero_division=0) for t in thresholds_plot]
rec_vals  = [recall_score(y_test, (y_prob_cw >= t).astype(int), zero_division=0) for t in thresholds_plot]

# Plot F1 as a function of threshold.
axes[0].plot(thresholds_plot, f1_vals,   lw=2, label='F1',        color='#1F77B4')

# Plot precision and recall curves on the same axis.
axes[0].plot(thresholds_plot, prec_vals, lw=2, label='Precision',  color='#2CA02C', linestyle='--')
axes[0].plot(thresholds_plot, rec_vals,  lw=2, label='Recall',     color='#D62728', linestyle='--')

# Mark the best threshold found earlier.
axes[0].axvline(best_threshold, color='black', lw=1.5, linestyle=':',
                label=f'Optimal t={best_threshold:.3f}')

# Mark the conventional default threshold 0.5 for comparison.
axes[0].axvline(0.5, color='gray', lw=1, linestyle=':', label='Default t=0.5')

# Axis labels and title.
axes[0].set_xlabel("Classification Threshold t")
axes[0].set_ylabel("Score")
axes[0].set_title("Precision, Recall, F1 vs Threshold\n(class_weight='balanced')")

# Show legend and grid.
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# Precision-Recall curve
# x-axis = recall
# y-axis = precision
#
# This curve is particularly useful in imbalanced classification because
# it focuses directly on positive-class retrieval quality.
axes[1].plot(recalls, precisions, lw=2, color='#1F77B4')

# Baseline precision under random guessing is approximately the positive-class prevalence.
# Here y_test.mean() is the fraction of positive examples in the test set.
axes[1].axhline(y_test.mean(), color='gray', lw=1, linestyle='--',
                label=f'Random (AP={y_test.mean():.3f})')

# Labels and title.
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve")

# Legend and grid.
axes[1].legend(); axes[1].grid(True, alpha=0.3)

# Improve layout, save figure, then display it.
plt.tight_layout()
plt.savefig("threshold_optimization.png", dpi=150, bbox_inches='tight')
# plt.show()
