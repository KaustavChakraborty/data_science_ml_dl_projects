"""
===================================================================
COMPREHENSIVE KNN CLASSIFICATION WITH SCIKIT-LEARN
===================================================================
Covers:
  - Feature scaling (mandatory for KNN)
  - Hyperparameter tuning (K, metric, weights) via GridSearchCV
  - Decision boundary visualization
  - Learning curves (bias-variance tradeoff)
  - Cross-validation
  - Metrics: accuracy, confusion matrix, classification report,
    ROC-AUC, precision-recall
  - Effect of K on performance
The code uses three built-in datasets from scikit-learn:
- make_moons         -> small nonlinear 2D toy data, useful for visualization
- breast_cancer      -> real binary classification dataset
- wine               -> multiclass dataset with correlated features
===================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold,
    learning_curve, cross_val_score, validation_curve
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay,
    precision_recall_curve, average_precision_score
)
from sklearn.decomposition import PCA

# Import our dataset generators (from Code 1)
# Assuming they are in the same file or imported
from sklearn.datasets import make_moons, load_breast_cancer, load_wine

import warnings
warnings.filterwarnings("ignore", category=UserWarning) # or MatplotlibDeprecationWarning

RANDOM_STATE = 42


# ---------------------------------------------------------------
# UTILITY: Decision Boundary Plot
# ---------------------------------------------------------------
def plot_decision_boundary(clf, X, y, ax, title, resolution=0.02, scaler=None):
    """
    Plots the 2D decision boundary of any classifier.
    
    Parameters
    ----------
    clf        : trained classifier with predict() method
    X          : 2D feature array (n_samples, 2) — already scaled
    y          : labels
    ax         : matplotlib axes object
    title      : plot title
    resolution : grid resolution (smaller = finer but slower)
    scaler     : if provided, inverse_transform for axis labels
    """
    # Define grid bounds (with margin)
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # Create mesh grid
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, resolution),
        np.arange(y_min, y_max, resolution)
    )

    # Predict on every grid point
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    Z = clf.predict(grid_points)
    Z = Z.reshape(xx.shape)

    # Color map
    n_classes = len(np.unique(y))
    cmap_light = plt.colormaps.get_cmap('Pastel1').resampled(n_classes)
    cmap_bold  = plt.colormaps.get_cmap('Set1').resampled(n_classes)

    # Fill decision regions
    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    ax.contour( xx, yy, Z, colors='gray', linewidths=0.5, alpha=0.5)

    # Plot data points
    for cls in np.unique(y):
        mask = y == cls
        ax.scatter(X[mask, 0], X[mask, 1],
                  c=[cmap_bold(cls)] * np.sum(mask),
                  s=25, edgecolors='k', linewidths=0.3,
                  label=f'Class {cls}', alpha=0.9, zorder=5)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Feature 1 (scaled)")
    ax.set_ylabel("Feature 2 (scaled)")
    ax.legend(fontsize=7)


# =============================================================================
# 1. EFFECT OF K: BIAS-VARIANCE TRADEOFF VISUALIZATION
# =============================================================================
def visualize_K_effect():
    """
    Show how different K values change the decision boundary.

    Why this matters
    ----------------
    K controls how "local" or "smooth" the KNN decision rule is.

    Small K:
        - very sensitive to nearby points
        - can create irregular / jagged boundaries
        - low bias, high variance
        - may overfit noise

    Large K:
        - averages over a larger neighborhood
        - smoother boundary
        - higher bias, lower variance
        - may underfit complex structure

    This function uses the moons dataset because:
    - it is nonlinear
    - it is 2D
    - it clearly shows how KNN adapts to geometry
    """

    print("\n=== EFFECT OF K ON DECISION BOUNDARY ===")

    # -------------------------------------------------------------------------
    # Step 1: Create a synthetic nonlinear binary dataset.
    # -------------------------------------------------------------------------
    # make_moons creates two interleaving crescent-shaped classes.
    #
    # n_samples=300:
    #     enough points for a nice-looking boundary
    #
    # noise=0.2:
    #     adds randomness so the classification task is not perfectly clean
    #
    # random_state:
    #     ensures reproducibility
    X, y = make_moons(n_samples=300, noise=0.2, random_state=RANDOM_STATE)

    
    # -------------------------------------------------------------------------
    # Step 2: Scale features.
    # -------------------------------------------------------------------------
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -------------------------------------------------------------------------
    # Step 3: Try several K values spanning very local to quite smooth.
    # -------------------------------------------------------------------------
    K_values = [1, 3, 7, 15, 31, 51]

    # Create a 2x3 panel so each K gets one subplot.
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("KNN Decision Boundaries: Effect of K (Moons Dataset)",
                fontsize=14, fontweight='bold')

    # -------------------------------------------------------------------------
    # Step 4: Fit one model per K and plot its decision boundary.
    # -------------------------------------------------------------------------
    for ax, K in zip(axes.ravel(), K_values):
        # Train on ALL data (for visualization only)
        knn = KNeighborsClassifier(n_neighbors=K, metric='euclidean')
        knn.fit(X_scaled, y)

        plot_decision_boundary(knn, X_scaled, y, ax, title=f"K = {K}")

    plt.tight_layout()
    plt.savefig("knn_K_effect.png", dpi=150, bbox_inches='tight')
    # plt.show()


# =============================================================================
# 2. VALIDATION CURVE: K vs ACCURACY
# =============================================================================
def plot_validation_curve_K():
    """
    Plot training and cross-validation accuracy as a function of K.

    For very small K:
        - training accuracy is often very high
        - validation accuracy may be lower because the model overfits

    As K increases:
        - training accuracy usually drops
        - validation accuracy may first improve, then eventually decline

    The best K is often near the peak of the validation curve.
    """
    print("\n=== VALIDATION CURVE: K vs ACCURACY ===")

    # -------------------------------------------------------------------------
    # Step 1: Build a larger moons dataset.
    # -------------------------------------------------------------------------
    X, y = make_moons(n_samples=600, noise=0.2, random_state=RANDOM_STATE)

    # -------------------------------------------------------------------------
    # Step 2: Construct a pipeline.
    # -------------------------------------------------------------------------
    # Pipeline is best practice because:
    # - scaler is learned only from training folds inside CV
    # - this prevents leakage from validation fold into preprocessing
    # - all steps are bundled into one reusable object
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('knn',    KNeighborsClassifier())
    ])

    # -------------------------------------------------------------------------
    # Step 3: Define the range of K values to test.
    # -------------------------------------------------------------------------
    # np.arange(1, 51, 2) = 1, 3, 5, ..., 49
    #
    # Odd K values are often used in binary classification to reduce the chance
    # of ties in majority voting.
    K_range = np.arange(1, 51, 2)

    # -------------------------------------------------------------------------
    # Step 4: Compute validation curve.
    # -------------------------------------------------------------------------
    # validation_curve automatically:
    # - fits the model for each K
    # - performs CV
    # - returns training scores and validation scores for every fold
    #
    # param_name='knn__n_neighbors':
    #   In a Pipeline, parameters are referenced as:
    #       <step_name>__<parameter_name>
    #   Here:
    #       step name = 'knn'
    #       parameter = 'n_neighbors'
    #
    # StratifiedKFold preserves class balance across folds.
    train_scores, val_scores = validation_curve(
        pipe, X, y,
        param_name='knn__n_neighbors',  # note: pipeline param naming
        param_range=K_range,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring='accuracy',
        n_jobs=-1  # use all CPUs
    )

    # -------------------------------------------------------------------------
    # Step 5: Summarize CV results.
    # -------------------------------------------------------------------------
    # For each K we have 5 training scores and 5 validation scores.
    # We average across folds to get a smooth curve.
    train_mean = np.mean(train_scores, axis=1)
    train_std  = np.std(train_scores,  axis=1)

    val_mean   = np.mean(val_scores,   axis=1)
    val_std    = np.std(val_scores,    axis=1)

    # Best K according to mean validation accuracy
    best_K = K_range[np.argmax(val_mean)]
    print(f"  Best K = {best_K} (CV accuracy = {np.max(val_mean):.4f})")


    # -------------------------------------------------------------------------
    # Step 6: Plot training and validation curves.
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    # Training accuracy curve
    ax.plot(K_range, train_mean, 'b-o', markersize=4, label='Training Accuracy')
    ax.fill_between(K_range, train_mean - train_std, train_mean + train_std,
                   alpha=0.15, color='blue')
    # Validation accuracy curve
    ax.plot(K_range, val_mean, 'r-s', markersize=4, label='CV Accuracy (5-fold)')
    ax.fill_between(K_range, val_mean - val_std, val_mean + val_std,
                   alpha=0.15, color='red')
    # Mark the best K visually
    ax.axvline(best_K, color='green', linestyle='--', label=f'Best K={best_K}')
    # Labeling
    ax.set_xlabel("Number of Neighbors (K)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Validation Curve: K vs Accuracy (Moons Dataset)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("knn_validation_curve.png", dpi=150, bbox_inches='tight')
    # plt.show()

    # Return the chosen K so it can be reused later if desired.
    return best_K


# =============================================================================
# 3. FULL KNN PIPELINE WITH GRID SEARCH
# =============================================================================
def knn_full_pipeline():
    """
    Full production-grade KNN pipeline:
        1. Load data (breast cancer — 30 features, binary)
        2. Train/test split with stratification
        3. Build Pipeline (StandardScaler + KNN)
        4. GridSearchCV over K, metric, weights
        5. Evaluate with multiple metrics
        6. Plot confusion matrix and ROC curve
    """
    print("\n=== FULL KNN PIPELINE: BREAST CANCER DATASET ===")

    # -------------------------------------------------------------------------
    # Step 1: Load dataset.
    # -------------------------------------------------------------------------
    bc = load_breast_cancer()
    X, y = bc.data, bc.target

    # Basic dataset summary
    print(f"  Data shape: {X.shape}")
    print(f"  Classes: {bc.target_names}")
    print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # -------------------------------------------------------------------------
    # Step 2: Train/test split with stratification.
    # -------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # -------------------------------------------------------------------------
    # Step 3: Build pipeline.
    # -------------------------------------------------------------------------
    # Pipeline ensures scaler is fit ONLY on training data
    # (prevents data leakage into test set)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('knn',    KNeighborsClassifier())
    ])

    # -------------------------------------------------------------------------
    # Step 4: Define hyperparameter search space.
    # -------------------------------------------------------------------------
    # We tune:
    #
    # knn__n_neighbors:
    #   controls local vs global smoothing
    #
    # knn__metric:
    #   choice of distance metric
    #
    # knn__weights:
    #   uniform  -> all K neighbors vote equally
    #   distance -> closer neighbors get higher influence
    #
    # knn__p:
    #   only relevant for Minkowski distance
    #   p=2 corresponds to Euclidean distance
    #
    # Note:
    #   Since metric is allowed to be euclidean, manhattan, or minkowski,
    #   setting p=[2] means:
    #   - if metric='minkowski', then p=2 -> Euclidean-style Minkowski
    #   - if metric='euclidean' or 'manhattan', p is effectively irrelevant
    param_grid = {
        'knn__n_neighbors': [3, 5, 7, 11, 15, 21, 31],
        'knn__metric':      ['euclidean', 'manhattan', 'minkowski'],
        'knn__weights':     ['uniform', 'distance'],
        'knn__p':           [2]   # only used when metric='minkowski'
    }

    # -------------------------------------------------------------------------
    # Step 5: Create cross-validation splitter.
    # -------------------------------------------------------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


    # -------------------------------------------------------------------------
    # Step 6: Set up GridSearchCV.
    # -------------------------------------------------------------------------
    # scoring='roc_auc':
    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',    # AUC is better than accuracy for imbalanced data
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )

    print("\n  Running GridSearchCV... ")
    grid_search.fit(X_train, y_train)

    # -------------------------------------------------------------------------
    # Step 7: Extract best model.
    # -------------------------------------------------------------------------
    print(f"\n  Best Parameters: {grid_search.best_params_}")
    print(f"  Best CV AUC:     {grid_search.best_score_:.4f}")

    # best_estimator_ is the full fitted pipeline with the best hyperparameters
    best_model = grid_search.best_estimator_

    # -------------------------------------------------------------------------
    # Step 8: Evaluate on held-out test set.
    # -------------------------------------------------------------------------
    # predict() gives discrete class labels
    y_pred      = best_model.predict(X_test)
    # predict_proba() gives class probabilities.
    # For binary classification, [:, 1] is the probability of the positive class.
    y_pred_prob = best_model.predict_proba(X_test)[:, 1]

    print(f"\n  === TEST SET METRICS ===")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"  ROC AUC:   {roc_auc_score(y_test, y_pred_prob):.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=bc.target_names))

    # -------------------------------------------------------------------------
    # Step 9: Plot confusion matrix and ROC curve.
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("KNN: Breast Cancer — Best Model Evaluation", fontsize=13)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=bc.target_names)
    disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
    axes[0].set_title("Confusion Matrix", fontsize=11)

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    # AUC summarizes ROC as one scalar
    auc_score = roc_auc_score(y_test, y_pred_prob)
    axes[1].plot(fpr, tpr, 'b-', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    # Diagonal line = random guessing baseline
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1, label='Random classifier')
    axes[1].fill_between(fpr, tpr, alpha=0.15, color='blue')
    axes[1].set_xlabel('False Positive Rate', fontsize=11)
    axes[1].set_ylabel('True Positive Rate', fontsize=11)
    axes[1].set_title('ROC Curve', fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("knn_evaluation.png", dpi=150, bbox_inches='tight')
    # plt.show()

    return best_model, grid_search


# =============================================================================
# 4. LEARNING CURVE: TRAINING SIZE vs PERFORMANCE
# =============================================================================
def plot_learning_curve():
    """
    Shows how performance changes as training set size grows.
    Useful for diagnosing:
        - Underfitting: both curves plateau at low accuracy
        - Overfitting: large gap between train and CV curves
        - Data hunger: CV curve still rising (need more data)

    Here we keep K fixed (K=7) and inspect whether adding more samples helps.
    """
    print("\n=== LEARNING CURVE ===")

    # -------------------------------------------------------------------------
    # Step 1: Load dataset.
    # -------------------------------------------------------------------------
    bc = load_breast_cancer()
    X, y = bc.data, bc.target

    # -------------------------------------------------------------------------
    # Step 2: Build a fixed pipeline.
    # -------------------------------------------------------------------------
    # Here we are not tuning K. We keep it fixed at 7 and study the effect of
    # increasing training-set size.
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('knn',    KNeighborsClassifier(n_neighbors=7, metric='euclidean'))
    ])

    # -------------------------------------------------------------------------
    # Step 3: Define training fractions.
    # -------------------------------------------------------------------------
    # np.linspace(0.1, 1.0, 20) creates 20 evenly spaced values between 10% and
    # 100% of the available training data used inside the CV learning-curve
    # computation.
    train_sizes_rel = np.linspace(0.1, 1.0, 20)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # -------------------------------------------------------------------------
    # Step 4: Compute learning curves.
    # -------------------------------------------------------------------------
    # learning_curve returns:
    # - actual number of training samples used
    # - training scores
    # - validation scores
    #
    # It repeatedly fits the model on progressively larger subsets.
    train_sizes, train_scores, val_scores = learning_curve(
        pipe, X, y,
        train_sizes=train_sizes_rel,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1
    )

    # -------------------------------------------------------------------------
    # Step 5: Aggregate over folds.
    # -------------------------------------------------------------------------
    train_mean = np.mean(train_scores, axis=1)
    train_std  = np.std(train_scores,  axis=1)

    val_mean   = np.mean(val_scores,   axis=1)
    val_std    = np.std(val_scores,    axis=1)

    # -------------------------------------------------------------------------
    # Step 6: Plot learning curves.
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_sizes, train_mean, 'b-o', markersize=5, label='Training Accuracy')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                   alpha=0.15, color='blue')
    ax.plot(train_sizes, val_mean, 'r-s', markersize=5, label='CV Accuracy (5-fold)')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                   alpha=0.15, color='red')
    ax.set_xlabel("Training Set Size", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("KNN Learning Curve (Breast Cancer, K=7)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("knn_learning_curve.png", dpi=150, bbox_inches='tight')
    # plt.show()


# =============================================================================
# 5. DISTANCE METRIC COMPARISON
# =============================================================================
def compare_distance_metrics():
    """
    Compares classification performance across different distance metrics.
    Useful for understanding which metric fits your data geometry.
    
    In soft matter: 
        - Euclidean: good for normalized physical properties
        - Mahalanobis: good for correlated features (e.g., density + viscosity)
        - Manhattan: robust to outliers in any single dimension
    """
    print("\n=== DISTANCE METRIC COMPARISON ===")

    # Use Wine dataset (13 features, meaningful correlations)
    wine = load_wine()
    X, y = wine.data, wine.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler  = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    metrics_to_test = {
        'Euclidean (p=2)':   KNeighborsClassifier(n_neighbors=7, metric='euclidean'),
        'Manhattan (p=1)':   KNeighborsClassifier(n_neighbors=7, metric='manhattan'),
        'Minkowski (p=3)':   KNeighborsClassifier(n_neighbors=7, metric='minkowski', p=3),
        'Chebyshev (p=inf)': KNeighborsClassifier(n_neighbors=7, metric='chebyshev'),
        'Cosine':            KNeighborsClassifier(n_neighbors=7, metric='cosine'),
    }

    print(f"\n  {'Metric':<25} {'Test Accuracy':>15} {'CV Accuracy (5-fold)':>22}")
    print(f"  {'-'*25} {'-'*15} {'-'*22}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    for name, clf in metrics_to_test.items():
        clf.fit(X_train_s, y_train)
        test_acc = clf.score(X_test_s, y_test)

        # CV scores on full dataset
        cv_scores = cross_val_score(
            Pipeline([('scaler', StandardScaler()), ('knn', clf)]),
            X, y, cv=cv, scoring='accuracy'
        )
        cv_acc = cv_scores.mean()
        cv_std = cv_scores.std()

        results[name] = (test_acc, cv_acc, cv_std)
        print(f"  {name:<25} {test_acc:>14.4f}  {cv_acc:.4f} ± {cv_std:.4f}")

    return results


# ---------------------------------------------------------------
# RUN ALL
# ---------------------------------------------------------------
if __name__ == "__main__":
    visualize_K_effect()
    best_K = plot_validation_curve_K()
    best_model, grid_search = knn_full_pipeline()
    plot_learning_curve()
    compare_distance_metrics()