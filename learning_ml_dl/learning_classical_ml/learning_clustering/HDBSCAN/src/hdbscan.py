"""
================================================================================
ANNOTATED VERSION OF THE ORIGINAL HDBSCAN TUTORIAL SCRIPT
================================================================================

This file is a deeply commented version of the original script. The goal here
is not just to say what each line does, but also why it is there and how it
changes the scientific or clustering outcome.

HOW TO READ THIS FILE
---------------------
1. Read the module-level notes first. They explain the overall architecture.
2. Then read `fit_hdbscan(...)` because that is the computational heart.
3. Then read the helper functions:
      - `load(...)`
      - `plot_condensed_tree(...)`
      - `summarise_hdbscan(...)`
      - `cluster_palette(...)`
4. After that, move through sections A → J in order. The script is written like
   a tutorial: each section demonstrates one conceptual capability of HDBSCAN.
5. Finally read the `__main__` block to see the full execution order.

WHAT THIS SCRIPT REALLY IS
--------------------------
This is not just a clustering script. It is a didactic analysis pipeline that
mixes:
  - data loading,
  - HDBSCAN fitting,
  - diagnostic summaries,
  - physical interpretation,
  - visualisation,
  - and comparison against ground truth or alternative methods.

So the output of the script is not a single model object. The real output is:
  - many HDBSCAN fits,
  - many diagnostic numbers,
  - and a directory full of explanatory plots.

GLOBAL EXECUTION PHILOSOPHY
---------------------------
The script repeatedly does the following pattern:

    raw data
      -> standardize
      -> fit HDBSCAN
      -> inspect labels / probabilities / outlier scores / persistence
      -> compare to truth or an expected physical picture
      -> plot everything

That repeated pattern is the main design principle of the whole file.

KEY CONSEQUENCES OF THE DESIGN
------------------------------
- Because the script standardizes almost every dataset before clustering,
  Euclidean distances become feature-balanced. This strongly affects the
  clustering outcome. Without scaling, a high-variance feature would dominate
  distance computations and distort HDBSCAN's density estimate.

- Because `prediction_data=True` is enabled inside `fit_hdbscan(...)`,
  the script can later compute soft memberships and approximate predictions for
  new points. That choice increases stored auxiliary information and is central
  to sections C and I.

- Because the code often evaluates against `y_true`, this script is partly a
  tutorial/benchmarking script rather than a blind unsupervised workflow.

- Because plotting is a first-class citizen throughout the file, many lines are
  about human interpretation, not algorithmic necessity. These lines do not
  change cluster assignments; they change how easy it is to understand them.

IMPORTANT "NO EFFECT" / MINOR-EFFECT DETAILS TO NOTICE
------------------------------------------------------
A few imported objects are not actually used in the final logic:
  - `matplotlib.colors as mcolors`
  - `KMeans`
  - `HDBSCAN as SK_HDBSCAN` from scikit-learn
  - `pdist`
These have no effect on the output because nothing calls them.

A few intermediate variables are also effectively unused:
  - `rows` and `clust_rows` inside `plot_condensed_tree(...)`
  - `feat_names` in `md_conformational_states(...)`
  - `test_scores` in `anomaly_detection_pipeline(...)`
They were likely left from exploratory development or future extensions.

ORIGINAL FILE HEADER (PRESERVED BELOW)
--------------------------------------

================================================================================
05_hdbscan.py  —  HDBSCAN: Hierarchical Density-Based Clustering
================================================================================

WHAT HDBSCAN DOES IN ONE SENTENCE
──────────────────────────────────
Runs DBSCAN across ALL density scales simultaneously, builds a hierarchy of
cluster births/deaths, then automatically extracts the most persistent
(stable) clusters — handling variable-density data and producing both hard
labels and soft membership probabilities.

KEY ADVANTAGES OVER DBSCAN
───────────────────────────
  ✓ No ε to tune — only min_cluster_size (intuitive: "how big is a cluster?")
  ✓ Handles clusters of very different densities in the same dataset
  ✓ Returns membership probabilities (0–1) for every point
  ✓ Returns outlier/anomaly scores (GLOSH) for every point
  ✓ Hierarchy is directly inspectable (condensed tree)
  ✓ More robust to parameter choice than DBSCAN

WHEN TO USE HDBSCAN
────────────────────
  ✓ Variable-density clusters (your primary reason to pick HDBSCAN over DBSCAN)
  ✓ You want probabilistic cluster assignments
  ✓ You want anomaly/outlier scores alongside cluster labels
  ✓ K is unknown and you don't want to set ε
  ✓ MD trajectories / conformational analysis (multi-scale basins)
  ✓ Large datasets (scales to millions with approximate NN)
  ✗ You need strict reproducibility across library versions → DBSCAN
  ✗ Simple, single-scale data you already understand → DBSCAN (simpler)

THE TWO KEY PARAMETERS
────────────────────────
  min_cluster_size  : Smallest group you'd call a cluster.
                      Start: 5. Tune UP if too many spurious micro-clusters.
  min_samples       : Controls density conservatism (default=min_cluster_size).
                      Tune UP to make cluster cores denser (more noise points).
                      Tune DOWN to pull borderline points into clusters.

SECTIONS
────────
  A  HDBSCAN basics — fit, labels, probabilities, outlier scores
  B  Condensed tree — reading the cluster hierarchy
  C  Soft membership probabilities — what they mean and how to use them
  D  GLOSH outlier scores — anomaly detection from clustering
  E  Parameter sensitivity: min_cluster_size and min_samples
  F  DBSCAN vs HDBSCAN — when and why HDBSCAN wins
  G  Soft-matter: MD conformational state extraction
  H  Soft-matter: multi-scale colloidal phases
  I  Soft-matter: anomaly detection in experimental trajectories
  J  Full inference pipeline: from raw data to physical interpretation

Run  00_generate_datasets.py  first.
Requirements: numpy  scipy  scikit-learn  hdbscan  matplotlib  seaborn  pandas
================================================================================

"""

# Standard library imports:
# - `os` is used to create the output directory.
# - `warnings` is used here only to suppress warnings globally.
#   This keeps tutorial output visually clean, but it can also hide important
#   diagnostics during debugging. So this improves presentation, not rigor.
import os, warnings

# This suppresses *all* warnings emitted by imported libraries.
# Effect on numerical/clustering results: none.
# Effect on user experience: quieter output, but potentially less transparency.
warnings.filterwarnings("ignore")

# Core numerical array library.
import numpy as np

# Matplotlib is used for all saved figures.
import matplotlib

# Force the non-interactive "Agg" backend so the script can run on servers,
# CI environments, remote clusters, or headless Linux machines.
# Effect on cluster assignments: none.
# Effect on output: figures are saved to disk instead of requiring a GUI.
matplotlib.use("Agg")

# Plotting API.
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Imported but not used in the current script.
# No effect on the outcome.
import matplotlib.colors as mcolors

# Used mainly for polished statistical plots such as heatmaps and color palettes.
import seaborn as sns

# Used to build labeled tabular views before plotting heatmaps.
import pandas as pd

# The *external* hdbscan package is the main engine actually used by the script.
# This is the implementation whose `HDBSCAN`, `approximate_predict`, and
# `membership_vector` functions power the main analysis.
import hdbscan as hdbscan_lib          # the 'hdbscan' package

# `DBSCAN` is used in the comparison section.
# `KMeans` and scikit-learn's `HDBSCAN` alias are imported but never used.
# Therefore KMeans and SK_HDBSCAN have no effect in this file as written.
from sklearn.cluster        import DBSCAN, KMeans, HDBSCAN as SK_HDBSCAN

# External validation / compactness metrics:
# - ARI compares predicted labels to known truth labels.
# - Silhouette measures cluster separation/compactness on assigned data.
from sklearn.metrics        import adjusted_rand_score, silhouette_score

# StandardScaler is crucial because HDBSCAN relies on distances.
# If features are not on comparable scales, density estimation becomes biased.
from sklearn.preprocessing  import StandardScaler

# PCA is used repeatedly for 2D visualization when feature space is >2D.
# PCA does not affect clustering itself when applied *after* fitting for plots.
from sklearn.decomposition  import PCA

# k-NN distances are used to derive DBSCAN eps heuristics in section F.
from sklearn.neighbors      import NearestNeighbors

# Synthetic data generator used in the DBSCAN-vs-HDBSCAN demonstration.
from sklearn.datasets       import make_blobs

# Imported but unused.
# No effect on the script outcome.
from scipy.spatial.distance import pdist

# Ensure the output directory exists before any plots are saved.
# If this line were removed and the directory did not already exist,
# plot saving would fail with a filesystem error.
os.makedirs("plots/hdbscan", exist_ok=True)

# Centralized random generator for reproducibility of injected anomalies,
# permutations, and synthetic sampling.
# Changing the seed changes the exact synthetic points, and therefore the
# exact cluster assignments and scores in the tutorial sections that simulate data.
RNG  = np.random.default_rng(42)
SEED = 42


# ── helpers ───────────────────────────────────────────────────────────────────

# ---------------------------------------------------------------------------
# HELPER: load(...)
# ---------------------------------------------------------------------------
# Purpose:
#   Read one `.npz` dataset from the local `data/` directory.
#
# Expected file structure:
#   The file must contain:
#     - X : feature matrix of shape (n_samples, n_features)
#     - y : ground-truth labels used for evaluation / illustration
#
# Why this matters:
#   Every analytical section assumes the same dataset contract. This keeps the
#   rest of the script simple, because all tutorial sections can call `load(...)`
#   and immediately receive features plus reference labels.
#
# What impacts outcome here:
#   The loader itself does not change the data. It only enforces the assumption
#   that the file already contains correctly prepared arrays.
# ---------------------------------------------------------------------------
def load(name):
    # Load the compressed NumPy archive from disk.
    d = np.load(f"data/{name}.npz")

    # Return:
    #   X -> feature matrix
    #   y -> reference labels
    # The loader assumes these keys exist. If they do not, the script will fail.
    return d["X"], d["y"]

# ---------------------------------------------------------------------------
# HELPER: plot_condensed_tree(...)
# ---------------------------------------------------------------------------
# Purpose:
#   Draw a simplified manual view of HDBSCAN's condensed cluster tree.
#
# Why this function exists:
#   The author comments that `hdbscan.plot()` has a matplotlib-version bug in
#   their environment, so this is a compatibility workaround.
#
# Conceptual importance:
#   The condensed tree is one of the main reasons to use HDBSCAN instead of
#   plain DBSCAN. It shows *which clusters persist across density scales*.
#   In HDBSCAN, persistence = credibility/stability.
#
# Important caveat:
#   This is not a mathematically exact re-implementation of the official plot.
#   It is a simplified, pedagogical visualization. So it communicates the idea
#   of persistence, but should not be mistaken for a perfect internal-state plot.
# ---------------------------------------------------------------------------
def plot_condensed_tree(clf, ax, title="Condensed Tree"):
    """
    Manual condensed-tree plot using the pandas representation.
    Avoids the hdbscan.plot() method which has a matplotlib-version bug.

    Each selected cluster is drawn as a coloured bar spanning its
    lambda birth → lambda death range, with width = cluster size.
    Unselected (noise-falloff) branches are grey.
    """
    try:
        # Convert the internal condensed-tree object into a pandas table that is
        # easier to manipulate manually.
        df = clf.condensed_tree_.to_pandas()
    except Exception:
        # If the tree is missing or incompatible, fail gracefully by drawing an
        # explanatory message on the axis instead of crashing the whole script.
        ax.text(0.5, 0.5, "Tree unavailable", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        return

    # Identify selected cluster ids
    # Only non-negative labels represent selected clusters.
    # Label -1 is reserved for noise and is intentionally excluded.
    selected = set(clf.labels_[clf.labels_ >= 0])

    # Use a categorical colormap for human-readable cluster coloring.
    cmap     = plt.cm.tab10

    # For each cluster: get birth λ, death λ, max size
    # Number of original data points.
    n_pts    = len(clf.labels_)

    # Draw one horizontal bar per selected cluster.
    for k in sorted(selected):
        # NOTE:
        # The next two variables are not actually used afterwards. They appear to
        # be remnants of a more exact plotting attempt. So they currently have no
        # effect on the figure.
        rows = df[df["child"] > n_pts - 1]  # cluster (not leaf) rows
        clust_rows = df[(df["child_size"] > 1) & (df["parent"] >= n_pts)]

        # Mask selecting all points assigned to cluster k in the final output.
        mask = clf.labels_ == k

        # Simplified estimate of the cluster's "birth" density scale.
        # This is not the exact official HDBSCAN persistence calculation, but a
        # practical proxy for a readable bar chart.
        lam_birth = df[df["child_size"] >= mask.sum()]["lambda_val"].min() if len(df) else 0

        # Use the maximum lambda in the tree as a common "death" endpoint.
        # This again is a simplified visual convention.
        lam_death = df["lambda_val"].max()

        # Draw the persistence-style bar:
        #   horizontal position = lambda range
        #   y-position          = cluster id
        #   label               = cluster id and final membership count
        ax.barh(k, lam_death - lam_birth, left=lam_birth,
                height=0.6, color=cmap(k % 10), alpha=0.8,
                edgecolor="k", lw=0.8,
                label=f"C{k} (n={mask.sum()})")

    # Also draw noise/falling-off branches in grey
    # Leaf rows correspond to branches that have shrunk down to singleton points.
    # Plotting these as faint gray marks hints at how points peel off into noise.
    # The upper limit of 500 avoids clutter and keeps the plot readable.
    leaf_rows = df[df["child_size"] == 1]
    if len(leaf_rows) > 0 and len(leaf_rows) < 500:
        for _, row in leaf_rows.iterrows():
            ax.plot([row["lambda_val"], row["lambda_val"] * 1.05],
                    [row["parent"] % len(selected), row["parent"] % len(selected)],
                    color="lightgray", lw=0.3, alpha=0.4)

    ax.set_xlabel("λ = 1/distance (density scale)")
    ax.set_ylabel("Cluster id")
    ax.set_title(title, fontweight="bold", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.2)

# ---------------------------------------------------------------------------
# HELPER / CORE ENGINE: fit_hdbscan(...)
# ---------------------------------------------------------------------------
# Purpose:
#   Construct and fit an HDBSCAN model with the tutorial's preferred defaults.
#
# Why this is the heart of the script:
#   Nearly every major section calls this function. So any default you change
#   here propagates through the entire tutorial.
#
# Parameters with strongest effect on results:
#   min_cluster_size:
#       Minimum size of a group that you are willing to call a cluster.
#       Increasing it suppresses tiny clusters and often increases noise.
#
#   min_samples:
#       Controls how conservative density estimation is.
#       Larger values require denser local neighborhoods to qualify as cluster.
#       This often produces more noise and tighter cluster cores.
#
#   cluster_selection_method:
#       'eom' gives coarser, persistence-optimized clusters.
#       'leaf' gives finer sub-clusters from the hierarchy.
#
#   gen_min_span_tree:
#       Needed only when you want tree/MST visual diagnostics.
#       Turning it off usually reduces overhead but loses some introspection.
#
# Less obvious but important choice:
#   prediction_data=True
#       This enables later soft-membership and approximate prediction routines.
#       It is essential for sections C and I.
# ---------------------------------------------------------------------------
def fit_hdbscan(X, min_cluster_size=15, min_samples=None,
                cluster_selection_method="eom",
                gen_min_span_tree=True):
    """
    Fit HDBSCAN and return clf with all attributes populated.

    cluster_selection_method
    ─────────────────────────
    'eom'  (Excess of Mass, default):
        Selects clusters that maximise total cluster persistence.
        Tends to give fewer, larger clusters.
        Best for: well-separated clusters, clean data.

    'leaf' :
        Always selects leaf clusters in the condensed tree
        (finest-grained clusters possible).
        Tends to give more, smaller clusters.
        Best for: hierarchical exploration, finding sub-clusters.
    """
    clf = hdbscan_lib.HDBSCAN(
        # Smallest allowed cluster in the extracted flat solution.
        min_cluster_size         = min_cluster_size,

        # Local density conservatism. `None` means "use the same value as
        # min_cluster_size", which is a common robust default.
        min_samples              = min_samples,

        # How to collapse the hierarchy into final flat labels.
        cluster_selection_method = cluster_selection_method,

        # Distance metric used for neighborhood structure.
        # Because the script almost always standardizes first, Euclidean distance
        # becomes a reasonable default in these examples.
        metric                   = "euclidean",

        # Whether to keep the minimum-spanning-tree information around for
        # diagnostics and plotting.
        gen_min_span_tree        = gen_min_span_tree,

        # Required for approximate prediction / membership-based utilities.
        prediction_data          = True,
    )

    # Perform the actual fit. After this line the model gains attributes like:
    #   labels_, probabilities_, outlier_scores_, condensed_tree_, ...
    clf.fit(X)
    return clf

# ---------------------------------------------------------------------------
# HELPER: summarise_hdbscan(...)
# ---------------------------------------------------------------------------
# Purpose:
#   Print a compact summary of how many clusters were found, how much noise was
#   produced, and optionally how well the result matches known truth labels.
#
# Why useful:
#   HDBSCAN often produces both clusters and noise. Reporting only the number of
#   clusters would hide whether the method classified half the data as noise.
#
# Important detail:
#   ARI is computed only on non-noise points. That means this metric answers:
#   "Among points the algorithm was confident enough to cluster, how well did it
#    label them?"
#   It does NOT penalize noise assignments directly in the same way as a full
#   all-point label comparison would.
# ---------------------------------------------------------------------------
def summarise_hdbscan(clf, y_true=None, prefix=""):
    """Print cluster summary, ARI, noise fraction."""
    # Final flat labels extracted from the HDBSCAN hierarchy.
    labels   = clf.labels_

    # Total number of points.
    n        = len(labels)

    # Count actual clusters while excluding the noise label (-1).
    n_clus   = len(set(labels)) - (1 if -1 in labels else 0)

    # Count points rejected as noise.
    n_noise  = (labels == -1).sum()
    print(f"  {prefix}clusters={n_clus}  "
          f"noise={n_noise}/{n} ({100*n_noise/n:.1f}%)")
    if y_true is not None and n_clus > 1:
        valid = labels != -1
        if valid.sum() > 5:
            ari = adjusted_rand_score(y_true[valid], labels[valid])
            print(f"  {prefix}ARI (non-noise) = {ari:.4f}")

# ---------------------------------------------------------------------------
# HELPER: cluster_palette(...)
# ---------------------------------------------------------------------------
# Purpose:
#   Convert labels (and optionally membership probabilities) into RGBA colors.
#
# What this changes:
#   Only the appearance of the plots. It does not change any clustering result.
#
# Why the opacity logic is meaningful:
#   When probabilities are supplied, alpha is tied to membership confidence.
#   This makes core points visually darker and boundary points visually lighter.
#   So one glance at a scatter plot shows both:
#     - which cluster a point belongs to, and
#     - how confidently it belongs there.
# ---------------------------------------------------------------------------
def cluster_palette(labels, probs=None):
    """
    Colors per point: noise=grey, clusters=tab10.
    If probs supplied, alpha scales with membership probability.
    """
    cmap   = plt.cm.tab10
    colors = []

    # Build one RGBA color per point.
    for i, lbl in enumerate(labels):
        if lbl == -1:
            # Noise points are always shown as semi-transparent gray so they are
            # visually distinct from accepted clusters.
            colors.append([0.5, 0.5, 0.5, 0.3])
        else:
            # Cluster color determined by cluster id.
            c = list(cmap(int(lbl) % 10))

            # If probabilities are given, store confidence in the alpha channel.
            # `clip(..., 0.15, 1.0)` prevents extremely small values from making
            # assigned points invisible.
            if probs is not None:
                c[3] = float(np.clip(probs[i], 0.15, 1.0))
            colors.append(c)
    return colors


# ════════════════════════════════════════════════════════════════════════════
# A — HDBSCAN BASICS
# ════════════════════════════════════════════════════════════════════════════

def hdbscan_basics():
    """
    Fit HDBSCAN on three datasets.  Demonstrate all key output attributes.

    Key attributes after fitting
    ────────────────────────────
    .labels_           (n,) int.  -1 = noise.  0,1,... = cluster id.
    .probabilities_    (n,) float ∈ [0,1].  Membership confidence.
                       1.0 = deep core of cluster.  0.0 = borderline/noise.
    .outlier_scores_   (n,) float ∈ [0,1].  GLOSH anomaly score.
                       High score = this point is unusual for its cluster.
    .cluster_persistence_  (K,) float.  Stability of each cluster.
                       Higher = cluster survives wider density range = more real.
    .condensed_tree_   Cluster hierarchy object (plot with .plot()).
    .minimum_spanning_tree_  MST used internally (plot with .plot()).
    """
    # Section A is the "API familiarization" section.
    # It does not ask a research question; it teaches what HDBSCAN returns.
    # Outcome impact:
    #   - `min_cluster_size` is fixed per dataset here, so differences in output
    #     mostly reflect dataset geometry, not hyperparameter sweeps.
    print("\n[A] HDBSCAN basics — key attributes")

    datasets = [
        ("blobs_easy",    15, "Isotropic Blobs"),
        ("circles",       15, "Concentric Circles"),
        ("moons",         15, "Two Moons"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(18, 17))
    fig.suptitle("HDBSCAN Basics — Labels / Probabilities / Outlier Scores",
                 fontsize=13, fontweight="bold")

    for row, (dname, mcs, title) in enumerate(datasets):
        X, y = load(dname)
        Xs   = StandardScaler().fit_transform(X)
        clf  = fit_hdbscan(Xs, min_cluster_size=mcs)

        n_clus  = len(set(clf.labels_)) - (1 if -1 in clf.labels_ else 0)
        n_noise = (clf.labels_ == -1).sum()
        ari     = (adjusted_rand_score(y, clf.labels_)
                   if n_clus > 1 else 0.0)

        print(f"\n  {title}  (min_cluster_size={mcs})")
        print(f"    clusters={n_clus}  noise={n_noise}  ARI={ari:.4f}")
        print(f"    prob range : [{clf.probabilities_.min():.3f}, "
              f"{clf.probabilities_.max():.3f}]")
        print(f"    outlier range: [{clf.outlier_scores_.min():.3f}, "
              f"{clf.outlier_scores_.max():.3f}]")
        if hasattr(clf, "cluster_persistence_"):
            print(f"    persistence: {np.round(clf.cluster_persistence_, 3)}")

        # Col 0: hard labels coloured
        ax = axes[row, 0]
        ax.scatter(Xs[:,0], Xs[:,1],
                   c=cluster_palette(clf.labels_), s=18, alpha=0.85)
        ax.set_title(f"{title}\n"
                     f"K={n_clus}  noise={n_noise}  ARI={ari:.3f}",
                     fontsize=9, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

        # Col 1: membership probability (alpha = confidence)
        ax = axes[row, 1]
        sc = ax.scatter(Xs[:,0], Xs[:,1],
                        c=clf.probabilities_, cmap="plasma",
                        s=18, vmin=0, vmax=1, alpha=0.9)
        plt.colorbar(sc, ax=ax, label="Membership prob")
        ax.set_title("Membership probability\n1=deep core  0=borderline",
                     fontsize=9, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

        # Col 2: outlier scores
        ax = axes[row, 2]
        sc = ax.scatter(Xs[:,0], Xs[:,1],
                        c=clf.outlier_scores_, cmap="hot_r",
                        s=18, vmin=0, vmax=1, alpha=0.9)
        plt.colorbar(sc, ax=ax, label="GLOSH outlier score")
        ax.set_title("GLOSH outlier score\nHigh=anomalous",
                     fontsize=9, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("plots/hdbscan/A_basics.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  saved → plots/hdbscan/A_basics.png")


# ════════════════════════════════════════════════════════════════════════════
# B — CONDENSED TREE: READING THE CLUSTER HIERARCHY
# ════════════════════════════════════════════════════════════════════════════

def condensed_tree_analysis(X, y_true, min_cluster_size=20, name="dataset"):
    """
    The condensed tree is the core output unique to HDBSCAN.

    How to read it
    ──────────────
    X-axis  : cluster membership count (how many points are in the cluster)
    Y-axis  : λ = 1/distance (higher λ = denser, more closely packed)
    Width   : how many points are in this cluster branch at this density

    A tall, wide bar = cluster that:
      • Contains many points (wide)
      • Persists over a large density range (tall)
      → This is a REAL, STABLE cluster

    A short, narrow bar = cluster that:
      • Contains few points (narrow)
      • Only appears at one specific density scale (short)
      → This is probably noise or an artefact

    Selected clusters (highlighted in colour) = what HDBSCAN reports as output.

    Cluster persistence = area under the bar = stability score.
    """
    # Section B focuses on *hierarchy interpretation* rather than just final labels.
    # Turning `gen_min_span_tree=True` matters here because the tree diagnostics
    # would be unavailable otherwise.
    print(f"\n[B] Condensed tree analysis — {name}")
    Xs  = StandardScaler().fit_transform(X)
    clf = fit_hdbscan(Xs, min_cluster_size=min_cluster_size,
                      gen_min_span_tree=True)

    n_clus = len(set(clf.labels_)) - (1 if -1 in clf.labels_ else 0)
    ari    = (adjusted_rand_score(y_true, clf.labels_)
              if n_clus > 1 else 0.0)
    print(f"  clusters={n_clus}  ARI={ari:.4f}")
    if hasattr(clf, "cluster_persistence_"):
        for k, p in enumerate(clf.cluster_persistence_):
            print(f"  C{k} persistence={p:.4f}")

    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle(f"Condensed Tree & Hierarchy — {name}  "
                 f"(min_cluster_size={min_cluster_size})",
                 fontsize=13, fontweight="bold")

    # 1. Condensed tree (key HDBSCAN plot)
    ax1 = fig.add_subplot(gs[0, :2])
    plot_condensed_tree(clf, ax1,
                        title="Condensed Cluster Tree\n"
                              "Width=cluster size  Height=density range survived\n"
                              "Coloured branches = selected clusters")

    # 2. Single linkage tree (full dendrogram before condensing) — use scipy
    from scipy.cluster.hierarchy import dendrogram
    ax2 = fig.add_subplot(gs[0, 2])
    try:
        sl_df = clf.single_linkage_tree_.to_pandas()
        # Convert to scipy linkage format: [child1, child2, distance, size]
        Z_sl  = sl_df[["left", "right", "distance", "size"]].values \
                if "left" in sl_df.columns else None
        if Z_sl is not None:
            dendrogram(Z_sl, ax=ax2, truncate_mode="lastp", p=30,
                       no_labels=True, color_threshold=0)
        else:
            ax2.text(0.5, 0.5, "MST unavailable", ha="center", va="center",
                     transform=ax2.transAxes)
    except Exception:
        ax2.text(0.5, 0.5, "Single-linkage\ntree unavailable",
                 ha="center", va="center", transform=ax2.transAxes)
    ax2.set_title("Single-Linkage Tree\n(before condensing)",
                  fontsize=9, fontweight="bold")
    ax2.set_xlabel(""); ax2.set_ylabel("Distance")
    ax2.tick_params(axis="x", labelsize=6)

    # 3. True labels
    Xp = PCA(n_components=2).fit_transform(Xs) if Xs.shape[1] > 2 else Xs
    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(Xp[:,0], Xp[:,1], c=y_true, cmap="tab10", s=18, alpha=0.8)
    ax.set_title("True labels", fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    # 4. HDBSCAN labels
    ax = fig.add_subplot(gs[1, 1])
    ax.scatter(Xp[:,0], Xp[:,1],
               c=cluster_palette(clf.labels_, clf.probabilities_),
               s=18, alpha=0.9)
    ax.set_title(f"HDBSCAN result  ARI={ari:.3f}\n"
                 "(opacity = membership probability)",
                 fontweight="bold", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

    # 5. Persistence bar chart
    ax = fig.add_subplot(gs[1, 2])
    if hasattr(clf, "cluster_persistence_") and len(clf.cluster_persistence_) > 0:
        ks   = list(range(len(clf.cluster_persistence_)))
        pers = clf.cluster_persistence_
        bars = ax.bar(ks, pers,
                      color=sns.color_palette("tab10", len(ks)),
                      edgecolor="k", alpha=0.85)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=9)
        ax.set_xlabel("Cluster"); ax.set_ylabel("Persistence (stability)")
        ax.set_title("Cluster persistence\nHigher = more real / stable",
                     fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "< 2 clusters", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")

    fname = f"plots/hdbscan/B_tree_{name.replace(' ','_')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {fname}")


# ════════════════════════════════════════════════════════════════════════════
# C — SOFT MEMBERSHIP PROBABILITIES
# ════════════════════════════════════════════════════════════════════════════

def soft_membership_analysis(X, y_true, min_cluster_size=20, name="dataset"):
    """
    Membership probabilities tell you HOW CONFIDENTLY a point belongs to
    its assigned cluster.  This is information DBSCAN and K-means never give.

    How to use probabilities in research
    ─────────────────────────────────────
    prob ≈ 1.0  → deep core member; archetypal representative of the state
    prob ≈ 0.5  → transitional configuration; on the boundary
    prob ≈ 0.0  → soft noise (assigned to nearest cluster but barely there)

    Practical applications
    ───────────────────────
    1. Filter to high-confidence representatives (prob > 0.8) for visualisation
    2. Build centroid features from only high-confidence members
    3. Identify transition-state configurations (prob 0.3–0.6, non-noise label)
    4. Weight cluster statistics by probability (e.g. probability-weighted mean)
    """
    # Section C studies not just whether a point belongs to a cluster, but how
    # strongly it belongs. This is where HDBSCAN starts behaving more like a
    # state-discovery tool than a rigid partitioner.
    print(f"\n[C] Soft membership analysis — {name}")
    Xs  = StandardScaler().fit_transform(X)
    clf = fit_hdbscan(Xs, min_cluster_size=min_cluster_size)

    labels = clf.labels_
    probs  = clf.probabilities_
    n_clus = len(set(labels)) - (1 if -1 in labels else 0)

    if n_clus < 2:
        print("  < 2 clusters, skipping")
        return

    Xp   = PCA(n_components=2).fit_transform(Xs) if Xs.shape[1] > 2 else Xs
    cmap = plt.cm.tab10

    print(f"  {n_clus} clusters found")
    print(f"  Prob distribution (non-noise):")
    non_noise = labels != -1
    thresholds = [0.8, 0.5, 0.3]
    for t in thresholds:
        frac = (probs[non_noise] >= t).mean()
        print(f"    prob ≥ {t}: {100*frac:.1f}% of cluster points")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f"Soft Membership Probabilities — {name}",
                 fontsize=13, fontweight="bold")

    # 1. All points, opacity = probability
    ax = axes[0, 0]
    ax.scatter(Xp[:,0], Xp[:,1],
               c=cluster_palette(labels, probs), s=20)
    ax.set_title("All points\n(opacity = membership probability)",
                 fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    # 2. High-confidence core members only (prob > 0.8)
    ax = axes[0, 1]
    core_mask = probs > 0.8
    ax.scatter(Xp[~core_mask, 0], Xp[~core_mask, 1],
               c="lightgray", s=10, alpha=0.3, label="low prob (<0.8)")
    ax.scatter(Xp[core_mask, 0], Xp[core_mask, 1],
               c=[cmap(int(l) % 10) for l in labels[core_mask]],
               s=25, alpha=0.9, label="high conf (≥0.8)")
    ax.set_title(f"High-confidence members (prob≥0.8)\n"
                 f"n={core_mask.sum()} / {len(labels)}",
                 fontweight="bold", fontsize=9)
    ax.legend(fontsize=8); ax.set_xticks([]); ax.set_yticks([])

    # 3. Transition-state points (0.2 < prob < 0.6, non-noise)
    trans_mask = (probs > 0.2) & (probs < 0.6) & (labels != -1)
    ax = axes[0, 2]
    ax.scatter(Xp[labels != -1, 0], Xp[labels != -1, 1],
               c=[cmap(int(l)%10) for l in labels[labels != -1]],
               s=12, alpha=0.25)
    ax.scatter(Xp[trans_mask, 0], Xp[trans_mask, 1],
               c="yellow", s=60, edgecolors="darkorange",
               lw=1.5, zorder=5,
               label=f"transitional (n={trans_mask.sum()})")
    ax.set_title("Transitional configurations\n"
                 "(prob 0.2–0.6: boundary / inter-state)",
                 fontweight="bold", fontsize=9)
    ax.legend(fontsize=8); ax.set_xticks([]); ax.set_yticks([])

    # 4. Probability histograms per cluster
    ax = axes[1, 0]
    for k in range(n_clus):
        m = labels == k
        if m.sum() > 0:
            ax.hist(probs[m], bins=20, alpha=0.6,
                    color=cmap(k % 10), edgecolor="k", lw=0.3,
                    label=f"C{k} (n={m.sum()})", density=True)
    ax.set_xlabel("Membership probability")
    ax.set_ylabel("Density")
    ax.set_title("Probability distribution per cluster\n"
                 "Left-skewed = diffuse  Right-skewed = compact",
                 fontweight="bold", fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 5. Probability-weighted cluster centroid vs unweighted
    ax = axes[1, 1]
    for k in range(n_clus):
        m = labels == k
        if m.sum() == 0: continue
        X_k  = Xp[m]
        p_k  = probs[m]
        c_uw = X_k.mean(axis=0)                              # unweighted
        c_w  = np.average(X_k, weights=p_k, axis=0)         # prob-weighted
        ax.scatter(*c_uw, s=200, marker="^",
                   color=cmap(k%10), edgecolors="k", lw=1.5,
                   zorder=6, label=f"C{k} unweighted")
        ax.scatter(*c_w,  s=200, marker="*",
                   color=cmap(k%10), edgecolors="white", lw=1,
                   zorder=7)
        ax.plot([c_uw[0], c_w[0]], [c_uw[1], c_w[1]],
                color=cmap(k%10), lw=1.5, ls="--", alpha=0.7)
    ax.scatter(Xp[:, 0], Xp[:, 1],
               c=cluster_palette(labels, probs), s=12, alpha=0.3, zorder=1)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0], marker="^", color="w", mfc="gray",
               ms=10, mec="k", label="Unweighted centroid"),
        Line2D([0],[0], marker="*", color="w", mfc="gray",
               ms=12, mec="w", label="Prob-weighted centroid"),
    ], fontsize=8)
    ax.set_title("Centroid shift: unweighted (△) vs prob-weighted (★)",
                 fontweight="bold", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

    # 6. Cumulative probability coverage
    ax = axes[1, 2]
    all_p = np.sort(probs[probs > 0])[::-1]
    ax.plot(np.linspace(0, 100, len(all_p)), np.cumsum(all_p) / all_p.sum(),
            lw=2, color="steelblue")
    for t in [0.5, 0.8]:
        thresh_frac = 100 * (probs > t).mean()
        ax.axvline(thresh_frac, ls="--", lw=1.5,
                   label=f"prob>{t}: {thresh_frac:.0f}% of pts")
    ax.set_xlabel("% of points (highest prob first)")
    ax.set_ylabel("Cumulative probability mass")
    ax.set_title("Coverage: top N% of points carry\nwhat fraction of membership?",
                 fontweight="bold", fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fname = f"plots/hdbscan/C_soft_membership_{name.replace(' ','_')}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {fname}")


# ════════════════════════════════════════════════════════════════════════════
# D — GLOSH OUTLIER SCORES
# ════════════════════════════════════════════════════════════════════════════

def outlier_score_analysis(X, y_true, min_cluster_size=15, name="dataset",
                           inject_outliers=True):
    """
    GLOSH (Global-Local Outlier Score from Hierarchies).

    What it measures
    ────────────────
    For each point: how anomalous is it relative to the densest cluster in
    its neighbourhood?  Score ∈ [0, 1].

    0.0 = perfectly typical, deep inside the densest part of a cluster
    1.0 = maximally anomalous — isolated point far from any dense region

    How GLOSH differs from other outlier methods
    ─────────────────────────────────────────────
    • Local Outlier Factor (LOF): compares density ratios of k-NN
    • Isolation Forest: based on random tree splits
    • GLOSH: comes free from HDBSCAN; uses the same density hierarchy

    Threshold for outlier flagging
    ───────────────────────────────
    Common practice: flag points with score > np.percentile(scores, 90)
    For soft matter: flag top 1-5% as anomalous configurations.
    """
    # Section D reuses the same HDBSCAN fit to get anomaly scores "for free".
    # This is powerful because clustering and outlier detection are now tied to
    # the same density model.
    print(f"\n[D] GLOSH outlier analysis — {name}")

    Xs = StandardScaler().fit_transform(X)

    # Optionally inject known outliers for benchmarking
    y_anom = np.zeros(len(Xs), dtype=int)
    if inject_outliers:
        n_inj = max(5, len(Xs) // 30)
        inj   = RNG.uniform(Xs.min()*1.5, Xs.max()*1.5, (n_inj, Xs.shape[1]))
        Xs    = np.vstack([Xs, inj])
        y_anom = np.concatenate([y_anom, np.ones(n_inj, int)])
        print(f"  Injected {n_inj} known outliers for evaluation")

    clf = fit_hdbscan(Xs, min_cluster_size=min_cluster_size)
    scores = clf.outlier_scores_

    # Choose threshold (90th percentile)
    threshold   = np.percentile(scores, 90)
    flagged     = scores >= threshold
    n_flagged   = flagged.sum()

    if inject_outliers:
        # How many injected outliers are in top-10%?
        n_known     = y_anom.sum()
        n_recovered = y_anom[flagged].sum()
        recall      = n_recovered / n_known if n_known > 0 else 0
        print(f"  Threshold (90th pct): {threshold:.4f}")
        print(f"  Flagged: {n_flagged}  "
              f"Recovered injected: {n_recovered}/{n_known}  "
              f"Recall={recall:.3f}")
    else:
        print(f"  Threshold (90th pct): {threshold:.4f}")
        print(f"  Flagged as anomalous: {n_flagged} ({100*n_flagged/len(Xs):.1f}%)")

    Xp = PCA(n_components=2).fit_transform(Xs) if Xs.shape[1] > 2 else Xs

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f"GLOSH Outlier Scores — {name}", fontsize=13, fontweight="bold")

    # 1. GLOSH score map
    ax = axes[0, 0]
    sc = ax.scatter(Xp[:,0], Xp[:,1], c=scores, cmap="hot_r",
                    s=18, vmin=0, vmax=1, alpha=0.85)
    plt.colorbar(sc, ax=ax, label="GLOSH score")
    ax.set_title("GLOSH scores on all points\nHot = anomalous",
                 fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    # 2. Flagged outliers highlighted
    ax = axes[0, 1]
    ax.scatter(Xp[~flagged, 0], Xp[~flagged, 1],
               c=cluster_palette(clf.labels_[~flagged]),
               s=18, alpha=0.7, label="normal")
    ax.scatter(Xp[flagged, 0], Xp[flagged, 1],
               c="red", s=80, marker="D", edgecolors="darkred",
               lw=1.2, zorder=6, label=f"outlier (top 10%)")
    if inject_outliers:
        known = y_anom.astype(bool)
        ax.scatter(Xp[known, 0], Xp[known, 1],
                   s=120, marker="*", c="yellow", edgecolors="k",
                   lw=1, zorder=7, label="injected outliers")
    ax.set_title(f"Flagged outliers (score > {threshold:.3f})",
                 fontweight="bold")
    ax.legend(fontsize=8); ax.set_xticks([]); ax.set_yticks([])

    # 3. Score distribution
    ax = axes[0, 2]
    ax.hist(scores[~flagged], bins=40, color="steelblue",
            alpha=0.7, edgecolor="k", lw=0.3,
            density=True, label="normal")
    ax.hist(scores[flagged],  bins=20, color="tomato",
            alpha=0.8, edgecolor="k", lw=0.3,
            density=True, label=f"flagged (≥{threshold:.2f})")
    ax.axvline(threshold, color="red", ls="--", lw=2,
               label="threshold")
    ax.set_xlabel("GLOSH score"); ax.set_ylabel("Density")
    ax.set_title("Score distribution\n"
                 "Bimodal = clear separation normal/anomalous",
                 fontweight="bold", fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 4. Score vs membership probability scatter
    ax = axes[1, 0]
    ax.scatter(clf.probabilities_[~flagged], scores[~flagged],
               c="steelblue", s=12, alpha=0.4, label="normal")
    ax.scatter(clf.probabilities_[flagged],  scores[flagged],
               c="tomato", s=50, marker="D", edgecolors="darkred",
               lw=0.8, alpha=0.9, label="flagged")
    ax.set_xlabel("Membership probability")
    ax.set_ylabel("GLOSH outlier score")
    ax.set_title("Probability vs Outlier score\n"
                 "Top-left = noise / outlier\n"
                 "Bottom-right = core cluster member",
                 fontweight="bold", fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 5. Sensitivity: % flagged vs threshold percentile
    pcts    = list(range(70, 99, 2))
    n_flags = [(scores >= np.percentile(scores, p)).sum() for p in pcts]
    ax = axes[1, 1]
    ax.plot(pcts, n_flags, "o-", color="tomato", lw=2, ms=7)
    ax.axvline(90, color="blue", ls="--", lw=2, label="90th pct (default)")
    ax.set_xlabel("Threshold percentile")
    ax.set_ylabel("# flagged outliers")
    ax.set_title("Flagging rate vs threshold\n"
                 "Choose based on expected anomaly rate",
                 fontweight="bold", fontsize=9)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # 6. Score sorted (anomaly ranking)
    ax = axes[1, 2]
    sorted_scores = np.sort(scores)[::-1]
    ax.semilogy(np.arange(1, len(sorted_scores)+1), sorted_scores + 1e-6,
                color="steelblue", lw=2)
    ax.axhline(threshold, color="red", ls="--", lw=2,
               label=f"threshold={threshold:.3f}")
    ax.axvline(n_flagged, color="red", ls=":", lw=1.5)
    ax.set_xlabel("Point rank (highest score first)")
    ax.set_ylabel("GLOSH score (log)")
    ax.set_title("Anomaly ranking (log scale)\n"
                 "Steep drop = clear outlier/normal separation",
                 fontweight="bold", fontsize=9)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fname = f"plots/hdbscan/D_outliers_{name.replace(' ','_')}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {fname}")


# ════════════════════════════════════════════════════════════════════════════
# E — PARAMETER SENSITIVITY
# ════════════════════════════════════════════════════════════════════════════

def parameter_sensitivity(X, y_true, name="dataset"):
    """
    Sweep min_cluster_size and min_samples.

    Key difference from DBSCAN sensitivity:
    ─────────────────────────────────────────
    HDBSCAN is MUCH less sensitive to parameters than DBSCAN.
    A 2× change in min_cluster_size rarely changes the result dramatically.
    A 5× change in DBSCAN's eps can flip results completely.

    Reading the grid
    ────────────────
    min_cluster_size too small  → many spurious micro-clusters
    min_cluster_size too large  → real small clusters absorbed as noise
    min_samples too large       → dense cores required → more noise
    min_samples too small       → borderline points pulled into clusters
    """
    # Section E explicitly explores hyperparameter robustness.
    # This section is important because users often over-trust a single run.
    print(f"\n[E] Parameter sensitivity — {name}")
    Xs = StandardScaler().fit_transform(X)

    mcs_vals = [5, 10, 20, 35, 60]
    ms_vals  = [None, 3, 8, 15]     # None = default (= min_cluster_size)

    n_c  = np.zeros((len(ms_vals), len(mcs_vals)), int)
    nf   = np.zeros((len(ms_vals), len(mcs_vals)))
    aris = np.zeros((len(ms_vals), len(mcs_vals)))

    for i, ms in enumerate(ms_vals):
        for j, mcs in enumerate(mcs_vals):
            clf  = fit_hdbscan(Xs, min_cluster_size=mcs, min_samples=ms,
                               gen_min_span_tree=False)
            lbl  = clf.labels_
            nc   = len(set(lbl)) - (1 if -1 in lbl else 0)
            n_c[i, j] = nc
            nf [i, j] = (lbl == -1).sum() / len(lbl)
            if y_true is not None and nc > 1:
                valid = lbl != -1
                if valid.sum() > 5:
                    aris[i, j] = adjusted_rand_score(y_true[valid], lbl[valid])

    ms_labels = [f"ms=default" if ms is None else f"ms={ms}"
                 for ms in ms_vals]

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    fig.suptitle(f"HDBSCAN Parameter Sensitivity — {name}",
                 fontsize=13, fontweight="bold")

    def hm(ax, data, title, fmt, cmap, vmin=None, vmax=None):
        df = pd.DataFrame(data, index=ms_labels,
                          columns=[f"mcs={v}" for v in mcs_vals])
        sns.heatmap(df, annot=True, fmt=fmt, cmap=cmap, ax=ax,
                    vmin=vmin, vmax=vmax, linewidths=0.5, linecolor="gray",
                    cbar_kws={"shrink": 0.8})
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_xlabel("min_cluster_size"); ax.set_ylabel("min_samples")

    hm(axes[0], n_c,  "# Clusters found",       "d",   "Blues")
    hm(axes[1], nf,   "Noise fraction",          ".2f", "Reds",   0, 0.5)
    hm(axes[2], aris, "ARI vs true labels",      ".2f", "RdYlGn", 0, 1)

    fname = f"plots/hdbscan/E_sensitivity_{name.replace(' ','_')}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {fname}")


# ════════════════════════════════════════════════════════════════════════════
# F — DBSCAN vs HDBSCAN: WHEN HDBSCAN WINS
# ════════════════════════════════════════════════════════════════════════════

def dbscan_vs_hdbscan():
    """
    The one dataset that exposes DBSCAN's fundamental limitation:
    clusters at different density scales.

    Dataset: 3 clusters
      - Tight, dense cluster (small spread)
      - Medium cluster
      - Diffuse, spread-out cluster (large spread)

    DBSCAN: any single ε either misses the diffuse cluster (too small ε)
            or merges the tight clusters (too large ε).
    HDBSCAN: handles all three densities simultaneously.
    """
    # Section F is a controlled thought experiment: construct data that is
    # intentionally hard for single-scale DBSCAN and easy for multi-scale HDBSCAN.
    print("\n[F] DBSCAN vs HDBSCAN — varying density showcase")

    # Build three clusters at very different densities
    X0, _ = make_blobs(n_samples=200, centers=[[0,  0]], cluster_std=0.3,
                       random_state=SEED)
    X1, _ = make_blobs(n_samples=200, centers=[[5,  0]], cluster_std=0.9,
                       random_state=SEED)
    X2, _ = make_blobs(n_samples=200, centers=[[2.5, 5]], cluster_std=2.2,
                       random_state=SEED)
    X     = np.vstack([X0, X1, X2])
    y     = np.array([0]*200 + [1]*200 + [2]*200)
    Xs    = StandardScaler().fit_transform(X)

    # DBSCAN at three ε values showing the dilemma
    from sklearn.neighbors import NearestNeighbors
    nbrs   = NearestNeighbors(n_neighbors=8).fit(Xs)
    dists, _ = nbrs.kneighbors(Xs)
    k_d    = np.sort(dists[:, -1])[::-1]
    eps_lo = float(np.percentile(k_d, 15))
    eps_md = float(np.percentile(k_d, 40))
    eps_hi = float(np.percentile(k_d, 70))

    db_lo = DBSCAN(eps=eps_lo, min_samples=8).fit_predict(Xs)
    db_md = DBSCAN(eps=eps_md, min_samples=8).fit_predict(Xs)
    db_hi = DBSCAN(eps=eps_hi, min_samples=8).fit_predict(Xs)

    # HDBSCAN — one call, no ε
    clf_h = fit_hdbscan(Xs, min_cluster_size=20, gen_min_span_tree=False)

    def nc_ari(lbl):
        nc  = len(set(lbl)) - (1 if -1 in lbl else 0)
        nns = (lbl == -1).sum()
        valid = lbl != -1
        ari = adjusted_rand_score(y[valid], lbl[valid]) if valid.sum() > 5 else 0.0
        return nc, nns, ari

    cases = [
        (y,          "True labels",    None),
        (db_lo,      f"DBSCAN ε={eps_lo:.2f}\n(ε too small)", None),
        (db_md,      f"DBSCAN ε={eps_md:.2f}\n(ε medium)",    None),
        (db_hi,      f"DBSCAN ε={eps_hi:.2f}\n(ε too large)", None),
        (clf_h.labels_,"HDBSCAN\n(no ε needed)", clf_h),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle("Varying-Density Challenge: DBSCAN vs HDBSCAN",
                 fontsize=13, fontweight="bold")

    for ax, (lbl, title, clf) in zip(axes, cases):
        if clf is None:
            ax.scatter(X[:,0], X[:,1], c=lbl, cmap="tab10",
                       s=18, alpha=0.8, vmin=-1)
        else:
            ax.scatter(X[:,0], X[:,1],
                       c=cluster_palette(lbl, clf.probabilities_),
                       s=18, alpha=0.9)
        if lbl is not y:
            nc, nns, ari = nc_ari(lbl)
            tc = "forestgreen" if ari > 0.85 else "red" if ari < 0.5 else "orange"
            ax.set_title(f"{title}\n"
                         f"K={nc}  noise={nns}  ARI={ari:.3f}",
                         fontsize=9, fontweight="bold", color=tc)
            print(f"  {title.split(chr(10))[0]:<20}: "
                  f"K={nc}  noise={nns}  ARI={ari:.3f}")
        else:
            ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

    # Annotate density circles (approximate extents)
    circle_specs = [
        (0.0,  0.0, 0.9),
        (5.0,  0.0, 2.7),
        (2.5,  5.0, 6.6),
    ]
    for ax in axes:
        for cx_, cy_, r_ in circle_specs:
            circle = plt.Circle((cx_, cy_), r_, fill=False, color="white",
                                ls=":", lw=1.5, alpha=0.4, transform=ax.transData)
            ax.add_patch(circle)

    plt.tight_layout()
    plt.savefig("plots/hdbscan/F_dbscan_vs_hdbscan.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved → plots/hdbscan/F_dbscan_vs_hdbscan.png")


# ════════════════════════════════════════════════════════════════════════════
# G — SOFT-MATTER: MD CONFORMATIONAL STATE EXTRACTION
# ════════════════════════════════════════════════════════════════════════════

def md_conformational_states():
    """
    Conformational clustering from a simulated MD trajectory.

    Physical setup
    ──────────────
    Polymer backbone described by: φ, ψ (dihedral angles), Rg, d_ee.
    MD produces a time series of frames, each a point in feature space.
    Frames cluster into metastable basins = conformational states.

    Why HDBSCAN is ideal here
    ─────────────────────────
    1. The basins have different population densities (helical state
       is more populated than rare kinked state).
    2. Transition-state frames are genuinely in between — soft membership
       correctly gives them low probability.
    3. Outlier scores flag configurations that have never been seen before
       (useful for enhanced sampling).
    4. No need to pre-specify K (unknown in real MD).

    Research inference extracted
    ────────────────────────────
    - State centroids (representative structures to visualise/save)
    - State populations = Boltzmann weights
    - Transition-state frames (low prob, non-noise label)
    - Rare/anomalous frames (high GLOSH score)
    - Cluster persistence = metastability (longer-lived = more stable)
    """
    # Section G reframes the abstract clustering result as a soft-matter /
    # molecular simulation interpretation problem.
    print("\n[G] MD conformational state extraction")
    X, y_true = load("polymer_conf")
    state_names = {0:"Extended(β)", 1:"Helical(α)", 2:"Collapsed", 3:"Kinked"}

    # Simulate MD time-ordering: interleave transitions
    # (in real MD the trajectory order matters; here we shuffle to simulate
    #  a converged ergodic trajectory)
    idx  = RNG.permutation(len(X))
    X    = X[idx]; y_true = y_true[idx]

    Xs  = StandardScaler().fit_transform(X)
    clf = fit_hdbscan(Xs, min_cluster_size=40, min_samples=10)

    labels = clf.labels_
    probs  = clf.probabilities_
    scores = clf.outlier_scores_

    n_states = len(set(labels)) - (1 if -1 in labels else 0)
    n_trans  = labels == -1
    ari      = (adjusted_rand_score(y_true[labels != -1], labels[labels != -1])
                if n_states > 1 else 0.0)

    print(f"  States found: {n_states}  noise={n_trans.sum()}  ARI={ari:.4f}")

    # ── Physical inference ────────────────────────────────────────────────────
    # 1. State populations (prob-weighted)
    print("\n  State populations (probability-weighted):")
    # These are human-readable names for the four MD features.
    # `feat_names` is not used later in the current code, so it has no direct
    # effect on the outcome. It is informational only.
    feat_names = ["φ", "ψ", "Rg", "d_ee"]

    # Fit a scaler in original feature space so that probability-weighted
    # centroids computed in original coordinates can be projected back into PCA
    # space for plotting.
    scaler     = StandardScaler().fit(X)

    state_info = []
    for k in sorted(set(labels)):
        if k == -1: continue
        m      = labels == k
        pop    = probs[m].sum() / probs[probs > 0].sum()   # fraction
        # Prob-weighted centroid in original space
        Xk_orig = X[m]
        pk      = probs[m]
        centroid = np.average(Xk_orig, weights=pk, axis=0)
        n_core   = (probs[m] > 0.8).sum()
        n_trans_k= ((probs[m] > 0.1) & (probs[m] < 0.5)).sum()
        pers     = (clf.cluster_persistence_[k]
                    if k < len(clf.cluster_persistence_) else 0.0)
        state_info.append(dict(k=k, pop=pop, centroid=centroid,
                               n_core=n_core, n_trans=n_trans_k, pers=pers))
        print(f"  S{k}: pop={pop:.3f}  "
              f"n={m.sum()}  core(p>0.8)={n_core}  "
              f"trans(0.1<p<0.5)={n_trans_k}  "
              f"persist={pers:.4f}")

    # 2. Most anomalous frames
    top_anom = np.argsort(scores)[::-1][:5]
    print(f"\n  Top-5 anomalous frames (GLOSH):")
    for i in top_anom:
        print(f"    frame {i:>5}: score={scores[i]:.4f}  "
              f"label={labels[i]}  prob={probs[i]:.3f}")

    # ── Figure ────────────────────────────────────────────────────────────────
    pca = PCA(n_components=2)
    Xp  = pca.fit_transform(Xs)
    cmap = plt.cm.tab10

    fig  = plt.figure(figsize=(20, 14))
    gs   = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)
    fig.suptitle("HDBSCAN: MD Conformational State Analysis",
                 fontsize=13, fontweight="bold")

    # 1. True states
    ax = fig.add_subplot(gs[0, 0])
    for k, nm in state_names.items():
        m = y_true == k
        ax.scatter(Xp[m,0], Xp[m,1], c=[cmap(k)], s=12, alpha=0.55, label=nm)
    ax.set_title("True conformational states", fontweight="bold")
    ax.legend(fontsize=7); ax.grid(alpha=0.2)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")

    # 2. HDBSCAN labels with prob opacity
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(Xp[:,0], Xp[:,1],
               c=cluster_palette(labels, probs), s=12, alpha=0.9)
    # Mark state centroids
    for si in state_info:
        cp = pca.transform(scaler.transform([si["centroid"]]))[0]
        ax.scatter(*cp, s=300, marker="*", c=[cmap(si["k"] % 10)],
                   edgecolors="k", lw=1.5, zorder=8)
        ax.annotate(f"S{si['k']}", cp, fontsize=9, fontweight="bold",
                    ha="center", va="bottom",
                    xytext=(0, 8), textcoords="offset points")
    ax.set_title(f"HDBSCAN states  ARI={ari:.3f}\n"
                 "(opacity=prob, ★=centroid)", fontweight="bold", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

    # 3. Condensed tree
    ax = fig.add_subplot(gs[0, 2])
    plot_condensed_tree(clf, ax, title="Condensed tree\n(cluster hierarchy)")

    # 4. Transition frames
    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(Xp[~n_trans, 0], Xp[~n_trans, 1],
               c=cluster_palette(labels[~n_trans], probs[~n_trans]),
               s=12, alpha=0.4)
    ax.scatter(Xp[n_trans, 0], Xp[n_trans, 1],
               c="red", s=50, marker="D", edgecolors="darkred",
               lw=0.8, zorder=6, label=f"noise/transition ({n_trans.sum()})")
    ax.set_title("Noise = transition-state frames",
                 fontweight="bold", fontsize=9)
    ax.legend(fontsize=8); ax.set_xticks([]); ax.set_yticks([])

    # 5. State population pie
    ax = fig.add_subplot(gs[1, 1])
    pops  = [si["pop"] for si in state_info]
    ks    = [si["k"]   for si in state_info]
    cols  = [cmap(k % 10) for k in ks]
    ax.pie(pops,
           labels=[f"S{k}\n{p:.1%}" for k, p in zip(ks, pops)],
           colors=cols, autopct="", startangle=90,
           wedgeprops={"edgecolor":"k","lw":0.8})
    ax.set_title("Probability-weighted\nstate populations",
                 fontweight="bold", fontsize=9)

    # 6. GLOSH score trajectory
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(np.arange(len(scores)), scores, lw=0.8, color="steelblue", alpha=0.7)
    threshold_g = float(np.percentile(scores, 95))
    ax.axhline(threshold_g, color="red", ls="--", lw=1.5,
               label=f"95th pct = {threshold_g:.3f}")
    ax.scatter(top_anom, scores[top_anom],
               c="red", s=80, zorder=6, label="Top 5 anomalies")
    ax.set_xlabel("MD frame index"); ax.set_ylabel("GLOSH outlier score")
    ax.set_title("Anomaly score along trajectory\n"
                 "Peaks = rare/transition configurations",
                 fontweight="bold", fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.savefig("plots/hdbscan/G_md_conformations.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved → plots/hdbscan/G_md_conformations.png")


# ════════════════════════════════════════════════════════════════════════════
# H — SOFT-MATTER: MULTI-SCALE COLLOIDAL PHASES
# ════════════════════════════════════════════════════════════════════════════

def colloidal_phases_hdbscan():
    """
    Compare HDBSCAN 'eom' vs 'leaf' selection on colloidal phase data.

    eom  (Excess of Mass):   finds the 3 main phases (coarse view)
    leaf (finest clusters):  finds sub-structure within each phase

    This directly maps to a research question:
    "Are there sub-populations within the liquid phase?
     (e.g., gel-like vs fluid-like regions)"
    """
    # Section H contrasts two hierarchy-flattening strategies on the same data.
    # This directly shows that cluster extraction is not only about density
    # estimation; it is also about how you choose to cut the hierarchy.
    print("\n[H] Colloidal phases — eom vs leaf selection")
    X, y_true = load("colloidal_phases")
    feat_names = ["ψ₆", "ρ_local", "Q_nematic"]
    phase_names = {0:"Gas", 1:"Liquid", 2:"Crystal"}
    Xs = StandardScaler().fit_transform(X)

    clf_eom  = fit_hdbscan(Xs, min_cluster_size=40,
                            cluster_selection_method="eom",
                            gen_min_span_tree=False)
    clf_leaf = fit_hdbscan(Xs, min_cluster_size=40,
                            cluster_selection_method="leaf",
                            gen_min_span_tree=False)

    for name, clf in [("eom", clf_eom), ("leaf", clf_leaf)]:
        n_c  = len(set(clf.labels_)) - (1 if -1 in clf.labels_ else 0)
        valid = clf.labels_ != -1
        ari  = (adjusted_rand_score(y_true[valid], clf.labels_[valid])
                if n_c > 1 else 0.0)
        print(f"  {name}: K={n_c}  noise={((clf.labels_==-1).sum())}  "
              f"ARI={ari:.4f}")

    pca  = PCA(n_components=2)
    Xp   = pca.fit_transform(Xs)
    cmap = plt.cm.tab10

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("HDBSCAN: Colloidal Phases — EoM vs Leaf Selection",
                 fontsize=13, fontweight="bold")

    # True
    ax = axes[0, 0]
    for k, nm in phase_names.items():
        m = y_true == k
        ax.scatter(Xp[m,0], Xp[m,1], c=[cmap(k)],
                   s=12, alpha=0.55, label=nm)
    ax.set_title("True phase labels", fontweight="bold")
    ax.legend(fontsize=9); ax.set_xticks([]); ax.set_yticks([])

    # EoM result
    ax = axes[0, 1]
    valid = clf_eom.labels_ != -1
    ari_eom = adjusted_rand_score(y_true[valid], clf_eom.labels_[valid]) if valid.sum() > 5 else 0
    ax.scatter(Xp[:,0], Xp[:,1],
               c=cluster_palette(clf_eom.labels_, clf_eom.probabilities_),
               s=12, alpha=0.85)
    n_eom = len(set(clf_eom.labels_)) - (1 if -1 in clf_eom.labels_ else 0)
    ax.set_title(f"HDBSCAN EoM: K={n_eom}\nARI={ari_eom:.3f}  "
                 f"(coarse phases)",
                 fontweight="bold", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

    # Leaf result
    ax = axes[0, 2]
    valid = clf_leaf.labels_ != -1
    ari_leaf = adjusted_rand_score(y_true[valid], clf_leaf.labels_[valid]) if valid.sum() > 5 else 0
    ax.scatter(Xp[:,0], Xp[:,1],
               c=cluster_palette(clf_leaf.labels_, clf_leaf.probabilities_),
               s=12, alpha=0.85)
    n_leaf = len(set(clf_leaf.labels_)) - (1 if -1 in clf_leaf.labels_ else 0)
    ax.set_title(f"HDBSCAN Leaf: K={n_leaf}\nARI={ari_leaf:.3f}  "
                 f"(fine sub-structure)",
                 fontweight="bold", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

    # Per-cluster centroid analysis (eom)
    ax = axes[1, 0]
    scaler = StandardScaler().fit(X)
    centroids_orig = []
    for k in sorted(set(clf_eom.labels_)):
        if k == -1: continue
        m = clf_eom.labels_ == k
        c = np.average(X[m], weights=clf_eom.probabilities_[m], axis=0)
        centroids_orig.append((k, c))
    df_c = pd.DataFrame([c for _, c in centroids_orig],
                        index=[f"C{k}" for k, _ in centroids_orig],
                        columns=feat_names)
    sns.heatmap(df_c, annot=True, fmt=".3f", cmap="YlOrRd",
                ax=ax, linewidths=0.5)
    ax.set_title("EoM centroids (original units)\nProb-weighted means",
                 fontweight="bold", fontsize=9)

    # Membership probabilities histogram
    ax = axes[1, 1]
    for k in sorted(set(clf_eom.labels_)):
        if k == -1: continue
        m = clf_eom.labels_ == k
        ax.hist(clf_eom.probabilities_[m], bins=25, alpha=0.6,
                color=cmap(k % 10), edgecolor="k", lw=0.3,
                density=True, label=f"C{k}")
    ax.set_xlabel("Membership probability")
    ax.set_ylabel("Density")
    ax.set_title("Membership distributions per phase\n"
                 "Left-peak = diffuse phase boundary",
                 fontweight="bold", fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Outlier scores per phase
    ax = axes[1, 2]
    for k in sorted(set(clf_eom.labels_)):
        if k == -1: continue
        m = clf_eom.labels_ == k
        ax.hist(clf_eom.outlier_scores_[m], bins=25, alpha=0.6,
                color=cmap(k % 10), edgecolor="k", lw=0.3,
                density=True, label=f"C{k}")
    ax.axvline(np.percentile(clf_eom.outlier_scores_, 95),
               color="red", ls="--", lw=1.5, label="95th pct")
    ax.set_xlabel("GLOSH outlier score")
    ax.set_ylabel("Density")
    ax.set_title("Outlier scores per phase\n"
                 "Right tail = anomalous configurations",
                 fontweight="bold", fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/hdbscan/H_colloidal_phases.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved → plots/hdbscan/H_colloidal_phases.png")


# ════════════════════════════════════════════════════════════════════════════
# I — ANOMALY DETECTION IN EXPERIMENTAL TRAJECTORIES
# ════════════════════════════════════════════════════════════════════════════

def anomaly_detection_pipeline():
    """
    Full anomaly detection pipeline mimicking an experimental time series.

    Scenario
    ────────
    A colloidal experiment runs for N frames. Most frames are in one of
    three normal phases. A few rare frames represent:
      (a) Phase transition events (intermediate states)
      (b) Measurement artefacts (sensor noise spike)
      (c) Novel phases never seen in training

    Pipeline
    ────────
    1. Fit HDBSCAN on training frames
    2. Use GLOSH to score ALL frames (train + test)
    3. Threshold to flag anomalies
    4. Separate the flagged frames by type (transition vs artefact)
    5. Report and visualise
    """
    # Section I simulates a train/test anomaly-detection setting.
    # Unlike earlier sections, it asks: "can a model fit on normal data help us
    # identify unusual future observations?"
    print("\n[I] Anomaly detection pipeline")
    X_train, y_train = load("colloidal_phases")
    Xs_train = StandardScaler().fit_transform(X_train)

    # Simulate test data: normal + anomalies
    # Type 1: transition events (between phase 0 and 1)
    n_trans  = 30
    X_trans  = (X_train[y_train == 0][:n_trans] +
                X_train[y_train == 1][:n_trans]) / 2
    X_trans += RNG.normal(0, 0.05, X_trans.shape)

    # Type 2: artefacts (random)
    n_art   = 20
    X_art   = RNG.uniform(X_train.min(axis=0), X_train.max(axis=0),
                           (n_art, X_train.shape[1]))

    # Type 3: novel phase (different feature range)
    n_novel = 25
    X_novel = RNG.normal([0.95, 0.30, 0.95], 0.04, (n_novel, 3))

    X_test  = np.vstack([X_trans, X_art, X_novel])
    y_type  = np.array(["transition"]*n_trans + ["artefact"]*n_art +
                       ["novel phase"]*n_novel)
    is_anom = np.ones(len(X_test), bool)

    # All data combined
    scaler   = StandardScaler().fit(X_train)
    Xs_all   = scaler.transform(np.vstack([X_train, X_test]))
    y_all    = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_test))])

    # Fit on training only, score on all
    clf      = fit_hdbscan(scaler.transform(X_train),
                           min_cluster_size=40, gen_min_span_tree=False)
    # Predict labels for test points using approximate prediction
    # Approximate prediction assigns new points to the nearest compatible
    # cluster learned from training data. This does NOT refit the model.
    test_lbl, test_prob = hdbscan_lib.approximate_predict(
        clf, scaler.transform(X_test)
    )

    # This computes soft cluster-membership vectors for test points.
    # In this script the resulting variable is never used later, so it has no
    # effect on the final figures or metrics. It is still conceptually useful
    # because it shows how one could inspect soft assignment to known clusters.
    test_scores = hdbscan_lib.membership_vector(clf, scaler.transform(X_test))

    # GLOSH on full dataset (refit including test)
    clf_full = fit_hdbscan(Xs_all, min_cluster_size=40,
                           gen_min_span_tree=False)
    scores_all = clf_full.outlier_scores_
    score_test = scores_all[len(X_train):]

    threshold   = np.percentile(scores_all[:len(X_train)], 97)
    flagged_idx = np.where(score_test >= threshold)[0]
    print(f"  Training threshold (97th pct): {threshold:.4f}")
    print(f"  Flagged test points: {len(flagged_idx)} / {len(X_test)}")

    # How well did we recover each anomaly type?
    for atype in ["transition", "artefact", "novel phase"]:
        m     = y_type == atype
        n_tot = m.sum()
        n_rec = (score_test[m] >= threshold).sum()
        print(f"    {atype:<14}: {n_rec}/{n_tot} recovered "
              f"({100*n_rec/n_tot:.0f}%)")

    pca = PCA(n_components=2)
    Xp  = pca.fit_transform(Xs_all)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("HDBSCAN Anomaly Detection in Experimental Trajectory",
                 fontsize=12, fontweight="bold")

    cmap = plt.cm.tab10
    type_cols = {"transition": "gold", "artefact": "red", "novel phase": "cyan"}

    # 1. GLOSH on all points
    ax = axes[0]
    sc = ax.scatter(Xp[:,0], Xp[:,1], c=scores_all, cmap="hot_r",
                    s=12, vmin=0, vmax=1, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="GLOSH score")
    ax.set_title("GLOSH scores (train + test)",
                 fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    # 2. Training vs test, anomaly types
    ax = axes[1]
    train_idx = np.arange(len(X_train))
    ax.scatter(Xp[train_idx, 0], Xp[train_idx, 1],
               c=cluster_palette(clf_full.labels_[:len(X_train)]),
               s=10, alpha=0.3, label="train (normal)")
    test_start = len(X_train)
    for atype in ["transition", "artefact", "novel phase"]:
        m = y_type == atype
        i = np.where(m)[0] + test_start
        ax.scatter(Xp[i, 0], Xp[i, 1],
                   c=type_cols[atype], s=60, marker="*",
                   edgecolors="k", lw=0.8, zorder=6,
                   label=atype)
    ax.set_title("Anomaly types in feature space",
                 fontweight="bold")
    ax.legend(fontsize=8); ax.set_xticks([]); ax.set_yticks([])

    # 3. Score distribution: train vs each anomaly type
    ax = axes[2]
    ax.hist(scores_all[:len(X_train)], bins=40, color="steelblue",
            alpha=0.6, density=True, label="training (normal)")
    for atype, col in type_cols.items():
        m = y_type == atype
        ax.hist(score_test[m], bins=15, color=col,
                alpha=0.7, density=True, label=atype, edgecolor="k", lw=0.3)
    ax.axvline(threshold, color="red", ls="--", lw=2,
               label=f"threshold={threshold:.3f}")
    ax.set_xlabel("GLOSH score"); ax.set_ylabel("Density")
    ax.set_title("Score distributions by type\n"
                 "Rightward shift = more anomalous",
                 fontweight="bold")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/hdbscan/I_anomaly_detection.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved → plots/hdbscan/I_anomaly_detection.png")


# ════════════════════════════════════════════════════════════════════════════
# J — FULL INFERENCE PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def full_inference_pipeline():
    """
    Complete research-grade HDBSCAN workflow from raw data to interpretation.

    Steps
    ─────
    1.  Load and standardise
    2.  Parameter scan (min_cluster_size)
    3.  Fit HDBSCAN with best parameters
    4.  Validate: silhouette, ARI if ground truth available
    5.  Characterise clusters: centroid, spread, compactness, population
    6.  Identify core representatives (prob > 0.8)
    7.  Identify transition configurations (0.2 < prob < 0.5)
    8.  Flag anomalies (GLOSH > 95th pct)
    9.  Summarise as a physical table
    10. Visualise everything
    """
    # Section J assembles the tutorial pieces into a research-style workflow:
    # select parameters, fit, validate, summarize, detect anomalies, and present.
    print("\n[J] Full inference pipeline — colloidal phases")
    X, y_true = load("colloidal_phases")
    feat_names  = ["ψ₆", "ρ_local", "Q_nematic"]
    phase_names = {0:"Gas", 1:"Liquid", 2:"Crystal"}

    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)

    # ── Step 2: parameter scan ────────────────────────────────────────────────
    print("  Scanning min_cluster_size...")
    best_mcs, best_ari = 5, -1
    scan_results = []
    for mcs in [10, 20, 30, 50, 75, 100]:
        clf  = fit_hdbscan(Xs, min_cluster_size=mcs, gen_min_span_tree=False)
        lbl  = clf.labels_
        nc   = len(set(lbl)) - (1 if -1 in lbl else 0)
        valid = lbl != -1
        ari  = (adjusted_rand_score(y_true[valid], lbl[valid])
                if nc > 1 and valid.sum() > 5 else 0.0)
        sil  = (silhouette_score(Xs[valid], lbl[valid])
                if nc > 1 and valid.sum() > 5 else 0.0)
        scan_results.append((mcs, nc, ari, sil))
        print(f"    mcs={mcs:>4}: K={nc}  ARI={ari:.4f}  Sil={sil:.4f}")
        if ari > best_ari:
            best_ari = ari; best_mcs = mcs

    print(f"  → Best mcs={best_mcs}  ARI={best_ari:.4f}")

    # ── Step 3: fit final model ───────────────────────────────────────────────
    clf     = fit_hdbscan(Xs, min_cluster_size=best_mcs, gen_min_span_tree=True)
    labels  = clf.labels_
    probs   = clf.probabilities_
    scores  = clf.outlier_scores_
    n_clus  = len(set(labels)) - (1 if -1 in labels else 0)

    # ── Steps 5–8: characterise ───────────────────────────────────────────────
    thresh_outlier = np.percentile(scores, 95)
    results_table  = []

    print(f"\n  {'State':>6} {'N':>6} {'Pop%':>6} "
          f"{'ψ₆':>8} {'ρ':>8} {'Q':>8} "
          f"{'Core%':>7} {'Trans%':>8} {'Sil':>6} {'Persist':>8}")
    print("  " + "─" * 72)

    valid_all = labels != -1
    for k in sorted(set(labels)):
        if k == -1: continue
        m    = labels == k
        Xk   = X[m]; pk = probs[m]
        pop  = m.sum() / valid_all.sum()
        cent = np.average(Xk, weights=pk, axis=0)
        core_f  = (pk > 0.8).mean()
        trans_f = ((pk > 0.1) & (pk < 0.5)).mean()
        sil_k   = 0.0  # will be overwritten below if valid
        pers    = (clf.cluster_persistence_[k]
                   if k < len(clf.cluster_persistence_) else 0.0)

        # Silhouette for this cluster vs others
        from sklearn.metrics import silhouette_samples
        if valid_all.sum() > n_clus and n_clus > 1:
            sv_k = silhouette_samples(Xs[valid_all], labels[valid_all])
            nn_m = np.where(valid_all)[0]
            # get indices of k-cluster within valid array
            k_valid = np.array([i for i,idx in enumerate(np.where(valid_all)[0])
                                 if labels[idx] == k])
            sil_k   = sv_k[k_valid].mean() if len(k_valid) > 0 else 0.0

        print(f"  {k:>6} {m.sum():>6} {pop:>6.1%} "
              f"{cent[0]:>8.4f} {cent[1]:>8.4f} {cent[2]:>8.4f} "
              f"{core_f:>7.2f} {trans_f:>8.2f} "
              f"{sil_k:>6.3f} {pers:>8.4f}")
        results_table.append(dict(
            State=k, N=m.sum(), Pop=pop, psi6=cent[0], rho=cent[1],
            Q=cent[2], Core=core_f, Trans=trans_f, Sil=sil_k, Persist=pers
        ))

    noise_m   = labels == -1
    anom_m    = scores >= thresh_outlier
    print(f"\n  Noise points: {noise_m.sum()} ({100*noise_m.mean():.1f}%)")
    print(f"  Anomalies (GLOSH>95th): {anom_m.sum()} ({100*anom_m.mean():.1f}%)")

    # ── Visualise ─────────────────────────────────────────────────────────────
    pca  = PCA(n_components=2)
    Xp   = pca.fit_transform(Xs)
    cmap = plt.cm.tab10

    fig  = plt.figure(figsize=(20, 14))
    gs   = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.38)
    fig.suptitle("Full HDBSCAN Inference Pipeline — Colloidal Phases",
                 fontsize=13, fontweight="bold")

    # 1. Parameter scan
    ax = fig.add_subplot(gs[0, 0])
    mcs_s, aris_s, sils_s = zip(*[(r[0],r[2],r[3]) for r in scan_results])
    ax.plot(mcs_s, aris_s, "o-", color="steelblue", lw=2, ms=8, label="ARI")
    ax.plot(mcs_s, sils_s, "s--", color="darkorange", lw=2, ms=8, label="Silhouette")
    ax.axvline(best_mcs, color="red", ls=":", lw=2, label=f"best={best_mcs}")
    ax.set_xlabel("min_cluster_size"); ax.set_ylabel("Score")
    ax.set_title("Parameter scan", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 2. True labels
    ax = fig.add_subplot(gs[0, 1])
    for k, nm in phase_names.items():
        m = y_true == k
        ax.scatter(Xp[m,0], Xp[m,1], c=[cmap(k)], s=12, alpha=0.55, label=nm)
    ax.set_title("True phases", fontweight="bold")
    ax.legend(fontsize=8); ax.set_xticks([]); ax.set_yticks([])

    # 3. HDBSCAN result
    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(Xp[:,0], Xp[:,1],
               c=cluster_palette(labels, probs), s=12, alpha=0.85)
    valid = labels != -1
    ari_final = adjusted_rand_score(y_true[valid], labels[valid]) if valid.sum() > 5 else 0
    ax.set_title(f"HDBSCAN mcs={best_mcs}\nARI={ari_final:.3f}",
                 fontweight="bold", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

    # 4. Condensed tree
    ax = fig.add_subplot(gs[0, 3])
    plot_condensed_tree(clf, ax, title="Condensed tree")

    # 5. Membership probabilities
    ax = fig.add_subplot(gs[1, 0])
    sc = ax.scatter(Xp[:,0], Xp[:,1], c=probs,
                    cmap="plasma", s=12, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label="Membership prob")
    ax.set_title("Membership probabilities", fontweight="bold", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

    # 6. GLOSH scores
    ax = fig.add_subplot(gs[1, 1])
    sc = ax.scatter(Xp[:,0], Xp[:,1], c=scores,
                    cmap="hot_r", s=12, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label="GLOSH score")
    ax.set_title("GLOSH outlier scores", fontweight="bold", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

    # 7. Transition + anomaly overlay
    ax = fig.add_subplot(gs[1, 2])
    ax.scatter(Xp[valid_all, 0], Xp[valid_all, 1],
               c=cluster_palette(labels[valid_all], probs[valid_all]),
               s=10, alpha=0.35)
    trans_m = (probs > 0.1) & (probs < 0.5) & (labels != -1)
    ax.scatter(Xp[trans_m, 0], Xp[trans_m, 1],
               c="yellow", s=50, edgecolors="orange", lw=1,
               zorder=5, label=f"transition (n={trans_m.sum()})")
    ax.scatter(Xp[anom_m, 0], Xp[anom_m, 1],
               c="red", s=70, marker="D", edgecolors="darkred",
               lw=0.8, zorder=6, label=f"anomaly (n={anom_m.sum()})")
    ax.set_title("Transition (○) and anomaly (◆) overlay",
                 fontweight="bold", fontsize=9)
    ax.legend(fontsize=8); ax.set_xticks([]); ax.set_yticks([])

    # 8. Physical summary table
    ax = fig.add_subplot(gs[1, 3])
    ax.axis("off")
    if results_table:
        col_labels = ["State","N","Pop","ψ₆","ρ","Q","Core%","Sil"]
        rows = [[f"S{r['State']}",
                 str(r['N']),
                 f"{r['Pop']:.0%}",
                 f"{r['psi6']:.3f}",
                 f"{r['rho']:.3f}",
                 f"{r['Q']:.3f}",
                 f"{r['Core']:.0%}",
                 f"{r['Sil']:.3f}"]
                for r in results_table]
        tbl = ax.table(cellText=rows, colLabels=col_labels,
                       loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.2, 1.6)
        # Header row colour
        for j in range(len(col_labels)):
            tbl[0, j].set_facecolor("#2c3e50")
            tbl[0, j].set_text_props(color="white", fontweight="bold")
        ax.set_title("Physical summary table", fontweight="bold", fontsize=9)

    plt.savefig("plots/hdbscan/J_full_pipeline.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved → plots/hdbscan/J_full_pipeline.png")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# MAIN EXECUTION BLOCK
# ---------------------------------------------------------------------------
# This is where the tutorial is actually run end-to-end.
#
# Execution order matters:
#   A introduces the outputs
#   B explains the hierarchy
#   C explains probabilities
#   D explains outliers
#   E explores robustness
#   F motivates why HDBSCAN is worth using
#   G/H/I show physical or experimental use cases
#   J combines everything into a full workflow
#
# So the main block is effectively the pedagogical table of contents.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 65)
    print("  HDBSCAN — COMPLETE TUTORIAL")
    print("=" * 65)

    # A. Learn the main HDBSCAN output attributes on simple datasets.
    hdbscan_basics()

    # Load reusable datasets for later sections.
    X_blob, y_blob = load("blobs_easy")
    X_coll, y_coll = load("colloidal_phases")
    X_poly, y_poly = load("polymer_conf")

    # B. Read condensed trees on an easy synthetic dataset.
    condensed_tree_analysis(X_blob, y_blob,
                            min_cluster_size=30,
                            name="isotropic blobs")
    #    Then read the same idea on a more physically motivated phase dataset.
    condensed_tree_analysis(X_coll, y_coll,
                            min_cluster_size=40,
                            name="colloidal phases")

    # C. Examine membership-confidence structure on colloidal phases.
    soft_membership_analysis(X_coll, y_coll,
                             min_cluster_size=40,
                             name="colloidal phases")

    # D. Examine anomaly scores on the same phase data.
    outlier_score_analysis(X_coll, y_coll,
                           min_cluster_size=40,
                           name="colloidal phases")

    # E. Show how robust the result is to HDBSCAN hyperparameters.
    parameter_sensitivity(X_coll, y_coll,
                          name="colloidal phases")
    parameter_sensitivity(X_blob, y_blob,
                          name="blobs easy")

    # F. Demonstrate why HDBSCAN wins on variable-density structure.
    dbscan_vs_hdbscan()
    # G. Translate clustering output into metastable conformational states.
    md_conformational_states()
    # H. Compare coarse vs fine hierarchy extraction on colloidal phase data.
    colloidal_phases_hdbscan()
    # I. Simulate anomaly detection on future trajectory frames.
    anomaly_detection_pipeline()
    # J. Run a full research-style workflow.
    full_inference_pipeline()

    print("\n" + "=" * 65)
    print("  HDBSCAN COMPLETE — see plots/hdbscan/")
    print("=" * 65)
