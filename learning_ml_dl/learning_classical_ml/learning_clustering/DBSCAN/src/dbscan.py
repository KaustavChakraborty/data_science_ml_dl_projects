"""
================================================================================
04_dbscan.py  —  DBSCAN: Density-Based Clustering with Noise
================================================================================

WHAT DBSCAN DOES IN ONE SENTENCE
─────────────────────────────────
Finds clusters as dense regions separated by sparse regions, and explicitly
labels sparse points as noise — no forcing every point into a cluster.

CORE CONCEPTS (brief, applied)
───────────────────────────────
  ε (eps)         : neighbourhood radius — "how close counts as nearby?"
  MinPts          : minimum neighbours to be a core point — "how dense is dense?"

  Core point      : has ≥ MinPts neighbours within ε → sits inside a cluster
  Border point    : within ε of a core, but not core itself → cluster edge
  Noise point     : neither → labelled -1, not assigned to any cluster

  A cluster = maximal set of core points mutually reachable through ε-chains,
              plus their border points.

WHEN TO USE DBSCAN
───────────────────
  ✓ K unknown
  ✓ Non-convex, irregular cluster shapes
  ✓ Genuine noise/outliers in the data
  ✓ All clusters at roughly the same density
  ✓ 2-D or 3-D spatial data (particle positions, microscopy)
  ✗ Clusters of very different densities  → use HDBSCAN
  ✗ High dimensions (d > 15)             → distances concentrate
  ✗ Need probabilistic membership        → use HDBSCAN

SECTIONS
────────
  A  DBSCAN from scratch — see every step of the algorithm
  B  k-distance plot — the principled way to choose ε
  C  sklearn DBSCAN — every parameter explained with comments
  D  Parameter sensitivity grid (eps × min_samples)
  E  Cluster diagnostics: what to measure after fitting
  F  Comparison with K-means on datasets where DBSCAN wins
  G  Soft-matter: colloidal aggregate detection
  H  Soft-matter: defect identification in 2-D crystal
  I  Noise analysis: who are the noise points?

Run  00_generate_datasets.py  first.
Requirements: numpy  scipy  scikit-learn  matplotlib  seaborn  pandas
================================================================================
"""

# =============================================================================
# HOW TO READ THIS FILE
# =============================================================================
# This is a *teaching script*, not just a compact production script.  The goal
# is to show DBSCAN from three complementary viewpoints:
#
# 1. Algorithmic viewpoint
#    - Section A re-implements DBSCAN manually so you can see the exact logic:
#      neighbourhood construction -> core-point test -> breadth-first cluster
#      expansion -> leftover points marked as noise.
#
# 2. Practical ML workflow viewpoint
#    - Sections B–F show what you actually do in real work: choose eps with a
#      k-distance plot, understand sklearn parameters, sweep parameters, inspect
#      diagnostics, and compare against K-means on cases where geometry matters.
#
# 3. Domain-science viewpoint
#    - Sections G–I reinterpret DBSCAN in soft-matter language: aggregates,
#      defect regions, and noise points as potentially meaningful states.
#
# If you are revising later, the most important ideas to keep in mind are:
#
#   • DBSCAN does NOT ask for K in advance.
#   • It identifies dense regions and leaves sparse points unlabeled (-1).
#   • eps controls neighbourhood radius.  min_samples controls density strictness.
#   • Standardisation changes the geometry seen by DBSCAN, so it can completely
#     change the result.  This script standardises most datasets on purpose.
#   • DBSCAN is excellent for non-convex shapes and explicit noise, but struggles
#     when cluster densities differ too much, because one global eps cannot fit
#     every density scale at once.
#
# The extra comments inserted in this annotated version are there to explain not
# only *what* the code is doing, but also *why that choice changes the outcome*.
# =============================================================================

import os, warnings
warnings.filterwarnings("ignore")
# The script suppresses warnings to keep tutorial output visually clean.
# This is convenient for teaching plots, but in serious debugging sessions
# you would often *not* suppress warnings because they can reveal numerical
# or API issues early.

import numpy as np
import matplotlib
# Force a non-interactive backend so figures can be generated in batch mode,
# remote terminals, CI, or headless servers.  This means the script saves files
# to disk instead of trying to open interactive GUI windows.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
from collections import Counter

from sklearn.cluster        import DBSCAN, KMeans
from sklearn.neighbors      import NearestNeighbors
from sklearn.preprocessing  import StandardScaler
from sklearn.decomposition  import PCA
from sklearn.metrics        import adjusted_rand_score, silhouette_score
from sklearn.datasets       import make_blobs
from scipy.spatial.distance import pdist
from scipy.signal           import argrelmin

os.makedirs("plots/dbscan", exist_ok=True)
# Reproducibility is crucial in tutorial code because you want the same figures
# every time you rerun the notebook/script.  Using a fixed Generator means the
# random synthetic datasets and the soft-matter toy systems are repeatable.
RNG  = np.random.default_rng(42)
SEED = 42


# ── helper ────────────────────────────────────────────────────────────────────
# Helper functions keep repeated logic out of the main analytical sections.
# None of them performs clustering by itself; they only support I/O, colouring,
# or concise textual summaries.
def load(name):
    """
    Load a synthetic dataset saved as a NumPy .npz bundle.

    Expected file structure
    -----------------------
    data/<name>.npz containing:
      - X : feature matrix, shape (n_samples, n_features)
      - y : ground-truth labels used only for evaluation/visual teaching

    Important conceptual point
    --------------------------
    DBSCAN itself is unsupervised and never uses y.  The script loads y only so
    that it can compute ARI or make "true label" comparison panels.
    """
    d = np.load(f"data/{name}.npz")
    return d["X"], d["y"]

def cluster_palette(labels):
    """Return colour array: noise=black, clusters=tab10.

    Why this matters
    ----------------
    In DBSCAN, the label -1 is semantically special: it means "noise" or
    "unassigned".  Making noise dark/grey-black in every plot reinforces the
    idea that DBSCAN is *not* forcing every point into a cluster.

    The modulo-10 colour mapping is only a visual convenience.  Cluster IDs do
    not carry an ordering meaning; C0 is not more important than C1.
    """
    cmap   = plt.cm.tab10
    colors = []
    for lbl in labels:
        if lbl == -1:
            colors.append([0.1, 0.1, 0.1, 0.6])   # dark grey for noise
        else:
            colors.append(cmap(int(lbl) % 10))
    return colors

def summarise_labels(labels, prefix=""):
    """Print cluster counts and noise fraction.

    This is a small helper, but it reinforces a key DBSCAN habit: after every
    fit, you should immediately look at

      - how many clusters were found, and
      - how many points were rejected as noise.

    For DBSCAN, these two numbers are often more informative than a single score.
    """
    n      = len(labels)
    noise  = (labels == -1).sum()
    n_clus = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"  {prefix}clusters={n_clus}  noise={noise}/{n} "
          f"({100*noise/n:.1f}%)")
    for k in sorted(set(labels)):
        tag = "NOISE" if k == -1 else f"C{k}"
        print(f"    {tag:>6}: {(labels==k).sum()} pts")


# ════════════════════════════════════════════════════════════════════════════
# A — DBSCAN FROM SCRATCH
# ════════════════════════════════════════════════════════════════════════════

class DBSCANScratch:
    """
    DBSCAN implemented step-by-step in pure NumPy.

    Purpose of this class
    ---------------------
    The sklearn implementation is fast and robust, but its internals are hidden.
    This scratch version makes the mechanics visible.  It is intentionally
    simple, computationally naive, and therefore pedagogically useful.

    What is intentionally *not* optimised here
    ------------------------------------------
    - pairwise distances are recomputed directly instead of using a tree index
    - the BFS queue uses list.pop(0), which is not ideal for very large n
    - everything is held in memory for clarity rather than scalability

    In other words: correctness and readability are prioritised over speed.

    Algorithm (exactly as Ester et al. 1996)
    ─────────────────────────────────────────
    1. Build ε-neighbourhood for every point.
    2. Mark core points (neighbourhood size ≥ MinPts).
    3. For each unvisited core point, start a new cluster and
       grow it by BFS: add all reachable core points and their
       border points.
    4. Any point still unvisited after all BFS passes = noise (-1).

    Time:  O(n²)  naively  [O(n log n) with spatial index]
    Space: O(n²)  for the full neighbour lists
    """

    def __init__(self, eps=0.5, min_samples=5):
        # Store the two fundamental DBSCAN hyperparameters.
        #
        # eps:
        #   Radius of the local neighbourhood.  Increase eps and more points see
        #   each other as neighbours, which usually means fewer noise points and
        #   more cluster merging.
        #
        # min_samples:
        #   Density threshold.  Increase it and DBSCAN becomes stricter about
        #   what counts as a dense region.  That usually means more noise and
        #   fewer / smaller clusters.
        self.eps         = eps
        self.min_samples = min_samples

    def fit_predict(self, X):
        """
        Fit DBSCAN and directly return labels.

        This mirrors sklearn.fit_predict(...) in spirit.  The method proceeds in
        the original DBSCAN order: neighbourhood discovery first, then core-point
        classification, then cluster expansion.
        """
        # Convert input to a dense NumPy float array so all later vectorised
        # distance calculations behave predictably.
        X      = np.asarray(X, dtype=float)
        n      = len(X)
        # DBSCAN uses -1 as the conventional noise label.
        # Starting from "everything is noise" is conceptually elegant because a
        # point only earns membership in a cluster once a density-reachable path
        # from a core point has been established.
        labels = np.full(n, -1, dtype=int)   # start: all noise

        # ── Step 1: find epsilon-neighbours for every point ─────────────────────
        # neighbours[i] = list of indices within eps of point i
        neighbours = []
        for i in range(n):
            # Distance from point i to every point, including itself.
            # Because the self-distance is 0, the point automatically appears in
            # its own eps-neighbourhood.  This matches sklearn's interpretation
            # that min_samples counts the point itself.
            dists = np.sqrt(np.sum((X - X[i]) ** 2, axis=1))

            # Store *indices* rather than the points themselves.  The indices are
            # enough to drive later BFS expansion and they preserve the original
            # identity of each sample.
            neighbours.append(np.where(dists <= self.eps)[0].tolist())

        # ── Step 2: mark core points ──────────────────────────────────────
        # A core point is simply one whose local neighbourhood is dense enough.
        # If eps is too small, very few points become core points.
        # If min_samples is too small, many points become core points.
        is_core = np.array([len(nb) >= self.min_samples
                            for nb in neighbours])

        # ── Step 3: grow clusters via BFS from each unvisited core ────────
        # cluster_id increments every time we discover a *new connected dense
        # component*.  visited prevents us from reprocessing the same points.
        cluster_id = 0
        visited    = np.zeros(n, dtype=bool)

        for i in range(n):
            # Only an unvisited core point can seed a new cluster.
            #
            # Why not border points?
            # A border point may lie within eps of multiple cores, so letting a
            # border point seed a cluster would make the algorithm ambiguous.
            if visited[i] or not is_core[i]:
                continue

            # Start new cluster; BFS queue initialised with point i
            visited[i]    = True
            labels[i]     = cluster_id
            queue         = list(neighbours[i])

            # Breadth-first search over density reachability:
            # start from the seed core point and repeatedly absorb points that
            # belong to its eps-connected dense component.
            while queue:
                # pop(0) gives FIFO behaviour, so this is a literal BFS.
                j = queue.pop(0)
                if not visited[j]:
                    visited[j] = True

                    # The moment point j is reached from the current core-based
                    # component, it becomes part of the current cluster.  This is
                    # true whether j is itself a core point or merely a border
                    # point.
                    labels[j]  = cluster_id        # border or core

                    # Only core points can further expand the cluster frontier.
                    # This is the key asymmetry in DBSCAN:
                    #   core  -> can recruit neighbours
                    #   border -> can belong, but cannot recruit further points
                    if is_core[j]:                 # only core points expand
                        queue.extend(neighbours[j])
                elif labels[j] == -1:
                    # This handles the case where j had previously been visited
                    # but still carried the default noise label.  If a cluster
                    # later reaches it from a core, j should be reclassified as a
                    # border point of that cluster.
                    labels[j] = cluster_id         # border reached from core

            cluster_id += 1

        # ── Step 4: anything still labelled -1 is noise ───────────────────
        # Store learned attributes in sklearn-like style so later code can use
        # this object almost like a tiny estimator.
        self.labels_   = labels
        self.is_core_  = is_core
        return labels


def demo_scratch():
    """Run scratch DBSCAN on circles and compare to sklearn.

    Why circles?
    ------------
    Concentric circles are a canonical example where density connectivity works
    and centroid-based clustering fails.  They are therefore ideal for a first
    DBSCAN demonstration.
    """
    print("\n[A] DBSCAN from scratch")
    X, y = load("circles")
    # Standardisation rescales both coordinate axes to unit variance.
    # For isotropic 2-D toy data this may look harmless, but it is still a real
    # modelling decision: eps is interpreted *after* scaling, not in the raw
    # input units.
    Xs   = StandardScaler().fit_transform(X)

    db_s  = DBSCANScratch(eps=0.25, min_samples=8)
    lbl_s = db_s.fit_predict(Xs)

    db_sk = DBSCAN(eps=0.25, min_samples=8).fit(Xs)
    lbl_k = db_sk.labels_

    ari_s = adjusted_rand_score(y, lbl_s)
    ari_k = adjusted_rand_score(y, lbl_k)
    print(f"  Scratch : ARI={ari_s:.4f}")
    print(f"  Sklearn : ARI={ari_k:.4f}  (should match)")
    summarise_labels(lbl_s, "Scratch: ")

    # Plot side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("DBSCAN from Scratch — Concentric Circles",
                 fontsize=13, fontweight="bold")

    for ax, (lbl, title) in zip(axes, [
            (y,     "True labels"),
            (lbl_s, f"Scratch  ARI={ari_s:.3f}"),
            (lbl_k, f"Sklearn  ARI={ari_k:.3f}"),
    ]):
        ax.scatter(Xs[:, 0], Xs[:, 1],
                   c=cluster_palette(lbl), s=18, alpha=0.85)
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    # Annotate point types on scratch result
    core_mask   = db_s.is_core_
    border_mask = (~core_mask) & (lbl_s != -1)
    noise_mask  = lbl_s == -1
    axes[1].scatter(Xs[core_mask,   0], Xs[core_mask,   1],
                    s=4, c="white", alpha=0.4, label="core")
    axes[1].scatter(Xs[border_mask, 0], Xs[border_mask, 1],
                    s=35, edgecolors="yellow", facecolors="none",
                    lw=1.2, label="border")
    axes[1].scatter(Xs[noise_mask,  0], Xs[noise_mask,  1],
                    s=60, marker="x", c="red", lw=1.5, label="noise")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("plots/dbscan/A_scratch_demo.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved => plots/dbscan/A_scratch_demo.png")


# ════════════════════════════════════════════════════════════════════════════
# B — k-DISTANCE PLOT: PRINCIPLED epsilon SELECTION
# ════════════════════════════════════════════════════════════════════════════

def kdistance_plot(X, min_samples=5, name="dataset"):
    """
    k-Distance Plot — the standard method to choose epsilon.

    Method
    ──────
    1. For each point, find its distance to its k-th nearest neighbour
       (k = min_samples).
    2. Sort these distances in DESCENDING order.
    3. Plot.  The plot has two regimes:
         Flat region (left)  : these points are in dense areas (small kNN dist)
         Steep region (right): these points are in sparse areas (large kNN dist)
    4. The "knee" between them is the optimal epsilon.
       Below knee => core points.  Above knee => noise candidates.

    Why this works
    ──────────────
    Setting epsilon at the knee captures all dense points as core points, while
    leaving genuinely sparse points as noise.  Setting epsilon too small (below knee)
    => too much noise.  Too large (above knee) => noise absorbed into clusters.

    We also auto-detect the knee using the maximum curvature of the curve.
    """
    print(f"\n[B] k-distance plot — {name}  (k={min_samples})")
    # Again we standardise first so the neighbour radius is measured in a
    # dimensionless, scale-balanced feature space.  Without this, one large-scale
    # feature could dominate the neighbour search.
    Xs = StandardScaler().fit_transform(X)

    # Compute k-th NN distances
    # For each point, sklearn returns distances to its nearest neighbours.
    # The last column dists[:, -1] is the distance to the k-th nearest neighbour
    # where k = min_samples.  Points in dense regions have small such distances;
    # sparse/outlier points have large ones.
    nbrs   = NearestNeighbors(n_neighbors=min_samples).fit(Xs)
    dists, _ = nbrs.kneighbors(Xs)

    # Sort from largest to smallest so the sparse-region points appear on the
    # left/high end of the curve and dense-region points collect in the flatter
    # lower tail.  This makes the knee easier to see visually.
    k_dists  = np.sort(dists[:, -1])[::-1]     # descending

    # Auto-detect knee: point of maximum second derivative
    # (largest positive curvature in the descending sorted distance curve)
    # Raw k-distance curves can be jagged, so the script smooths them with a
    # simple moving average.  This is only for knee detection and plotting; the
    # original unsmoothed values are still kept in k_dists.
    smooth  = np.convolve(k_dists, np.ones(15)/15, mode="same")
    d2      = np.gradient(np.gradient(smooth))
    knee_i  = int(np.argmax(d2[10:-10])) + 10   # skip edge effects
    eps_knee = k_dists[knee_i]

    print(f"  Auto-detected knee: epsilon ~ {eps_knee:.4f}  (index {knee_i})")

    # Show DBSCAN result with suggested eps
    db   = DBSCAN(eps=eps_knee, min_samples=min_samples).fit(Xs)
    n_c  = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    n_ns = (db.labels_ == -1).sum()
    print(f"  DBSCAN at epsilon={eps_knee:.3f}: {n_c} clusters, "
          f"{n_ns} noise pts ({100*n_ns/len(Xs):.1f}%)")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"k-Distance Plot & ε Selection — {name} (k={min_samples})",
                 fontsize=13, fontweight="bold")

    # 1. k-distance curve
    ax = axes[0]
    ax.plot(np.arange(len(k_dists)), k_dists,
            lw=1.5, color="steelblue", alpha=0.8)
    ax.plot(np.arange(len(smooth)),  smooth,
            lw=2.5, color="navy", ls="--", label="smoothed")
    ax.axvline(knee_i,   color="red", ls="--", lw=2,
               label=f"knee at index {knee_i}")
    ax.axhline(eps_knee, color="red", ls=":",  lw=1.5,
               label=f"ε = {eps_knee:.3f}")
    ax.fill_between(np.arange(knee_i+1), k_dists[:knee_i+1],
                    alpha=0.10, color="red", label="sparse / noise zone")
    ax.fill_between(np.arange(knee_i, len(k_dists)),
                    k_dists[knee_i:], alpha=0.10, color="green",
                    label="dense / core zone")
    ax.set_xlabel("Points (sorted descending by k-dist)",  fontsize=10)
    ax.set_ylabel(f"{min_samples}-NN distance",           fontsize=10)
    ax.set_title("k-Distance Plot\n<= sparse | dense =>",   fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 2. DBSCAN result at suggested ε
    ax = axes[1]
    ax.scatter(Xs[:, 0], Xs[:, 1],
               c=cluster_palette(db.labels_), s=18, alpha=0.8)
    ax.set_title(f"DBSCAN at epsilon={eps_knee:.3f}\n"
                 f"{n_c} clusters, {n_ns} noise pts",
                 fontweight="bold", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    # 3. Effect of eps around knee
    # This local sweep is pedagogically useful because it shows the typical
    # DBSCAN phase transition behaviour around a plausible eps value:
    # too small -> fragmentation / lots of noise; too large -> cluster merging.
    eps_vals = eps_knee * np.array([0.4, 0.6, 0.8, 1.0, 1.3, 1.7, 2.5])
    n_clusters_list, noise_fracs = [], []
    for e in eps_vals:
        db_  = DBSCAN(eps=e, min_samples=min_samples).fit(Xs)
        n_c_ = len(set(db_.labels_)) - (1 if -1 in db_.labels_ else 0)
        nf_  = (db_.labels_ == -1).sum() / len(Xs)
        n_clusters_list.append(n_c_)
        noise_fracs.append(nf_)

    ax2 = axes[2]
    ax2_r = ax2.twinx()
    ax2.plot(eps_vals, n_clusters_list, "o-",
             color="steelblue", lw=2, ms=8, label="# clusters")
    ax2_r.plot(eps_vals, noise_fracs, "s--",
               color="tomato", lw=2, ms=8, label="noise fraction")
    ax2.axvline(eps_knee, color="red", ls=":", lw=2, label=f"ε_knee={eps_knee:.2f}")
    ax2.set_xlabel("epsilon"); ax2.set_ylabel("# clusters", color="steelblue")
    ax2_r.set_ylabel("Noise fraction", color="tomato")
    ax2.set_title("epsilon sweep: clusters & noise vs epsilon", fontweight="bold")
    lines1, lab1 = ax2.get_legend_handles_labels()
    lines2, lab2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, lab1 + lab2, fontsize=8)
    ax2.grid(alpha=0.3)

    fname = f"plots/dbscan/B_kdistance_{name.replace(' ','_')}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved => {fname}")
    return eps_knee


# ════════════════════════════════════════════════════════════════════════════
# C — SKLEARN DBSCAN: EVERY PARAMETER EXPLAINED
# ════════════════════════════════════════════════════════════════════════════

def sklearn_dbscan_annotated(X, y_true, eps, min_samples, name="dataset"):
    """
    Fully annotated sklearn DBSCAN call.

    PARAMETERS
    ──────────
    eps              ε neighbourhood radius.
                     CRITICAL — most sensitive parameter.
                     Use k-distance plot to set this.

    min_samples      Minimum neighbours (including self) to be a core point.
                     Rule of thumb: 2 × n_features.  Min=3, never 2.
                     Higher → stricter density → more noise points.

    metric           Distance function.  Default 'euclidean'.
                     Also: 'manhattan', 'cosine', 'haversine' (lat/lon),
                     'precomputed' (pass your own n×n distance matrix).

    algorithm        Spatial index for neighbour queries.
                     'auto'      : sklearn chooses best for your data
                     'ball_tree' : O(n log n) for low-d; best for d < 20
                     'kd_tree'   : similar; good for d < 20, strict Minkowski
                     'brute'     : O(n²) exact; use when n < 2000 or d large

    leaf_size        Tree leaf size (ball_tree/kd_tree only).
                     Trades build time vs query time.  Default 30.

    n_jobs           Parallelism for neighbour queries.  -1 = all cores.
                     Important for large n.

    POST-FIT ATTRIBUTES
    ───────────────────
    .labels_         (n,) int.  -1 = noise.  0,1,2,... = cluster ids.
                     Note: sklearn does NOT guarantee contiguous cluster ids
                     if you call fit on a subset.

    .core_sample_indices_   indices of all core points in the input data.

    .components_     actual X values of core points (X[core_sample_indices_])
    """
    print(f"\n[C] sklearn DBSCAN annotated — {name}")
    # Again we standardise first so the neighbour radius is measured in a
    # dimensionless, scale-balanced feature space.  Without this, one large-scale
    # feature could dominate the neighbour search.
    Xs = StandardScaler().fit_transform(X)

    # This block is the transition from theory to the real estimator you would
    # use in practice.  The choices here directly affect neighbour queries,
    # scalability, and the semantic strictness of clusters vs noise.
    db = DBSCAN(
        eps         = eps,
        min_samples = min_samples,
        metric      = "euclidean",   # change to 'precomputed' for custom dists
        algorithm   = "auto",        # let sklearn pick ball_tree/kd_tree
        leaf_size   = 30,
        n_jobs      = -1,
    )
    db.fit(Xs)

    labels      = db.labels_
    n_clusters  = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise     = (labels == -1).sum()
    n_core      = len(db.core_sample_indices_)
    # sklearn exposes core points explicitly but not border points as a stored
    # attribute.  Border count is inferred as whatever remains after subtracting
    # core and noise from the total.
    n_border    = len(Xs) - n_noise - n_core

    print(f"  epsilon={eps:.3f}  min_samples={min_samples}")
    print(f"  Clusters : {n_clusters}")
    print(f"  Core pts : {n_core}  ({100*n_core/len(Xs):.1f}%)")
    print(f"  Border   : {n_border}  ({100*n_border/len(Xs):.1f}%)")
    print(f"  Noise    : {n_noise}  ({100*n_noise/len(Xs):.1f}%)")
    if y_true is not None and n_clusters > 1:
        valid = labels != -1
        ari = adjusted_rand_score(y_true[valid], labels[valid])
        print(f"  ARI (non-noise): {ari:.4f}")

    return db, Xs


# ════════════════════════════════════════════════════════════════════════════
# D — PARAMETER SENSITIVITY GRID
# ════════════════════════════════════════════════════════════════════════════

def parameter_grid(X, y_true, name="dataset"):
    """
    Sweep eps × min_samples and measure:
      - Number of clusters found
      - Noise fraction
      - ARI vs true labels (where available)

    Reading the grid
    ────────────────
    • Too small ε  OR  too large min_samples → single cluster = 1 large blob,
      OR everything is noise.
    • Too large ε  OR  too small min_samples → all merged into 1 cluster.
    • The "sweet spot" is the region where n_clusters matches your expectation
      and noise fraction is reasonable (5-20% for typical lab data).
    """
    print(f"\n[D] Parameter sensitivity grid — {name}")
    Xs       = StandardScaler().fit_transform(X)
    # Using the median pairwise distance gives the eps sweep a scale tied to
    # the dataset itself.  This makes the grid less arbitrary than hard-coding
    # raw eps values that might only make sense for one dataset.
    all_d    = pdist(Xs)
    med_d    = np.median(all_d)

    eps_vals = np.round(med_d * np.array([0.15, 0.25, 0.40, 0.60,
                                          0.85, 1.20, 1.70]), 3)
    min_vals = [3, 5, 8, 12, 20]

    n_c  = np.zeros((len(min_vals), len(eps_vals)), int)
    nf   = np.zeros((len(min_vals), len(eps_vals)))
    aris = np.zeros((len(min_vals), len(eps_vals)))

    # This double loop is effectively a coarse response surface over the two
    # hyperparameters.  The three recorded outputs answer complementary questions:
    #
    #   n_c  : did DBSCAN find the expected number of dense components?
    #   nf   : did the chosen density threshold reject too much / too little data?
    #   aris : if ground truth exists, how well do the non-noise labels match it?
    for i, ms in enumerate(min_vals):
        for j, ep in enumerate(eps_vals):
            db   = DBSCAN(eps=ep, min_samples=ms, n_jobs=-1).fit(Xs)
            lbl  = db.labels_
            nc   = len(set(lbl)) - (1 if -1 in lbl else 0)
            n_c[i, j] = nc
            nf [i, j] = (lbl == -1).sum() / len(lbl)
            if y_true is not None and nc > 1:
                valid = lbl != -1
                if valid.sum() > 10:
                    aris[i, j] = adjusted_rand_score(y_true[valid], lbl[valid])

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    fig.suptitle(f"DBSCAN Parameter Grid — {name}", fontsize=13, fontweight="bold")

    def hm(ax, data, title, fmt, cmap, vmin=None, vmax=None):
        df = pd.DataFrame(data, index=[f"ms={m}" for m in min_vals],
                          columns=[f"ε={e}" for e in eps_vals])
        sns.heatmap(df, annot=True, fmt=fmt, cmap=cmap, ax=ax,
                    vmin=vmin, vmax=vmax,
                    linewidths=0.5, linecolor="gray",
                    cbar_kws={"shrink": 0.8})
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_xlabel("ε (as fraction of median pairwise distance)")
        ax.set_ylabel("min_samples")

    hm(axes[0], n_c,  "# Clusters found\n(target=true K)",
       "d",   "Blues")
    hm(axes[1], nf,   "Noise fraction\n(ideal: 0.05–0.20)",
       ".2f", "Reds", 0, 0.5)
    hm(axes[2], aris, "ARI vs true labels\n(1=perfect)",
       ".2f", "RdYlGn", 0, 1)

    fname = f"plots/dbscan/D_param_grid_{name.replace(' ','_')}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {fname}")


# ════════════════════════════════════════════════════════════════════════════
# E — CLUSTER DIAGNOSTICS: WHAT TO MEASURE AFTER FITTING
# ════════════════════════════════════════════════════════════════════════════

def cluster_diagnostics(X, y_true, eps, min_samples, name="dataset"):
    """
    After fitting DBSCAN, extract physically meaningful statistics.

    Metrics to report
    ─────────────────
    Per-cluster:
      - Size (n pts)
      - Core fraction (density indicator)
      - Centroid (mean position in feature space)
      - Spread (std dev per feature — cluster compactness)
      - Intra-cluster silhouette score

    Global:
      - Noise fraction
      - Mean silhouette (non-noise points only)
      - Number of clusters

    For soft matter
    ───────────────
    - Cluster size distribution (are aggregates fractal?)
    - Spatial extent of each cluster (radius of gyration)
    - Core-to-border ratio (how compact / diffuse are clusters?)
    """
    print(f"\n[E] Cluster diagnostics — {name}")
    # Again we standardise first so the neighbour radius is measured in a
    # dimensionless, scale-balanced feature space.  Without this, one large-scale
    # feature could dominate the neighbour search.
    Xs = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(Xs)
    labels = db.labels_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    non_noise  = labels != -1

    # Silhouette is undefined or uninformative when fewer than two non-noise
    # clusters exist.  That is why the function exits early here.
    if n_clusters < 2:
        print("  < 2 clusters — skipping silhouette")
        return

    sil_global = silhouette_score(Xs[non_noise], labels[non_noise])
    print(f"  Global: {n_clusters} clusters  noise={n_noise} "
          f"({100*n_noise/len(labels):.1f}%)  silhouette={sil_global:.4f}")

    # Per-cluster stats
    print(f"\n  {'Cluster':>8} {'Size':>6} {'Core%':>6} "
          f"{'Rg (scaled)':>12} {'Sil':>6}")
    print("  " + "─" * 48)

    # silhouette_samples gives one value per non-noise point.  The script later
    # averages those within each DBSCAN cluster to see which clusters are tight
    # and well-separated versus weak or ambiguous.
    from sklearn.metrics import silhouette_samples
    sv = silhouette_samples(Xs[non_noise], labels[non_noise])
    nn_idx = np.where(non_noise)[0]

    cluster_data = []
    for k in sorted(set(labels)):
        if k == -1:
            continue
        mask   = labels == k
        X_k    = Xs[mask]
        is_c_k = np.isin(np.where(mask)[0], db.core_sample_indices_)
        core_f = is_c_k.sum() / len(X_k)

        # Radius of gyration (spread from centroid)
        centroid = X_k.mean(axis=0)
        Rg       = np.sqrt(np.mean(np.sum((X_k - centroid)**2, axis=1)))

        # Silhouette for this cluster
        local_nn = np.isin(nn_idx, np.where(mask)[0])
        sil_k    = sv[local_nn].mean() if local_nn.sum() > 0 else 0.0

        print(f"  {k:>8} {len(X_k):>6} {core_f:>6.2f} "
              f"{Rg:>12.4f} {sil_k:>6.3f}")
        cluster_data.append(dict(k=k, size=len(X_k), core_f=core_f,
                                 Rg=Rg, sil=sil_k))

    # ── Visualise diagnostics ─────────────────────────────────────────────────
    # PCA is used here *only* as a plotting projection when dimensionality is
    # greater than 2.  The clustering was already done in Xs.  PCA does not feed
    # back into the fitted DBSCAN labels.
    Xp = PCA(n_components=2).fit_transform(Xs) if Xs.shape[1] > 2 else Xs

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f"DBSCAN Diagnostics — {name}", fontsize=13, fontweight="bold")

    cmap = plt.cm.tab10

    # 1. Cluster scatter with point-type annotation
    ax = axes[0, 0]
    core_idx   = db.core_sample_indices_
    all_idx    = np.arange(len(Xs))
    border_idx = np.array([i for i in all_idx
                           if labels[i] != -1 and i not in core_idx])
    noise_idx  = np.where(labels == -1)[0]

    for k in range(n_clusters):
        m = np.array([i for i in core_idx if labels[i] == k])
        if len(m): ax.scatter(Xp[m,0], Xp[m,1], c=[cmap(k)],
                              s=25, alpha=0.7)
    for k in range(n_clusters):
        m = np.array([i for i in border_idx if labels[i] == k])
        if len(m): ax.scatter(Xp[m,0], Xp[m,1], c=[cmap(k)],
                              s=60, edgecolors="black", lw=1.2, alpha=0.9)
    ax.scatter(Xp[noise_idx,0], Xp[noise_idx,1],
               c="gray", s=40, marker="x", lw=1.5, label="noise")
    from matplotlib.lines import Line2D
    legend_els = [Line2D([0],[0], marker="o", color="w",
                         markerfacecolor="gray", ms=8, label="core"),
                  Line2D([0],[0], marker="o", color="w",
                         markerfacecolor="gray", markeredgecolor="k",
                         ms=10, label="border"),
                  Line2D([0],[0], marker="x", color="gray",
                         ms=10, lw=2, label="noise")]
    ax.legend(handles=legend_els, fontsize=8)
    ax.set_title("Point types: core / border / noise", fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    # 2. Cluster size bar chart
    ax = axes[0, 1]
    ks    = [d["k"]    for d in cluster_data]
    sizes = [d["size"] for d in cluster_data]
    bars  = ax.bar(ks, sizes,
                   color=[cmap(k % 10) for k in ks],
                   edgecolor="k", alpha=0.85)
    ax.bar_label(bars, padding=2, fontsize=9)
    ax.axhline(np.mean(sizes), color="red", ls="--",
               label=f"mean={np.mean(sizes):.0f}")
    ax.set_xlabel("Cluster"); ax.set_ylabel("Size (# points)")
    ax.set_title("Cluster size distribution", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

    # 3. Core fraction per cluster
    ax = axes[0, 2]
    cf = [d["core_f"] for d in cluster_data]
    ax.bar(ks, cf, color=[cmap(k%10) for k in ks],
           edgecolor="k", alpha=0.85)
    ax.axhline(np.mean(cf), color="red", ls="--",
               label=f"mean={np.mean(cf):.2f}")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Cluster"); ax.set_ylabel("Core fraction")
    ax.set_title("Core fraction (density indicator)\n"
                 "High→dense core  Low→diffuse/hollow", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

    # 4. Rg (compactness) per cluster
    ax = axes[1, 0]
    rgs = [d["Rg"] for d in cluster_data]
    ax.bar(ks, rgs, color=[cmap(k%10) for k in ks],
           edgecolor="k", alpha=0.85)
    ax.set_xlabel("Cluster"); ax.set_ylabel("Radius of gyration (scaled)")
    ax.set_title("Cluster compactness (Rg)\nLow→tight  High→diffuse",
                 fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # 5. Silhouette per cluster
    ax = axes[1, 1]
    sils = [d["sil"] for d in cluster_data]
    cols = ["forestgreen" if s > 0.5 else "orange" if s > 0.25 else "red"
            for s in sils]
    ax.bar(ks, sils, color=cols, edgecolor="k", alpha=0.85)
    ax.axhline(sil_global, color="red", ls="--",
               label=f"global={sil_global:.3f}")
    ax.axhline(0.5, color="green", ls=":", lw=1, label="good threshold")
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("Cluster"); ax.set_ylabel("Mean silhouette")
    ax.set_title("Per-cluster silhouette\n>0.5 good  <0.25 weak",
                 fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

    # 6. Noise point distribution (distance to nearest cluster)
    ax = axes[1, 2]
    if noise_idx.size > 0:
        non_noise_X = Xs[non_noise]
        noise_X     = Xs[noise_idx]
        # Distance from each noise point to nearest non-noise point
        from scipy.spatial.distance import cdist
        dist_to_cluster = cdist(noise_X, non_noise_X).min(axis=1)
        ax.hist(dist_to_cluster, bins=30, color="tomato",
                edgecolor="k", lw=0.5, alpha=0.8)
        ax.axvline(eps, color="blue", ls="--", lw=2,
                   label=f"ε={eps:.3f}")
        ax.set_xlabel("Distance to nearest cluster point")
        ax.set_ylabel("Count")
        ax.set_title("Noise point proximity\n"
                     "Left of ε line = borderline noise", fontweight="bold")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No noise points", ha="center", va="center",
                fontsize=14, transform=ax.transAxes)
        ax.axis("off")

    fname = f"plots/dbscan/E_diagnostics_{name.replace(' ','_')}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {fname}")
    return cluster_data


# ════════════════════════════════════════════════════════════════════════════
# F — COMPARISON: DBSCAN VS K-MEANS WHERE DBSCAN WINS
# ════════════════════════════════════════════════════════════════════════════

def dbscan_vs_kmeans():
    """
    Systematic comparison on datasets designed to expose K-means weaknesses.

    Key insights to observe
    ───────────────────────
    Circles  : K-means cuts vertically (wrong).  DBSCAN follows ring shape.
    Moons    : K-means splits wrong.  DBSCAN traces crescents.
    Blobs+noise: K-means forces noise into clusters.  DBSCAN flags it.
    Varying density: DBSCAN struggles too (one ε) — sets up HDBSCAN motivation.
    """
    print("\n[F] DBSCAN vs K-means comparison")

    # The comparison section is deliberately designed around cases where the
    # assumptions of K-means and DBSCAN diverge.  K-means prefers convex,
    # centroid-representable groups and it always assigns every point.  DBSCAN
    # prefers dense connected regions and may reject points as noise.

    # Build a noisy blobs dataset manually (not in standard suite)
    X_nb, y_nb = make_blobs(n_samples=400, centers=3,
                             cluster_std=0.6, random_state=SEED)
    # Add genuine noise
    noise_pts  = RNG.uniform(-6, 6, (60, 2))
    X_nb  = np.vstack([X_nb, noise_pts])
    y_nb  = np.concatenate([y_nb, np.full(60, -1)])   # true noise = -1

    # Varying density dataset
    X_v1, _  = make_blobs(n_samples=300, centers=[[0,0]],
                           cluster_std=0.3, random_state=SEED)
    X_v2, _  = make_blobs(n_samples=300, centers=[[5,5]],
                           cluster_std=1.5, random_state=SEED)
    X_vd     = np.vstack([X_v1, X_v2])
    y_vd     = np.array([0]*300 + [1]*300)

    cases = [
        ("circles",  2,  0.25, 8,  "Concentric Circles"),
        ("moons",    2,  0.18, 8,  "Two Moons"),
        (None,       3,  0.40, 8,  "Blobs + Noise",     X_nb, y_nb),
        (None,       2,  0.35, 8,  "Varying Density",   X_vd, y_vd),
    ]

    fig, axes = plt.subplots(4, 3, figsize=(18, 22))
    fig.suptitle("DBSCAN vs K-means — Advantage Cases",
                 fontsize=14, fontweight="bold")

    for row, case in enumerate(cases):
        if case[0] is not None:
            X, y = load(case[0])
            name = case[4]
            K, eps, ms = case[1], case[2], case[3]
        else:
            _, K, eps, ms, name, X, y = case

        # Standardise each case before comparing algorithms so neither method
        # is unfairly advantaged by raw feature scale.
        Xs     = StandardScaler().fit_transform(X)
        km_lbl = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit_predict(Xs)
        db_lbl = DBSCAN(eps=eps, min_samples=ms, n_jobs=-1).fit_predict(Xs)

        valid_db  = db_lbl != -1
        ari_km = adjusted_rand_score(y[y != -1], km_lbl[y != -1]) if (y != -1).any() else \
                 adjusted_rand_score(y, km_lbl)
        # Important evaluation subtlety: for DBSCAN the script computes ARI only
        # on the non-noise subset.  This focuses the score on clustering quality
        # among assigned points rather than heavily penalising the explicit noise
        # mechanism that DBSCAN is designed to use.
        ari_db = (adjusted_rand_score(y[valid_db], db_lbl[valid_db])
                  if valid_db.sum() > 10 else 0.0)

        nc_db  = len(set(db_lbl)) - (1 if -1 in db_lbl else 0)
        noise_pct = 100 * (db_lbl == -1).sum() / len(db_lbl)

        for col, (lbl, title, c_override) in enumerate([
            (y,      f"{name}\n(True)",              None),
            (km_lbl, f"K-Means K={K}\nARI={ari_km:.3f}", "km"),
            (db_lbl, f"DBSCAN ε={eps}\n"
                     f"K={nc_db}  noise={noise_pct:.0f}%  ARI={ari_db:.3f}", "db"),
        ]):
            ax = axes[row, col]
            if c_override == "db":
                ax.scatter(Xs[:, 0], Xs[:, 1],
                           c=cluster_palette(lbl), s=14, alpha=0.8)
            else:
                ax.scatter(Xs[:, 0], Xs[:, 1],
                           c=lbl, cmap="tab10", s=14, alpha=0.8,
                           vmin=-1, vmax=max(set(lbl)))
            tc = ("forestgreen" if col > 0 and (
                      (col==1 and ari_km > 0.85) or
                      (col==2 and ari_db > 0.85))
                  else "red" if col > 0 and (
                      (col==1 and ari_km < 0.5) or
                      (col==2 and ari_db < 0.5))
                  else "black")
            ax.set_title(title, fontsize=9, fontweight="bold", color=tc)
            ax.set_xticks([]); ax.set_yticks([])

        print(f"  {name:<22}: K-means ARI={ari_km:.3f}  DBSCAN ARI={ari_db:.3f}")

    plt.tight_layout()
    plt.savefig("plots/dbscan/F_vs_kmeans.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved → plots/dbscan/F_vs_kmeans.png")


# ════════════════════════════════════════════════════════════════════════════
# G — SOFT-MATTER: COLLOIDAL AGGREGATE DETECTION
# ════════════════════════════════════════════════════════════════════════════

def colloidal_aggregate_detection():
    """
    Simulates a 2-D colloidal experiment: particles at (x,y) positions.

    Physical scenario
    ─────────────────
    A dilute colloidal suspension under attractive interactions forms
    irregular DLCA (Diffusion-Limited Cluster Aggregation) clusters.
    The clusters are:
      - Non-convex (fractal-like arms)
      - Different sizes
      - Embedded in a background of free particles (noise)

    This is exactly DBSCAN's sweet spot.

    What we extract
    ───────────────
    - Number of aggregates
    - Aggregate size distribution (power law? fractal dimension?)
    - Free particle fraction (noise)
    - Per-aggregate radius of gyration Rg
    - Identification of the largest aggregate (percolating cluster?)
    """
    print("\n[G] Soft-matter: colloidal aggregate detection")

    # n_total is not later used directly; the real system size emerges from the
    # sum of aggregate particles plus free particles.  Keeping the variable still
    # signals the intended experiment scale.
    n_total = 800

    # Generate fractal-like aggregates using random walk clusters
    def make_aggregate(center, n_pts, spread):
        # This is a random-walk growth model.  It is not a faithful DLCA simulator
        # in the strict physical sense, but it generates irregular connected arms
        # that are visually and geometrically suitable for demonstrating DBSCAN
        # on non-convex particle aggregates.
        pts = [center]
        for _ in range(n_pts - 1):
            base    = pts[RNG.integers(len(pts))]
            angle   = RNG.uniform(0, 2*np.pi)
            step    = RNG.exponential(spread)
            new_pt  = base + step * np.array([np.cos(angle), np.sin(angle)])
            pts.append(new_pt)
        return np.array(pts)

    # 5 aggregates of different sizes + free particles
    aggregates = [
        make_aggregate([2,  2],  80,  0.4),
        make_aggregate([8,  8],  40,  0.5),
        make_aggregate([14, 3],  120, 0.35),
        make_aggregate([5,  13], 25,  0.6),
        make_aggregate([12, 12], 60,  0.45),
    ]
    agg_labels_true = []
    for i, agg in enumerate(aggregates):
        agg_labels_true.extend([i] * len(agg))

    X_agg  = np.vstack(aggregates)
    # Free particles (genuine noise)
    X_free = RNG.uniform(-1, 17, (180, 2))
    X_all  = np.vstack([X_agg, X_free])
    y_all  = np.array(agg_labels_true + [-1]*180)

    # ── k-distance plot to find ε ─────────────────────────────────────────────
    # Standardise so aggregate density contrast is captured correctly
    from sklearn.preprocessing import StandardScaler as _SS
    scaler_agg = _SS().fit(X_all)
    X_all_s    = scaler_agg.transform(X_all)
    min_samples = 6
    nbrs     = NearestNeighbors(n_neighbors=min_samples).fit(X_all_s)
    dists, _ = nbrs.kneighbors(X_all_s)
    k_dists  = np.sort(dists[:, -1])[::-1]
    smooth   = np.convolve(k_dists, np.ones(10)/10, mode="same")
    d2       = np.gradient(np.gradient(smooth))
    knee_i   = int(np.argmax(d2[5:-5])) + 5
    # Note the design choice here: the code computes a knee index but then uses
    # the 30th percentile of k-distances as eps_auto.  This is a pragmatic choice
    # for cleaner separation in this synthetic colloid example, not a universal
    # theorem.  It is a reminder that domain knowledge sometimes overrides the
    # most naive automatic knee rule.
    eps_auto = float(np.percentile(k_dists, 30))   # 30th pct gives clean separation

    # ── Fit DBSCAN ────────────────────────────────────────────────────────────
    db     = DBSCAN(eps=eps_auto, min_samples=min_samples, n_jobs=-1)
    labels = db.fit_predict(X_all_s)

    n_agg  = len(set(labels)) - (1 if -1 in labels else 0)
    n_free = (labels == -1).sum()
    print(f"  Auto ε={eps_auto:.3f}  min_samples={min_samples}")
    print(f"  Found {n_agg} aggregates  +  {n_free} free particles "
          f"({100*n_free/len(labels):.1f}%)")

    # ── Physical inference ────────────────────────────────────────────────────
    agg_stats = []
    for k in sorted(set(labels)):
        if k == -1: continue
        pts  = X_all[labels == k]
        cent = pts.mean(axis=0)
        Rg   = np.sqrt(np.mean(np.sum((pts - cent)**2, axis=1)))
        agg_stats.append({"id": k, "n": len(pts), "Rg": Rg,
                          "cx": cent[0], "cy": cent[1]})

    print(f"\n  {'Agg':>4} {'Size':>6} {'Rg':>8}")
    for s in sorted(agg_stats, key=lambda x: -x["n"]):
        print(f"  {s['id']:>4} {s['n']:>6} {s['Rg']:>8.3f}")

    # The largest cluster is often the physically interesting one because it may
    # correspond to incipient percolation or the dominant aggregate.
    largest = max(agg_stats, key=lambda x: x["n"])
    print(f"\n  Largest aggregate: C{largest['id']}  "
          f"n={largest['n']}  Rg={largest['Rg']:.3f}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("DBSCAN: Colloidal Aggregate Detection",
                 fontsize=13, fontweight="bold")

    # 1. Raw data (no labels)
    ax = axes[0, 0]
    ax.scatter(X_all[:, 0], X_all[:, 1], c="steelblue", s=18, alpha=0.7)
    ax.set_title("Raw particle positions", fontweight="bold")
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    ax.set_aspect("equal"); ax.grid(alpha=0.2)

    # 2. k-distance plot
    ax = axes[0, 1]
    ax.plot(k_dists, color="steelblue", lw=1.5)
    ax.plot(smooth,  color="navy", lw=2, ls="--", label="smoothed")
    ax.axvline(knee_i,   color="red", ls="--", lw=2)
    ax.axhline(eps_auto, color="red", ls=":", lw=2,
               label=f"ε={eps_auto:.3f}")
    ax.set_xlabel("Points (sorted)"); ax.set_ylabel("6-NN distance")
    ax.set_title("k-Distance plot → ε selection", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # 3. DBSCAN result
    ax = axes[0, 2]
    ax.scatter(X_all[:, 0], X_all[:, 1],
               c=cluster_palette(labels), s=18, alpha=0.85)
    for s in agg_stats:
        ax.annotate(f"A{s['id']}\nn={s['n']}",
                    (s["cx"], s["cy"]),
                    ha="center", va="center",
                    fontsize=8, fontweight="bold",
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.2",
                              fc="black", alpha=0.5))
    ax.set_title(f"DBSCAN result: {n_agg} aggregates\n"
                 f"(black = free particles)",
                 fontweight="bold")
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    ax.set_aspect("equal"); ax.grid(alpha=0.2)

    # 4. Aggregate size distribution
    ax = axes[1, 0]
    sizes = sorted([s["n"] for s in agg_stats])
    ax.bar(range(len(sizes)), sorted(sizes, reverse=True),
           color=plt.cm.tab10.colors[:len(sizes)],
           edgecolor="k", alpha=0.85)
    ax.set_xlabel("Aggregate (sorted by size)")
    ax.set_ylabel("Number of particles")
    ax.set_title("Aggregate size distribution", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # 5. Rg vs Size scatter (fractal dimension signature)
    ax = axes[1, 1]
    ns  = np.array([s["n"]  for s in agg_stats])
    rgs = np.array([s["Rg"] for s in agg_stats])
    ax.scatter(ns, rgs, s=120, c=range(len(ns)),
               cmap="tab10", edgecolors="k", zorder=5)
    for s in agg_stats:
        ax.annotate(f"A{s['id']}", (s["n"]+1, s["Rg"]),
                    fontsize=9, fontweight="bold")
    # Rg ∝ N^(1/d_f): log-log fit for fractal dimension
    if len(ns) > 2:
        log_n   = np.log(ns); log_Rg = np.log(rgs)
        p       = np.polyfit(log_n, log_Rg, 1)
        d_f     = 1.0 / p[0]   # fractal dimension d_f
        x_fit   = np.linspace(ns.min(), ns.max(), 50)
        ax.plot(x_fit, np.exp(np.polyval(p, np.log(x_fit))),
                "r--", lw=2, label=f"Rg∝N^(1/d_f)  d_f≈{d_f:.2f}")
        ax.legend(fontsize=9)
        print(f"  Fractal dimension d_f ≈ {d_f:.2f}  "
              f"(DLCA ≈ 1.8, RLCA ≈ 2.1)")
    ax.set_xlabel("Aggregate size N")
    ax.set_ylabel("Radius of gyration Rg")
    ax.set_title("Rg vs N  →  fractal dimension", fontweight="bold")
    ax.grid(alpha=0.3)

    # 6. Core vs border visualisation
    ax = axes[1, 2]
    core_i   = db.core_sample_indices_
    for k in sorted(set(labels)):
        if k == -1: continue
        m_core   = [i for i in core_i   if labels[i] == k]
        m_border = [i for i in range(len(labels))
                    if labels[i] == k and i not in core_i]
        c = plt.cm.tab10(k % 10)
        if m_core:
            ax.scatter(X_all[m_core, 0],   X_all[m_core, 1],
                       c=[c], s=20, alpha=0.7, label=f"A{k} core")
        if m_border:
            ax.scatter(X_all[m_border, 0], X_all[m_border, 1],
                       c=[c], s=80, edgecolors="k", lw=1.5,
                       alpha=0.9)
    ax.scatter(X_all[labels==-1, 0], X_all[labels==-1, 1],
               c="gray", s=30, marker="x", lw=1.5)
    ax.set_title("Core (filled) vs Border (outlined) vs Free (×)",
                 fontweight="bold", fontsize=9)
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    ax.set_aspect("equal"); ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig("plots/dbscan/G_colloidal_aggregates.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved → plots/dbscan/G_colloidal_aggregates.png")


# ════════════════════════════════════════════════════════════════════════════
# H — SOFT-MATTER: DEFECT IDENTIFICATION IN 2-D CRYSTAL
# ════════════════════════════════════════════════════════════════════════════

def crystal_defect_detection():
    """
    Simulates a 2-D colloidal crystal with point defects and grain boundaries.

    Physical scenario
    ─────────────────
    A nearly perfect hexagonal lattice with:
      - Point vacancies (missing particles)
      - Interstitials (extra particles in wrong positions)
      - A grain boundary (two crystal grains meeting at an angle)

    Features used: local bond-orientational order ψ₆ and coordination number z.
    Particles in perfect crystal: high ψ₆, z=6.
    Particles near defects: low ψ₆, z≠6.

    DBSCAN clusters the DEFECT PARTICLES into connected defect regions.
    The perfect lattice background are noise (-1).
    """
    print("\n[H] Soft-matter: crystal defect identification")

    # Build hexagonal lattice
    def hex_lattice(nx, ny, a=1.0):
        # Construct an ideal 2-D triangular/hexagonal lattice in Cartesian space.
        # Every other row is offset by a/2, which produces the hexagonal packing
        # geometry seen in colloidal crystals.
        pts = []
        for i in range(nx):
            for j in range(ny):
                x = i * a + (j % 2) * a/2
                y = j * a * np.sqrt(3)/2
                pts.append([x, y])
        return np.array(pts)

    latt = hex_lattice(18, 18)
    # Add small thermal noise
    # Add small random displacements to mimic thermal fluctuations.  Without
    # this, the lattice would be unrealistically perfect and defect signatures in
    # psi6 / coordination would be too sharp and artificial.
    latt += RNG.normal(0, 0.04, latt.shape)

    # Introduce grain boundary: rotate right half by 15°
    cx    = latt[:, 0].max() / 2
    right = latt[:, 0] > cx
    theta = np.deg2rad(15)
    R     = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta),  np.cos(theta)]])
    latt[right] = (latt[right] - [cx, 0]) @ R.T + [cx, 0]

    # Remove some particles (vacancies)
    n_vac    = 12
    vac_idx  = RNG.choice(len(latt), n_vac, replace=False)
    mask_keep = np.ones(len(latt), bool)
    mask_keep[vac_idx] = False
    latt     = latt[mask_keep]

    # Add interstitials in wrong positions
    n_int = 8
    inter = latt[RNG.choice(len(latt), n_int)] + RNG.normal(0, 0.3, (n_int, 2))
    latt  = np.vstack([latt, inter])

    # Compute local features: coordination number z and bond-orient. ψ₆
    n_pts = len(latt)
    nbrs  = NearestNeighbors(n_neighbors=7).fit(latt)
    d, idx = nbrs.kneighbors(latt)

    # Coordination number = neighbours within 1.5a
    z = (d[:, 1:] < 1.5).sum(axis=1).astype(float)

    # ψ₆ = |mean exp(6iθ)| over neighbours
    psi6 = []
    for i in range(n_pts):
        nbs = idx[i, 1:7]
        angles = np.arctan2(latt[nbs, 1] - latt[i, 1],
                            latt[nbs, 0] - latt[i, 0])
        psi6.append(np.abs(np.mean(np.exp(6j * angles))))
    psi6 = np.array(psi6)

    # Feature engineering step:
    #
    #   1 - psi6        : zero for perfect local hexagonal order, larger when the
    #                     bond-angle environment is disordered.
    #   |z - 6| / 6     : zero for ideal six-fold coordination, larger when local
    #                     packing has missing or extra neighbours.
    #
    # This 2-D feature space is intentionally interpretable: perfect particles
    # sit near the origin; defective particles move away from it.
    # Feature matrix: [1 - ψ₆, |z - 6| / 6]
    # Defective points have HIGH values on both; perfect lattice has LOW values
    X_feat = np.column_stack([1 - psi6, np.abs(z - 6) / 6.0])

    # DBSCAN: find DEFECTIVE particles (high disorder)
    # Set eps to capture neighbouring defects as connected regions.
    # Only cluster TRULY disordered particles: pre-filter to high-disorder subset
    disorder_score = X_feat[:, 0] + X_feat[:, 1]           # combined disorder
    # Only the most disordered particles are passed to DBSCAN.  This is a very
    # intentional modelling decision: DBSCAN is used here to cluster *defects*,
    # not to cluster the whole lattice.  The ordered background is treated as
    # implicit noise / non-target structure.
    threshold      = np.percentile(disorder_score, 65)      # top 35% most disordered
    defect_mask    = disorder_score >= threshold
    X_defect       = X_feat[defect_mask]
    db_def         = DBSCAN(eps=0.35, min_samples=3).fit(X_defect)
    # Map back to full label array
    labels_full    = np.full(len(X_feat), -1, dtype=int)
    labels_full[defect_mask] = db_def.labels_
    # Renumber so perfect lattice = -1, defect clusters = 0,1,2,...
    labels  = labels_full

    n_defect_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_perfect         = (labels == -1).sum()
    print(f"  Defect clusters: {n_defect_clusters}")
    print(f"  Perfect lattice particles (noise label): {n_perfect}")
    print(f"  Defective particles: {(labels != -1).sum()}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("DBSCAN: Crystal Defect Identification",
                 fontsize=13, fontweight="bold")

    # 1. ψ₆ map
    ax = axes[0, 0]
    sc = ax.scatter(latt[:, 0], latt[:, 1], c=psi6,
                    cmap="RdYlGn", s=35, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label="ψ₆")
    ax.set_title("Bond-orientational order ψ₆\nGreen=crystal  Red=defect",
                 fontweight="bold")
    ax.set_aspect("equal"); ax.grid(alpha=0.2)

    # 2. Coordination number map
    ax = axes[0, 1]
    sc = ax.scatter(latt[:, 0], latt[:, 1], c=z,
                    cmap="RdYlGn", s=35, vmin=4, vmax=8)
    plt.colorbar(sc, ax=ax, label="Coordination z")
    ax.set_title("Coordination number z\nz=6 ideal (hexagonal)",
                 fontweight="bold")
    ax.set_aspect("equal"); ax.grid(alpha=0.2)

    # 3. Feature space (defect fingerprint)
    ax = axes[0, 2]
    ax.scatter(X_feat[labels==-1, 0], X_feat[labels==-1, 1],
               c="steelblue", s=20, alpha=0.4, label="perfect")
    for k in range(n_defect_clusters):
        m = labels == k
        ax.scatter(X_feat[m, 0], X_feat[m, 1],
                   c=[plt.cm.tab10(k % 10)], s=60,
                   edgecolors="k", lw=1, label=f"defect C{k}")
    ax.set_xlabel("1 − ψ₆  (disorder)")
    ax.set_ylabel("|z − 6| / 6  (coord. deviation)")
    ax.set_title("Feature space: DBSCAN clusters defects",
                 fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 4. DBSCAN result on crystal
    ax = axes[1, 0]
    ax.scatter(latt[labels==-1, 0], latt[labels==-1, 1],
               c="lightsteelblue", s=20, alpha=0.5, label="perfect lattice")
    for k in range(n_defect_clusters):
        m = labels == k
        ax.scatter(latt[m, 0], latt[m, 1],
                   c=[plt.cm.tab10(k % 10)], s=80,
                   edgecolors="k", lw=1.2, label=f"defect C{k}")
    ax.set_title("Defect clusters in real space",
                 fontweight="bold")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.2)

    # 5. ψ₆ histogram: perfect vs defective
    ax = axes[1, 1]
    ax.hist(psi6[labels == -1],  bins=30, color="steelblue",
            alpha=0.7, density=True, label="perfect (DBSCAN noise)")
    ax.hist(psi6[labels != -1],  bins=30, color="tomato",
            alpha=0.7, density=True, label="defective (DBSCAN cluster)")
    ax.set_xlabel("ψ₆"); ax.set_ylabel("Density")
    ax.set_title("ψ₆ distribution: perfect vs defective",
                 fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # 6. Grain boundary highlight
    ax = axes[1, 2]
    # Use the grain boundary cluster (usually the largest defect cluster)
    defect_sizes = [(k, (labels==k).sum())
                    for k in set(labels) if k != -1]
    defect_sizes.sort(key=lambda x: -x[1])
    ax.scatter(latt[labels==-1, 0], latt[labels==-1, 1],
               c="lightgray", s=18, alpha=0.4)
    for rank, (k, sz) in enumerate(defect_sizes):
        m   = labels == k
        tag = "Grain boundary" if rank == 0 else f"Point defects C{k}"
        ax.scatter(latt[m, 0], latt[m, 1],
                   c=[plt.cm.tab10(rank % 10)], s=90,
                   edgecolors="k", lw=1.2, label=f"{tag} (n={sz})")
    ax.axvline(cx, color="red", ls="--", lw=1.5, alpha=0.5,
               label="grain boundary axis")
    ax.set_title("Grain boundary vs point defects",
                 fontweight="bold")
    ax.set_aspect("equal")
    ax.legend(fontsize=8); ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig("plots/dbscan/H_crystal_defects.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved → plots/dbscan/H_crystal_defects.png")


# ════════════════════════════════════════════════════════════════════════════
# I — NOISE POINT ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

def noise_analysis(X, y_true, eps, min_samples, name="dataset"):
    """
    Deep-dive into noise points — who are they, and are they meaningful?

    In research, noise points can be:
    ─────────────────────────────────
    (a) Genuine outliers / experimental artefacts → discard
    (b) Transition-state / boundary configurations → scientifically interesting
    (c) A sign that ε is too small → tune parameters

    Diagnostic questions
    ────────────────────
    1. How far are noise points from the nearest cluster?
       (If all near-ε → borderline; increase ε slightly)
    2. Do noise points cluster among themselves?
       (If yes → hidden low-density cluster; reduce min_samples)
    3. Do noise points have unusual feature values?
       (If yes → they represent rare/interesting states)
    """
    print(f"\n[I] Noise analysis — {name}")
    # Again we standardise first so the neighbour radius is measured in a
    # dimensionless, scale-balanced feature space.  Without this, one large-scale
    # feature could dominate the neighbour search.
    Xs = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(Xs)
    labels = db.labels_

    noise_mask  = labels == -1
    clust_mask  = labels != -1
    n_noise     = noise_mask.sum()

    # This branch exists because DBSCAN can legitimately assign every point to
    # some cluster if eps is large enough or min_samples is permissive enough.
    # In that case there is no "noise population" left to analyse.
    if n_noise == 0:
        print("  No noise points with these parameters — increase eps or decrease min_samples")
        return

    print(f"  {n_noise} noise points ({100*n_noise/len(labels):.1f}%)")

    X_noise   = Xs[noise_mask]
    X_cluster = Xs[clust_mask]

    # Distance from each noise point to nearest cluster point
    from scipy.spatial.distance import cdist
    if len(X_cluster) > 0:
        # Distance-to-nearest-cluster is one of the most useful noise diagnostics.
        # Small distances imply borderline rejection; large distances suggest truly
        # isolated outliers or genuinely sparse regions.
        d_to_cluster = cdist(X_noise, X_cluster).min(axis=1)
    else:
        d_to_cluster = np.full(n_noise, np.nan)

    # Can noise points form their own clusters?
    if n_noise >= 5:
        db_noise = DBSCAN(eps=eps * 1.5, min_samples=3).fit(X_noise)
        n_sub = len(set(db_noise.labels_)) - (1 if -1 in db_noise.labels_ else 0)
        print(f"  Sub-clusters within noise (ε×1.5): {n_sub}")
    else:
        n_sub = 0

    pct_borderline = 100 * (d_to_cluster < eps * 1.2).mean()
    print(f"  Noise points within 1.2×ε of cluster: {pct_borderline:.1f}%")
    if pct_borderline > 60:
        print("  → Many borderline noise: consider increasing eps slightly")
    elif n_sub > 0:
        print("  → Sub-clusters in noise: consider reducing min_samples")
    else:
        print("  → Noise looks genuinely sparse/outlier-like")

    # PCA is used here *only* as a plotting projection when dimensionality is
    # greater than 2.  The clustering was already done in Xs.  PCA does not feed
    # back into the fitted DBSCAN labels.
    Xp = PCA(n_components=2).fit_transform(Xs) if Xs.shape[1] > 2 else Xs

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Noise Point Analysis — {name}", fontsize=12, fontweight="bold")

    # 1. Noise vs cluster in embedding space
    ax = axes[0]
    ax.scatter(Xp[clust_mask, 0], Xp[clust_mask, 1],
               c=labels[clust_mask], cmap="tab10",
               s=14, alpha=0.5, label="clustered")
    ax.scatter(Xp[noise_mask, 0], Xp[noise_mask, 1],
               c="red", s=45, marker="x", lw=1.8,
               label=f"noise ({n_noise})")
    ax.set_title("Noise points in embedding", fontweight="bold")
    ax.legend(fontsize=9); ax.set_xticks([]); ax.set_yticks([])

    # 2. Distance distribution
    ax = axes[1]
    ax.hist(d_to_cluster, bins=30, color="tomato",
            edgecolor="k", lw=0.5, alpha=0.8)
    ax.axvline(eps, color="blue", ls="--", lw=2,
               label=f"ε = {eps:.3f}")
    ax.axvline(eps * 1.2, color="blue", ls=":", lw=1.5,
               label=f"1.2×ε = {eps*1.2:.3f}")
    ax.set_xlabel("Distance to nearest cluster point")
    ax.set_ylabel("Count")
    ax.set_title("Noise proximity to clusters\n"
                 "Left of ε = borderline noise", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # 3. What happens if we lower min_samples?
    ms_vals = [max(2, min_samples - 4), max(2, min_samples - 2),
               min_samples, min_samples + 3, min_samples + 6]
    noise_counts = []
    for ms in ms_vals:
        lb = DBSCAN(eps=eps, min_samples=ms, n_jobs=-1).fit_predict(Xs)
        noise_counts.append((lb == -1).sum())

    ax = axes[2]
    ax.plot(ms_vals, noise_counts, "o-", color="steelblue", lw=2, ms=9)
    ax.axvline(min_samples, color="red", ls="--", lw=2,
               label=f"current ms={min_samples}")
    ax.set_xlabel("min_samples"); ax.set_ylabel("Noise count")
    ax.set_title("Noise vs min_samples\n"
                 "Sensitivity check", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fname = f"plots/dbscan/I_noise_{name.replace(' ','_')}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {fname}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

# =============================================================================
# EXECUTION ORDER IN MAIN
# =============================================================================
# The main block is arranged pedagogically rather than mathematically.  The user
# is taken from first principles to applied workflows in this order:
#
#   A  Understand DBSCAN mechanics directly
#   B  Learn how to choose eps
#   C  Inspect the real sklearn estimator
#   D  Explore parameter sensitivity
#   E  Measure post-fit diagnostics
#   F  Compare against a baseline algorithm
#   G  Apply to soft-matter aggregation
#   H  Apply to defect clustering in crystals
#   I  Reinterpret noise as an object of study
# =============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  DBSCAN — COMPLETE TUTORIAL")
    print("=" * 65)

    demo_scratch()                                                   # A

    X_circ, y_circ = load("circles")
    X_moon,  y_moon  = load("moons")
    X_blob,  y_blob  = load("blobs_easy")
    X_coll,  y_coll  = load("colloidal_phases")

    eps_circ = kdistance_plot(X_circ, min_samples=8, name="circles")  # B
    eps_moon = kdistance_plot(X_moon,  min_samples=8, name="moons")   # B


    sklearn_dbscan_annotated(                                         # C
        X_circ, y_circ, eps=eps_circ, min_samples=8, name="circles")

    parameter_grid(X_circ, y_circ, name="circles")                   # D
    parameter_grid(X_blob, y_blob, name="blobs")                     # D


    cluster_diagnostics(                                              # E
        X_circ, y_circ, eps=eps_circ, min_samples=8, name="circles")

    dbscan_vs_kmeans()                                                # F

    
    colloidal_aggregate_detection()                                   # G
    crystal_defect_detection()                                        # H

    noise_analysis(X_circ, y_circ,                                   # I
                   eps=eps_circ, min_samples=8, name="circles")

    print("\n" + "=" * 65)
    print("  DBSCAN COMPLETE — see plots/dbscan/")
    print("=" * 65)
