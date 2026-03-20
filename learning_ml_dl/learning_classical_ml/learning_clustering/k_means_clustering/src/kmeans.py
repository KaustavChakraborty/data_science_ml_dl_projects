"""
================================================================================
01_kmeans.py
K-MEANS CLUSTERING — FROM SCRATCH TO ADVANCED
================================================================================

This file is a complete learning module for K-means clustering.

It is designed in a layered way:

    A. Build K-means from scratch using NumPy
       -> understand the algorithm deeply

    B. Compare random initialization vs K-means++
       -> learn why initialization matters so much

    C. Use sklearn KMeans properly
       -> understand practical library usage

    D. Learn how to choose the number of clusters K
       -> use multiple metrics, not guesswork

    E. Visualize silhouette plots
       -> inspect clustering quality point-by-point

    F. Apply K-means to a soft-matter / research-style dataset
       -> see how clustering enters scientific workflows

    G. Study K-means failure modes
       -> understand where it breaks and why

    H. Compare standard KMeans and MiniBatchKMeans
       -> see how scalable clustering works for large datasets

IMPORTANT
---------
Run `00_generate_datasets.py` first.

This file expects datasets stored in:
    data/<dataset_name>.npz

Required packages
-----------------
numpy
scipy
scikit-learn
matplotlib
seaborn

================================================================================
HOW TO READ THIS FILE
================================================================================

If you are learning K-means for the first time, this is a good reading order:

1. Read the imports and helper loader
2. Read class KMeansScratch carefully
3. Focus especially on:
       _init_random
       _init_kmeanspp
       _lloyd
       fit
       predict

4. Then go to compare_init()
5. Then sklearn_demo()
6. Then choose_K()
7. Then failure_modes()

This file is both:
- a runnable script
- a teaching document

================================================================================
CORE IDEA OF K-MEANS
================================================================================

Given a dataset X and a chosen number of clusters K:

1. Start with K initial centroids
2. Assign each point to the nearest centroid
3. Recompute each centroid as the mean of the points assigned to it
4. Repeat until centroids stop moving much

That is K-means.

The quantity it tries to minimize is:

    J = sum over clusters of sum over points in cluster of
        squared distance(point, centroid)

This is called:
- WCSS = within-cluster sum of squares
- inertia
- K-means objective

================================================================================
"""

# =============================================================================
# SECTION 1 — IMPORTS
# =============================================================================
#
# This section imports everything needed by the tutorial.
#
# The imports come from:
# - Python standard library
# - NumPy / SciPy for numerical work
# - matplotlib for plotting
# - scikit-learn for production-ready clustering tools
#
# =============================================================================

import os
import warnings

# Suppress warnings so the tutorial output remains clean and readable.
# This is useful in long educational scripts where non-critical warnings can
# distract from the main teaching flow.
warnings.filterwarnings("ignore")

import numpy as np

# Use a non-interactive backend.
# This means plots are saved directly to disk instead of popping up as windows.
# Very useful when running on servers, remote machines, or script pipelines.
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# Counter will be used later when multiple K-selection methods vote for a
# best K. We then take the majority recommendation.
from collections import Counter

# Main clustering algorithms from sklearn
from sklearn.cluster import KMeans, MiniBatchKMeans

# Validation metrics and clustering quality metrics
from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    adjusted_mutual_info_score
)

# StandardScaler is essential because K-means is distance-based.
# Without scaling, features with large numeric magnitude dominate the distances.
from sklearn.preprocessing import StandardScaler

# PCA is used for dimensionality reduction in visualization.
from sklearn.decomposition import PCA

# cdist computes distances between all points and centroids efficiently.
from scipy.spatial.distance import cdist


# =============================================================================
# SECTION 2 — GLOBAL SETUP
# =============================================================================
#
# This section creates directories and random seeds used throughout the module.
#
# =============================================================================

# Folder where output plots from this module will be saved.
os.makedirs("plots/kmeans", exist_ok=True)

# Modern NumPy random generator object.
# Useful for our own manual implementations.
RNG = np.random.default_rng(42)

# Classic integer seed used for sklearn random_state arguments.
SEED = 42


# =============================================================================
# SECTION 3 — DATA LOADER
# =============================================================================
#
# This function loads datasets generated earlier by 00_generate_datasets.py.
#
# Every dataset is stored in:
#     data/<name>.npz
#
# and contains:
#     X : feature matrix
#     y : label vector
#
# =============================================================================

def load(name):
    """
    Load a dataset from disk.

    Parameters
    ----------
    name : str
        Base dataset name, for example:
            "blobs_easy"
            "colloidal_phases"
            "iris"

    Returns
    -------
    X : ndarray
        Feature matrix of shape (n_samples, n_features)

    y : ndarray
        Ground-truth labels of shape (n_samples,)

    Why this helper exists
    ----------------------
    Keeping dataset loading in one small function makes the rest of the code
    cleaner and easier to read.
    """
    d = np.load(f"data/{name}.npz")
    return d["X"], d["y"]


# ════════════════════════════════════════════════════════════════════════════
# A — K-MEANS FROM SCRATCH (PURE NUMPY)
# ════════════════════════════════════════════════════════════════════════════
#
# This is the most important educational part of the file.
#
# Instead of calling sklearn immediately, we implement K-means ourselves.
# This helps you see:
#
# - how centroids are initialized
# - how assignment works
# - how centroid updates are computed
# - how convergence is checked
# - why multiple restarts are useful
#
# If you understand this class, you understand K-means.
#
# ════════════════════════════════════════════════════════════════════════════

class KMeansScratch:
    """
    Pure NumPy implementation of K-means using Lloyd's algorithm.

    ---------------------------------------------------------------------------
    CONCEPT NOTE
    ---------------------------------------------------------------------------
    K-means solves the following problem:

        Given:
            - data points X
            - number of clusters K

        Find:
            - K centroids
            - assignment of every point to one centroid

        Such that:
            - points are close to their assigned centroid
            - the total squared distance is minimized

    Mathematically, the objective is:

        J = Σ_k Σ_{x_i in cluster k} ||x_i - μ_k||²

    where:
        μ_k = centroid of cluster k

    ---------------------------------------------------------------------------
    HOW LLOYD'S ALGORITHM WORKS
    ---------------------------------------------------------------------------
    K-means alternates between two steps:

    1. Assignment step
       Fix centroids, then assign each point to the nearest centroid.

    2. Update step
       Fix assignments, then recompute each centroid as the mean of all points
       assigned to that cluster.

    Repeat until convergence.

    ---------------------------------------------------------------------------
    WHY MULTIPLE RESTARTS?
    ---------------------------------------------------------------------------
    K-means can get stuck in different local minima depending on where the
    centroids start. So we often run it several times and keep the best result.

    ---------------------------------------------------------------------------
    PARAMETERS
    ---------------------------------------------------------------------------
    K : int
        Number of clusters.

    init : str
        Initialization strategy:
            "kmeans++" or "random"

    max_iter : int
        Maximum iterations per run.

    tol : float
        Stop when centroid movement becomes smaller than this threshold.

    n_init : int
        Number of independent restarts.

    random_state : int
        Seed for reproducibility.

    ---------------------------------------------------------------------------
    STORED ATTRIBUTES AFTER fit()
    ---------------------------------------------------------------------------
    labels_
    cluster_centers_
    inertia_
    all_histories
    """

    def __init__(self, K=3, init="kmeans++", max_iter=300,
                 tol=1e-4, n_init=10, random_state=42):
        """
        Store user-supplied K-means settings.

        Nothing is fitted here yet.
        This constructor only sets up the configuration.
        """
        self.K = K
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.rng = np.random.default_rng(random_state)

    # ─────────────────────────────────────────────────────────────────────────
    # INITIALIZATION METHODS
    # ─────────────────────────────────────────────────────────────────────────
    #
    # Initialization matters a lot in K-means.
    #
    # Poor initialization can lead to:
    # - slow convergence
    # - worse final clustering
    # - bad local minima
    #
    # Two strategies are implemented below:
    #
    # 1. random
    # 2. kmeans++
    #
    # ─────────────────────────────────────────────────────────────────────────

    def _init_random(self, X):
        """
        Randomly choose K distinct data points as the initial centroids.

        -----------------------------------------------------------------------
        BEGINNER NOTE
        -----------------------------------------------------------------------
        This is the simplest possible initialization strategy.

        Idea:
            pick K points from the dataset at random

        Advantage:
            very easy to implement

        Disadvantage:
            can be very bad if
            - two centroids start very close to each other
            - one real cluster gets no initial centroid
            - centroids start in noisy regions

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        centroids : ndarray of shape (K, n_features)
        """
        idx = self.rng.choice(len(X), size=self.K, replace=False)
        return X[idx].copy()

    def _init_kmeanspp(self, X):
        """
        K-means++ initialization.

        -----------------------------------------------------------------------
        BEGINNER NOTE
        -----------------------------------------------------------------------
        K-means++ improves initialization by spreading centroids out.

        It works like this:

        Step 1:
            Choose the first centroid randomly.

        Step 2:
            Compute how far every point is from the nearest already chosen
            centroid.

        Step 3:
            Points that are far away get higher probability of being chosen as
            the next centroid.

        Step 4:
            Repeat until K centroids are chosen.

        Why this is smart:
            it avoids putting many centroids in the same region too early.

        This usually gives:
        - lower final inertia
        - faster convergence
        - more stable clustering results

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        centroids : ndarray of shape (K, n_features)
        """

        # Number of data points in the dataset.
        # If X has shape (n_samples, n_features), then len(X) = n_samples. 
        n = len(X)

        # Choose the first centroid uniformly at random.
        centroids = [X[self.rng.integers(n)].copy()]

        # -----------------------------------------------------------------------
        # CHOOSE THE REMAINING K-1 CENTROIDS
        # -----------------------------------------------------------------------
        #
        # Each new centroid is chosen using distance-weighted sampling.
        #
        # If we need K total centroids and already have 1, then we repeat this
        # process K-1 more times.
        for _ in range(1, self.K):
            # -------------------------------------------------------------------
            # For every existing centroid c, compute squared distance from every
            # point in X to c.
            #
            # For one centroid c:
            #   np.sum((X - c) ** 2, axis=1)
            #
            # gives a vector of length n, where each entry is:
            #   squared distance of point i from centroid c
            #
            # We do this for every currently chosen centroid.
            #
            # np.stack(...) then makes a 2D array:
            #   shape = (number_of_current_centroids, n)
            #
            # After that, np.min(..., axis=0) keeps, for each point, only the
            # smallest squared distance among all chosen centroids.
            #
            # So finally:
            #
            #   D_sq[i] = squared distance of point i to its nearest currently
            #             chosen centroid
            #
            # This is the key quantity in K-means++.
            # -------------------------------------------------------------------            
            D_sq = np.min(
                np.stack([np.sum((X - c) ** 2, axis=1) for c in centroids]),
                axis=0,
            )

            # -------------------------------------------------------------------
            # CONVERT DISTANCES INTO PROBABILITIES
            # -------------------------------------------------------------------
            #
            # A point with larger D_sq should be more likely to be chosen.
            #
            # So we normalize the squared distances so that they sum to 1:
            #
            #   probs[i] = D_sq[i] / sum(D_sq)
            #
            # Then probs becomes a valid probability distribution.
            #
            # Interpretation:
            # - points close to existing centroids get small probability
            # - points far away get large probability
            probs = D_sq / D_sq.sum()

            # -------------------------------------------------------------------
            # SAMPLE THE NEXT CENTROID USING THOSE PROBABILITIES
            # -------------------------------------------------------------------
            #
            # self.rng.choice(n, p=probs) chooses one index from 0..n-1 according
            # to the probability distribution probs.
            #
            # Then X[...] gives the actual data point at that index.
            centroids.append(X[self.rng.choice(n, p=probs)].copy())

        return np.array(centroids)   # shape = (K, number_of_features)

    # ─────────────────────────────────────────────────────────────────────────
    # ONE FULL LLOYD RUN
    # ─────────────────────────────────────────────────────────────────────────
    #
    # This is the heart of K-means.
    #
    # A single run means:
    #   starting from one set of initial centroids
    #   -> repeatedly assign points
    #   -> recompute centroids
    #   -> stop when stable
    #
    # ─────────────────────────────────────────────────────────────────────────

    def _lloyd(self, X, centroids):
        """
        Run one full K-means optimization starting from the given centroids.

        ===========================================================================
        THE K-MEANS OBJECTIVE
        ===========================================================================
        K-means tries to minimize:

            J = sum over all points of squared distance to assigned centroid

        More formally:

            J = Σ_k Σ_{x_i in cluster k} ||x_i - μ_k||^2

        where:
            μ_k = centroid of cluster k

        Smaller J means:
        - tighter clusters
        - points closer to centroids
        - better K-means solution

        ===========================================================================
        WHAT HAPPENS IN EACH ITERATION?
        ===========================================================================
        Each iteration of Lloyd's algorithm has three core parts:

        PART 1 — ASSIGNMENT
            For every point, compute distance to every centroid and assign the point
            to the nearest one.

        PART 2 — OBJECTIVE EVALUATION
            Compute the current K-means objective J.

        PART 3 — CENTROID UPDATE
            For each cluster, compute the mean of all assigned points and move the
            centroid there.

        Then we check whether centroids moved only a tiny amount.
        If yes, stop.

        ===========================================================================
        WHY THIS WORKS
        ===========================================================================
        K-means works because each of these two operations improves or preserves
        the objective:

        Assignment step:
            If centroids are fixed, the best cluster for a point is the nearest one.

        Update step:
            If assignments are fixed, the best centroid is the mean of the points in
            that cluster.

        So each full iteration makes J go down or stay the same.

        ===========================================================================
        DISTANCE COMPUTATION TRICK
        ===========================================================================
        A direct way to compute distances would be to loop over:
        - every point
        - every centroid

        But that would be slow in Python.

        Instead, this code uses the algebraic identity:

            ||x - c||² = ||x||² - 2 x·c + ||c||²

        This allows vectorized NumPy computation of the full distance matrix.

        ===========================================================================
        EMPTY CLUSTER PROBLEM
        ===========================================================================
        Sometimes a centroid may end up with no assigned points.

        Then its mean is undefined.

        In that case, this code handles it by reinitializing that centroid to a
        random data point.

        That keeps the algorithm running safely.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        centroids : ndarray of shape (K, n_features)
            Initial centroids for this run.

        Returns
        -------
        labels : ndarray
            Final cluster assignments.

        centroids : ndarray
            Final centroids.

        J : float
            Final K-means objective value (inertia / WCSS).

        history : list of float
            Objective value per iteration.
        """

        # Number of samples and number of features.
        n, d = X.shape

        # Store the K-means objective at every iteration.
        # This is useful for understanding convergence behavior.
        history = []


        # -----------------------------------------------------------------------
        # MAIN LLOYD ITERATION LOOP
        # -----------------------------------------------------------------------
        #
        # Repeat the assign-update cycle until:
        # - centroids stop moving enough, or
        # - max_iter iterations are reached
        for _ in range(self.max_iter):

            # ================================================================
            # STEP 1 — ASSIGNMENT STEP
            # ================================================================
            #
            # We want a matrix D_sq of shape (n, K) such that:
            #
            #   D_sq[i, k] = squared distance between point i and centroid k
            #
            # Instead of computing this with slow nested loops, we use the identity:
            #
            #   ||x - c||² = ||x||² - 2 x·c + ||c||²
            #
            # Term by term:
            #
            # np.sum(X ** 2, axis=1, keepdims=True)
            #   gives shape (n, 1)
            #   each row = squared norm of one data point
            #
            # X @ centroids.T
            #   gives shape (n, K)
            #   each entry = dot product between point i and centroid k
            #
            # np.sum(centroids ** 2, axis=1)
            #   gives shape (K,)
            #   each entry = squared norm of one centroid
            #
            # Broadcasting combines them into a full (n, K) matrix.
            D_sq = (
                np.sum(X ** 2, axis=1, keepdims=True)
                - 2.0 * (X @ centroids.T)
                + np.sum(centroids ** 2, axis=1)
            )

            # Due to floating-point precision, tiny negative values may appear.
            # But squared distances must be nonnegative.
            D_sq = np.maximum(D_sq, 0.0)

            # Assign each point to the nearest centroid.
            labels = np.argmin(D_sq, axis=1)

            # ================================================================
            # STEP 2 — COMPUTE CURRENT OBJECTIVE J
            # ================================================================
            #
            # D_sq[np.arange(n), labels]
            # picks, for each point i, the squared distance to the centroid that
            # point was assigned to.
            #
            # Summing those gives the total K-means objective value:
            #
            #   J = total within-cluster sum of squares
            J = D_sq[np.arange(n), labels].sum()
            history.append(J)

            # ================================================================
            # STEP 3 — UPDATE CENTROIDS
            # ================================================================
            #
            # For each cluster k:
            #   take all points assigned to k
            #   compute their mean
            #   that mean becomes the new centroid
            #
            # ================================================================
            new_c = np.zeros_like(centroids)

            # Update each centroid one by one.
            for k in range(self.K):

                # mask is a boolean array marking which points belong to cluster k.
                #
                # Example:
                # labels = [0,2,1,0,1]
                # for k=1, mask becomes:
                # [False, False, True, False, True]
                mask = labels == k

                # If the cluster has at least one point:
                #   centroid = mean of its points
                #
                # Else:
                #   re-seed using a random point
                # Why mean?
                # Because for squared Euclidean distance, the mean is the point that
                # minimizes total squared distance to all cluster members.
                new_c[k] = (
                    X[mask].mean(axis=0) if mask.sum() > 0
                    else X[self.rng.integers(n)]
                )

            # ================================================================
            # STEP 4 — CONVERGENCE CHECK
            # ================================================================
            #
            # Compute how much each centroid moved.
            # If the largest movement is smaller than tolerance, stop.
            #
            # ================================================================
            shift = np.sqrt(np.sum((new_c - centroids) ** 2, axis=1)).max()

            # Replace old centroids with updated ones.
            centroids = new_c

            # Stop early if centroids moved very little.
            if shift < self.tol:
                break
        
        # Return final results for this one run.
        return labels, centroids, J, history

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC FIT METHOD
    # ─────────────────────────────────────────────────────────────────────────
    #
    # This method performs multiple K-means restarts and keeps the best one.
    #
    # ─────────────────────────────────────────────────────────────────────────

    def fit(self, X):
        """
        Fit K-means to data using multiple independent restarts.

        -----------------------------------------------------------------------
        BEGINNER NOTE
        -----------------------------------------------------------------------
        Why not run K-means only once?

        Because K-means can converge to different answers depending on how
        centroids were initialized.

        So this method:
        - runs K-means several times
        - stores the objective history of each run
        - keeps the best run (smallest final inertia)

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self
        """

        # Convert input data to a NumPy array of floats
        X = np.asarray(X, dtype=float)

        # Start with infinity so that the first real run will definitely improve it
        best_J = np.inf

        # Store the objective history of every restart.
        # Each element added later will be a list like:
        #   [J_iter_1, J_iter_2, ..., J_iter_t]
        self.all_histories = []

        # Repeat K-means multiple times from different initial centroid choices.
        # This reduces the risk of ending with a poor local minimum.
        for _ in range(self.n_init):
            
            # ---------------------------------------------------------------
            # Choose initial centroids for this restart
            # ---------------------------------------------------------------
            # If the user asked for "kmeans++", use the smarter spread-out
            # initialization. Otherwise, fall back to random initialization.
            c0 = (
                self._init_kmeanspp(X)
                if self.init == "kmeans++"
                else self._init_random(X)
            )

            # ---------------------------------------------------------------
            # Run one full Lloyd optimization
            # ---------------------------------------------------------------
            # _lloyd() performs the iterative K-means process:
            #   1. assign points to nearest centroid
            #   2. recompute centroids as means
            #   3. repeat until convergence
            #
            # Returned values:
            #   lbl  -> final labels for this run
            #   c    -> final centroids for this run
            #   J    -> final inertia / WCSS for this run
            #   hist -> objective value at each iteration
            lbl, c, J, hist = self._lloyd(X, c0)

            # Save this run's convergence history for later inspection.
            self.all_histories.append(hist)

            # Keep the best solution found so far.
            if J < best_J:
                best_J = J
                self.labels_ = lbl
                self.cluster_centers_ = c
                self.inertia_ = J

        return self

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC PREDICT METHOD
    # ─────────────────────────────────────────────────────────────────────────
    #
    # Once centroids are learned, we can assign new unseen points to the
    # nearest centroid.
    #
    # ─────────────────────────────────────────────────────────────────────────

    def predict(self, X):
        """
        Assign new points to the nearest learned centroid.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray
            Cluster index for each new point.

        Interpretation
        --------------
        This does not "re-fit" K-means.
        It only uses the already learned centroids and asks:
            which centroid is nearest to each point?
        """
        X = np.asarray(X, dtype=float)
        return np.argmin(cdist(X, self.cluster_centers_, "sqeuclidean"), axis=1)


# ════════════════════════════════════════════════════════════════════════════
# B — INITIALIZATION COMPARISON
# ════════════════════════════════════════════════════════════════════════════
#
# This experiment compares:
# - random initialization
# - k-means++ initialization
#
# over many repeated runs.
#
# ════════════════════════════════════════════════════════════════════════════

def compare_init(X, K=3, n_runs=50):
    """
    Compare random initialization and K-means++ empirically.

    ---------------------------------------------------------------------------
    BEGINNER NOTE
    ---------------------------------------------------------------------------
    For each random seed, this function runs:

        1. KMeans(init="random", n_init=1)
        2. KMeans(init="k-means++", n_init=1)

    Then it records:
    - final inertia
    - number of iterations to converge

    Finally it plots histograms showing:
    - how often each method gets lower inertia
    - how fast each method converges

    Parameters
    ----------
    X : ndarray
        Input data.

    K : int, default=3
        Number of clusters.

    n_runs : int, default=50
        Number of independent trials.

    Why n_init=1 here?
    ------------------
    Because we want to compare single initializations fairly.
    If we used many restarts internally, the comparison between random and
    k-means++ would become less direct.
    """
    print("\n[B] Init comparison: random vs K-means++")

    # Scale first because K-means uses Euclidean distances.
    Xs = StandardScaler().fit_transform(X)

    # res is a nested dictionary that stores the outcomes of repeated runs.
    # This makes later plotting and summary calculations easy.
    res = {
        "random": {"inertia": [], "iters": []},
        "k-means++": {"inertia": [], "iters": []}
    }


    # -----------------------------------------------------------------------
    # Repeat the experiment many times with different random seeds.
    #
    # For each seed, we run both initialization methods once.
    # This creates a fair side-by-side comparison across many independent
    # trials.
    # -----------------------------------------------------------------------
    for seed in range(n_runs):
        for init in ("random", "k-means++"):

            # Create one KMeans object for this specific trial.
            #
            # Key settings:
            # n_clusters = K
            # init       = current method being tested
            # n_init     = 1 (important for fair one-shot comparison)
            # max_iter   = upper bound on Lloyd iterations
            # random_state = seed for reproducibility
            km = KMeans(
                n_clusters=K,
                init=init,
                n_init=1,
                max_iter=300,
                random_state=seed
            )

            # Fit K-means on the scaled data.
            km.fit(Xs)

            res[init]["inertia"].append(km.inertia_)
            res[init]["iters"].append(km.n_iter_)

    # -----------------------------------------------------------------------
    # Create a 1x2 figure:
    # - left  = inertia histograms
    # - right = convergence iteration histograms
    # -----------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Initialisation: Random vs K-means++",
                 fontsize=13, fontweight="bold")

    col = {"random": "tomato", "k-means++": "steelblue"}

    # -----------------------------------------------------------------------
    # LEFT PANEL — HISTOGRAM OF FINAL INERTIA
    #
    # This shows the distribution of final K-means objective values over all
    # repeated runs.
    #
    # Lower inertia means tighter clusters under the K-means objective.
    # -----------------------------------------------------------------------
    for init in ("random", "k-means++"):
        ax1.hist(
            res[init]["inertia"],
            bins=20,
            alpha=0.6,
            color=col[init],
            edgecolor="k",
            lw=0.5,
            label=init
        )

    # -----------------------------------------------------------------------
    # RIGHT PANEL — HISTOGRAM OF ITERATIONS TO CONVERGENCE
    #
    # This shows how quickly each initialization method tends to converge.
    #
    # If one method consistently needs fewer iterations, it suggests that its
    # starting centroids are already closer to a good solution.
    # -----------------------------------------------------------------------
    for init in ("random", "k-means++"):
        ax2.hist(
            res[init]["iters"],
            bins=range(1, 51),
            alpha=0.6,
            color=col[init],
            edgecolor="k",
            lw=0.5,
            label=init
        )

    ax1.set_xlabel("Final WCSS")
    ax1.set_ylabel("Count (50 runs)")
    ax1.set_title("Lower is better", fontweight="bold")
    ax1.legend()
    ax1.grid(alpha=0.3)

    avg_r = np.mean(res["random"]["iters"])
    avg_p = np.mean(res["k-means++"]["iters"])

    ax2.set_xlabel("Iterations to convergence")
    ax2.set_ylabel("Count")
    ax2.set_title(f"K++: {avg_p:.1f} avg iters vs Random: {avg_r:.1f}",
                  fontweight="bold")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/kmeans/init_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    for init in ("random", "k-means++"):
        v = res[init]["inertia"]
        print(f"  {init:<12}: mean inertia {np.mean(v):.2f} ± {np.std(v):.2f}")

    print("  saved  plots/kmeans/init_comparison.png")


# ════════════════════════════════════════════════════════════════════════════
# C — SKLEARN KMeans DEMO
# ════════════════════════════════════════════════════════════════════════════
#
# This section shows how to use sklearn's implementation properly and how to
# interpret its parameters and outputs.
#
# ════════════════════════════════════════════════════════════════════════════

def sklearn_demo(X, y_true, K=3, name="dataset"):
    """
    Demonstrate sklearn KMeans with full parameter explanation.

    ---------------------------------------------------------------------------
    PURPOSE OF THIS FUNCTION
    ---------------------------------------------------------------------------
    To understand how the professional implementation works.

    Important sklearn KMeans parameters:
    ------------------------------------
    n_clusters
        Number of clusters K to fit.

    init
        How centroids are initialized.

    n_init
        Number of independent restarts.

    max_iter
        Maximum number of Lloyd iterations per restart.

    tol
        Convergence threshold - If centroid movement becomes sufficiently small, the run stops early.

    algorithm
        Internal implementation style ("lloyd", "elkan", etc.).

    random_state
        Reproducibility.

    ===========================================================================
    WHAT METRICS ARE COMPUTED?
    ===========================================================================
    This function reports three types of quantities:

    1. inertia_
       The K-means objective value.
       Lower means tighter clusters under the K-means criterion.

    2. External validation metrics
       These compare predicted clusters to known true labels.

       ARI = Adjusted Rand Index
           Measures agreement between predicted and true grouping, corrected for
           chance.

       AMI = Adjusted Mutual Information
           Another label-agreement metric, also corrected for chance.

       These are useful here because the dataset is synthetic or benchmarked,
       so we know the true labels.

    3. Internal validation metric
       silhouette_score
           Measures cluster separation and compactness using only X and the
           predicted labels, without needing true labels.

    ===========================================================================
    WHY BOTH EXTERNAL AND INTERNAL METRICS?
    ===========================================================================
    External metrics tell you:
        "Did K-means recover the known true grouping?"

    Internal metrics tell you:
        "Does the clustering look geometrically good in feature space?"

    In real unsupervised problems, you usually do not have true labels, so
    internal metrics become especially important.

    ===========================================================================
    INPUT
    ===========================================================================
    Parameters
    ----------
    X : ndarray
        Input data matrix of shape (n_samples, n_features).

    y_true : ndarray
        Known true labels.

        These are NOT used during clustering.
        They are used only after fitting for benchmarking.

    K : int, default=3
        Number of clusters to fit.

    name : str, default="dataset"
        Dataset name used in the printed section header.

    ===========================================================================
    OUTPUT
    ===========================================================================
    Returns
    -------
    km : fitted KMeans object
        The trained sklearn KMeans model.

    Xs : ndarray
        The scaled version of X that was used for fitting.

    ===========================================================================
    PRACTICAL INTERPRETATION OF OUTPUT
    ===========================================================================
    Example output line:

        inertia=123.4567  iters=5  ARI=0.98  AMI=0.97  silhouette=0.71

    Interpretation:
    - inertia is the final WCSS
    - iters tells you how quickly it converged
    - ARI and AMI near 1 mean clustering aligns well with true labels
    - silhouette near 1 means clusters are well separated geometrically

    ===========================================================================
    """

    print(f"\n[C] sklearn KMeans — {name}")

    Xs = StandardScaler().fit_transform(X)

    km = KMeans(
        n_clusters=K,
        init="k-means++",
        n_init=10,
        max_iter=300,
        tol=1e-4,
        algorithm="lloyd",
        random_state=SEED,
        verbose=0,
    )
    # Fit the KMeans model to the scaled data.
    km.fit(Xs)

    # External metrics compare cluster labels against known true labels.
    ari = adjusted_rand_score(y_true, km.labels_)
    ami = adjusted_mutual_info_score(y_true, km.labels_)

    # Internal metric uses only clustered geometry.
    sil = silhouette_score(Xs, km.labels_)

    print(
        f"  inertia={km.inertia_:.4f}  iters={km.n_iter_}  "
        f"ARI={ari:.4f}  AMI={ami:.4f}  silhouette={sil:.4f}"
    )

    return km, Xs


# ════════════════════════════════════════════════════════════════════════════
# D — CHOOSING THE NUMBER OF CLUSTERS K
# ════════════════════════════════════════════════════════════════════════════
#
# One of the biggest practical questions in clustering is:
#
#     "How do I choose K?"
#
# There is no single universal answer.
#
# This file compares several common criteria:
# - Elbow
# - Silhouette
# - Calinski-Harabasz
# - Davies-Bouldin
# - Gap statistic
#
# ════════════════════════════════════════════════════════════════════════════

def _gap_statistic(Xs, K_range, B=20):
    """
    Compute the Gap statistic for a range of candidate cluster counts K

    ===========================================================================
    BIG PICTURE
    ===========================================================================
    The Gap statistic is a method for choosing the number of clusters K.

    Its main idea is very intuitive:

        "Does my data cluster better than random data would cluster?"

    If the answer is yes, that suggests the cluster structure is meaningful.

    If the answer is no, then what looks like clustering may just be an effect
    of random scatter in space.

    ===========================================================================
    CORE IDEA
    ===========================================================================
    For each candidate K:

        1. Cluster the real data
           -> compute W_k

        2. Generate B reference datasets with no true clusters
           -> cluster each one
           -> compute W*_k for each reference dataset

        3. Compare:
           Gap(K) = average(log(W*_k)) - log(W_k)

    If the real data clusters much better than the random reference datasets,
    then W_k will be much smaller than W*_k, and the gap will be large.

    ===========================================================================
    WHAT IS W_k?
    ===========================================================================
    W_k is the within-cluster dispersion for K clusters.

    In this implementation, it is represented by KMeans inertia:

        inertia = within-cluster sum of squared distances

    So:
        smaller W_k  -> tighter clusters
        larger W_k   -> looser clusters

    Large gap means:
        real data clusters much better than random data

    ===========================================================================
    WHAT IS THE STANDARD ERROR TERM s_k?
    ===========================================================================
    Since the reference datasets are random, their clustering quality varies
    from one random sample to another.

    So the Gap statistic also computes an uncertainty estimate:

        s_k = std(log(W*_k)) * sqrt(1 + 1/B)

    This is used in the standard selection rule:

        Choose the smallest K such that

            Gap(K) >= Gap(K+1) - s_{K+1}

    This means:
    - do not increase K unless the next K gives a clearly better gap
    - prefer a smaller simpler K when the improvement is not significant

    Parameters
    ----------
    Xs : ndarray
        Scaled feature matrix.

    K_range : iterable
        Candidate K values.

    B : int, default=20
        Number of reference datasets.

    Returns
    -------
    gaps : ndarray    Gap statistic value for each K in K_range.
    sks  : ndarray    Standard error estimate for each K in K_range.
        Standard error estimates used in the standard gap rule.

    ===========================================================================
    HOW THE REFERENCE DATA IS GENERATED
    ===========================================================================
    The reference datasets are generated uniformly inside the bounding box of
    the real data.

    That means:
    - for each feature dimension
    - find the minimum and maximum value in the real data
    - sample random points uniformly between those limits

    This creates data with roughly the same scale and rectangular extent as the
    real dataset, but without the real cluster structure.
    """

    # -----------------------------------------------------------------------
    # Determine the bounding box of the real data.
    #
    # lo[j] = minimum value of feature j
    # hi[j] = maximum value of feature j
    #
    # These bounds are used to generate uniform random reference datasets.
    # -----------------------------------------------------------------------
    lo, hi = Xs.min(axis=0), Xs.max(axis=0)
    # Number of samples and number of features
    n, d = Xs.shape
    # Random generator for reproducible reference dataset creation
    rng = np.random.default_rng(SEED)

    # Lists to store:
    # - gap value for each K
    # - standard error estimate for each K
    gaps, sks = [], []

    # -----------------------------------------------------------------------
    # Loop over candidate K values one by one.
    # For each K, compare the real data to random reference datasets.
    # -----------------------------------------------------------------------
    for K in K_range:
        # ================================================================
        # PART 1 — CLUSTER THE REAL DATA
        # ================================================================
        #
        # Fit KMeans with the current K on the real scaled dataset.
        #
        # km.inertia_ is the within-cluster sum of squares W_k.
        km = KMeans(n_clusters=K, n_init=5, max_iter=200, random_state=SEED)
        km.fit(Xs)
        log_Wk = np.log(km.inertia_ + 1e-10)

        # ================================================================
        # PART 2 — CLUSTER RANDOM REFERENCE DATASETS
        # ================================================================
        #
        # We now create B datasets that have:
        # - the same number of points n
        # - the same number of features d
        # - the same bounding-box range as the real data
        #
        # but no true clustering structure
        #
        # For each reference dataset:
        #   1. generate uniform random points
        #   2. cluster them using KMeans with the same K
        #   3. record log(inertia)

        # lW_ref will collect those B values
        lW_ref = []
        for b in range(B):
            # Generate one uniform random reference dataset inside the bounding
            # box [lo, hi].
            #
            # Shape = (n, d), matching the real dataset.
            Xr = rng.uniform(lo, hi, (n, d))
            km_r = KMeans(n_clusters=K, n_init=3, max_iter=100, random_state=b)
            km_r.fit(Xr)
            lW_ref.append(np.log(km_r.inertia_ + 1e-10))

        lW_ref = np.array(lW_ref)

        # ================================================================
        # PART 3 — COMPUTE GAP(K)
        # ================================================================
        #
        # Gap(K) = average(log(W*_k)) - log(W_k)
        #
        # Interpretation:
        # - if real data clusters much better than random data:
        #       log(W_k) will be much smaller
        #       so gap will be larger
        gap = lW_ref.mean() - log_Wk

        # ================================================================
        # PART 4 — COMPUTE UNCERTAINTY ESTIMATE s_k
        # ================================================================
        #
        # The reference datasets are random, so their log-dispersion values
        # vary.
        #
        # This variability is summarized by:
        #
        #   s_k = std(log(W*_k)) * sqrt(1 + 1/B)
        #
        # This is the standard error-like quantity used in the original Gap
        # statistic selection rule.
        sk = np.std(lW_ref) * np.sqrt(1 + 1/B)

        # Store results for this K
        gaps.append(gap)
        sks.append(sk)

    return np.array(gaps), np.array(sks)


def choose_K(X, K_max=10, name="dataset"):
    """
    Compare multiple methods for selecting the number of clusters K.

    ===========================================================================
    BIG PICTURE
    ===========================================================================
    One of the most important practical problems in K-means is:

        "How many clusters should I choose?"

    K-means requires K in advance.
    But in most real problems, K is not known beforehand.

    So this function tries several candidate values of K and evaluates them
    using multiple well-known criteria.

    Instead of trusting only one metric, it compares several and shows them
    together in one diagnostic figure.

    ---------------------------------------------------------------------------
    BEGINNER NOTE
    ---------------------------------------------------------------------------
    This function runs K-means for K = 2, 3, ..., K_max
    and computes several quality criteria.

    METHODS USED
    ------------
    1. Elbow
       Look for a bend in the inertia curve.

    2. Silhouette
       Higher is better.

    3. Calinski-Harabasz
       Higher is better.

    4. Davies-Bouldin
       Lower is better.

    5. Gap statistic
       Choose K using reference-random comparison.

    Parameters
    ----------
    X : ndarray
        Input feature matrix.

    K_max : int, default=10
        Largest K to test.

    name : str
        Dataset name for titles and filenames.

    Returns
    -------
    dict
        A dictionary containing recommended K values from each method.
    """
    print(f"\n[D] Choosing K — {name}")

    Xs = StandardScaler().fit_transform(X)

    # Candidate K values to test.
    # We begin at K=2 because K=1 is not useful for these cluster comparison metrics in this setting.
    K_range = list(range(2, K_max + 1))

    # These lists will store the metric values for each tested K.
    inertias, sils, chs, dbs = [], [], [], []

    # -----------------------------------------------------------------------
    # For each candidate K:
    #   1. fit KMeans
    #   2. compute internal clustering metrics
    #   3. store them for later analysis and plotting
    # -----------------------------------------------------------------------
    for K in K_range:
        km = KMeans(n_clusters=K, n_init=10, max_iter=300, random_state=SEED)
        km.fit(Xs)

        # Inertia / WCSS:
        # total within-cluster sum of squares
        inertias.append(km.inertia_)

        # Mean silhouette score:
        # higher means better separated clusters
        sils.append(silhouette_score(Xs, km.labels_))

        # Calinski-Harabasz index:
        # higher is better
        chs.append(calinski_harabasz_score(Xs, km.labels_))

        # Davies-Bouldin index:
        # lower is better
        dbs.append(davies_bouldin_score(Xs, km.labels_))

    # -----------------------------------------------------------------------
    # Compute the Gap statistic separately.
    #
    # This compares real clustering quality against random reference datasets.
    # gaps = gap value for each K
    # sks  = associated standard-error estimates
    # -----------------------------------------------------------------------
    gaps, sks = _gap_statistic(Xs, K_range, B=20)
    K_arr = np.array(K_range)

    # ------------------------------------------------------------
    # Recommended K from each method
    # ------------------------------------------------------------

    # Elbow heuristic: use second discrete difference to detect a bend.
    d2 = np.diff(np.diff(inertias))
    K_elbow = K_range[int(np.argmax(d2)) + 1]

    # ------------------------------------------------------------
    # Silhouette method
    # ------------------------------------------------------------
    #
    # Choose the K with the largest mean silhouette score.
    K_sil = K_range[int(np.argmax(sils))]

    # Calinski-Harabasz: maximize
    K_ch = K_range[int(np.argmax(chs))]

    # Davies-Bouldin: minimize
    K_db = K_range[int(np.argmin(dbs))]

    # Gap rule
    K_gap = K_range[0]
    for i in range(len(gaps) - 1):
        if gaps[i] >= gaps[i+1] - sks[i+1]:
            K_gap = K_range[i]
            break

    # Print recommended K from each method.
    print(f"  Elbow          K* = {K_elbow}")
    print(f"  Silhouette     K* = {K_sil}  (max={max(sils):.3f})")
    print(f"  Calinski-H     K* = {K_ch}")
    print(f"  Davies-Bouldin K* = {K_db}")
    print(f"  Gap statistic  K* = {K_gap}")

    # -----------------------------------------------------------------------
    # PLOT ALL K-SELECTION DIAGNOSTICS
    # -----------------------------------------------------------------------
    #
    # We make a 2x3 figure:
    #
    # [0,0] Elbow / inertia
    # [0,1] Silhouette
    # [0,2] Calinski-Harabasz
    # [1,0] Davies-Bouldin
    # [1,1] Gap statistic
    # [1,2] Consensus / voting panel
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"K-Selection — {name}", fontsize=13, fontweight="bold")

    def vline(ax, k):
        """
        Draw a red dashed vertical line showing the recommended K.
        """
        ax.axvline(k, color="red", ls="--", alpha=0.7, label=f"K*={k}")

    specs = [
        (
            axes[0, 0], K_arr, inertias,
            "Elbow  (look for kink)",
            "WCSS", "steelblue", K_elbow, False
        ),
        (
            axes[0, 1], K_arr, sils,
            "Silhouette  (maximise)",
            "Mean sil", "forestgreen", K_sil, False
        ),
        (
            axes[0, 2], K_arr, chs,
            "Calinski-Harabász  (maximise)",
            "CH index", "darkorange", K_ch, False
        ),
        (
            axes[1, 0], K_arr, dbs,
            "Davies-Bouldin  (minimise)",
            "DB index", "purple", K_db, False
        ),
    ]

    # Plot the first four K-selection curves.
    for ax, x, y, title, ylabel, c, kstar, _ in specs:
        ax.plot(x, y, "o-", color=c, lw=2, ms=7)
        vline(ax, kstar)
        ax.set_xlabel("K")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)

    # Gap statistic panel
    ax = axes[1, 1]
    ax.errorbar(
        K_arr, gaps, yerr=sks,
        fmt="o-",
        color="darkred",
        lw=2,
        ms=7,
        capsize=4
    )
    vline(ax, K_gap)
    ax.set_xlabel("K")
    ax.set_ylabel("Gap statistic")
    ax.set_title("Gap statistic  (Tibshirani et al.)", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    # Consensus / voting panel
    ax = axes[1, 2]
    methods = ["Elbow", "Sil", "CH", "DB", "Gap"]
    votes = [K_elbow, K_sil, K_ch, K_db, K_gap]
    cols = ["steelblue", "forestgreen", "darkorange", "purple", "darkred"]

    bars = ax.bar(methods, votes, color=cols, alpha=0.85, edgecolor="k")
    ax.bar_label(bars, fmt="%d", padding=2, fontsize=12, fontweight="bold")
    ax.set_ylim(0, K_max + 1)
    ax.set_ylabel("Suggested K")
    ax.set_title("Method consensus", fontweight="bold")

    # Majority vote among methods.
    majority = Counter(votes).most_common(1)[0][0]
    ax.axhline(
        majority,
        color="red",
        ls="--",
        lw=2,
        label=f"Majority = {majority}"
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Save the diagnostic figure.
    fname = f"plots/kmeans/choose_K_{name.replace(' ','_')}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  saved  {fname}")

    return dict(zip(methods, votes))


# ════════════════════════════════════════════════════════════════════════════
# E — SILHOUETTE PLOTS
# ════════════════════════════════════════════════════════════════════════════
#
# Mean silhouette score is useful, but the full silhouette plot tells more.
#
# It shows:
# - the distribution of silhouette values within each cluster
# - whether clusters are balanced
# - whether some points are badly assigned
#
# ════════════════════════════════════════════════════════════════════════════

def silhouette_plot(X, K_candidates, name="dataset"):
    """
    Create silhouette plots for several candidate K values.

    ===========================================================================
    BIG PICTURE
    ===========================================================================
    A single mean silhouette score is useful, but it hides a lot of detail.

    Two different clusterings may have similar mean silhouette values while
    having very different internal structure.

    The full silhouette plot gives a much richer view.

    It shows:
    - how silhouette values are distributed inside each cluster
    - whether some clusters are much better than others
    - whether some points are poorly assigned
    - whether cluster sizes are balanced or very uneven

    ---------------------------------------------------------------------------
    BEGINNER NOTE
    ---------------------------------------------------------------------------
    Silhouette value for one point measures:
    - how close it is to its own cluster
    - how far it is from the nearest other cluster

    Interpretation:
        near +1 -> very well assigned
        near  0 -> on boundary
        below 0 -> likely misassigned

    This function creates a full silhouette visualization for each K.

    ===========================================================================
    WHY THIS FUNCTION MATTERS
    ===========================================================================
    Suppose two different choices of K both have reasonable mean silhouette.

    The full silhouette plot can reveal things that the mean alone hides:
    - one cluster may be very weak
    - one cluster may be too small
    - many points may have negative silhouette
    - cluster balance may be poor

    So this function is a more diagnostic visualization than just printing a
    single number.

    Parameters
    ----------
    X : ndarray
        Input data.

    K_candidates : iterable of int
        Candidate K values.

    name : str
        Dataset name for titles / filenames.
    """
    print(f"\n[E] Silhouette plot — {name}")

    Xs = StandardScaler().fit_transform(X)
    # Colormap used to color different clusters.
    cmap = plt.cm.tab10

    fig, axes = plt.subplots(1, len(K_candidates), figsize=(5 * len(K_candidates), 6))

    if len(K_candidates) == 1:
        axes = [axes]

    # -----------------------------------------------------------------------
    # Loop over each candidate K and build one silhouette plot panel.
    # -----------------------------------------------------------------------
    for ax, K in zip(axes, K_candidates):
        # Fit K-means for the current K and get predicted labels
        km = KMeans(n_clusters=K, n_init=10, random_state=SEED)
        lbl = km.fit_predict(Xs)

        # Compute silhouette value for every point.
        #
        # sv[i] = silhouette value of point i
        sv = silhouette_samples(Xs, lbl)
        # Mean silhouette score across all points
        mu = sv.mean()

        # y_lo tracks the lower vertical position for the next cluster band.
        #
        # We start from 10 instead of 0 to leave a little top margin.
        y_lo = 10

        # -------------------------------------------------------------------
        # Draw one silhouette band per cluster.
        # -------------------------------------------------------------------
        for k in range(K):
            # Extract silhouette values of the current cluster k and sort them.
            #
            # Sorting produces the characteristic silhouette shape:
            # thin at one end, wider where many points have similar values, etc.
            vals = np.sort(sv[lbl == k])

            # y_hi is the upper boundary of this cluster's band
            y_hi = y_lo + len(vals)

            ax.fill_betweenx(
                np.arange(y_lo, y_hi),
                0,
                vals,
                facecolor=cmap(k),
                alpha=0.8
            )

            ax.text(
                -0.05,
                (y_lo + y_hi) / 2,
                str(k),
                ha="right",
                va="center",
                fontsize=9
            )

            y_lo = y_hi + 10

        # -------------------------------------------------------------------
        # Mean silhouette reference line.
        #
        # This helps compare the cluster bands against the overall average.
        # -------------------------------------------------------------------
        ax.axvline(mu, color="red", ls="--", lw=1.5, label=f"mean={mu:.3f}")

        ax.set_title(f"K={K}  mean sil={mu:.3f}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Silhouette s_i")
        ax.set_xlim([-0.2, 1.0])
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    fig.suptitle(f"Silhouette Plots — {name}",
                 fontsize=12, fontweight="bold", y=1.01)

    plt.tight_layout()
    fname = f"plots/kmeans/silhouette_{name.replace(' ','_')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  saved → {fname}")



# ════════════════════════════════════════════════════════════════════════════
# FAILURE MODES OF K-MEANS
# ════════════════════════════════════════════════════════════════════════════
#
# K-means is powerful, but it does NOT solve every clustering problem well.
#
# It works best when clusters are:
# - compact
# - roughly spherical / round
# - similar in size
# - separated mainly by centroid distance
# It tends to struggle when clusters are:  
# - non-convex
# - nested
# - elongated
# - strongly unequal in size or density
#
# This section demonstrates those failures clearly.
#
# ════════════════════════════════════════════════════════════════════════════

def failure_modes():
    """
    Visualize classic cases where K-means performs poorly.

    ===========================================================================
    CASES INCLUDED
    ===========================================================================
    This function tests K-means on four important failure cases:

    1. Concentric circles
       One cluster forms an inner ring and the other forms an outer ring.

       Why K-means struggles:
       - centroids summarize clusters by their mean location
       - but both rings are centered around nearly the same point
       - K-means cannot represent "one cluster surrounding another"

    2. Two moons
       The clusters are curved, interleaved, and non-convex.

       Why K-means struggles:
       - K-means tends to split space using straight centroid-based boundaries
       - moon-shaped clusters are not naturally separated that way

    3. Anisotropic blobs
       Clusters are elongated and rotated.

       Why K-means struggles:
       - K-means prefers compact, roughly round clusters
       - elongated covariance structure can cause incorrect partitions

    4. Unequal-size blobs
       One cluster is large and diffuse, another is small and tight.

       Why K-means struggles:
       - K-means tends to favor balanced centroid-based partitions
       - large diffuse clusters may get split
       - small clusters may be absorbed or merged

    Output
    ------
    A 2x4 figure:
        top row    = true labels
        bottom row = K-means output
        So for each failure case, you can visually compare:
    - what the actual grouping is
    - what K-means thinks the grouping is 

    """

    print("\n[G] Failure modes")

    # -----------------------------------------------------------------------
    # Each tuple describes one failure-case dataset:
    #
    #   nm    -> dataset name to load from disk
    #   K     -> number of clusters given to K-means
    #   title -> human-readable display title
    #
    # These datasets were created earlier specifically to expose weaknesses
    # of centroid-based clustering.
    # -----------------------------------------------------------------------

    cases = [
        ("circles",       2, "Concentric Circles"),
        ("moons",         2, "Two Moons"),
        ("blobs_aniso",   3, "Anisotropic Blobs"),
        ("blobs_unequal", 3, "Unequal-Size Blobs")
    ]

    # -----------------------------------------------------------------------
    # Create a 2-row x 4-column plotting grid.
    #
    # Row 0 -> true labels
    # Row 1 -> K-means predictions
    #
    # Each column corresponds to one failure case.
    # -----------------------------------------------------------------------

    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    fig.suptitle("K-Means Failure Modes  (top=true, bottom=K-means)",
                 fontsize=13, fontweight="bold")

    # -----------------------------------------------------------------------
    # Loop over the failure-case datasets one by one.
    #
    # col is the subplot column index.
    # nm, K, title come from the cases list.
    # -----------------------------------------------------------------------
    for col, (nm, K, title) in enumerate(cases):
        # Load the dataset:
        # X = features
        # y = true labels
        X, y = load(nm)

        Xs = StandardScaler().fit_transform(X)

        # Fit K-means and get predicted cluster labels
        km = KMeans(n_clusters=K, n_init=10, random_state=SEED)
        pred = km.fit_predict(Xs)

        # Compare predicted labels to true labels using ARI
        ari = adjusted_rand_score(y, pred)
        # -------------------------------------------------------------------
        # For each dataset, produce two plots:
        #
        # row = 0 -> true labels
        # row = 1 -> predicted labels
        #
        # lbl = labels to color by
        # ttl = title of the subplot
        # -------------------------------------------------------------------
        for row, (lbl, ttl) in enumerate([
                (y,    f"{title}\n(True)"),
                (pred, f"K-Means K={K}\nARI={ari:.3f}")]):

            ax = axes[row, col]

            # Scatter plot in the original 2D feature space
            ax.scatter(X[:, 0], X[:, 1], c=lbl,
                       cmap="tab10", s=14, alpha=0.75)

            # On the predicted row only, overlay learned centroids.
            #
            # Important note:
            # km.cluster_centers_ are in the scaled feature space because the
            # model was fit on Xs.
            #
            # In this script they are still plotted directly for a visual cue.
            # The intent is mainly pedagogical: to show where K-means thinks
            # the centers are.
            if row == 1:
                ax.scatter(km.cluster_centers_[:, 0],
                           km.cluster_centers_[:, 1],
                           c="red", s=180, marker="*", zorder=5)

            tc = ("red" if row == 1 and ari < 0.5
                  else "darkorange" if row == 1 and ari < 0.9
                  else "black")

            ax.set_title(ttl, fontsize=9, fontweight="bold", color=tc)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("plots/kmeans/failure_modes.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    print("  saved plots/kmeans/failure_modes.png")


# ════════════════════════════════════════════════════════════════════════════
# H — MINI-BATCH K-MEANS
# ════════════════════════════════════════════════════════════════════════════
#
# Standard K-means can be expensive on very large datasets because each
# iteration uses the full dataset.
#
# MiniBatchKMeans uses random mini-batches instead, making it much faster.
#
# ════════════════════════════════════════════════════════════════════════════

def minibatch_demo(n=80_000):
    """
    Compare standard KMeans with MiniBatchKMeans on a large dataset.

    ===========================================================================
    BIG PICTURE
    ===========================================================================
    Standard K-means can become expensive when the dataset is very large.

    Why?
    Because in each iteration it repeatedly uses the full dataset to:
    - compute point-to-centroid assignments
    - update the centroids

    If the number of samples n is huge, this can take significant time.

    MiniBatchKMeans is a faster approximate alternative.

    Instead of using the full dataset in every update step, it uses small random
    subsets called mini-batches.

    This makes training much faster, especially for large data.

    ---------------------------------------------------------------------------
    BEGINNER NOTE
    ---------------------------------------------------------------------------
    Mini-batch K-means is an approximate version of K-means.

    Instead of recomputing centroids from the whole dataset each time,
    it updates centroids using small randomly chosen batches.

    Advantages:
    - much faster
    - good for large datasets

    Trade-off:
    - final inertia may be slightly worse
    - but often not by much

    Parameters
    ----------
    n : int, default=80_000
        Number of synthetic data points to generate.
    """
    import time

    print(f"\n[H] Mini-batch demo  (n={n:,})")

    # -----------------------------------------------------------------------
    # Generate a large synthetic dataset with 6 cluster centers.
    #
    # We do not need true labels here because the main goal is to compare:
    # - runtime
    # - inertia
    # - label agreement between the two clustering methods
    # -----------------------------------------------------------------------
    from sklearn.datasets import make_blobs
    Xbig, _ = make_blobs(n_samples=n, centers=6,
                         cluster_std=1.5, random_state=SEED)

    # Full KMeans timing
    t0 = time.time()
    km = KMeans(n_clusters=6, n_init=3, max_iter=100,
                random_state=SEED).fit(Xbig)
    t_full = time.time() - t0

    # MiniBatchKMeans timing
    # batch_size=1024 means:
    # each update uses a randomly chosen subset of 1024 points
    #
    # reassignment_ratio=0.01 controls how aggressively under-used centers may
    # be re-seeded.
    t0 = time.time()
    mb = MiniBatchKMeans(n_clusters=6, init="k-means++", n_init=3,
                         batch_size=1024, max_iter=100,
                         reassignment_ratio=0.01,
                         random_state=SEED).fit(Xbig)
    t_mb = time.time() - t0

    # -----------------------------------------------------------------------
    # PRINT COMPARISON RESULTS
    # -----------------------------------------------------------------------
    #
    # km.inertia_ and mb.inertia_ are the final objective values.
    #
    # Inertia loss measures how much worse mini-batch is, relative to full
    # KMeans, in percentage terms.
    #
    # ARI agreement compares the two final labelings directly.
    # -----------------------------------------------------------------------
    print(f"  Full KMeans  : {t_full:.2f}s   inertia={km.inertia_:.1f}")
    print(f"  MiniBatch    : {t_mb:.2f}s   inertia={mb.inertia_:.1f}")
    print(f"  Speed-up     : {t_full/t_mb:.1f}×")
    print(f"  Inertia loss : {100*(mb.inertia_-km.inertia_)/km.inertia_:.2f}%")
    print(f"  ARI agreement: {adjusted_rand_score(km.labels_, mb.labels_):.4f}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION BLOCK
# ════════════════════════════════════════════════════════════════════════════
#
# This is the script entry point.
#
# When the file is run directly:
# - it performs all demonstrations
# - saves all plots
# - prints summaries
#
# This makes the file a complete tutorial pipeline.
#
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  K-MEANS CLUSTERING — COMPLETE TUTORIAL")
    print("=" * 65)

    '''
    ===========================================================================
    WHAT THIS MAIN BLOCK DOES
    ===========================================================================
    It runs the tutorial in a structured order:

    A. Load a simple easy dataset
    B. Compare scratch K-means with sklearn KMeans
    C. Compare initialization strategies
    D. Demonstrate sklearn usage
    E. Explore how to choose K
    F. Build silhouette plots
    G. Run a research-style soft-matter example
    H. Show failure modes
    I. Compare full KMeans and MiniBatchKMeans
    '''

    # ------------------------------------------------------------------------
    # LOAD A SIMPLE BENCHMARK DATASET
    # ------------------------------------------------------------------------
    #
    # We begin with "blobs_easy" because it is one of the cleanest and most
    # favorable datasets for K-means:
    # - clusters are well separated
    # - clusters are compact
    # - clusters are roughly spherical
    #
    # So it is a good sanity-check dataset.
    X_blob, y_blob = load("blobs_easy")

    # Scale before K-means because distance-based methods are sensitive to
    # feature magnitude.
    Xs_blob = StandardScaler().fit_transform(X_blob)

    # ------------------------------------------------------------------------
    # A — scratch implementation vs sklearn
    # ------------------------------------------------------------------------
    print("\n[A] Scratch vs sklearn")

    # Fit the scratch implementation.
    kms = KMeansScratch(K=3, init="kmeans++", n_init=10, random_state=SEED)
    kms.fit(Xs_blob)

    print(f"  Scratch : inertia={kms.inertia_:.4f}  "
          f"ARI={adjusted_rand_score(y_blob, kms.labels_):.4f}")

    # Fit sklearn KMeans on the same scaled data.
    km_sk = KMeans(n_clusters=3, n_init=10, random_state=SEED).fit(Xs_blob)

    print(f"  Sklearn : inertia={km_sk.inertia_:.4f}  "
          f"ARI={adjusted_rand_score(y_blob, km_sk.labels_):.4f}")

    # ------------------------------------------------------------------------
    # B — initialization comparison
    # ------------------------------------------------------------------------
    # Compare random initialization vs k-means++ over many runs to show that
    # initialization quality matters.
    compare_init(X_blob)

    # ------------------------------------------------------------------------
    # C — sklearn usage demo
    # ------------------------------------------------------------------------
    sklearn_demo(X_blob, y_blob, K=3, name="easy blobs")

    # ------------------------------------------------------------------------
    # D — choose K
    # ------------------------------------------------------------------------
    choose_K(X_blob, K_max=8, name="easy blobs")

    # Optional extra experiment on iris.
    choose_K(*load("iris"), K_max=8) if False else None

    # ------------------------------------------------------------------------
    # E — silhouette plots
    # ------------------------------------------------------------------------
    # Visualize per-point silhouette values for several candidate K values.
    silhouette_plot(X_blob, [2, 3, 4, 5], "easy blobs")

    # ------------------------------------------------------------------------
    # failure modes
    # ------------------------------------------------------------------------
    failure_modes()

    # ------------------------------------------------------------------------
    # mini-batch comparison
    # ------------------------------------------------------------------------
    minibatch_demo()

    print("\n" + "=" * 65)
    print("  K-MEANS MODULE COMPLETE — see plots/kmeans/")
    print("=" * 65)
