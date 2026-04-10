"""
================================================================================
hierarchical_clustering.py
================================================================================

This is a heavily documented version of the user's original hierarchical
clustering tutorial script.

Primary goal of this file
-------------------------
This file is meant for *reading and learning*, not just running.

It keeps the original structure and logic, but adds detailed explanations about:

1. what each import does,
2. why each preprocessing step exists,
3. what each function returns,
4. how each block changes the final clustering outcome,
5. which parts affect science / statistics,
6. which parts affect only visualization,
7. where implementation choices are pedagogical versus production-quality,
8. a few subtle caveats in the current implementation.

High-level summary of what the original script does
---------------------------------------------------
The script is a tutorial-style exploration of agglomerative hierarchical
clustering. It contains six main pieces:

A. A "from scratch" NumPy implementation of agglomerative clustering
   so that the merge process is transparent.

B. A comparison of four linkage methods
   (single, complete, average, Ward)
   on selected datasets.

C. Dendrogram analysis for one dataset, including
   cophenetic correlation and merge-distance jumps.

D. A simple strategy for choosing the number of clusters K.

E. A domain-specific example:
   polymer conformational hierarchy.

F. A performance heatmap comparing linkage methods across datasets.

A few important interpretive notes
----------------------------------
1. StandardScaler has a major effect on results.
   Hierarchical clustering is distance-based, so rescaling features changes
   the geometry of the data and therefore changes the merges.

2. PCA here is only for plotting.
   It affects the scatter plots that you look at, but not the clustering labels
   produced in the script, because clustering is done on the standardized data
   `Xs`, not on the PCA projection `Xp`.

3. The scratch implementation is educational, not optimized.
   It uses explicit loops and repeated distance updates. That is great for
   understanding but not ideal for large datasets.

4. The scratch implementation of Ward is conceptually correct for selecting
   merges because it uses the Ward merge cost (increase in within-cluster sum
   of squares), but the exact numeric heights stored in the linkage matrix may
   not match SciPy's exact Ward height convention one-for-one.

5. Some imports in the original file are unused:
      - inconsistent
      - AgglomerativeClustering
   This is not harmful, but it tells you the file may have evolved from a
   broader tutorial where those pieces were planned or used earlier.

6. The section banner at the top mentions "Connectivity constraints", but the
   actual file does not implement that section. The final section present is the
   linkage performance heatmap.
================================================================================
"""

# -----------------------------------------------------------------------------
# Standard-library imports
# -----------------------------------------------------------------------------

import os
import warnings

# The original file suppresses warnings globally.
# Effect on outcome:
#   - This does NOT change the mathematics of clustering.
#   - It only changes what gets printed to the console.
# Practical consequence:
#   - Cleaner output.
# Risk:
#   - You may miss useful warnings from libraries.
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Core numerical stack
# -----------------------------------------------------------------------------

import numpy as np

# -----------------------------------------------------------------------------
# Plotting stack
# -----------------------------------------------------------------------------

import matplotlib

# Force a non-interactive backend.
# Effect on outcome:
#   - Does NOT change clustering results.
#   - Only changes how plots are rendered.
# Why do this?
#   - It makes the script safe to run on headless machines / servers / clusters.
#   - `Agg` writes image files without requiring a display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Seaborn is used only for the final heatmap.
# Effect on outcome:
#   - None on clustering.
#   - Purely visual / presentation layer.
import seaborn as sns

# Pandas is used to convert the result dictionary into a table for the heatmap.
# Again, presentation / convenience only.
import pandas as pd

# -----------------------------------------------------------------------------
# SciPy hierarchical clustering tools
# -----------------------------------------------------------------------------

from scipy.cluster.hierarchy import (
    dendrogram,   # draw dendrograms
    linkage,      # compute SciPy linkage matrix from data
    fcluster,     # cut linkage matrix into flat cluster labels
    cophenet,     # compute cophenetic correlation / distances
    inconsistent, # imported in original script but not used anywhere below
)

from scipy.spatial.distance import (
    pdist,       # condensed pairwise distance vector
    squareform,  # convert between condensed and square distance representations
)

# -----------------------------------------------------------------------------
# scikit-learn utilities
# -----------------------------------------------------------------------------

from sklearn.cluster import AgglomerativeClustering
# NOTE:
#   This import was unused in the original script.
#   In this extended documented version, we also use it below in
#   `AgglomerativeSklearnExplorer` to show how scikit-learn stores the merge
#   tree internally through attributes like `children_` and `distances_`.

from sklearn.metrics import (
    adjusted_rand_score,  # compares predicted labels to ground truth
    silhouette_score,     # unsupervised internal cluster quality metric
)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# -----------------------------------------------------------------------------
# Filesystem setup and RNG
# -----------------------------------------------------------------------------

# Create the output directory that will hold generated plot images.
# Effect on outcome:
#   - No effect on clustering.
#   - Prevents file-save errors later.
os.makedirs("plots/hierarchical", exist_ok=True)

# Reproducible random number generator used for:
#   - subsampling in polymer_hierarchy()
#   - subsampling pairwise distances in dendrogram_analysis()
RNG = np.random.default_rng(42)

# The original file also defines a bare integer seed.
# In this exact script it is not used later.
SEED = 42


def load(name):
    """
    Load a dataset from `data/{name}.npz`.

    Expected file structure
    -----------------------
    Each `.npz` file must contain:
        X : feature matrix of shape (n_samples, n_features)
        y : true labels of shape (n_samples,)

    Why this helper exists
    ----------------------
    It centralizes the expected dataset format, so every later function can say

        X, y = load("blobs_easy")

    instead of repeatedly writing the `np.load(...)` logic.

    Effect on outcome
    -----------------
    This function itself does not alter the data. It only reads it.
    However, if the `.npz` files do not have keys named exactly `X` and `y`,
    the script will fail here.
    """
    d = np.load(f"data/{name}.npz")
    return d["X"], d["y"]


# =============================================================================
# A — AGGLOMERATIVE FROM SCRATCH (pure NumPy)
# =============================================================================

class AgglomerativeScratch:
    """
    Educational bottom-up agglomerative clustering implemented directly in NumPy.

    Why this class is in the script
    -------------------------------
    SciPy can already compute hierarchical clustering very efficiently with
    `linkage()`. So this class is not here because SciPy lacks the feature.

    It is here because pedagogy matters:
    this class exposes the actual mechanics of hierarchical clustering.

    Conceptual algorithm
    --------------------
    Start with n singleton clusters:
        {0}, {1}, {2}, ..., {n-1}

    Then repeatedly:
      1. find the two closest active clusters,
      2. merge them,
      3. record the merge in linkage-matrix format,
      4. update distances from the new merged cluster to all others.

    Continue until only one cluster remains.

    Linkage methods implemented here
    --------------------------------
    single:
        distance between two clusters = closest cross-cluster pair

    complete:
        distance between two clusters = farthest cross-cluster pair

    average:
        distance = average over all cross-cluster pairs

    ward:
        merge the pair that produces the smallest increase in within-cluster
        sum of squares (WCSS)

    Important caveat about this implementation
    ------------------------------------------
    For single / complete / average:
        the script updates cluster distances using the Lance–Williams recurrence.

    For ward:
        it does not use the same recursive distance matrix update.
        Instead, it recomputes the Ward merge cost from centroids and cluster
        sizes when comparing active clusters.

    That is perfectly reasonable pedagogically.
    It preserves the merge-order logic.
    But the numeric "distance" values stored in the linkage matrix for Ward
    may not exactly match SciPy's internal convention for dendrogram heights.
    """

    def __init__(self, method="ward"):
        """
        Store the chosen linkage rule.

        Parameters
        ----------
        method : str
            One of:
                "single", "complete", "average", "ward"

        Effect on outcome
        -----------------
        This single choice dramatically changes the clustering behavior:

        - single:
            can connect elongated / chained structures,
            but is vulnerable to chaining noise.

        - complete:
            favors compact, tight clusters,
            tends to resist chaining.

        - average:
            compromises between single and complete.

        - ward:
            prefers merges that minimally increase variance;
            often works very well on compact roughly spherical clusters.
        """
        self.method = method

    def _lw_update(self, d_ai, d_bi, d_ab, na, nb, ni):
        """
        Compute distance from new merged cluster c = a ∪ b to another cluster i
        using the Lance–Williams recurrence.

        Parameters
        ----------
        d_ai : float
            Distance between cluster a and cluster i.
        d_bi : float
            Distance between cluster b and cluster i.
        d_ab : float
            Distance between cluster a and cluster b.
            Included for general Lance–Williams formulations.
        na, nb, ni : int
            Sizes of clusters a, b, and i.

        Returns
        -------
        float
            Updated distance d(c, i).

        Why this matters
        ----------------
        In agglomerative clustering, once two clusters merge, you must define
        how the new cluster relates to the remaining clusters.

        This choice is the heart of the method.
        The update formula determines future merges, so it strongly shapes the
        resulting dendrogram and final flat clustering.

        Method-specific meaning
        -----------------------
        single linkage:
            new cluster is close to i if either parent was close to i

        complete linkage:
            new cluster is far from i unless both parents were close to i

        average linkage:
            new distance is a size-weighted average of the two old distances
        """
        nc = na + nb  # total size of new cluster c

        if self.method == "single":
            # Equivalent to min(d_ai, d_bi), written in a symmetric formula form.
            # Why it matters:
            #   This makes it easy for clusters to connect through a "bridge"
            #   or a chain of nearby points.
            return 0.5 * (d_ai + d_bi) - 0.5 * abs(d_ai - d_bi)

        elif self.method == "complete":
            # Equivalent to max(d_ai, d_bi).
            # Why it matters:
            #   This punishes wide / spread-out merges.
            #   It tends to produce tighter, more compact clusters.
            return 0.5 * (d_ai + d_bi) + 0.5 * abs(d_ai - d_bi)

        elif self.method == "average":
            # Weighted by cluster sizes.
            # Why it matters:
            #   Larger parent clusters contribute proportionally more to the
            #   new distance. This stabilizes behavior compared with single.
            return (na * d_ai + nb * d_bi) / nc

        else:
            # Ward is intentionally excluded here because this helper implements
            # only the formulas used for single / complete / average.
            raise ValueError(f"LW not implemented for {self.method}")

    def fit(self, X):
        """
        Build the full dendrogram and store it as `self.Z_`.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        self

        Output stored
        -------------
        self.Z_ : ndarray of shape (n_samples - 1, 4)

        This follows SciPy linkage-matrix convention:
            Z[t, 0] = label of first merged cluster at step t
            Z[t, 1] = label of second merged cluster at step t
            Z[t, 2] = merge distance / height
            Z[t, 3] = size of new merged cluster

        Why `fit()` is the core of the class
        ------------------------------------
        This is where the algorithm actually runs:
            - distance matrix creation
            - repeated closest-pair search
            - merge bookkeeping
            - cluster-size updates
            - centroid updates for Ward
            - linkage matrix assembly
        """
        # Convert input to a floating-point NumPy array.
        # Why:
        #   distance calculations and weighted means should use numeric arrays.
        X = np.asarray(X, dtype=float)

        # Number of points.
        n = len(X)

        # ---------------------------------------------------------------------
        # Step 1: compute all pairwise Euclidean distances between points
        # ---------------------------------------------------------------------
        # `pdist(X)` returns the condensed pairwise distance vector.
        # `squareform(...)` converts that into an n x n matrix.
        #
        # Effect on outcome:
        #   Euclidean distance is the geometric basis for all later merges.
        #   If you changed the metric, you would change the clustering.
        D = squareform(pdist(X))

        # Prevent self-merges by setting diagonal entries to infinity.
        # Without this, each point would appear distance 0 from itself and could
        # be incorrectly selected as the closest pair.
        np.fill_diagonal(D, np.inf)

        # ---------------------------------------------------------------------
        # Step 2: initialize cluster metadata
        # ---------------------------------------------------------------------

        # Initially each point is its own cluster, so every cluster size is 1.
        sizes = [1] * n

        # Centroids are only really needed for Ward, but the original code stores
        # them unconditionally for simplicity.
        centroids = X.copy().tolist()

        # `active` stores the currently existing cluster "slots".
        # At the start these are 0..n-1, one for each point.
        active = list(range(n))

        # `labels` maps internal storage positions to the external cluster labels
        # expected by SciPy linkage format.
        #
        # Initially labels are just the original point indices 0..n-1.
        labels = list(range(n))

        # New merged clusters in SciPy linkage format are labeled n, n+1, ...
        next_label = n

        def cluster_dist(a, b):
            """
            Distance between two currently active clusters `a` and `b`.

            Why this inner helper exists
            ----------------------------
            It centralizes the rule for comparing cluster pairs, so the outer
            loop can stay generic regardless of linkage method.

            Important distinction
            ---------------------
            For single / complete / average:
                the distance is read from the current cluster distance matrix D.

            For ward:
                the code computes the merge cost directly from centroids
                and cluster sizes.
            """
            if self.method == "ward":
                mu_a = np.array(centroids[a])
                mu_b = np.array(centroids[b])
                na_ = sizes[a]
                nb_ = sizes[b]

                # Ward merge cost:
                # proportional to increase in total within-cluster variance.
                #
                # Why it matters:
                #   This linkage strongly prefers merging clusters whose
                #   centroids are close *and* whose sizes make the merge cheap.
                #
                # Consequence:
                #   It tends to create compact, variance-minimizing clusters.
                #
                # Subtle note:
                #   The comment in the original code mentions a sqrt-based form,
                #   but the implementation actually stores the SSE-like merge
                #   cost without taking sqrt. For selecting the minimum merge
                #   this is fine; the ordering is preserved.
                return (2 * na_ * nb_ / (na_ + nb_)) * np.sum((mu_a - mu_b) ** 2)

            else:
                # For non-Ward methods, D already holds the current cluster
                # distances after successive Lance–Williams updates.
                return D[a, b]

        # This will accumulate the linkage rows step by step.
        Z = []

        # ---------------------------------------------------------------------
        # Main agglomeration loop
        # ---------------------------------------------------------------------
        # We start with n clusters and need exactly n-1 merges to end with 1.
        for step in range(n - 1):

            # Search for the closest pair among currently active clusters.
            best_d = np.inf
            best_a = best_b = -1

            # Double loop over active clusters.
            # This is simple but O(m^2) at each step, where m is current number
            # of active clusters. Fine for tutorial-scale data, not ideal for
            # very large datasets.
            for ii, a in enumerate(active):
                for jj, b in enumerate(active):
                    if ii >= jj:
                        # Skip duplicates and self-pairs.
                        continue

                    d = cluster_dist(a, b)

                    if d < best_d:
                        best_d = d
                        best_a = a
                        best_b = b

            # Sizes of the two clusters being merged.
            na_ = sizes[best_a]
            nb_ = sizes[best_b]
            nc_ = na_ + nb_

            # Record this merge in linkage format.
            #
            # Why this matters:
            #   The linkage matrix is the complete compressed representation of
            #   the dendrogram. Everything later (dendrogram plotting,
            #   cutting into K clusters, etc.) depends on these rows.
            Z.append([labels[best_a], labels[best_b], best_d, nc_])

            # Compute centroid of the merged cluster.
            # This is essential for Ward because future Ward costs depend on
            # centroid positions and cluster sizes.
            new_centroid = (
                (
                    na_ * np.array(centroids[best_a])
                    + nb_ * np.array(centroids[best_b])
                ) / nc_
            ).tolist()

            # Create a new internal cluster slot.
            #
            # Why `max(active) + 1`?
            #   This ensures a fresh index not currently used by any active
            #   cluster slot in the distance matrix / metadata arrays.
            new_idx = max(active) + 1

            # Append new cluster metadata.
            sizes.append(nc_)
            centroids.append(new_centroid)
            labels.append(next_label)
            next_label += 1

            # -------------------------------------------------------------
            # Update distance structure for non-Ward methods
            # -------------------------------------------------------------
            if self.method != "ward":
                d_new = {}

                # Compute the new cluster's distance to every other active cluster.
                for c in active:
                    if c in (best_a, best_b):
                        continue

                    d_new[c] = self._lw_update(
                        D[best_a, c],  # old distance from parent a to c
                        D[best_b, c],  # old distance from parent b to c
                        best_d,        # distance between a and b
                        na_, nb_, sizes[c]
                    )

                # Expand the square distance matrix by one row and one column
                # so the newly created cluster can be represented.
                D_new = np.full((len(D) + 1, len(D) + 1), np.inf)

                # Copy old distances into the top-left block.
                D_new[:len(D), :len(D)] = D

                # Fill in distances between the new cluster and existing clusters.
                for c, v in d_new.items():
                    D_new[new_idx, c] = v
                    D_new[c, new_idx] = v

                D = D_new
                np.fill_diagonal(D, np.inf)

            # -------------------------------------------------------------
            # Update active set
            # -------------------------------------------------------------
            # Remove the two merged clusters and add the new merged cluster.
            active = [c for c in active if c not in (best_a, best_b)]
            active.append(new_idx)

        # Convert linkage rows to an array and store on the instance.
        self.Z_ = np.array(Z)
        return self

    def get_labels(self, K):
        """
        Cut the stored dendrogram into exactly K flat clusters.

        Parameters
        ----------
        K : int
            Desired number of clusters.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Zero-based cluster labels.

        Why this works
        --------------
        `fcluster(..., criterion="maxclust")` traverses the linkage matrix and
        finds a flat partition with at most / exactly K clusters depending on
        the tree structure and criterion.

        Why subtract 1
        --------------
        SciPy returns labels starting at 1.
        The original code prefers labels starting at 0, which is conventional in
        Python / NumPy and easier for color indexing.
        """
        return fcluster(self.Z_, t=K, criterion="maxclust") - 1


def sklearn_model_to_linkage(model):
    """
    Reconstruct a SciPy-style linkage matrix from a fitted scikit-learn
    `AgglomerativeClustering` model.

    Why this helper is needed
    -------------------------
    scikit-learn and SciPy expose hierarchical clustering in two different ways:

    1. SciPy stores the tree directly as a linkage matrix `Z`.
       That is exactly the format expected by `scipy.cluster.hierarchy.dendrogram`.

    2. scikit-learn stores the tree in model attributes such as:
           model.children_
           model.distances_
           model.n_leaves_

    These contain the same essential merge information, but not in SciPy's
    exact 4-column linkage format. This helper converts one representation into
    the other so we can use SciPy's dendrogram plotting tools.

    Expected sklearn attributes
    ---------------------------
    children_  : shape (n_samples - 1, 2)
        For each merge step, the two child node IDs that were merged.

        Convention:
        - 0, 1, ..., n_samples-1 refer to original data points.
        - n_samples, n_samples+1, ... refer to newly formed internal nodes.

    distances_ : shape (n_samples - 1,)
        Merge heights / distances for each merge.

    n_leaves_ : int
        Number of original observations.

    Returns
    -------
    Z : ndarray of shape (n_samples - 1, 4)
        A standard linkage matrix with columns:
            [child_1, child_2, merge_distance, cluster_size]

    Why the cluster_size column matters
    -----------------------------------
    SciPy's dendrogram uses the 4th column to know how many original leaves lie
    under each internal merge node. That affects plotting, leaf counts, and the
    correctness of the tree representation.
    """
    children = model.children_
    distances = model.distances_
    n_samples = model.n_leaves_

    # `counts[i]` will store how many original data points live under the
    # internal node with ID `n_samples + i`.
    counts = np.zeros(children.shape[0], dtype=float)

    for i, merge in enumerate(children):
        count = 0

        for child_idx in merge:
            if child_idx < n_samples:
                # Child is an original observation (a leaf), so it contributes 1.
                count += 1
            else:
                # Child is itself an internal merge node created earlier.
                # Its subtree size was already computed and stored in `counts`.
                count += counts[child_idx - n_samples]

        counts[i] = count

    Z = np.column_stack([children, distances, counts]).astype(float)
    return Z


class AgglomerativeSklearnExplorer:
    """
    Pedagogical wrapper around `sklearn.cluster.AgglomerativeClustering`.

    Why this class exists
    ---------------------
    The original script imports `AgglomerativeClustering` but never uses it.
    This wrapper adds a parallel path to the tutorial:

    - `AgglomerativeScratch` shows the algorithm conceptually from first
      principles.
    - `AgglomerativeSklearnExplorer` shows how a production ML library exposes
      the same hierarchy through estimator attributes.

    What this class lets you inspect
    --------------------------------
    After fitting a full tree, you can examine:

    - `children_`  : which two nodes were merged at each step
    - `distances_` : the merge height at each step
    - `Z_`         : a reconstructed SciPy linkage matrix

    This is especially useful if you want to understand how scikit-learn's
    estimator API relates to the more classical SciPy dendrogram workflow.

    Important implementation detail
    -------------------------------
    To recover the full hierarchy, we fit the estimator with:

        n_clusters=None
        distance_threshold=0.0
        compute_distances=True

    That forces the model to keep building merges until only one cluster
    remains, and it stores the merge distances needed for a dendrogram.

    Then, when you ask for labels at a particular K, we re-fit the estimator
    with `n_clusters=K` to obtain the flat partition exactly as scikit-learn
    would normally return it.
    """

    def __init__(self, method="ward", metric="euclidean"):
        """
        Parameters
        ----------
        method : str
            Linkage criterion: "single", "complete", "average", or "ward".

        metric : str
            Distance metric used by scikit-learn for non-Ward linkage.
            Ward is only valid with Euclidean geometry.

        Effect on outcome
        -----------------
        Exactly as in the scratch / SciPy implementations, `method` strongly
        changes the clustering tree:

        - single    -> prone to chaining, good for filamentary structures
        - complete  -> compact clusters, more conservative merges
        - average   -> compromise behavior
        - ward      -> variance-minimizing, often strong on blob-like data
        """
        self.method = method
        self.metric = metric

    def _make_model(self, *, full_tree=False, n_clusters=None):
        """
        Construct the underlying scikit-learn estimator.

        Why there are two modes
        -----------------------
        Full-tree mode:
            used when we want to inspect all merges and build a dendrogram.

        Flat-clustering mode:
            used when we want scikit-learn's labels for a specific K.
        """
        kwargs = {"linkage": self.method}

        if full_tree:
            kwargs.update(
                n_clusters=None,
                distance_threshold=0.0,
                compute_distances=True,
            )
        else:
            kwargs.update(n_clusters=n_clusters)

        # In modern scikit-learn, the keyword is `metric`.
        # In older versions, the keyword was `affinity`.
        # The try/except below keeps this code robust across both APIs.
        if self.method == "ward":
            kwargs["metric"] = "euclidean"
        else:
            kwargs["metric"] = self.metric

        try:
            return AgglomerativeClustering(**kwargs)
        except TypeError:
            if "metric" in kwargs:
                kwargs["affinity"] = kwargs.pop("metric")
            return AgglomerativeClustering(**kwargs)

    def fit(self, X):
        """
        Fit the *full* agglomerative tree on the provided data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input feature matrix.

        Important note
        --------------
        This class does not standardize `X` internally.
        That is deliberate.

        Why?
        ----
        Scaling is a separate modeling choice, not part of the clustering
        algorithm itself. The surrounding analysis functions decide whether to
        standardize the data before passing it here.
        """
        X = np.asarray(X, dtype=float)
        self.X_ = X

        self.full_model_ = self._make_model(full_tree=True)
        self.full_model_.fit(X)

        # Store the most informative sklearn attributes explicitly so they are
        # easy to inspect from outside the class.
        self.children_ = self.full_model_.children_
        self.distances_ = self.full_model_.distances_
        self.n_leaves_ = self.full_model_.n_leaves_

        # Convert sklearn's representation into SciPy linkage format.
        self.Z_ = sklearn_model_to_linkage(self.full_model_)
        return self

    def get_labels(self, K):
        """
        Re-fit scikit-learn's estimator to obtain exactly K flat clusters.

        Why re-fit instead of cutting `self.Z_` with `fcluster`?
        -------------------------------------------------------
        Because the goal of this class is to show *scikit-learn's own estimator
        behavior*. Re-fitting with `n_clusters=K` answers the question:

            "What labels would scikit-learn itself return for this K?"

        In practice this usually agrees with cutting the full tree, but using
        the estimator directly keeps the demonstration faithful to the library's
        own API.
        """
        if not hasattr(self, "X_"):
            raise RuntimeError("Call fit(X) before get_labels(K).")

        cut_model = self._make_model(full_tree=False, n_clusters=K)
        return cut_model.fit_predict(self.X_)


# =============================================================================
# B — LINKAGE COMPARISON
# =============================================================================

def linkage_comparison(X, y_true, K, name="dataset"):
    """
    Compare four linkage methods on the same dataset.

    What this function is trying to answer
    --------------------------------------
    "If I keep the dataset fixed and only change linkage criterion,
    how much does the clustering quality change?"

    Inputs
    ------
    X : ndarray
        Data matrix.
    y_true : ndarray
        Ground-truth labels, used only for evaluation with ARI.
    K : int
        Number of clusters to extract from each dendrogram.
    name : str
        Descriptive name used in titles and output filename.

    Outputs
    -------
    aris : dict
        Mapping {method -> ARI score}

    Figures produced
    ----------------
    2 x 4 panel:
      top row    -> dendrogram for each linkage
      bottom row -> scatter plot of extracted K clusters

    Important methodological point
    ------------------------------
    This function standardizes X *before* clustering:

        Xs = StandardScaler().fit_transform(X)

    This is one of the most important choices in the whole script because
    hierarchical clustering is distance-based.
    """
    print(f"\n[B] Linkage comparison — {name}")

    # Standardize each feature to roughly zero mean and unit variance.
    #
    # Why this strongly affects outcome:
    #   If one feature has a much larger numeric scale than another,
    #   Euclidean distance will be dominated by that feature.
    #
    # After scaling:
    #   each feature contributes more comparably.
    Xs = StandardScaler().fit_transform(X)

    methods = ["single", "complete", "average", "ward"]

    # Purely for visual identity in titles.
    colors = {
        "single": "royalblue",
        "complete": "darkorange",
        "average": "forestgreen",
        "ward": "crimson",
    }

    fig = plt.figure(figsize=(22, 12))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.40, wspace=0.30)

    fig.suptitle(
        f"Linkage Comparison — {name}",
        fontsize=18,
        fontweight="bold"
    )

    # If data are higher-dimensional, reduce to 2D for plotting only.
    #
    # Critical point:
    #   The clustering is NOT performed on Xp.
    #   The clustering is performed on Xs.
    #
    # Therefore:
    #   PCA changes what the human sees in the scatter plots,
    #   but does NOT change the cluster assignments computed below.
    Xp = PCA(n_components=2).fit_transform(Xs) if Xs.shape[1] > 2 else Xs

    # Store ARI scores for all methods.
    aris = {}

    for col, method in enumerate(methods):
        # Build hierarchical clustering tree using SciPy.
        Z = linkage(Xs, method=method)

        # Cut tree into K clusters.
        lbl = fcluster(Z, t=K, criterion="maxclust") - 1

        # -----------------------------------------------------------------
        # Top row: dendrogram
        # -----------------------------------------------------------------
        ax_d = fig.add_subplot(gs[0, col])

        dendrogram(
            Z,
            ax=ax_d,
            truncate_mode="lastp",
            p=20,
            show_leaf_counts=True,
            leaf_rotation=90,
            leaf_font_size=7,

            # This threshold controls branch coloring.
            # It does NOT change the clustering result itself.
            # It only changes dendrogram coloring.
            #
            # The idea:
            #   use a threshold near the height corresponding to the K-cluster cut
            #   so colored branches visually match the K-partition reasonably well.
            color_threshold=0.7 * Z[-K + 1, 2],
            above_threshold_color="gray",
        )

        ax_d.set_title(
            f"{method.capitalize()} Linkage",
            fontsize=16,
            fontweight="bold",
            color=colors[method]
        )
        ax_d.set_ylabel("Merge distance", fontsize=8)
        ax_d.tick_params(axis="both", labelsize=7)

        # -----------------------------------------------------------------
        # Bottom row: flat clustering visualized in 2D
        # -----------------------------------------------------------------
        ax_s = fig.add_subplot(gs[1, col])

        ax_s.scatter(
            Xp[:, 0],
            Xp[:, 1],
            c=lbl,
            cmap="tab10",
            s=20,
            alpha=0.7
        )

        # ARI uses ground truth and is insensitive to label permutation.
        # Why ARI matters:
        #   It tells us how well the clustering recovered the known classes.
        ari = adjusted_rand_score(y_true, lbl)

        # Silhouette does not use ground truth.
        # It asks whether points are close to their own cluster and far from others.
        sil = silhouette_score(Xs, lbl)

        aris[method] = ari

        # The title color itself encodes rough quality.
        # Again, purely a display choice.
        c = "forestgreen" if ari > 0.85 else "red" if ari < 0.5 else "orange"

        ax_s.set_title(
            f"K={K}  ARI={ari:.3f}  Sil={sil:.3f}",
            fontsize=9,
            fontweight="bold",
            color=c
        )

        ax_s.set_xticks([])
        ax_s.set_yticks([])

    fname = f"plots/hierarchical/linkage_cmp_{name.replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()

    best = max(aris, key=aris.get)

    print(f"  ARIs: { {k: f'{v:.3f}' for k, v in aris.items()} }")
    print(f"  Best linkage: {best}  ARI={aris[best]:.3f}")
    print(f"  saved → {fname}")

    return aris


# =============================================================================
# B2 — SCIKIT-LEARN AGGLOMERATIVE ANALYSIS
# =============================================================================


def sklearn_agglomerative_analysis(X, y_true, K, name="dataset"):
    """
    Analyze agglomerative clustering using scikit-learn's estimator API.

    Why this function is useful
    ---------------------------
    The SciPy-based parts of this script revolve around two functions:

        linkage(...)   -> build hierarchical tree in matrix form
        fcluster(...)  -> cut the tree into K clusters

    That is a very "hierarchical clustering textbook" workflow.

    scikit-learn presents the same method differently:

        model = AgglomerativeClustering(...)
        model.fit(X)
        model.labels_
        model.children_
        model.distances_

    This function makes that representation visible and plots it in a way that
    mirrors the existing SciPy comparison section.

    What is plotted
    ---------------
    For each linkage method:

    Row 1: truncated dendrogram reconstructed from sklearn attributes
    Row 2: flat K-cluster partition in 2D (via PCA if needed)
    Row 3: merge-distance trajectory across the whole agglomeration process

    Why the third row matters
    -------------------------
    The merge-distance curve shows *when* the algorithm starts making expensive
    merges. Large late-stage jumps often indicate natural coarse cluster
    boundaries.
    """
    print(f"\n[B2] scikit-learn AgglomerativeClustering analysis — {name}")

    # Standardize first for the same reason as elsewhere in the script:
    # agglomerative clustering is distance-based, so feature scale matters.
    Xs = StandardScaler().fit_transform(X)

    # PCA is only for plotting.
    Xp = PCA(n_components=2).fit_transform(Xs) if Xs.shape[1] > 2 else Xs

    methods = ["single", "complete", "average", "ward"]
    colors = {
        "single": "royalblue",
        "complete": "darkorange",
        "average": "forestgreen",
        "ward": "crimson",
    }

    fig = plt.figure(figsize=(22, 16))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.42, wspace=0.30)
    fig.suptitle(
        f"scikit-learn AgglomerativeClustering — {name}",
        fontsize=24,
        fontweight="bold",
    )

    summary = {}

    for col, method in enumerate(methods):
        ag = AgglomerativeSklearnExplorer(method=method)
        ag.fit(Xs)
        lbl = ag.get_labels(K)
        Z = ag.Z_

        ari = adjusted_rand_score(y_true, lbl)
        sil = silhouette_score(Xs, lbl)
        summary[method] = {
            "ARI": ari,
            "Silhouette": sil,
            "Final merge": float(ag.distances_[-1]),
        }

        # Print a small textual diagnostic so you can inspect the raw estimator
        # representation, not only the plots.
        first_merges = ag.children_[:min(5, len(ag.children_))].tolist()
        print(
            f"  {method:<10}: ARI={ari:.3f} | Sil={sil:.3f} | "
            f"first merges={first_merges} | final merge={ag.distances_[-1]:.3f}"
        )

        # -------------------------------------------------------------
        # Row 1 — Dendrogram reconstructed from sklearn attributes
        # -------------------------------------------------------------
        ax_d = fig.add_subplot(gs[0, col])

        if K < len(Z) + 1:
            cut_h = 0.5 * (Z[-K, 2] + Z[-(K - 1), 2])
        else:
            cut_h = Z[-1, 2]

        dendrogram(
            Z,
            ax=ax_d,
            truncate_mode="lastp",
            p=20,
            show_leaf_counts=True,
            leaf_rotation=90,
            leaf_font_size=7,
            color_threshold=cut_h,
            above_threshold_color="gray",
        )
        ax_d.axhline(cut_h, color="black", ls="--", lw=1.2)
        ax_d.set_title(
            f"{method.capitalize()} linkage\nchildren_ + distances_ → dendrogram",
            fontsize=9,
            fontweight="bold",
            color=colors[method],
        )
        ax_d.set_ylabel("Merge distance", fontsize=8)
        ax_d.tick_params(axis="both", labelsize=7)

        # -------------------------------------------------------------
        # Row 2 — Flat clustering result at the chosen K
        # -------------------------------------------------------------
        ax_s = fig.add_subplot(gs[1, col])
        ax_s.scatter(Xp[:, 0], Xp[:, 1], c=lbl, cmap="tab10", s=20, alpha=0.7)
        title_color = "forestgreen" if ari > 0.85 else "red" if ari < 0.5 else "orange"
        ax_s.set_title(
            f"K={K}  ARI={ari:.3f}  Sil={sil:.3f}",
            fontsize=9,
            fontweight="bold",
            color=title_color,
        )
        ax_s.set_xticks([])
        ax_s.set_yticks([])

        # -------------------------------------------------------------
        # Row 3 — Merge-distance trajectory across agglomeration steps
        # -------------------------------------------------------------
        ax_m = fig.add_subplot(gs[2, col])
        merge_steps = np.arange(1, len(ag.distances_) + 1)
        ax_m.plot(merge_steps, ag.distances_, "o-", ms=3, lw=1.5)

        # Highlight the last three merges because they often encode the most
        # global / coarse cluster structure.
        top = min(3, len(merge_steps))
        ax_m.plot(
            merge_steps[-top:],
            ag.distances_[-top:],
            "o",
            ms=6,
            color="red",
        )

        ax_m.set_title("Merge distances over steps", fontsize=9, fontweight="bold")
        ax_m.set_xlabel("Merge step")
        ax_m.set_ylabel("Distance")
        ax_m.grid(alpha=0.3)

    fname = f"plots/hierarchical/sklearn_agglomerative_{name.replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()

    print("  Summary:")
    for method, stats in summary.items():
        print(
            f"    {method:<10}: "
            f"ARI={stats['ARI']:.3f} | "
            f"Sil={stats['Silhouette']:.3f} | "
            f"final merge={stats['Final merge']:.3f}"
        )
    print(f"  saved → {fname}")
    return summary


# =============================================================================
# C — DENDROGRAM ANALYSIS
# =============================================================================

def dendrogram_analysis(X, y_true, name="dataset"):
    """
    Analyse a Ward dendrogram in more depth.

    Main ideas demonstrated
    -----------------------
    1. The dendrogram itself is a nested merge history.
    2. Cophenetic correlation measures how faithfully the tree preserves
       original pairwise distances.
    3. Large jumps in merge distance can suggest natural cluster boundaries.
    4. A cut height can be interpreted as a choice of number of clusters K.

    Why Ward is used here
    ---------------------
    Ward often gives a clean, interpretable dendrogram for compact,
    blob-like datasets because it merges clusters in a variance-minimizing way.
    """
    print(f"\n[C] Dendrogram analysis — {name}")

    # Standardize before distance-based clustering.
    Xs = StandardScaler().fit_transform(X)

    # Build Ward linkage.
    Z = linkage(Xs, method="ward")

    # -------------------------------------------------------------------------
    # Cophenetic correlation
    # -------------------------------------------------------------------------
    #
    # `cophenet(Z, pdist(Xs))` returns:
    #   c      -> cophenetic correlation coefficient
    #   d_coph -> cophenetic distances for all sample pairs
    #
    # Definitions:
    #   d_ij     = true distance between points i and j in data space
    #   c_ij     = height in the dendrogram where i and j first become joined
    #
    # If c is high:
    #   the tree is a faithful geometric summary of the data.
    c, d_coph = cophenet(Z, pdist(Xs))

    print(
        f"  Cophenetic r (Ward): {c:.4f}  "
        f"({'Excellent' if c > 0.75 else 'Good' if c > 0.60 else 'Poor'})"
    )

    # Compare cophenetic quality across linkage methods.
    # This helps answer:
    #   Which linkage yields a tree that best preserves pairwise geometry?
    for method in ["single", "complete", "average", "ward"]:
        Zi = linkage(Xs, method=method)
        ci, _ = cophenet(Zi, pdist(Xs))
        print(f"  Cophenetic ({method:<10}): {ci:.4f}")

    # -------------------------------------------------------------------------
    # Heuristic for choosing K from merge-distance jumps
    # -------------------------------------------------------------------------
    #
    # Z[:, 2] stores merge heights in ascending order.
    # Late merges happen high in the tree.
    merge_d = Z[:, 2]

    # Reversing merge distances puts the largest / latest merges first.
    # Taking np.diff then measures jump sizes between successive late merges.
    #
    # Terminology note:
    # The original comment says "second difference", but np.diff on a reversed
    # array is actually a first difference of the reversed sequence.
    accel = np.diff(merge_d[::-1])

    # Convert jump index into a candidate K.
    #
    # This is heuristic, not guaranteed theory.
    # It is saying:
    #   "where does the dendrogram suddenly start forcing much larger merges?"
    K_accel = int(np.argmax(accel)) + 2

    print(f"  Suggested K from distance acceleration: {K_accel}")

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    fig.suptitle(
        f"Dendrogram Analysis — {name} (Ward's Linkage)",
        fontsize=14,
        fontweight="bold"
    )

    # 1) Dendrogram
    ax1 = fig.add_subplot(gs[0, :])

    dendrogram(
        Z,
        ax=ax1,
        truncate_mode="lastp",
        p=30,
        show_leaf_counts=True,
        leaf_rotation=45,
        leaf_font_size=8,

        # Again: visual only, not algorithmic.
        color_threshold=0.7 * np.percentile(merge_d, 85),
        above_threshold_color="dimgray",
    )

    # Draw several possible cut heights corresponding to K = 2..6.
    cut_Ks = range(2, 7)
    colours_cut = ["red", "blue", "green", "orange", "purple"]

    for K_cut, cc in zip(cut_Ks, colours_cut):
        # For a linkage with n-1 merges:
        #   the threshold yielding K clusters lies between two neighboring
        #   high-level merge heights.
        #
        # This is an approximate visual annotation for interpretation.
        h = (
            (merge_d[-(K_cut)] + merge_d[-(K_cut - 1)]) / 2
            if K_cut <= len(merge_d)
            else merge_d[-1]
        )
        ax1.axhline(h, color=cc, ls="--", lw=1.5, label=f"K={K_cut}")

    ax1.set_title(
        "Dendrogram (last 30 merges)  —  dashed lines = cut heights",
        fontsize=10,
        fontweight="bold"
    )
    ax1.set_xlabel("Cluster (count in parentheses)")
    ax1.set_ylabel("Ward merge distance")
    ax1.legend(fontsize=9, loc="upper left")

    # 2) Merge-distance jump bar chart
    ax2 = fig.add_subplot(gs[1, 0])

    last = min(20, len(merge_d))
    x = np.arange(len(merge_d) - last, len(merge_d))

    bars = ax2.bar(
        x,
        merge_d[-last:],
        color="steelblue",
        alpha=0.75,
        edgecolor="k",
        lw=0.4
    )

    # Highlight top-3 largest late merges.
    # Interpretively:
    #   these are candidate places where the data resist being merged further.
    top3 = np.argsort(merge_d[-last:])[::-1][:3]
    for t in top3:
        bars[t].set_color("red")
        bars[t].set_alpha(0.9)

    ax2.set_xlabel("Merge step (last 20)")
    ax2.set_ylabel("Merge distance")
    ax2.set_title(
        "Large jumps = natural cluster boundaries",
        fontsize=9,
        fontweight="bold"
    )
    ax2.grid(alpha=0.3)

    # 3) Cophenetic scatter plot
    ax3 = fig.add_subplot(gs[1, 1])

    true_d = pdist(Xs)

    # To keep plotting manageable, the script randomly subsamples pair distances.
    # Effect on outcome:
    #   none on statistics already computed,
    #   only speeds up / declutters the plot.
    idx = RNG.choice(len(true_d), size=min(5000, len(true_d)), replace=False)

    ax3.scatter(
        true_d[idx],
        d_coph[idx],
        alpha=0.15,
        s=4,
        color="steelblue"
    )

    lim = max(true_d.max(), d_coph.max())
    ax3.plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfect r=1")

    ax3.set_xlabel("Original pairwise distance")
    ax3.set_ylabel("Cophenetic distance")
    ax3.set_title(
        f"Cophenetic correlation  r = {c:.4f}",
        fontsize=9,
        fontweight="bold"
    )
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    fname = f"plots/hierarchical/dendrogram_{name.replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  saved → {fname}")

    return K_accel


# =============================================================================
# D — CHOOSING K FROM DENDROGRAM
# =============================================================================

def choose_K_hierarchical(X, y_true, K_max=10, name="dataset"):
    """
    Compare candidate values of K for Ward hierarchical clustering.

    What this function is doing conceptually
    ----------------------------------------
    Hierarchical clustering gives you a full tree.
    But many users still want one final partition with a fixed number of
    clusters K.

    This function asks:
      "If I cut the Ward dendrogram at K = 2, 3, ..., K_max,
       which K seems best?"

    It uses two criteria:
      1. silhouette score   -> unsupervised / internal
      2. ARI                -> supervised / external (only if y_true exists)

    Important distinction
    ---------------------
    The clustering model itself is the same Ward tree each time.
    Only the cut level changes.
    """
    print(f"\n[D] Choosing K — {name}")

    Xs = StandardScaler().fit_transform(X)
    Z = linkage(Xs, method="ward")

    K_range = range(2, K_max + 1)
    sils, aris = [], []

    for K in K_range:
        # Flat labels obtained by cutting the same tree at a different level.
        lbl = fcluster(Z, t=K, criterion="maxclust") - 1

        # Silhouette:
        #   higher is better
        #   interprets compactness and separation
        sils.append(silhouette_score(Xs, lbl))

        # ARI:
        #   only available if ground truth is known
        if y_true is not None:
            aris.append(adjusted_rand_score(y_true, lbl))

    K_sil = list(K_range)[int(np.argmax(sils))]
    K_ari = list(K_range)[int(np.argmax(aris))] if aris else None

    print(f"  Best K by silhouette : {K_sil}  ({max(sils):.3f})")
    if K_ari:
        print(f"  Best K by ARI        : {K_ari}  ({max(aris):.3f})")

    K_arr = np.array(list(K_range))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"K Selection (Ward's) — {name}", fontweight="bold")

    # Left panel: silhouette vs K
    axes[0].plot(K_arr, sils, "o-", color="steelblue", lw=2)
    axes[0].axvline(K_sil, color="red", ls="--", label=f"K*={K_sil}")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Silhouette")
    axes[0].set_title("Silhouette (maximise)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Right panel: ARI vs K, only if labels are known
    if aris:
        axes[1].plot(K_arr, aris, "o-", color="forestgreen", lw=2)
        axes[1].axvline(K_ari, color="red", ls="--", label=f"K*={K_ari}")
        axes[1].set_xlabel("K")
        axes[1].set_ylabel("ARI")
        axes[1].set_title("ARI vs true labels")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    fname = f"plots/hierarchical/chooseK_{name.replace(' ', '_')}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  saved → {fname}")


# =============================================================================
# E — SOFT-MATTER: POLYMER CONFORMATIONAL HIERARCHY
# =============================================================================

def polymer_hierarchy():
    """
    Domain-specific example: clustering polymer conformations.

    What this section is trying to show
    -----------------------------------
    Hierarchical clustering is especially natural when the data truly have
    multi-level structure.

    In this example, conformational states may have:
      - a coarse partition (compact vs extended)
      - a finer partition (specific state basins)

    Hierarchical clustering is attractive because both levels can live in
    the same dendrogram.

    Dataset assumptions
    -------------------
    The loaded dataset `polymer_conf` is expected to contain features such as:
      - φ, ψ dihedral angles
      - Rg   (radius of gyration)
      - d_ee (end-to-end distance)

    Those features encode shape / compactness / geometry of conformations.
    """
    print("\n[E] Polymer conformation hierarchy")

    X, y_true = load("polymer_conf")

    # Human-readable names for ground-truth states.
    state = {
        0: "Extended(β)",
        1: "Helical(α)",
        2: "Collapsed",
        3: "Kinked"
    }

    # Subsample for clarity.
    #
    # Why this matters:
    #   - It makes plots less crowded.
    #   - It reduces computational load somewhat.
    #
    # Tradeoff:
    #   - You lose some information from the full dataset.
    idx = RNG.choice(len(X), size=800, replace=False)
    X_sub = X[idx]
    y_sub = y_true[idx]

    # Standardize before clustering.
    Xs = StandardScaler().fit_transform(X_sub)

    # Ward linkage on the standardized subset.
    Z = linkage(Xs, method="ward")

    # Cophenetic correlation tells whether the hierarchy is geometrically sensible.
    c, _ = cophenet(Z, pdist(Xs))
    print(f"  Cophenetic r (Ward): {c:.4f}")

    # Cut the same dendrogram at two levels:
    #   K=2 for coarse hierarchy
    #   K=4 for fine hierarchy
    lbl2 = fcluster(Z, t=2, criterion="maxclust") - 1
    lbl4 = fcluster(Z, t=4, criterion="maxclust") - 1

    # For the coarse comparison, the script manually groups states into:
    #   compact   -> labels 1 and 2
    #   extended  -> labels 0 and 3
    #
    # This is a scientifically meaningful recoding:
    # it tests whether the hierarchy first separates conformations by broad
    # compactness class before distinguishing finer state details.
    compact_mask = np.isin(y_sub, [1, 2])
    y_coarse = compact_mask.astype(int)

    ari2 = adjusted_rand_score(y_coarse, lbl2)
    ari4 = adjusted_rand_score(y_sub, lbl4)

    print(f"  ARI K=2 (compact vs extended) : {ari2:.4f}")
    print(f"  ARI K=4 (all states)          : {ari4:.4f}")

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------
    # PCA again is only for plotting.
    pca = PCA(n_components=2)
    Xp = pca.fit_transform(Xs)

    cmap = plt.cm.tab10

    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    fig.suptitle(
        "Polymer Conformation Hierarchy — Ward's Agglomerative",
        fontsize=14,
        fontweight="bold"
    )

    # Dendrogram panel
    ax_d = fig.add_subplot(gs[0, :2])

    dendrogram(
        Z,
        ax=ax_d,
        truncate_mode="lastp",
        p=30,
        show_leaf_counts=True,
        leaf_rotation=45,
        leaf_font_size=8,

        # Choosing Z[-4, 2] roughly emphasizes the 4-cluster structure.
        # Again, this affects branch colors only.
        color_threshold=Z[-4, 2],
        above_threshold_color="gray"
    )

    # Draw cut heights corresponding to K=2 and K=4.
    # These lines help visually connect the tree to the flat partitions.
    h2 = (Z[-2, 2] + Z[-1, 2]) / 2
    h4 = (Z[-4, 2] + Z[-3, 2]) / 2

    ax_d.axhline(h2, color="blue", ls="--", lw=2, label="K=2 cut")
    ax_d.axhline(h4, color="red", ls="--", lw=2, label="K=4 cut")

    ax_d.set_title(
        "Dendrogram — two-level hierarchy visible",
        fontsize=10,
        fontweight="bold"
    )
    ax_d.set_ylabel("Ward merge distance")
    ax_d.legend(fontsize=9)

    # Silhouette vs K for this polymer subset
    ax_s = fig.add_subplot(gs[0, 2])

    K_range = range(2, 8)
    sils = [
        silhouette_score(Xs, fcluster(Z, t=K, criterion="maxclust") - 1)
        for K in K_range
    ]

    ax_s.plot(list(K_range), sils, "o-", color="steelblue", lw=2)
    ax_s.axvline(4, color="red", ls="--", label="K=4 (known)")
    ax_s.set_xlabel("K")
    ax_s.set_ylabel("Silhouette")
    ax_s.set_title("Silhouette vs K")
    ax_s.legend()
    ax_s.grid(alpha=0.3)

    # Bottom-left: true states
    ax = fig.add_subplot(gs[1, 0])
    for lbl, nm in state.items():
        m = y_sub == lbl
        ax.scatter(Xp[m, 0], Xp[m, 1], c=[cmap(lbl)], s=18, alpha=0.6, label=nm)

    ax.set_title("True states", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")

    # Bottom-middle: K=2 clustering
    ax = fig.add_subplot(gs[1, 1])
    for k in range(2):
        m = lbl2 == k
        ax.scatter(
            Xp[m, 0],
            Xp[m, 1],
            c=[cmap(k)],
            s=18,
            alpha=0.6,
            label=f"C{k} n={m.sum()}"
        )

    ax.set_title(f"K=2  ARI={ari2:.3f}", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # Bottom-right: K=4 clustering
    ax = fig.add_subplot(gs[1, 2])
    for k in range(4):
        m = lbl4 == k
        ax.scatter(
            Xp[m, 0],
            Xp[m, 1],
            c=[cmap(k)],
            s=18,
            alpha=0.6,
            label=f"C{k} n={m.sum()}"
        )

    ax.set_title(f"K=4  ARI={ari4:.3f}", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    plt.savefig(
        "plots/hierarchical/polymer_hierarchy.png",
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()

    print("  saved → plots/hierarchical/polymer_hierarchy.png")


# =============================================================================
# F — PERFORMANCE HEATMAP
# =============================================================================

def performance_heatmap():
    """
    Summarize performance of linkage methods across dataset types.

    Why this function is useful
    ---------------------------
    Individual examples can be misleading.
    This function collects several datasets and asks:

        "Which linkage methods work well for which data geometries?"

    Evaluation metric
    -----------------
    ARI (Adjusted Rand Index), using known ground-truth labels.

    Why ARI is suitable here
    ------------------------
    - invariant to label permutations,
    - corrected for chance,
    - 1.0 means perfect recovery,
    - near 0 means random-like agreement.
    """
    print("\n[F] Performance heatmap: linkage × dataset")

    # Dataset registry:
    #   key   -> human-readable row label
    #   value -> (filename, expected K)
    datasets = {
        "Isotropic":   ("blobs_easy",    3),
        "Anisotropic": ("blobs_aniso",   3),
        "Unequal":     ("blobs_unequal", 3),
        "Circles":     ("circles",       2),
        "Moons":       ("moons",         2),
    }

    methods = ["single", "complete", "average", "ward"]
    results = {}

    for ds_name, (fname, K) in datasets.items():
        X, y = load(fname)

        # Standardization again matters strongly because ARI comparisons across
        # linkage methods would otherwise partly reflect raw feature scales.
        Xs = StandardScaler().fit_transform(X)

        row = {}

        for method in methods:
            Z = linkage(Xs, method=method)
            lbl = fcluster(Z, t=K, criterion="maxclust") - 1
            row[method] = adjusted_rand_score(y, lbl)

        results[ds_name] = row

        print(
            f"  {ds_name:<14}: "
            + " | ".join(f"{m}: {v:.3f}" for m, v in row.items())
        )

    # Convert nested dict to table form for heatmap plotting.
    df = pd.DataFrame(results).T

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.heatmap(
        df,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        ax=ax,
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"label": "ARI (1=perfect)"}
    )

    ax.set_title(
        "Agglomerative Clustering: ARI by Linkage × Dataset",
        fontsize=13,
        fontweight="bold"
    )
    ax.set_xlabel("Linkage method")
    ax.set_ylabel("Dataset")

    plt.tight_layout()
    plt.savefig(
        "plots/hierarchical/linkage_heatmap.png",
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()

    print("  saved → plots/hierarchical/linkage_heatmap.png")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    """
    The script's execution order is tutorial-like:

    A. first show a tiny scratch example,
    B. then compare linkages on standard datasets,
    C. then inspect a dendrogram more deeply,
    D. then scan over possible K values,
    E. then show a domain-specific polymer example,
    F. finally summarize performance as a heatmap.

    This sequence is deliberate.
    It moves from:
        mechanics  -> comparison  -> interpretation  -> model selection
        -> scientific application -> overall summary
    """

    print("=" * 65)
    print("  AGGLOMERATIVE HIERARCHICAL CLUSTERING — TUTORIAL")
    print("=" * 65)

    # -------------------------------------------------------------------------
    # A — scratch demo on a tiny manually specified dataset
    # -------------------------------------------------------------------------
    print("\n[A] Scratch agglomerative (10 points)")

    # A small 2D dataset designed so that there are two obvious groups:
    #   one near the origin / lower-left,
    #   another near the upper-right.
    #
    # Why this matters:
    #   With only 10 points, the scratch implementation is easy to sanity-check.
    Xsmall = np.array([
        [1,   1],
        [1.5, 1.2],
        [5,   5],
        [3,   4],
        [4,   4],
        [3.5, 3.5],
        [0,   0],
        [0.5, 0.3],
        [5.5, 5],
        [4.5, 5.5],
    ])

    # Ground truth labels for evaluation.
    ysmall = np.array([0, 0, 1, 1, 1, 1, 0, 0, 1, 1])

    for method in ("single", "complete", "average", "ward"):
        ag = AgglomerativeScratch(method=method)
        ag.fit(Xsmall)
        lbl = ag.get_labels(K=2)

        print(f"  {method:<10}: ARI={adjusted_rand_score(ysmall, lbl):.3f}")

    # -------------------------------------------------------------------------
    # A2 — scikit-learn estimator view of the same agglomerative idea
    # -------------------------------------------------------------------------
    # This produces a richer diagnostic plot showing:
    #   1. the reconstructed dendrogram,
    #   2. the flat clustering at K=2,
    #   3. the full merge-distance trajectory.
    #
    # Because the dataset is tiny, it is easy to inspect the actual merge tree.
    sklearn_agglomerative_analysis(Xsmall, ysmall, K=2, name="tiny toy set")

    # -------------------------------------------------------------------------
    # B — linkage comparison on canonical datasets
    # -------------------------------------------------------------------------
    X_blob, y_blob = load("blobs_easy")
    linkage_comparison(X_blob, y_blob, K=3, name="isotropic blobs")

    X_moon, y_moon = load("moons")
    linkage_comparison(X_moon, y_moon, K=2, name="two moons")

    # -------------------------------------------------------------------------
    # C — detailed dendrogram analysis
    # -------------------------------------------------------------------------
    dendrogram_analysis(X_blob, y_blob, name="isotropic blobs")

    # -------------------------------------------------------------------------
    # D — choose K for the same isotropic blob dataset
    # -------------------------------------------------------------------------
    choose_K_hierarchical(X_blob, y_blob, K_max=8, name="isotropic blobs")

    # -------------------------------------------------------------------------
    # E — domain-specific polymer hierarchy
    # -------------------------------------------------------------------------
    polymer_hierarchy()

    # -------------------------------------------------------------------------
    # F — overall performance summary
    # -------------------------------------------------------------------------
    performance_heatmap()

    print("\n" + "=" * 65)
    print("  HIERARCHICAL MODULE COMPLETE — see plots/hierarchical/")
    print("=" * 65)
