# Agglomerative Hierarchical Clustering — Deep-Dive Tutorial Project

This project is a **full learning notebook in script form** for understanding agglomerative hierarchical clustering from first principles, through standard SciPy usage, up to scientific interpretation on synthetic and soft-matter-inspired datasets.

It is designed so that, at some distant future date, you can return to this repository, read only this README, and recover:

- what hierarchical clustering is,
- how the algorithm actually works,
- what each linkage criterion means mathematically,
- how the code in this project is structured,
- how to read every generated figure,
- how to choose the number of clusters `K`,
- when each linkage succeeds or fails,
- and what the polymer example is trying to teach physically.

---

## Table of Contents

1. [Project purpose](#project-purpose)
2. [What this project teaches](#what-this-project-teaches)
3. [High-level overview of hierarchical clustering](#high-level-overview-of-hierarchical-clustering)
4. [Why this project is useful](#why-this-project-is-useful)
5. [Repository structure](#repository-structure)
6. [Environment and dependencies](#environment-and-dependencies)
7. [How to run](#how-to-run)
8. [What the script produces](#what-the-script-produces)
9. [Core mathematical ideas](#core-mathematical-ideas)
10. [Understanding the linkage methods](#understanding-the-linkage-methods)
11. [How the scratch implementation works](#how-the-scratch-implementation-works)
12. [How to read a linkage matrix](#how-to-read-a-linkage-matrix)
13. [How to read a dendrogram](#how-to-read-a-dendrogram)
14. [Metrics used in this project](#metrics-used-in-this-project)
15. [Section-by-section walkthrough of the code](#section-by-section-walkthrough-of-the-code)
16. [Detailed interpretation of each dataset](#detailed-interpretation-of-each-dataset)
17. [How to choose `K`](#how-to-choose-k)
18. [How to interpret the polymer hierarchy section](#how-to-interpret-the-polymer-hierarchy-section)
19. [How to interpret the performance heatmap](#how-to-interpret-the-performance-heatmap)
20. [Most important conceptual lessons](#most-important-conceptual-lessons)
21. [Pitfalls, caveats, and known issues](#pitfalls-caveats-and-known-issues)
22. [Suggested extensions](#suggested-extensions)
23. [Revision guide: if you only have 10 minutes](#revision-guide-if-you-only-have-10-minutes)
24. [Revision guide: if you have 1 hour](#revision-guide-if-you-have-1-hour)
25. [Glossary](#glossary)

---

## Project purpose

The goal of this project is **not just to run hierarchical clustering**, but to *learn it deeply*.

Many clustering tutorials stop at:

```python
Z = linkage(X, method="ward")
labels = fcluster(Z, t=3, criterion="maxclust")
```

and never really explain:

- what `Z` contains,
- what exactly is being merged at each step,
- why different linkage methods produce different dendrograms,
- why single linkage can be perfect for moons and terrible for unequal blobs,
- why Ward excels on compact blob-like clusters,
- why silhouette can be misleading on non-convex data,
- and how hierarchical clustering can reveal *nested* structure rather than just flat labels.

This project fills that gap.

---

## What this project teaches

By the end of this project, you should understand all of the following.

### Algorithmic understanding

You will know how agglomerative clustering proceeds step by step:

1. start with `n` singleton clusters,
2. compute pairwise distances,
3. repeatedly merge the two closest clusters,
4. update cluster-to-cluster distances after each merge,
5. record the merge history in a linkage matrix,
6. cut the final tree at a chosen height or chosen number of clusters.

### Mathematical understanding

You will understand the difference between:

- **single linkage**: nearest-neighbor cluster distance,
- **complete linkage**: farthest-neighbor cluster distance,
- **average linkage**: average cross-cluster distance,
- **Ward linkage**: merge cost based on increase in within-cluster variance.

### Geometric understanding

You will see that the correct linkage depends on cluster geometry:

- compact spherical blobs,
- elongated anisotropic blobs,
- unequal cluster sizes,
- non-convex rings,
- non-convex moons,
- physically meaningful hierarchical states.

### Practical understanding

You will learn how to use:

- SciPy’s `linkage`, `dendrogram`, `fcluster`, `cophenet`,
- `StandardScaler`,
- PCA for visualization,
- ARI and silhouette for evaluation,
- dendrogram cut heights for selecting `K`.

### Scientific understanding

You will also see that clustering is not only about “best labels.” Sometimes the real value lies in the **hierarchical map itself**, especially in scientific systems such as polymer conformational landscapes.

---

## High-level overview of hierarchical clustering

Agglomerative hierarchical clustering is a **bottom-up clustering algorithm**.

Instead of directly assigning each point to one of `K` clusters from the start, it builds a **tree of merges**.

At the beginning:

- every sample is its own cluster.

Then repeatedly:

- the two closest clusters are merged.

This continues until:

- all samples belong to one final cluster.

The entire process creates a tree called a **dendrogram**.

This is important because hierarchical clustering does **more than produce one partition**. It produces a whole nested family of partitions:

- many small clusters at low cut heights,
- fewer larger clusters at high cut heights,
- one giant cluster at the very top.

So hierarchical clustering answers not only:

> “What are the clusters?”

but also:

> “How do clusters merge into larger structures?”

That is why it is so useful when the data contain **multi-scale structure**.

---

## Why this project is useful

This project is useful because it combines **three levels of learning** in one place.

### Level 1 — from scratch intuition

The custom `AgglomerativeScratch` class shows the algorithm in pure NumPy. This is where you learn what the algorithm is actually doing internally.

### Level 2 — library-level workflow

The SciPy-based sections show how hierarchical clustering is normally done in practice:

- `linkage(...)`
- `dendrogram(...)`
- `fcluster(...)`

### Level 3 — interpretation and scientific reasoning

The comparison plots, dendrogram analysis, `K` selection plots, polymer example, and final heatmap teach how to *think* about clustering results rather than just run code.

---

## Repository structure

A typical project layout for this tutorial is expected to look like this:

```text
.
├── 00_generate_datasets.py
├── hierarchical_v1.py                # original clustering tutorial script
├── hierarchical_v1_annotated.py      # heavily documented version
├── hierarchical_v1_annotated_with_sklearn.py   # optional extended version
├── data/
│   ├── blobs_easy.npz
│   ├── blobs_aniso.npz
│   ├── blobs_unequal.npz
│   ├── circles.npz
│   ├── moons.npz
│   └── polymer_conf.npz
└── plots/
    └── hierarchical/
        ├── linkage_cmp_isotropic_blobs.png
        ├── linkage_cmp_two_moons.png
        ├── dendrogram_isotropic_blobs.png
        ├── chooseK_isotropic_blobs.png
        ├── polymer_hierarchy.png
        └── linkage_heatmap.png
```

The data are expected to be stored in `.npz` files with keys:

- `X`: feature matrix
- `y`: true labels

---

## Environment and dependencies

This project uses the following Python ecosystem.

### Core dependencies

- `numpy`
- `scipy`
- `matplotlib`
- `seaborn`
- `pandas`
- `scikit-learn`

### Libraries and what they are used for

#### NumPy
Used for:

- raw array manipulation,
- manual distance bookkeeping,
- centroid updates,
- random subsampling.

#### SciPy
Used for:

- pairwise distances via `pdist` and `squareform`,
- linkage matrix creation via `linkage`,
- dendrogram visualization via `dendrogram`,
- flat cluster extraction via `fcluster`,
- cophenetic correlation via `cophenet`.

#### scikit-learn
Used for:

- `StandardScaler`,
- `PCA`,
- `adjusted_rand_score`,
- `silhouette_score`.

#### Matplotlib / seaborn / pandas
Used for:

- all plots,
- subplot grids,
- the final performance heatmap table.

---

## How to run

### Step 1 — generate the datasets

Run the dataset-generation script first.

```bash
python 00_generate_datasets.py
```

This creates the `.npz` files inside `data/`.

### Step 2 — run the main tutorial

```bash
python hierarchical_v1.py
```

or if you are using your expanded version:

```bash
python hierarchical_v1_annotated.py
```

or:

```bash
python hierarchical_v1_annotated_with_sklearn.py
```

### Step 3 — inspect the output plots

Look inside:

```text
plots/hierarchical/
```

Each image corresponds to one conceptual lesson in hierarchical clustering.

---

## What the script produces

The project is organized into conceptual sections.

### A — Agglomerative clustering from scratch

A tiny 10-point dataset is clustered using a manual NumPy implementation so that you can see the merge process conceptually.

### B — Linkage comparison

The script compares the four linkage methods on different datasets using:

- a dendrogram for each method,
- a scatter plot of the flat clustering after cutting at chosen `K`.

### C — Dendrogram analysis

This section focuses on:

- merge heights,
- cut lines,
- cophenetic correlation,
- jump structure,
- and how to read the dendrogram beyond just extracting labels.

### D — Choosing `K`

This section compares:

- silhouette versus `K`,
- ARI versus `K`.

### E — Polymer conformational hierarchy

This is the physically interesting part:

- hierarchical clustering on polymer conformational features,
- comparison between coarse and fine cuts,
- interpretation in terms of conformational basins.

### F — Performance heatmap

A compact summary of how each linkage behaves on different geometries.

---

## Core mathematical ideas

### 1. Pairwise distance matrix

For a dataset with samples

\[
X = \{x_1, x_2, \dots, x_n\},
\]

the Euclidean pairwise distances are

\[
d_{ij} = \lVert x_i - x_j \rVert.
\]

At the start of agglomerative clustering, every point is its own cluster, so cluster distances are initially just point-to-point distances.

### 2. Repeated merging

At each step, the algorithm chooses the two clusters `A` and `B` with the smallest cluster-to-cluster distance under the selected linkage rule.

Then it creates a new merged cluster:

\[
C = A \cup B.
\]

### 3. Linkage matrix

Every merge is recorded in a row of the linkage matrix:

```text
[id_left, id_right, merge_height, cluster_size]
```

If there are `n` samples, then the linkage matrix has shape:

\[
(n-1) \times 4.
\]

### 4. Dendrogram

The linkage matrix is simply a compact numerical encoding of the dendrogram.

The dendrogram is the visual tree of all merges.

---

## Understanding the linkage methods

This is the heart of hierarchical clustering.

### Single linkage

For two clusters `A` and `B`:

\[
d_{\text{single}}(A,B) = \min_{a \in A,\, b \in B} d(a,b)
\]

Interpretation:

- distance between clusters is the distance between their **closest pair of points**.

Behavior:

- tends to connect clusters through local bridges,
- excellent for connected curved shapes,
- vulnerable to **chaining**,
- can fail badly when there are noisy bridges or unequal densities.

Good for:

- circles,
- moons,
- filamentary structures.

### Complete linkage

\[
d_{\text{complete}}(A,B) = \max_{a \in A,\, b \in B} d(a,b)
\]

Interpretation:

- distance between clusters is the distance between their **farthest pair of points**.

Behavior:

- prefers compact clusters,
- resists chaining,
- can be too conservative,
- may split extended or curved structures.

Good for:

- compact, well-separated clusters,
- elongated but still separated clusters.

### Average linkage

\[
d_{\text{average}}(A,B) = \frac{1}{|A||B|} \sum_{a\in A}\sum_{b\in B} d(a,b)
\]

Interpretation:

- distance between clusters is the **average** distance over all cross-cluster pairs.

Behavior:

- compromise between single and complete,
- often robust,
- less extreme than either chaining or worst-case compactness.

Good for:

- many general-purpose datasets,
- moderately irregular shapes,
- datasets where complete is too strict and single is too permissive.

### Ward linkage

Ward does not primarily think in terms of nearest or farthest cross-cluster points.

Instead, it merges the pair of clusters that causes the **smallest increase in within-cluster sum of squares**.

A standard form of the merge cost is:

\[
\Delta J(A,B) = \frac{|A||B|}{|A|+|B|} \lVert \mu_A - \mu_B \rVert^2
\]

where:

- `|A|`, `|B|` are cluster sizes,
- `\mu_A`, `\mu_B` are cluster centroids.

Interpretation:

- Ward tries to keep clusters compact and variance-minimizing.

Behavior:

- excellent for blob-like clusters,
- usually strong on spherical or near-spherical groups,
- poor on strongly non-convex manifolds like circles.

Good for:

- isotropic blobs,
- many compact Gaussian-like datasets,
- practical settings where compactness is the desired bias.

---

## How the scratch implementation works

The custom `AgglomerativeScratch` class is where the algorithm becomes truly understandable.

### Initialization

The class starts with one cluster per sample.

Internally it tracks:

- cluster sizes,
- cluster centroids,
- active cluster IDs,
- labels for linkage-matrix compatibility,
- and the current cluster-distance representation.

### Pairwise distances

For non-Ward methods, it builds the full pairwise distance matrix using:

- `pdist(X)`
- `squareform(...)`

The diagonal is set to infinity so a cluster is never merged with itself.

### Merge search

At each iteration it loops over all active cluster pairs and finds the pair with the smallest cluster distance.

This is simple and transparent, though not optimized for very large `n`.

### Distance update

For single, complete, and average linkage, the code uses the **Lance–Williams recurrence**.

This is a standard recursive formula for updating distances after a merge without recomputing all pointwise distances from scratch.

### Why Lance–Williams matters

Suppose clusters `A` and `B` merge into `C = A ∪ B`.

Then you need the distance from `C` to every other active cluster `I`.

Lance–Williams gives a generic update rule in terms of:

- `d(A,I)`
- `d(B,I)`
- `d(A,B)`
- cluster sizes.

This is elegant because it shows that many linkage rules are just different parameter choices of one common recurrence family.

### Ward in the scratch class

Ward is handled separately through centroid and size logic rather than the same `_lw_update(...)` routine.

That is conceptually helpful because Ward is not just another “nearest/farthest/average” distance rule. It is a variance-based merge criterion.

### Output

At the end of `fit(...)`, the object stores:

- `self.Z_` — the linkage matrix.

Then `get_labels(K)` cuts that dendrogram into `K` flat clusters using `fcluster(...)`.

---

## How to read a linkage matrix

Every row in the linkage matrix corresponds to one merge.

For `n` original points:

- original samples have IDs `0, 1, ..., n-1`,
- the first merged cluster gets ID `n`,
- the next merged cluster gets ID `n+1`,
- and so on.

Each row is:

```text
[left_id, right_id, merge_height, new_cluster_size]
```

### Example idea

If a row is:

```text
[2, 8, 0.31, 2]
```

it means:

- point 2 and point 8 were merged,
- the merge happened at height 0.31,
- the new cluster has size 2.

If a later row is:

```text
[10, 9, 0.44, 3]
```

it means:

- the already-created cluster with ID 10 merged with point 9,
- at height 0.44,
- producing a cluster of size 3.

This is how the entire tree is encoded numerically.

---

## How to read a dendrogram

A dendrogram is one of the most information-rich plots in clustering.

### Leaves

The leaves at the bottom correspond to samples or truncated subclusters.

### Vertical axis

The y-axis is the merge height.

This is the dissimilarity level at which two branches join.

### Horizontal ordering

The left-right order is usually **not meaningful**. It may change without altering the clustering.

What matters is:

- who merges with whom,
- and at what height.

### Cutting the dendrogram

If you draw a horizontal line across the dendrogram:

- every connected component below that line is a cluster.

This is how a flat clustering is extracted from the hierarchy.

### Large vertical gaps

If there is a large gap between merge heights, it often indicates a natural scale separation.

Interpretation:

- many low merges = within-cluster consolidation,
- a large later merge = forcing distinct groups together.

This is why big jumps are often used as a cue for choosing `K`.

---

## Metrics used in this project

### Adjusted Rand Index (ARI)

ARI measures similarity between predicted cluster labels and true labels.

- `1.0` = perfect agreement,
- around `0` = random-level agreement,
- negative = worse than random after adjustment.

Important:

ARI is only available when true labels are known.

### Silhouette score

Silhouette measures how well each sample lies inside its assigned cluster compared with neighboring clusters.

Rough interpretation:

- high silhouette: compact and well-separated clusters,
- low silhouette: overlapping or poorly separated clusters.

Important caveat:

Silhouette favors **compact Euclidean clusters**. It can be misleading for non-convex shapes such as moons and circles.

### Cophenetic correlation

Cophenetic correlation compares:

- original pairwise distances,
- versus dendrogram cophenetic distances.

It measures how faithfully the tree preserves the pairwise geometry of the data.

High cophenetic correlation means:

- the dendrogram is a good summary of the data geometry.

Important:

It is **not** the same as label accuracy.

You can have:

- high cophenetic correlation but imperfect labels,
- or low silhouette but correct manifold clustering.

---

## Section-by-section walkthrough of the code

### Section A — Scratch agglomerative clustering

Purpose:

- teach the algorithm from the inside.

What happens:

- a tiny 10-point toy dataset is clustered,
- each linkage is run,
- the final `K=2` clustering is evaluated with ARI.

Why this matters:

- it demonstrates that very different merge criteria can still produce the same top-level partition on an easy dataset,
- while hiding differences in internal hierarchy.

### Section B — Linkage comparison

Purpose:

- compare single, complete, average, and Ward side-by-side.

What happens:

- the data are standardized,
- each linkage is computed using SciPy,
- truncated dendrograms are drawn,
- flat clusters are extracted at fixed `K`,
- ARI and silhouette are reported.

Why this matters:

- you see both the tree and the final clustering,
- you learn that the same dataset can look very different depending on linkage,
- and you learn that “best linkage” depends on data geometry.

### Section C — Dendrogram analysis

Purpose:

- move beyond “get labels” into “understand the tree.”

What happens:

- Ward dendrogram is computed,
- cophenetic correlation is measured,
- merge-distance jumps are visualized,
- a cophenetic scatter plot is drawn.

Why this matters:

- this section teaches how to use the hierarchy itself as an analytical object.

### Section D — Choosing `K`

Purpose:

- compare flat-clustering quality across different cut levels.

What happens:

- the Ward dendrogram is cut at `K = 2, 3, ..., K_max`,
- silhouette and ARI are plotted versus `K`.

Why this matters:

- it shows when geometry-based and truth-based criteria agree,
- and when they may disagree.

### Section E — Polymer conformational hierarchy

Purpose:

- show that hierarchical clustering is useful for scientific systems with nested states.

What happens:

- a polymer conformational dataset is loaded,
- a Ward dendrogram is computed,
- coarse `K=2` and finer `K=4` cuts are compared,
- true states and clustering results are visualized in PCA space,
- silhouette versus `K` is also shown.

Why this matters:

- real scientific data often have overlapping or nested states,
- hierarchy can be more meaningful than one flat partition.

### Section F — Performance heatmap

Purpose:

- compress the whole tutorial into one comparative summary.

What happens:

- each dataset is clustered with each linkage,
- ARI values are assembled into a matrix,
- the matrix is shown as a heatmap.

Why this matters:

- it reveals the geometry-to-linkage relationship at a glance.

---

## Detailed interpretation of each dataset

### 1. Isotropic blobs

These are compact, roughly spherical, well-separated clusters.

Expected behavior:

- all methods should work well,
- Ward should be especially natural.

What this dataset teaches:

- the easy textbook case,
- compact clusters align with silhouette,
- compact clusters align with Ward’s bias.

### 2. Anisotropic blobs

These clusters are elongated or stretched in some directions.

Expected behavior:

- complete, average, and Ward often work well,
- single linkage may suffer from chaining along elongated directions.

What this dataset teaches:

- not all non-spherical data are non-clusterable,
- but chaining becomes more dangerous.

### 3. Unequal blobs

These clusters differ in size, density, or both.

Expected behavior:

- single linkage can fail dramatically,
- average and Ward are often more robust.

What this dataset teaches:

- cluster imbalance is a serious challenge,
- geometric closeness alone is not enough.

### 4. Circles

These are non-convex ring structures.

Expected behavior:

- single linkage can be excellent,
- compactness-based methods fail badly.

What this dataset teaches:

- connectedness and compactness are different notions of clustering.

### 5. Moons

These are curved crescent-shaped manifolds.

Expected behavior:

- single linkage often succeeds,
- complete, average, and Ward may split each moon incorrectly.

What this dataset teaches:

- silhouette can prefer the wrong answer if the wrong answer is more compact.

### 6. Polymer conformations

These are scientifically motivated states defined by features such as:

- dihedral angles,
- radius of gyration,
- end-to-end distance.

Expected behavior:

- hierarchy may be real even if label recovery is imperfect,
- multiple scales of structure may coexist,
- a physically meaningful number of states may differ from the silhouette-optimal number.

What this dataset teaches:

- clustering in science is often about structure discovery, not perfect classification.

---

## How to choose `K`

Choosing `K` is one of the hardest parts of clustering.

This project teaches that there is no single universal answer.

### Method 1 — Dendrogram cuts

Look for large vertical gaps in the dendrogram.

Interpretation:

- low merges = within-cluster consolidation,
- large later jumps = between-cluster fusion.

A horizontal line in a large gap often gives a natural `K`.

### Method 2 — Silhouette vs `K`

Choose the `K` that maximizes silhouette.

Good when:

- clusters are compact and well-separated.

Be careful when:

- clusters are curved or non-convex,
- clusters contain physically meaningful substructure.

### Method 3 — ARI vs `K`

Use this only when true labels are available.

Useful for:

- benchmarking,
- synthetic datasets,
- validation studies.

### Most important lesson

The “best `K`” depends on what you mean by “best.”

- best geometric compactness,
- best match to ground truth labels,
- best physical interpretability,
- or best hierarchical summary.

These are not always the same.

---

## How to interpret the polymer hierarchy section

This section deserves special attention because it is the most realistic scientific example in the project.

### Why it is different from toy datasets

Toy datasets are often designed to be clean.

The polymer example is different because physical states can be:

- broad,
- overlapping,
- internally heterogeneous,
- and nested.

That means:

- moderate ARI does not imply failure,
- and a rising silhouette at larger `K` may indicate sub-basin structure rather than discovery of new macrostates.

### The `K=2` cut

This corresponds to a coarse conceptual split:

- compact-like versus extended-like conformations.

If ARI is moderate rather than perfect, that is still meaningful. It suggests the hierarchy captures a coarse physical organization but not a perfectly sharp binary separation.

### The `K=4` cut

This corresponds to a finer-grained decomposition into four states.

If ARI improves only modestly relative to `K=2`, the interpretation is often:

- the known physical states overlap in feature space,
- or the features are only partially sufficient to separate them.

### Silhouette in the polymer setting

If silhouette keeps increasing beyond the known four states, do **not** immediately conclude that the physical system truly has more than four states.

A more careful interpretation is:

- the conformational landscape may contain sub-basins,
- splitting broad states into tighter subclusters improves compactness,
- but those extra clusters may not correspond to distinct macrostates.

This is a very important scientific lesson.

---

## How to interpret the performance heatmap

The heatmap is the most compressed summary in the project.

It tells you which linkage criterion suits which geometry.

### Single linkage

Think of single linkage as the **connectivity-based** option.

Strengths:

- circles,
- moons,
- filamentary structures,
- manifold-like connected shapes.

Weaknesses:

- chaining,
- noisy bridges,
- unequal densities,
- accidental cluster connection.

### Complete linkage

Think of complete linkage as the **strict compactness** option.

Strengths:

- compact separation,
- anti-chaining behavior,
- often good on anisotropic but still separated blobs.

Weaknesses:

- may split curved or extended connected structures.

### Average linkage

Think of average linkage as the **middle-ground compromise**.

Strengths:

- often robust,
- less extreme than single or complete,
- good general-purpose behavior.

Weaknesses:

- still not ideal for strongly non-convex manifolds.

### Ward linkage

Think of Ward as the **variance-minimizing blob detector**.

Strengths:

- excellent on compact clusters,
- often strong on many practical structured datasets,
- frequently best when cluster identity is about low within-cluster spread.

Weaknesses:

- poor on circles and other strongly non-convex structures,
- imposes a compactness bias that may not match the data-generating geometry.

---

## Most important conceptual lessons

If you remember only a handful of ideas from this project, remember these.

### Lesson 1

Hierarchical clustering produces a **tree**, not just labels.

That tree contains more information than any single flat partition.

### Lesson 2

Different linkage methods define “cluster closeness” differently.

So the clustering outcome is not determined only by the data. It is also determined by the **geometry bias** of the linkage rule.

### Lesson 3

Single linkage is not “bad.” It is **specialized**.

It is excellent for connected manifolds, but fragile on unequal or noisy datasets.

### Lesson 4

Ward is not “universally best.” It is best when cluster identity is close to **compact variance-minimizing blobs**.

### Lesson 5

Silhouette is useful but not universal.

It is aligned with compact Euclidean clustering, not necessarily with the true geometry of non-convex structures.

### Lesson 6

High cophenetic correlation means the dendrogram is a good geometric summary. It does **not** automatically mean the clustering labels are perfect.

### Lesson 7

In scientific data, moderate clustering scores can still correspond to meaningful hierarchical organization.

---

## Pitfalls, caveats, and known issues

This section is especially important if you revisit the project later.

### 1. PCA is only for visualization

In the comparison and polymer plots, PCA is used to reduce the data to 2D for scatter plotting.

Important:

- clustering is done in the full standardized feature space,
- PCA does **not** determine the cluster assignments.

### 2. Standardization matters a lot

The project standardizes features before most serious analyses.

This prevents features with larger numerical scale from dominating Euclidean distance.

If you remove standardization, results may change substantially.

### 3. Tie-breaking in “best linkage”

If multiple linkage methods have identical ARI, the printed “best linkage” may simply reflect Python’s tie-breaking order in `max(...)`, not a meaningful superiority.

### 4. Silhouette can be misleading on non-convex data

This is especially important for circles and moons.

The correct manifold clustering may have a lower silhouette than an incorrect but more compact Euclidean partition.

### 5. The current automatic “distance acceleration” heuristic should be treated cautiously

In the dendrogram analysis section, the current heuristic based on reversed merge distances and `np.diff(...)` can produce nonsensical `K` suggestions on some datasets.

So:

- trust the dendrogram and the jump plot,
- do not blindly trust the current automatic `K_accel` number.

### 6. The header mentions connectivity constraints, but the base script may not actually implement that section

If you revisit the project and do not find a connectivity-constraint demonstration, that is because the script header is broader than the functions currently implemented.

### 7. Some imports may be unused in the base version

Depending on which script version you are using:

- `AgglomerativeClustering` may be imported but not used,
- `inconsistent` may be imported but not used.

That is not a conceptual issue, but it is useful to know when cleaning or extending the code.

---

## Suggested extensions

Once you fully understand the current project, the next natural extensions are the following.

### 1. Add scikit-learn `AgglomerativeClustering` analysis

Useful for:

- exploring `children_`,
- exploring `distances_`,
- comparing SciPy and scikit-learn representations,
- reconstructing dendrograms from estimator internals.

### 2. Add connectivity constraints

Useful for:

- spatial clustering,
- graph-constrained agglomeration,
- preventing implausible long-range merges.

### 3. Add alternative distance metrics

For example:

- cosine distance,
- Manhattan distance,
- angular distance,
- periodic-angle-aware distances for dihedral data.

### 4. Add optimal leaf ordering

This improves dendrogram readability without changing the cluster tree itself.

### 5. Add cluster-stability analysis

Useful for:

- bootstrap validation,
- subsample robustness,
- confidence in merges.

### 6. Add domain-aware polymer features

For scientific work, consider:

- periodic treatment of angles,
- contact maps,
- RMSD-based distances,
- torsional embedding,
- secondary-structure-aware descriptors.

---

## Revision guide: if you only have 10 minutes

Read only these sections:

1. [High-level overview of hierarchical clustering](#high-level-overview-of-hierarchical-clustering)
2. [Understanding the linkage methods](#understanding-the-linkage-methods)
3. [How to read a dendrogram](#how-to-read-a-dendrogram)
4. [How to choose `K`](#how-to-choose-k)
5. [Most important conceptual lessons](#most-important-conceptual-lessons)

If you do that, you will recover the backbone of the project.

---

## Revision guide: if you have 1 hour

Follow this order.

### Pass 1 — rebuild the concepts

Read:

- project purpose,
- overview of hierarchical clustering,
- linkage methods,
- metrics,
- dendrogram interpretation.

### Pass 2 — rebuild the code structure

Read:

- section-by-section walkthrough,
- scratch implementation,
- linkage matrix interpretation.

### Pass 3 — rebuild the scientific judgment

Read:

- dataset interpretations,
- polymer interpretation,
- performance heatmap,
- pitfalls and caveats.

### Pass 4 — then rerun the code

Run:

```bash
python hierarchical_v1.py
```

and compare the generated figures against what the README says they should mean.

If the figures make sense without surprise, you have recovered the project.

---

## Glossary

### Agglomerative clustering
Bottom-up clustering that starts with singleton clusters and repeatedly merges the closest pair.

### Dendrogram
Tree visualization of the complete merge history.

### Linkage criterion
Rule used to define distance between clusters.

### Single linkage
Cluster distance based on closest pair of points.

### Complete linkage
Cluster distance based on farthest pair of points.

### Average linkage
Cluster distance based on average cross-cluster distance.

### Ward linkage
Merge rule that minimizes increase in within-cluster variance.

### Linkage matrix
Numerical encoding of the hierarchical merge tree.

### Cophenetic distance
Dendrogram height at which two samples first become part of the same cluster.

### Cophenetic correlation
Correlation between original pairwise distances and cophenetic distances.

### Silhouette
Compactness-and-separation metric for a flat clustering.

### ARI
Adjusted Rand Index; label agreement measure corrected for chance.

### Chaining
Tendency of single linkage to connect samples through local bridges into long structures.

### Non-convex cluster
A cluster whose natural shape is curved or hollow rather than blob-like.

### Hierarchical structure
Nested organization where small groups combine into larger groups across scales.

---

## Final takeaway

This project teaches a very important principle:

> clustering is not just about applying an algorithm. It is about matching the algorithm’s geometric bias to the structure that is meaningful in your data.

If you remember that one sentence, you will already be thinking about hierarchical clustering the right way.
