# observations.md

# K-Means Clustering — Detailed Observations and Experimental Analysis

## Project overview

This project studies **K-means clustering** from both a **theoretical** and a **practical** perspective. The workflow includes:

- a **scratch implementation** of K-means using NumPy
- a comparison with **scikit-learn's KMeans**
- an empirical study of **initialization strategies**
- a systematic study of **how to choose the number of clusters \(K\)**
- a detailed examination using **silhouette plots**
- a visualization of **failure modes**
- a runtime-quality comparison between **standard KMeans** and **MiniBatchKMeans**

The experiments show not only where K-means performs extremely well, but also where it fails due to geometric assumptions built into the algorithm.

---

# 1. Core understanding developed through this project

K-means is a centroid-based clustering algorithm that partitions a dataset into \(K\) groups by minimizing the **within-cluster sum of squares (WCSS)**, also called **inertia**. The algorithm works by repeatedly:

1. assigning each point to the nearest centroid
2. recomputing each centroid as the mean of the points assigned to it
3. stopping when centroid movement becomes sufficiently small

This project makes one thing especially clear:

> K-means is highly effective when the data contains compact, well-separated, approximately spherical clusters, but its performance degrades when the data geometry violates those assumptions.

That single statement is supported throughout all experiments in this repository.

---

# 2. Validation of the scratch implementation

## Observation

The NumPy-based `KMeansScratch` implementation produces the same result as `sklearn.cluster.KMeans` on the `blobs_easy` dataset.

Reported result:

- Scratch inertia = `24.8783`
- sklearn inertia = `24.8783`
- Scratch ARI = `1.0000`
- sklearn ARI = `1.0000`

## Interpretation

This is a strong validation of the scratch implementation.

Matching inertia means:

- both implementations are optimizing the same objective
- the distance computations, assignments, centroid updates, and stopping logic are correct
- the custom implementation reaches the same final local optimum as sklearn on this dataset

Matching ARI means:

- both methods recover the true cluster structure perfectly
- the grouping itself is correct, not just the objective value

## Important nuance

The exact numeric labels assigned to clusters do not matter. Cluster `0` in one run may correspond to cluster `2` in another run. Metrics like **ARI** handle label permutation correctly, so ARI = 1 means the clustering structure is exactly recovered even if cluster IDs differ.

## Practical takeaway

The scratch implementation is not only pedagogically useful but also empirically correct on a clean benchmark. This gives confidence in the manual implementation before moving on to more advanced analysis.

---

# 3. Behavior on the easy benchmark dataset

## Dataset: `blobs_easy`

This dataset consists of:

- 3 compact Gaussian blobs
- clear separation between clusters
- approximately isotropic shape
- balanced cluster geometry

This is the near-ideal environment for K-means.

## Observation

Every major experiment on this dataset indicates that K-means is an excellent fit.

Evidence:

- perfect ARI
- perfect AMI
- high silhouette score
- clear elbow at \(K=3\)
- unanimous agreement across all K-selection methods
- highly stable performance under k-means++ initialization

## Interpretation

`blobs_easy` satisfies the assumptions of K-means unusually well:

- each cluster can be represented by a centroid
- Euclidean distance is meaningful
- within-cluster spread is small
- between-cluster separation is large

This dataset is therefore an ideal pedagogical benchmark for understanding how K-means behaves under favorable conditions.

---

# 4. Initialization matters: random vs k-means++

## Experiment summary

The project compares two initialization strategies across many repeated runs:

- `init="random"`
- `init="k-means++"`

with `n_init=1` in each case so that each run reflects a single initialization event.

Reported result:

- random: mean inertia = `119.47 ± 178.41`
- k-means++: mean inertia = `24.88 ± 0.00`

Average convergence iterations:

- random ≈ `4.3`
- k-means++ ≈ `2.0`

## Main observation

Initialization has a **very strong effect** on both:

- final clustering quality
- optimization speed

## What the results mean

### Random initialization

Random initialization shows:

- very large variability in final inertia
- occasional convergence to the optimal solution
- repeated convergence to much worse local minima
- slower average convergence

This suggests that random initialization often places multiple centroids in the same true cluster or fails to place an initial centroid in an important region of the data.

### K-means++

K-means++ shows:

- nearly zero variability in final inertia
- consistently optimal or near-optimal clustering
- much faster convergence
- dramatically improved robustness

This indicates that the k-means++ initialization successfully spreads the starting centroids across the natural data structure.

## Important nuance

The success of k-means++ here is unusually clean because the dataset itself is easy. On more complex datasets, k-means++ will still usually help, but it may not always collapse to a single perfect final inertia value the way it does here.

## Practical takeaway

K-means++ should be preferred over naive random initialization in almost all practical use cases.

---

# 5. sklearn KMeans performance summary

The sklearn demo reports:

- inertia = `24.8783`
- iterations = `2`
- ARI = `1.0000`
- AMI = `1.0000`
- silhouette = `0.8751`

## Interpretation of each number

### Inertia = 24.8783

This is the final within-cluster sum of squares on the scaled dataset.

Interpretation:
- the clusters are very compact
- points are close to their assigned centroids
- the K-means objective is minimized very effectively

### Iterations = 2

Convergence in only 2 iterations indicates:

- the initialization was already very close to the final optimum
- the data structure is simple and strongly clusterable
- the optimization landscape is easy for K-means on this dataset

### ARI = 1.0000

The clustering matches the true labels perfectly.

### AMI = 1.0000

Mutual information agreement is also perfect, confirming that the learned partition and the true partition are identical in grouping structure.

### Silhouette = 0.8751

This is a very high value.

Interpretation:
- points are much closer to their own cluster than to neighboring clusters
- the separation between groups is very strong
- cluster assignments are highly reliable

## Practical takeaway

This is an almost ideal K-means result. The algorithm is not merely acceptable here; it is exceptionally well matched to the geometry of the data.

---

# 6. Choosing the number of clusters: detailed analysis of `choose_K_easy_blobs.png`

This is one of the most important outputs of the project.

The following methods were compared for \(K = 2, 3, \dots, 8\):

- Elbow
- Silhouette
- Calinski-Harabasz
- Davies-Bouldin
- Gap statistic

Reported result:

- Elbow: `K = 3`
- Silhouette: `K = 3`
- Calinski-Harabasz: `K = 3`
- Davies-Bouldin: `K = 3`
- Gap statistic: `K = 3`

## Why this result is significant

All five methods agree on the same answer. That means the conclusion is supported from multiple independent perspectives:

- compactness
- separation
- between/within-cluster ratio
- cluster similarity penalty
- comparison to random-reference clustering

This level of agreement is unusually strong and indicates that the dataset has a very clear natural clustering structure.

---

## 6.1 Elbow method panel

### Observation

The WCSS curve shows:

- a very large drop from \(K=2\) to \(K=3\)
- much smaller decreases for \(K > 3\)

### Interpretation

This is the classic elbow pattern.

At \(K=2\):
- one centroid is forced to represent what are actually two distinct real blobs
- within-cluster dispersion remains large

At \(K=3\):
- each true blob can get its own centroid
- WCSS decreases dramatically

For \(K>3\):
- extra centroids mostly split already compact clusters
- improvement continues, but only marginally

### Nuance

Inertia always decreases with increasing \(K\), so the best \(K\) is **not** found by minimizing inertia directly. Instead, one looks for the point where the improvement becomes much less significant. In this dataset, that turning point is very clearly \(K=3\).

---

## 6.2 Silhouette score panel

### Observation

Approximate values:

- \(K=2\): `0.703`
- \(K=3\): `0.875`
- \(K=4\): `0.701`
- \(K=5\): `0.536`
- \(K>=6\): about `0.35`

### Interpretation

This panel provides some of the clearest evidence for the correct number of clusters.

At \(K=3\):
- silhouette reaches its maximum
- the increase relative to \(K=2\) is large
- cluster assignments are clearly strongest here

At \(K=2\):
- the score is still reasonably good
- but not optimal
- this suggests that merging two true blobs still gives a somewhat coherent partition, though not the true one

At \(K=4\):
- the score returns to about the same level as \(K=2\)
- but now the error is different: instead of merging true groups, K-means begins splitting a natural group

At \(K=5\) and above:
- silhouette drops substantially
- this indicates over-fragmentation
- points become closer to rival subclusters inside the same original blob

### Nuance

A very important lesson emerges here:

> A wrong value of \(K\) can still produce a respectable silhouette score.

For example, both \(K=2\) and \(K=4\) produce values around `0.70`, but those correspond to different structural mistakes:
- \(K=2\): under-clustering by merging groups
- \(K=4\): over-clustering by splitting groups

This is why the **full silhouette plots** are essential and not just the average score.

---

## 6.3 Calinski-Harabasz panel

### Observation

The CH index peaks sharply at \(K=3\).

### Interpretation

The Calinski-Harabasz score rewards:

- high between-cluster separation
- low within-cluster dispersion

At \(K=3\), that trade-off is optimal:
- each true group is captured cleanly
- clusters are tight and far apart

At \(K=2\):
- within-cluster spread is too large because two true groups are merged

At \(K>3\):
- splitting compact true groups does not create meaningful new between-cluster structure
- the score declines

### Practical takeaway

The CH index strongly supports the interpretation that the natural structure of the data is exactly three clusters.

---

## 6.4 Davies-Bouldin panel

### Observation

The Davies-Bouldin index has a clear minimum at \(K=3\).

### Interpretation

Davies-Bouldin measures how similar clusters are to one another, combining:
- internal spread
- distance to other clusters

Lower values are better.

At \(K=3\):
- clusters are compact
- clusters are far from one another
- similarity between clusters is minimized

At larger \(K\):
- one or more true clusters are split into nearby subclusters
- these subclusters become similar and close to each other
- the DB index rises

### Nuance

This metric is particularly good at penalizing unnecessary splits. That is why it becomes much worse once \(K\) exceeds the natural number of groups.

---

## 6.5 Gap statistic panel

### Observation

The Gap statistic jumps strongly from \(K=2\) to \(K=3\), then becomes approximately flat or slightly lower afterward.

### Interpretation

The Gap statistic asks:

> How much better does the real data cluster than random reference data?

At \(K=3\), the answer becomes substantially stronger than at \(K=2\).

That means:
- three-cluster structure is genuinely meaningful
- the dataset is not merely being partitioned well because of random geometry
- the improvement at \(K=3\) is structurally real

For \(K>3\):
- real data still clusters somewhat better than random data
- but the additional benefit over \(K=3\) is not substantial

### Important nuance

The standard gap rule prefers the **smallest sufficient K**, not necessarily the absolute largest raw gap value. That makes it conservative and helps avoid overfitting the number of clusters.

### Practical takeaway

The Gap statistic provides a more statistically grounded confirmation that \(K=3\) is the correct choice.

---

## 6.6 Consensus / voting panel

### Observation

All five methods vote for `K = 3`.

### Interpretation

This is a very strong result.

Each method is measuring something different:
- Elbow measures diminishing returns in WCSS reduction
- Silhouette measures pointwise assignment quality
- CH measures between/within spread ratio
- DB measures inter-cluster similarity
- Gap compares real clustering to randomness

When all of them agree, the evidence for the chosen \(K\) becomes very compelling.

## Final conclusion from `choose_K_easy_blobs.png`

The dataset has a clear and unambiguous natural clustering structure at:

\[
K = 3
\]

This conclusion is supported simultaneously by five independent cluster-selection principles.

---

# 7. Detailed analysis of `silhouette_easy_blobs.png`

This is arguably the most instructive diagnostic figure in the project.

It shows silhouette bands for:

- \(K=2\)
- \(K=3\)
- \(K=4\)
- \(K=5\)

A silhouette plot is more informative than a single average silhouette number because it reveals:

- the distribution of pointwise assignment quality
- cluster balance
- weak clusters
- tails of poorly assigned points
- whether problems are global or localized

---

## 7.1 K = 2

Mean silhouette ≈ `0.703`

### Observation

The clustering looks reasonably good on average.

Most points appear to have positive silhouette values, and many are moderately large.

### Interpretation

This indicates that even with only 2 clusters, the partition is not completely unnatural. That makes sense because the three true blobs are well separated, so K-means can merge two true blobs into one larger cluster while still preserving some overall separation.

### Nuance

This is a very important lesson:

> A clustering can have a decent silhouette score and still be wrong.

At \(K=2\), the algorithm is under-clustering:
- it is merging two true groups
- one resulting cluster is doing too much representational work

This gives a reasonably good but not truly correct partition.

### What to learn from it

Do not interpret a silhouette score around `0.70` as automatically “correct.” It can still represent a merged-cluster solution.

---

## 7.2 K = 3

Mean silhouette ≈ `0.875`

### Observation

This is the strongest panel by far.

The silhouette bands are:
- wide
- consistently shifted toward high values
- balanced across clusters
- almost entirely positive
- close to the right side of the axis

### Interpretation

This indicates an almost ideal clustering:

- each cluster is highly coherent internally
- each cluster is strongly separated from the others
- points overwhelmingly belong much more strongly to their assigned cluster than to any alternative cluster

### Why this is better than just “highest mean”

The average silhouette is excellent, but more importantly:

- all three clusters look strong
- there is no weak cluster dragging the quality down
- there are essentially no dubious assignments

This means the success is not driven by only one or two strong clusters. It is a genuinely high-quality partition across the entire dataset.

### Practical conclusion

This panel provides the strongest evidence that \(K=3\) is the natural clustering.

---

## 7.3 K = 4

Mean silhouette ≈ `0.701`

### Observation

The mean score is close to that of \(K=2\), but the structure is different.

### Interpretation

This is an example of **over-clustering**.

K-means likely takes one of the true blobs and splits it into two nearby artificial subclusters. Those two subclusters are each internally coherent enough to keep the silhouette from collapsing, but they are close to each other, so separation is weaker.

### Nuance

This is one of the deepest lessons in the silhouette analysis:

> Similar average silhouette values can correspond to very different kinds of mistakes.

- \(K=2\): not enough clusters, true groups merged
- \(K=4\): too many clusters, true groups split

So the mean score alone is not enough. The band structure matters.

### What to learn from it

A silhouette score around `0.70` is not enough to conclude that \(K=4\) is acceptable. The full plot shows that the internal geometry is weaker than at \(K=3\).

---

## 7.4 K = 5

Mean silhouette ≈ `0.536`

### Observation

This panel is clearly weaker.

The silhouette bands become:
- more fragmented
- more uneven
- shifted leftward
- less convincing overall

### Interpretation

At this point, K-means is strongly over-fragmenting the natural structure.

The original compact blobs are being carved into several smaller centroid-based regions. These artificial subclusters are not far from one another, so points become less confidently assigned.

### Important nuance

A drop from `0.875` to `0.536` is major.

Even if many values remain positive, the clustering quality has degraded substantially:
- cohesion is reduced
- separation is reduced
- the partition no longer matches the true natural grouping

### Practical takeaway

Values of \(K\) larger than the true natural count quickly create artificial structure that is mathematically legal for K-means but not meaningful for the data.

---

## Final conclusion from `silhouette_easy_blobs.png`

The silhouette figure teaches several important nuances:

1. \(K=3\) is clearly optimal, both in mean score and in full cluster-by-cluster quality.
2. \(K=2\) is an under-clustered solution that still looks moderately good on average.
3. \(K=4\) is an over-clustered solution that can look deceptively acceptable if only the average is considered.
4. \(K=5\) and beyond clearly degrade the structure.

### Most important lesson

> The full silhouette plot is much more informative than the mean silhouette score alone.

It shows **how** a solution is wrong, not just **how much** it is wrong.

---

# 8. Failure modes: what the experiments teach

The failure-mode figure studies four datasets:

- concentric circles
- two moons
- anisotropic blobs
- unequal-size blobs

This section is crucial because it shows that K-means has geometric biases.

---

## 8.1 Concentric circles

Reported ARI ≈ `-0.002`

### Interpretation

This is essentially complete failure.

Why?
- the true clusters are nested rings
- both rings have nearly the same center
- K-means can only partition based on centroid distance
- so it cuts the rings in an angular or wedge-like fashion rather than separating inner vs outer ring

### Lesson

K-means cannot represent nested non-convex structure.

---

## 8.2 Two moons

Reported ARI ≈ `0.445`

### Interpretation

This is a partial failure.

Why?
- the true clusters are curved and interleaved
- K-means prefers straight-line / Voronoi-style partitions
- it can capture some rough separation, but not the correct nonlinear geometry

### Lesson

K-means struggles on non-convex manifolds.

---

## 8.3 Anisotropic blobs

Reported ARI ≈ `0.980`

### Interpretation

This result is interesting because it is much better than one might expect.

The dataset was included as a stress case because elongated clusters can challenge K-means, but here the elongation is not severe enough to destroy centroid-based separability. The means remain well separated, so K-means still performs very well.

### Lesson

“K-means struggles on anisotropic data” is a tendency, not an absolute rule.

---

## 8.4 Unequal-size blobs

Reported ARI ≈ `0.684`

### Interpretation

This is a moderate failure.

The large diffuse cluster is not handled as well as the compact clusters. K-means tends to split or distort large spread-out clusters because it prefers balanced Voronoi-style partitions.

### Lesson

K-means is sensitive to cluster size imbalance and density imbalance.

---

# 9. MiniBatchKMeans analysis

Reported result:

- Full KMeans: `0.38s`, inertia = `308565.7`
- MiniBatchKMeans: `0.05s`, inertia = `309405.3`
- Speed-up = `7.5×`
- Inertia loss = `0.27%`
- ARI agreement = `0.9617`

## Interpretation

This is an excellent trade-off.

MiniBatchKMeans is much faster while preserving almost all clustering quality.

### Speed-up

A 7.5× speed-up is substantial.

### Inertia loss

Only `0.27%` worse than full KMeans means the approximation is extremely accurate in terms of the K-means objective.

### ARI agreement

An ARI of `0.9617` between the two clusterings means the resulting partitions are very similar.

## Practical takeaway

MiniBatchKMeans is a strong practical alternative for large datasets when speed matters.

---

# 10. Main conceptual lessons learned from the project

## 10.1 K-means is powerful when the geometry is right

It works extremely well for:
- compact clusters
- spherical or near-spherical groups
- balanced densities
- centroid-separable structure

## 10.2 Initialization matters a lot

- random initialization can be unstable
- k-means++ dramatically improves both quality and convergence speed

## 10.3 Choosing K should be evidence-based

Multiple metrics should be checked together. On clean datasets, they may agree strongly. On harder datasets, disagreement can itself be informative.

## 10.4 Silhouette mean is useful but not sufficient

The full silhouette plot gives much deeper insight than the average score alone.

## 10.5 K-means fails for understandable geometric reasons

Its failure is not random. It struggles when the data violates its centroid-and-Euclidean assumptions.

## 10.6 Approximate clustering can be very valuable

MiniBatchKMeans demonstrates that in large-scale settings, near-optimal clustering can often be obtained much faster.

---

# 11. Final conclusion

This project successfully demonstrates K-means from first principles to practical usage.

The experiments show:

- the scratch implementation is correct
- sklearn behavior is well understood
- k-means++ is far superior to naive random initialization
- the correct number of clusters in the easy benchmark is unambiguously \(K=3\)
- silhouette analysis reveals not only the best K but also the character of wrong K values
- failure cases illustrate the geometric limits of K-means
- MiniBatchKMeans offers an excellent practical speed-quality trade-off

Overall, the project gives a strong conceptual and practical foundation for understanding when K-means should be trusted, when it should be questioned, and how to diagnose its behavior rigorously.
