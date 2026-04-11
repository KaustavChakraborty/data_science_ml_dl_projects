# Observations.md — Hierarchical Clustering Project

## Purpose of this document

This file records the **main empirical observations, interpretations, patterns, successes, failures, and cautionary lessons** from the hierarchical clustering project. It is written as a long-form revision document so that, at a much later time, I can come back to this project and quickly recover:

- what each experiment was trying to show,
- what the outputs actually showed,
- what the linkage methods were doing,
- why some methods worked and others failed,
- how to interpret dendrograms, ARI, silhouette, and cophenetic correlation together,
- and what broader clustering principles emerged from the full set of results.

This is **not** a code walkthrough. The README is for the full project structure and theory. This file is specifically for the **observed outcomes and their interpretation**.

---

## Big-picture conclusion of the whole project

The single most important conclusion from this project is:

> **There is no universally best linkage method. The best linkage depends on the geometry that defines a meaningful cluster in the dataset.**

Across all experiments, the results showed a very clear pattern:

- **Ward linkage** is excellent when the true clusters are compact, blob-like, and variance-defined.
- **Single linkage** is uniquely powerful when the true clusters are non-convex connected structures such as circles or moons.
- **Complete linkage** is often strong when one wants compact clusters and resistance to chaining.
- **Average linkage** is frequently the most balanced compromise across different dataset types.
- Metrics such as **silhouette** can be very helpful on convex cluster structure, but can become misleading on non-convex manifolds.
- Dendrograms are valuable not only for flat clustering, but also for understanding **coarse-to-fine hierarchy** in the data.

A second major lesson is:

> **Good clustering must always be interpreted relative to the structure one is trying to recover.**

If the desired notion of a cluster is “compact low-variance group,” Ward is often appropriate.
If the desired notion is “connected curved manifold,” single linkage may be the only correct method.
If the desired notion is “physically interpretable states with internal substructure,” dendrogram interpretation may matter more than flat labels.

---

# 1. Scratch agglomerative clustering on the tiny toy dataset

## Main outcome

On the tiny 10-point toy dataset, all four linkage methods produced perfect 2-cluster recovery:

- single: ARI = 1.000
- complete: ARI = 1.000
- average: ARI = 1.000
- ward: ARI = 1.000

The scikit-learn-based analysis also confirmed this and showed that all methods produced:

- `K = 2`
- `ARI = 1.000`
- `Silhouette ≈ 0.724`

## Interpretation

This toy dataset had a very strong and visually obvious two-cluster structure:

- one small lower-left group,
- one upper-right group,
- clear separation between them,
- and no ambiguous bridge points.

Because the top-level separation was so strong, all linkage methods agreed on the same final flat partition when cut at `K=2`.

## Important observation

Although the final labels were identical, the **merge histories were not identical**.

This is a key lesson:

> Different linkage rules can produce different dendrograms but the same flat clustering at a chosen K.

This is especially important when teaching hierarchical clustering. Agreement in ARI at one chosen K does **not** imply the full trees are the same.

## Single-linkage-specific observation on the toy set

Single linkage tended to grow clusters through local nearest-neighbor bridges. This was visible in its merge ordering and merge-distance trajectory.

This confirmed the classic chaining behavior of single linkage.

On this simple dataset, chaining did not cause failure, because the two big groups were so well separated that local bridges stayed within the correct macro-cluster.

## Main lesson from the toy example

The toy example was ideal for understanding the mechanics of hierarchical clustering because it showed:

- what a linkage matrix is doing,
- what a dendrogram is encoding,
- how cluster labels are extracted by cutting the tree,
- and how different linkage definitions can agree at the coarse level while differing internally.

---

# 2. Linkage comparison on isotropic blobs

## Main numerical result

All four linkage methods achieved perfect recovery:

- single = 1.000
- complete = 1.000
- average = 1.000
- ward = 1.000

Silhouette at `K=3` was also very high:

- silhouette ≈ 0.875

## Interpretation

This dataset is the classic easy case for clustering:

- three compact blobs,
- nearly isotropic shape,
- strong separation,
- no curved structure,
- no strong density imbalance,
- no bridges or chaining paths.

In this geometry, essentially every reasonable agglomerative linkage can recover the correct partition.

## Important observation about the “best linkage” line

The code reported:

> Best linkage: single

But this should **not** be interpreted as a scientific preference for single linkage on isotropic blobs.

It was only the consequence of tie-breaking in the code: since all ARIs were equal, Python returned the first method in the dictionary/order.

Scientifically, this result means:

> All linkage methods tied perfectly on isotropic blobs.

## What this tells me about silhouette

Silhouette is highly trustworthy here because the dataset matches the assumptions behind silhouette:

- clusters are compact,
- separation is strong,
- cluster shape is roughly convex.

So the very high silhouette is fully consistent with the perfect ARI.

## Deeper conceptual lesson

On clean isotropic blobs:

- linkage choice matters very little,
- both label-based and geometry-based metrics agree,
- dendrogram structure is stable,
- K-selection is easy,
- and the dataset behaves like the ideal textbook case.

This dataset serves as the “control” experiment for the project.

---

# 3. Linkage comparison on two moons

## Main numerical result

The methods behaved very differently:

- single = 1.000
- complete = 0.483
- average = 0.583
- ward = 0.583

The silhouette scores did **not** align with ARI in the way one might expect:

- single had the best ARI, but lower silhouette than some incorrect solutions.

## Interpretation

This was one of the most important results in the whole project.

The two-moons dataset is **non-convex**. The true clusters are not compact round blobs. Each cluster is an extended curved manifold.

That means the correct clustering is based on **connectedness along the moon**, not on compact Euclidean grouping.

## Why single linkage was perfect

Single linkage only asks for the closest cross-cluster pair. That allows it to chain along each moon through local neighbor connections.

This is exactly the right inductive bias for the two-moons dataset.

Usually, chaining is treated as a weakness of single linkage. But on two moons, it becomes the central reason for success.

So this result taught a very important lesson:

> A property that is a weakness on one dataset can be the decisive strength on another.

## Why complete / average / Ward were imperfect

These methods prefer more compact clusters.

- Complete is strict and dislikes elongated shapes.
- Average is more moderate but still Euclidean/compactness-oriented.
- Ward explicitly prefers low-variance compact groups.

Since each moon is long and curved, those methods tend to split the moons into geometrically compact pieces rather than preserving the entire moon as a single cluster.

Hence the lower ARIs.

## Important lesson about silhouette on non-convex data

This experiment showed clearly that silhouette can be misleading.

The correct moon clustering is not especially compact in Euclidean space, so silhouette does not reward it as strongly as one might expect.

Some geometrically incorrect solutions can obtain comparable or even better silhouette because they partition the moons into tighter convex chunks.

Therefore:

> On non-convex manifold datasets, silhouette should not be trusted blindly.

This is one of the most important cautionary notes in the whole project.

## Main lesson from two moons

The result demonstrates that:

- cluster geometry matters more than metric scores alone,
- manifold-shaped clusters need connectivity-aware linkage,
- and single linkage can be exactly right when cluster identity is about connected shape.

---

# 4. Dendrogram analysis on isotropic blobs (Ward)

## Main numerical result

- Cophenetic correlation (Ward) = 0.9841
- Single = 0.9682
- Complete = 0.9825
- Average = 0.9853
- Ward = 0.9841

The dendrogram showed very large jumps at the final merges.

The current “distance acceleration” heuristic, however, gave a nonsensical suggested K of 558.

## Interpretation of the dendrogram

The dendrogram showed:

- many low-cost within-blob merges,
- then a large late-stage jump when forced to merge the three main blobs into fewer groups.

This is exactly what one wants for a clean 3-cluster dataset.

The meaningful structure is:

- low merge heights = merges within a true blob,
- high late merge heights = forced merges between different true blobs.

Thus the dendrogram strongly supported `K=3`.

## Interpretation of the cophenetic correlation

A cophenetic correlation near 0.984 means the dendrogram represents the original pairwise distance geometry extremely well.

This was expected because isotropic blobs are clean, well separated, and highly compatible with a tree-like hierarchical representation.

It is notable that all methods achieved very high cophenetic correlations on this dataset. That means the dataset itself is very friendly to hierarchical summarization.

## Important caution: the K-acceleration heuristic was flawed

The printed value `K = 558` is obviously meaningless for a dataset with 3 visible blobs.

This means the implementation of the automatic K-selection heuristic based on reversed merge-distance differences was not conceptually correct.

This is a crucial observation for future revision:

> The dendrogram itself was excellent, but the current automatic “distance acceleration” K-selection formula was not trustworthy.

Therefore the plot should be interpreted visually and in conjunction with silhouette/ARI, not through that printed K value.

## Main lesson

This section showed that dendrograms are powerful for interpretation, but automatic heuristics built on top of them must be validated carefully.

---

# 5. K selection on isotropic blobs

## Main numerical result

- Best K by silhouette = 3 (0.875)
- Best K by ARI = 3 (1.000)

## Interpretation

This result was as clean as possible.

The silhouette curve and the ARI curve both peaked at exactly the true number of clusters.

This means:

- the correct clustering is also the most geometrically natural one,
- the dataset has a clear and stable cluster count,
- and unsupervised and supervised criteria agree completely.

## Detailed reading of the curves

### At K = 2

Silhouette is decent, but lower than at 3.
ARI is also much lower.

This means that with only 2 clusters, the model is forced to merge two genuine blobs together.

### At K = 3

This is the optimum:

- each true blob gets one cluster,
- compactness is maximized,
- separation is maximized,
- and labels match perfectly.

### At K > 3

Both silhouette and ARI decline.

This indicates over-segmentation:

- true blobs are being split into multiple artificial subclusters,
- which hurts label agreement,
- and also reduces cluster separation because nearby subclusters now compete with each other.

## Main lesson

The isotropic K-selection plot is the clean textbook demonstration of:

- under-clustering for too-small K,
- optimal clustering at the natural K,
- over-clustering for too-large K.

---

# 6. Polymer conformation hierarchy

## Main numerical result

- Cophenetic r (Ward) = 0.7475
- ARI at K=2 (compact vs extended) = 0.6197
- ARI at K=4 (all states) = 0.6479

The silhouette curve kept increasing at larger K and reached its best values well above 4.

## High-level interpretation

This was the most scientifically realistic dataset in the project.

Unlike blobs or moons, the polymer conformational data were not expected to form perfectly isolated idealized clusters.

Instead, the data appeared to contain:

- coarse physical hierarchy,
- overlapping or partially mixed states,
- diffuse basins,
- sub-basin structure within larger states,
- and only moderate separability of the labeled classes.

## Interpretation of the dendrogram

The dendrogram suggested a genuine **two-level hierarchy**:

- a broad coarse split at the top,
- with finer subdivision lower down.

This is physically meaningful because conformational ensembles often organize in nested ways:

- broad macrostate classes,
- within which there are more specific conformational subtypes.

So the dendrogram was valuable not because it perfectly reproduced labels, but because it revealed a plausible physical hierarchy.

## Interpretation of K = 2 result

At `K=2`, the clustering was compared to a coarse binary interpretation: compact vs extended.

ARI ≈ 0.62 means the coarse split is partially meaningful but not perfect.

This suggests that the data do have a broad binary organization, but the mapping from that binary split to the true physical labels is not exact.

That makes sense in a conformational setting because:

- some states may share compactness characteristics but differ in dihedrals,
- some states may overlap in one observable and differ in another,
- and broad physical categories are usually fuzzy rather than perfectly separable.

## Interpretation of K = 4 result

At `K=4`, the ARI improves only slightly to ≈ 0.648.

This is very important.

If the four labeled states were cleanly separated in the chosen feature space, the ARI should have risen dramatically at K=4.
Instead, the improvement is modest.

This tells me that:

- the labels are real and meaningful,
- but they are not sharply isolated in the chosen features,
- and hierarchical clustering cannot recover them perfectly because the state distributions overlap.

## Interpretation of the PCA panels

The PCA plots visually supported this conclusion.

Some states appeared compact and distinct, while others were broad, diffuse, or partially overlapping.

Therefore the moderate ARIs are not evidence of algorithm failure. Rather, they reflect genuine structure in the data:

- some states are easy,
- some are broad ensembles,
- some are internally heterogeneous.

## Interpretation of the silhouette-vs-K curve

This panel is easy to misread unless stated explicitly.

Silhouette continued to improve for K greater than 4, even reaching its best values near K=7.

This does **not** mean the system truly contains seven physical macrostates.

More likely, it means:

- broad conformational states contain sub-basin structure,
- splitting those into tighter geometric clusters increases silhouette,
- but those extra clusters may not correspond to distinct named physical states.

This is a crucial physical interpretation lesson:

> Higher silhouette at larger K can reflect geometric substructure, not necessarily new physically meaningful macrostates.

## Interpretation of cophenetic correlation

A cophenetic correlation of 0.7475 is respectable but far below the values for isotropic blobs.

This tells me that the polymer data are only moderately tree-like.

That is exactly what I would expect for a real conformational landscape:

- partially hierarchical,
- partially overlapping,
- not perfectly representable by a simple dendrogram.

## Main lesson from the polymer section

This was the most realistic demonstration that hierarchical clustering is not always about recovering one “correct” flat labeling.

Sometimes the important output is the **hierarchy itself**:

- broad coarse division,
- finer substructure,
- imperfect separability of states,
- and the possibility that geometric clustering reveals sub-basin structure beyond named classes.

---

# 7. Performance heatmap — linkage × dataset

## Main numerical result

### Isotropic
- single = 1.000
- complete = 1.000
- average = 1.000
- ward = 1.000

### Anisotropic
- single = 0.568
- complete = 0.975
- average = 0.956
- ward = 0.961

### Unequal
- single = -0.004
- complete = 0.484
- average = 0.800
- ward = 0.822

### Circles
- single = 1.000
- complete = 0.175
- average = 0.129
- ward = 0.020

### Moons
- single = 1.000
- complete = 0.483
- average = 0.583
- ward = 0.583

## Interpretation: the heatmap is a geometry-to-linkage dictionary

This heatmap is one of the most important summaries of the entire project.

It shows that each linkage has a characteristic inductive bias.

---

## 7.1 Single linkage

### Observed pattern

Single linkage was:

- perfect on isotropic blobs,
- mediocre on anisotropic blobs,
- catastrophic on unequal blobs,
- perfect on circles,
- perfect on moons.

### Interpretation

Single linkage is the linkage of **connectedness** and **local chaining**.

That makes it excellent for non-convex manifolds such as circles and moons, because the true cluster identity is about being connected along a shape.

However, that same property makes it fragile on:

- unequal density,
- bridge-like noise,
- elongated chains between groups,
- and datasets where one wants compact rather than connected clusters.

### Main lesson

Single linkage is powerful but specialized.

It should be chosen when connected geometry is the target notion of a cluster, not when compactness is the goal.

---

## 7.2 Complete linkage

### Observed pattern

Complete linkage was:

- perfect on isotropic blobs,
- excellent on anisotropic blobs,
- only moderate on unequal blobs,
- poor on circles,
- mediocre on moons.

### Interpretation

Complete linkage strongly resists chaining because it judges cluster distance by the farthest cross-cluster pair.

This makes it good at preserving compact, tight, separated groups.

That is why it performed strongly on isotropic and anisotropic data.

However, because it dislikes elongated or curved clusters, it performed poorly on circles and moderately on moons.

### Main lesson

Complete linkage is useful when one wants compact clusters and wants to suppress chaining, but it is not suitable for strongly non-convex manifold structure.

---

## 7.3 Average linkage

### Observed pattern

Average linkage was:

- perfect on isotropic blobs,
- excellent on anisotropic blobs,
- strong on unequal blobs,
- poor on circles,
- intermediate on moons.

### Interpretation

Average linkage behaved like the most balanced method across datasets.

It was rarely the absolute best, but often robust.

It combines some resistance to chaining with less severity than complete linkage.

That is why it performed well on anisotropic and unequal data, but still could not fully handle non-convex manifold datasets like circles and moons.

### Main lesson

Average linkage is a strong compromise method when dataset geometry is not obviously manifold-shaped and one wants a robust middle ground.

---

## 7.4 Ward linkage

### Observed pattern

Ward linkage was:

- perfect on isotropic blobs,
- excellent on anisotropic blobs,
- best on unequal among the four methods in this experiment,
- terrible on circles,
- only moderate on moons.

### Interpretation

Ward is the linkage of **variance minimization**.

It is most natural for blob-like compact cluster structure.

It is especially strong when meaningful groups can be characterized as low-variance cohesive regions in feature space.

That is why it worked so well on isotropic, anisotropic, and unequal blob-like datasets.

But because it strongly prefers compact groups, it is fundamentally mismatched to circles and moons.

### Main lesson

Ward is an excellent default for compact Euclidean clusters, but it should not be used when the true cluster identity is non-convex connected structure.

---

# 8. Cross-dataset synthesis

## What the project as a whole demonstrated

The project showed three broad regimes of clustering geometry.

### Regime 1: easy compact blob structure
Examples:
- isotropic blobs

Observations:
- all methods succeed,
- K-selection is easy,
- silhouette and ARI agree,
- dendrograms are highly faithful.

### Regime 2: elongated / unequal but still blob-like structure
Examples:
- anisotropic blobs,
- unequal blobs

Observations:
- single linkage becomes unreliable,
- complete/average/Ward are much better,
- Ward and average often perform best,
- geometry is still cluster-like but more challenging.

### Regime 3: non-convex manifold structure
Examples:
- circles,
- moons

Observations:
- single linkage is uniquely appropriate,
- complete/average/Ward fail because they prefer compact convex groups,
- silhouette becomes questionable if it conflicts with manifold truth.

### Regime 4: realistic hierarchical physical state structure
Examples:
- polymer conformation hierarchy

Observations:
- no single flat partition is perfectly correct,
- hierarchy matters more than one chosen K,
- moderate ARI does not imply useless clustering,
- silhouette may favor sub-basin splitting,
- interpretation must be domain-aware.

---

# 9. Practical lessons for future use

## If I have a new dataset, what should I remember?

### Use Ward when:
- clusters are expected to be compact,
- Euclidean variance is meaningful,
- I want a strong default for blob-like structure,
- I care about clean low-variance groups.

### Use single when:
- clusters are non-convex,
- cluster identity is about connected shape,
- I want to follow manifolds such as arcs, rings, or chains,
- I understand that chaining is a feature, not a bug.

### Use complete when:
- I want compactness and strong resistance to chaining,
- but do not want as much variance-based bias as Ward.

### Use average when:
- I want a robust compromise,
- dataset geometry is mixed or uncertain,
- and I need a linkage less extreme than single or complete.

---

# 10. Metric-specific caution notes

## ARI

Use ARI when true labels exist.

It tells me how well my clustering matches known labels, but not whether those labels are the only meaningful structure in the data.

## Silhouette

Silhouette is powerful but only if its assumptions roughly match the data.

It works best for:
- compact,
- convex,
- well-separated clusters.

It can be misleading for:
- moons,
- circles,
- hierarchical sub-basin data,
- overlapping physical states.

## Cophenetic correlation

Cophenetic correlation tells me how well a dendrogram preserves pairwise distance geometry.

High cophenetic correlation means the tree is a good geometric summary.
It does **not** necessarily mean the labels are correct.

## Dendrogram cuts

Dendrograms should be interpreted visually and physically, not only through automated heuristics.

The current distance-acceleration heuristic in this project should not be trusted without fixing the implementation.

---

# 11. Main conceptual takeaways I should remember years later

If I forget everything else and only remember a few principles, they should be these:

1. **Hierarchical clustering is not one algorithm but a family of algorithms defined by the linkage rule.**
2. **Different linkage rules encode different definitions of what it means for clusters to be close.**
3. **Ward is best thought of as a variance-minimizing blob detector.**
4. **Single linkage is best thought of as a connectivity/manifold detector.**
5. **Silhouette is not universally reliable; it depends on cluster geometry.**
6. **Dendrograms are valuable not only for choosing K but for understanding hierarchy.**
7. **On realistic scientific data, moderate ARI can still coexist with meaningful hierarchical insight.**
8. **The right method depends on the geometry of the true structure, not on generic popularity.**

---

# 12. Final project-level conclusion

This project successfully demonstrated hierarchical clustering from multiple complementary angles:

- implementation from scratch,
- production-library usage,
- side-by-side linkage comparison,
- dendrogram interpretation,
- K selection,
- physically motivated hierarchy analysis,
- and cross-dataset benchmarking.

The strongest final insight is this:

> **Hierarchical clustering is most useful when treated as a geometry-sensitive, interpretation-heavy tool rather than a one-click clustering black box.**

The linkage rule is not a small parameter. It is the core modeling choice.

Choosing linkage is effectively choosing the definition of cluster proximity:

- nearest-point proximity,
- farthest-point proximity,
- average proximity,
- or variance-increase proximity.

Everything that happened in the project followed from that fact.

