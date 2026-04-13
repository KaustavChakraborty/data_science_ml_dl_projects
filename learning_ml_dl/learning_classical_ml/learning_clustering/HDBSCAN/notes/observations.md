# observations.md — Detailed Outcome-by-Outcome Analysis of the HDBSCAN Project

## 1. Purpose of this document

This document is an **interpretation companion** to the HDBSCAN project script and figures. It does not focus on implementation details of the code itself; instead, it explains what each generated result means, how to read the plots, why certain numerical patterns appear, and what the outcomes imply scientifically.

The project is structured as a tutorial-style inference pipeline rather than as a single benchmark script. It begins with simple synthetic datasets, moves through hierarchy, soft membership, outlier scoring, parameter sensitivity, and model comparison, and then ends with domain-inspired workflows for molecular dynamics, colloidal phases, anomaly detection, and a full inference pipeline.

A recurring theme across almost every section is the distinction between:

- **dense, stable cores**
- **boundary or transitional regions**
- **noise / rejected points**
- **rare or anomalous configurations**

That distinction is the conceptual heart of HDBSCAN.

---

## 2. The central scientific message of the project

Across all sections, the script shows that HDBSCAN is most useful when the data contains one or more of the following properties:

1. **unknown number of clusters**
2. **non-convex cluster shape**
3. **variable density across clusters**
4. **a need to identify ambiguous points instead of forcing them into labels**
5. **a need for confidence values**
6. **a need for hierarchy or anomaly scores**

The project repeatedly demonstrates that HDBSCAN does not merely "cluster points." It also tells you:

- which groups are **stable across density scales**
- which points are **deep-core representatives**
- which points are **borderline**
- which points are **better treated as noise**
- which points are **locally anomalous**

This is why the outputs often show **excellent purity among assigned points** even when many points are rejected as noise.

---

## 3. Important global interpretation rules for this project

Before discussing the sections one by one, there are several global interpretation rules that matter throughout the project.

### 3.1 Standardization matters everywhere

The script standardizes features before fitting HDBSCAN in nearly every analysis block. This means clustering and anomaly detection are done in balanced feature space, rather than being dominated by whichever physical descriptor has the largest raw numerical range.

So when a phase or conformational state is called “dense,” that density is in **standardized feature space**, not necessarily in raw spatial coordinates.

### 3.2 ARI is not always computed the same way

This is one of the most important caveats in the entire project.

In some sections, ARI is computed on **all labels including noise**.  
In other sections, ARI is computed only on **non-noise points**.

That difference changes interpretation dramatically.

- **ARI including noise** answers:  
  “How well does the entire partition, including rejected points, match the truth?”

- **ARI on non-noise points only** answers:  
  “How pure are the points that the algorithm decided to keep?”

This is why a model can have:
- very high ARI on retained points,
- but lower effective full-data agreement,
- simply because it rejects many ambiguous points as noise.

### 3.3 Noise is not automatically a mistake

In this project, noise often corresponds to:

- boundary material
- diffuse interfacial states
- transition-like configurations
- low-density fringe points
- points the model refuses to trust

In many scientific settings, this is a feature, not a flaw.

### 3.4 The custom condensed-tree plot is approximate

The project uses a manual condensed-tree helper for plotting. It is useful for qualitative viewing, but should not be interpreted as a perfect rendering of the official HDBSCAN condensed hierarchy.

The most trustworthy hierarchy quantity in the project is usually the printed or plotted **cluster persistence**, not the precise horizontal bar geometry in the custom condensed-tree panel.

---

## 4. Section A — HDBSCAN basics

This section introduces three classic benchmark datasets:

- isotropic blobs
- concentric circles
- two moons

The point of this section is not subtle scientific interpretation. It is to demonstrate the basic HDBSCAN outputs:

- hard labels
- membership probabilities
- outlier scores
- persistence

### 4.1 Isotropic blobs

This is the easy baseline.

Expected pattern:
- correct number of clusters
- zero or near-zero noise
- very high ARI
- probabilities highest near cluster centers
- outlier scores highest near cluster edges

Interpretation:
The result shows that HDBSCAN can trivially solve well-separated compact clusters. The interesting part is not the cluster labeling itself, but the fact that even in this simple case, HDBSCAN still gives useful internal structure:

- central points are **prototypical members**
- outer shell points are **less confident**
- edge points have **larger outlier scores**

This already demonstrates a major difference from algorithms like K-means: HDBSCAN tells you **how strongly** a point belongs to its group.

### 4.2 Concentric circles

This dataset matters because centroid-based methods often struggle here.

Interpretation:
HDBSCAN correctly recovers both rings, which shows that it is not limited to convex or spherical cluster geometry.

Key outcome:
Even when the labels are perfect, the persistence values of the two rings may differ. This tells you the two geometric structures are not equally strong in the density hierarchy.

So “correct clustering” and “equal hierarchical stability” are different things.

### 4.3 Two moons

This is another non-convex structure test.

Interpretation:
HDBSCAN successfully recovers both crescent-shaped manifolds. Probability usually drops near the moon tips and locally sparse regions, and outlier scores rise in those same areas.

That means the moon structure is recovered globally, but the internal confidence field still reflects local sparsity.

### 4.4 Main lesson from Section A

Section A proves three foundational things:

1. HDBSCAN handles simple compact clusters easily.
2. HDBSCAN handles non-convex geometry naturally.
3. HDBSCAN provides richer outputs than plain labels.

This section establishes the vocabulary that later sections use:
- **labels**
- **probabilities**
- **outlier scores**
- **persistence**

---

## 5. Section B — Condensed tree and hierarchy

This section asks not only “what clusters were found?” but also:

- how stable are they?
- over what density range do they survive?
- does the hierarchy support them as real branches?

### 5.1 On easy isotropic blobs

Outcome:
The algorithm finds the expected three clusters with excellent agreement.

Interpretation:
This shows the hierarchy is not only useful for ambiguous data. Even clean Gaussian blobs produce meaningful persistence values. When those persistence values are similar, it tells you the blobs are of roughly comparable density-scale stability.

### 5.2 On colloidal phases

This is where the hierarchy becomes more scientifically interesting.

The true phase panel may show three conceptual states, but the HDBSCAN result often keeps only the dense cores and rejects a significant fraction of points as grey/noise.

Interpretation:
This means the dataset does not consist of three perfectly isolated phase islands. Instead, it contains:

- three main dense phase regions
- a broad shell of ambiguous, diffuse, or interfacial points

Persistence values then tell you which phase cores are strongest in the hierarchy.

A phase with higher persistence:
- survives over a broader density range
- is more robust as a density-defined object
- is likely more “real” in the HDBSCAN sense

A phase with lower persistence:
- may still be valid,
- but is less robust,
- and may sit closer to overlap/interfacial material.

### 5.3 Main lesson from Section B

The condensed-tree analysis teaches that clustering is not only about group count.  
It is also about **hierarchical support**.

A cluster can exist and still be weak.  
A cluster can be correct and still be less persistent than another.

This section introduces the key concept that later domain analyses rely on:
**not all discovered states are equally real or equally stable.**

---

## 6. Section C — Soft membership probabilities

This section is one of the most important in the whole project because it turns a hard clustering result into a graded confidence landscape.

### 6.1 What membership probability means here

The probability is not a Bayesian class probability.  
It is a confidence-like score reflecting how strongly a point belongs to its assigned HDBSCAN cluster.

High values mean:
- deep inside the cluster core
- archetypal representative
- far from boundaries

Moderate values mean:
- peripheral member
- boundary-like
- weakly assigned

Very low values for assigned points mean:
- the point is being kept, but only barely

Noise points are not assigned meaningful cluster membership.

### 6.2 On colloidal phases

The project outcome usually shows:

- three main colored phase cores
- a substantial grey/noise region
- a subset of high-confidence members
- a smaller set of weak-but-still-assigned transitional members

Interpretation:
This says the phase diagram in feature space is not perfectly crisp. HDBSCAN finds three strong phase cores, but much of the surrounding material does not deserve confident assignment.

The percentages such as:
- probability ≥ 0.8
- probability ≥ 0.5
- probability ≥ 0.3

must be interpreted carefully. In the code, those printed values are often computed **only on non-noise points**. So they describe the confidence distribution among the accepted members, not among all data points.

This is why the project can simultaneously report:
- many highly confident assigned points
- and still show a large grey/noise population overall

There is no contradiction. It means the model is **strict**: once a point is kept, it is usually quite trustworthy.

### 6.3 Probability histograms per cluster

These histograms are a very useful diagnostic.

If a cluster’s histogram piles up near 1:
- the cluster is compact and internally coherent

If it has a long lower-probability tail:
- the cluster has diffuse boundaries
- or the cluster shape is elongated / heterogeneous

This is especially useful for identifying phases or conformational states that are internally clean versus those that are smeared into surrounding space.

### 6.4 Probability-weighted centroid shift

The project compares:
- unweighted centroid
- probability-weighted centroid

Interpretation:
If the weighted centroid shifts noticeably relative to the ordinary centroid, then low-confidence fringe points are pulling the ordinary center away from the true dense core.

This is a very useful scientific idea:
for representative structures or phase centroids, one often wants the **core-centered** summary, not the arithmetic average of all fringe members.

### 6.5 Main lesson from Section C

Section C teaches that clusters are not all-or-nothing objects.

A cluster often contains:
- core representatives
- soft boundary members
- a shell of points that may be better treated as transition/interfacial material
- rejected noise around it

This is one of the strongest practical reasons to use HDBSCAN rather than a method that labels every point equally.

---

## 7. Section D — GLOSH outlier analysis

This section explains outlier scoring through HDBSCAN’s built-in hierarchy.

### 7.1 What GLOSH means in this project

GLOSH scores quantify how anomalous a point is relative to the dense hierarchical structure learned by HDBSCAN.

A high score does not always mean:
- “this point is totally foreign”

It can also mean:
- “this point lies at an unusually sparse or atypical part of a cluster”

So in this project, GLOSH is often best interpreted as:
**hierarchy-relative atypicality**

### 7.2 Injected outliers

For evaluation, the code injects synthetic outliers into certain datasets and then measures recall after flagging the top score tail.

This is useful because it gives a semi-controlled benchmark:
- some anomalies are known
- the algorithm must recover them from density information alone

### 7.3 On colloidal phases

The project outcome typically shows that:
- many injected outliers are recovered
- some natural fringe/interfacial points are also flagged
- not all flagged points correspond to synthetic injections

Interpretation:
This is exactly what a good anomaly detector should do in a scientific dataset:
it should not only find artificial anomalies,
it should also highlight genuinely unusual real configurations.

A flagged point can therefore be:
- a synthetic injected anomaly
- a naturally sparse boundary point
- an interface configuration
- a rare event within a known phase

### 7.4 Probability vs outlier score scatter

This is one of the most conceptually rich plots.

It usually shows a pattern where:
- deep-core members sit at high probability and low outlier score
- weak or noisy points sit at low probability and higher outlier score

But the relation is not identical.  
A point can be:
- weakly assigned but not extremely anomalous
- anomalous while still being weakly attached to a cluster

So probability and GLOSH are related, but not redundant.

### 7.5 Main lesson from Section D

Outlier detection here is not just “find points far away from everything.”  
It is “find points that are unusually unsupported by the local hierarchical density structure.”

This is why GLOSH is especially useful in soft matter, MD, and state-space analyses where rare but physically meaningful configurations matter.

---

## 8. Section E — Parameter sensitivity

This section examines how HDBSCAN changes as the two main control parameters vary:

- `min_cluster_size`
- `min_samples`

### 8.1 Key conceptual difference

`min_cluster_size` controls:
- the smallest group you are willing to call a cluster

`min_samples` controls:
- how strict the local density requirement is

This distinction is crucial throughout the project.

### 8.2 Easy blobs

For the easy blob dataset, the heatmaps often show:
- fixed cluster count
- zero noise
- perfect ARI across almost all settings

Interpretation:
The data structure is so clear that parameter choice barely matters.

This is a sign of a trivial clustering problem.

### 8.3 Colloidal phases

For the colloidal phases dataset, the heatmaps show much more structure:

- most settings still recover 3 clusters
- some small-cluster or permissive settings create 2 or 4 clusters
- noise fraction varies substantially
- ARI on non-noise points often stays high except in unstable corners

Interpretation:
The coarse 3-phase answer is robust, but the treatment of interfacial/boundary material depends strongly on conservatism.

This is one of the most important scientific findings in the project:
**the main phases are stable, but the assignment of diffuse material is not.**

### 8.4 Role of default min_samples

When the code uses `min_samples=None`, HDBSCAN sets:
`min_samples = min_cluster_size`

This makes the model increasingly conservative as `min_cluster_size` rises.

That is why some rows/columns in the sensitivity scan show sharp growth in noise fraction: both minimum cluster size and density strictness are rising together.

### 8.5 Main lesson from Section E

For ambiguous physical data, tuning HDBSCAN is not mainly about “forcing the right answer.”  
It is about choosing a scientifically appropriate balance between:

- state purity
- cluster granularity
- amount of rejected material

This section teaches that parameter selection is really a choice about **how conservative your interpretation should be**.

---

## 9. Section F — DBSCAN vs HDBSCAN on variable-density data

This is one of the clearest conceptual demonstrations in the whole project.

### 9.1 Why this example matters

The synthetic dataset contains three clusters of very different density:

- one tight
- one medium
- one diffuse

This is the natural failure mode of DBSCAN, because DBSCAN uses a **single global epsilon**.

### 9.2 Small epsilon DBSCAN

With very small `eps`, DBSCAN only captures the densest cluster and throws away most of the rest as noise.

Interpretation:
This is the under-connection regime.
DBSCAN becomes too strict for diffuse structures.

### 9.3 Medium epsilon DBSCAN

At intermediate `eps`, DBSCAN partially recovers more of the data but tends to fragment or inconsistently connect the less dense regions.

Interpretation:
This is the compromise regime, but it still fails to give a clean single solution.

### 9.4 Large epsilon DBSCAN

At large `eps`, DBSCAN absorbs more of the diffuse cluster but risks over-connecting regions or creating incorrect merged structure.

Interpretation:
This is the over-connection regime.

### 9.5 HDBSCAN on the same data

HDBSCAN usually recovers the intended 3-cluster picture much more naturally.

Interpretation:
This is the core reason HDBSCAN exists. It effectively explores density scales hierarchically and then selects persistent structures, rather than forcing one global neighborhood radius to serve all density regimes.

### 9.6 Main lesson from Section F

The varying-density challenge shows that HDBSCAN is not just “DBSCAN without epsilon tuning.”  
It is a different conceptual tool designed for **multi-density structure**.

This is one of the strongest outcomes in the whole project.

---

## 10. Section G — MD conformational state extraction

This section is one of the most scientifically interesting parts of the project.

Frames are represented using four conformational descriptors:

- φ
- ψ
- Rg
- d_ee

Each MD frame becomes a point in feature space, and HDBSCAN is used to identify metastable basins.

### 10.1 Why the MD result matters

The true state labels contain four named conformational classes, but HDBSCAN often finds **five** states plus a substantial noise population.

Interpretation:
This means one of the true classes, especially the broad collapsed-like region, is not density-uniform. HDBSCAN splits it into multiple sub-basins while still rejecting a large transition-like cloud.

So the result is not simply “wrong number of states.”  
It is saying:
**the conformational landscape is hierarchically richer than the coarse truth labels.**

### 10.2 The five states

The project reports raw state sizes, probability-weighted populations, core-member counts, transition-like assigned members, and persistence.

Interpretation of the pattern:
- two dominant states carry the largest weighted populations
- two moderate states appear comparably important
- one small state appears very sharp but weakly persistent

That last point is especially important.

A state can be:
- compact
- high-confidence internally
- but low persistence

This usually means it is a narrow local niche rather than a broad robust basin.

### 10.3 Noise as transition-like material

The lower-left panel explicitly interprets noise frames as transition-like frames.

Scientifically, the safest wording is:
- they are likely inter-basin / diffuse / weakly supported frames
- many may indeed be transition or excursion states
- but HDBSCAN alone does not prove kinetic transition-state identity

Still, the concentration of noise around the broad right-hand manifold strongly suggests that part of the conformational landscape is highly heterogeneous and transition-rich.

### 10.4 Top anomalous frames

The top anomalous frames often belong to one dominant state but with low membership probability.

Interpretation:
These are not necessarily foreign structures.
They are usually **rare fringe configurations within a known basin**.

This is exactly the sort of result that could guide structural inspection in a real MD study.

### 10.5 Anomaly trace along “trajectory”

The code shuffles the frames before plotting the score trace.

Important implication:
This is not a true time-series kinetic trace.
It is better interpreted as an index-wise anomaly profile over a randomized frame order.

So it is useful for ranking rare frames, not for inferring temporal bursts or transition timing.

### 10.6 Main lesson from Section G

The MD pipeline shows that HDBSCAN can reveal:
- metastable basin structure
- sub-basin splitting
- core vs fringe conformations
- transition-rich regions
- rare frames worth inspection

This section is a strong demonstration of HDBSCAN as a conformational-state analysis tool.

---

## 11. Section H — Colloidal phases: EoM vs Leaf selection

This section asks whether the hierarchy contains meaningful fine substructure beyond the three main phases.

### 11.1 EoM versus leaf

- **EoM** tends to extract coarse, stable main clusters
- **Leaf** tends to extract finer leaf-level structure if it exists strongly enough

### 11.2 Observed outcome

In this project, both EoM and leaf often return:
- the same number of clusters
- the same noise count
- the same ARI on non-noise points

Interpretation:
At the chosen threshold, the colloidal phase dataset supports a robust **three-phase** picture and does not contain leaf-level substructure strong enough to survive as separate meaningful states.

This is a very valuable negative result:
sometimes the correct scientific conclusion is that the coarse phase decomposition is sufficient.

### 11.3 Probability-weighted centroids in physical units

The project computes centroids in original feature units:

- ψ₆
- ρ_local
- Q_nematic

These values are physically interpretable and allow the cluster labels to be mapped onto:
- gas-like
- liquid-like
- crystal-like

This is one of the strongest parts of the whole tutorial because it turns abstract cluster labels into physically meaningful phases.

### 11.4 Membership and outlier histograms per phase

These histograms tell you:
- which phases are internally compact
- which phases have broader diffuse shells
- which phases have longer anomaly tails

So the model is not just phase-labeling.  
It is also phase-characterizing.

### 11.5 Main lesson from Section H

The hierarchy does not always imply a richer final flat clustering.
In this case, leaf selection confirms the same coarse picture as EoM.

That strengthens confidence in the three-phase interpretation.

---

## 12. Section I — Anomaly detection pipeline

This section switches from unsupervised phase discovery to a detection-style question:

Can the HDBSCAN/GLOSH framework identify unusual future observations relative to a normal training set?

### 12.1 The anomaly types

The code constructs three kinds of test anomalies:

1. **transition points**
2. **artefacts**
3. **novel phase points**

This design is very insightful because these three are not equally easy to detect.

### 12.2 Outcome pattern

Typical result:
- transition recovery: poor or zero
- artefact recovery: partial
- novel phase recovery: very high or perfect

This is not accidental.

### 12.3 Why transitions are missed

The transition points are created between known phases, so they often remain close to the normal manifold in feature space.

Interpretation:
GLOSH is a density-anomaly detector, not a semantic intermediate-state detector. Smooth bridges inside known support do not necessarily look more anomalous than the 97th percentile of training scores.

So the detector is not good at “gentle interpolation anomalies.”

### 12.4 Why artefact detection is partial

Artefacts are random over a broad range.

Some land in obviously unsupported regions and are flagged.  
Others accidentally fall near already occupied support and are not extreme enough.

So partial recall is the expected behavior.

### 12.5 Why novel-phase detection is excellent

Novel-phase points form a coherent cloud in a distinct region of feature space.

Interpretation:
This is the anomaly type HDBSCAN/GLOSH is best suited for:
**a new, unsupported density island**

That is why novel-phase recall is usually perfect.

### 12.6 Main lesson from Section I

The anomaly detector is best at:
- genuinely new structure
- clear support mismatch

It is weaker at:
- transitional or interpolative states within the normal manifold

This is a powerful and honest limitation profile.

---

## 13. Section J — Full inference pipeline

This section combines parameter tuning, clustering, probabilities, anomalies, and physical summary into one final workflow.

### 13.1 Purpose of the pipeline

This is the project’s closest approximation to a real analysis scenario:

1. scan a parameter
2. choose the best setting
3. fit the final model
4. inspect true vs inferred phases
5. inspect hierarchy
6. inspect membership confidence
7. inspect outlier scores
8. overlay transitions and anomalies
9. produce a physical summary table

### 13.2 Parameter scan over min_cluster_size

The code scans several `min_cluster_size` values and records:
- number of clusters
- ARI
- silhouette

Important interpretation:
ARI is again computed on non-noise points.

So the “best” setting is really the one that gives the cleanest phase fidelity among retained points.

### 13.3 Best parameter choice

The project often finds that ARI improves up to a moderately large `min_cluster_size`, then collapses once the value becomes too large and the phase structure over-merges.

This teaches an important practical lesson:
- increasing `min_cluster_size` can improve purity
- but beyond a point, it destroys the physically meaningful number of states

### 13.4 Final three states

The final table typically reports:
- state size
- population
- ψ₆
- ρ
- Q
- core fraction
- transition fraction
- silhouette
- persistence

These statistics make the final inferred clusters interpretable as physical phases.

Typical mapping:
- high ψ₆, high ρ, high Q → crystal-like
- intermediate values → liquid-like
- very low values → gas-like

### 13.5 Zero transition points under strict settings

At a very conservative final setting, the project often reports:
- many noise points
- zero transition points among assigned members

Interpretation:
The model is so strict that it no longer keeps borderline points inside clusters. It keeps only trusted cores and rejects the rest.

This is a **high-precision, low-coverage** clustering regime.

### 13.6 Noise and anomalies in the final pipeline

A large noise fraction here does not imply failure.  
It implies that the model is choosing to represent only the strongest phase cores.

Similarly, the anomaly overlay often shows that anomalies concentrate in the diffuse interfacial cloud rather than inside dense phase basins.

That is a very coherent physical picture.

### 13.7 Main lesson from Section J

The full pipeline demonstrates how one can turn HDBSCAN results into a complete scientific report:

- choose a parameter rationally
- identify robust states
- characterize them physically
- separate core from fringe
- identify anomalous or rare material
- acknowledge rejected ambiguous regions

This is the most “publication-style” part of the project.

---

## 14. Cross-cutting patterns across the whole project

Looking across all sections together, several major patterns emerge.

### 14.1 HDBSCAN is strongest when the data contains ambiguity

On easy blobs, HDBSCAN looks almost boring because everything is obvious.  
Its real power appears when the data has:

- overlap
- variable density
- interfacial material
- sub-basin structure
- rare events

### 14.2 The project repeatedly favors purity over forced completeness

Again and again, the script chooses to preserve:
- clean dense cores
and reject:
- diffuse boundary material

This is why many outcomes show:
- strong non-noise ARI
- large noise fractions

That is not inconsistency. It is a deliberate modeling stance.

### 14.3 Membership probability and GLOSH together are extremely informative

Probability tells you:
- how safely a point belongs to its cluster

GLOSH tells you:
- how atypical it is relative to the hierarchy

Together they let you distinguish:
- core members
- fringe members
- noise
- rare but still cluster-affiliated points
- genuinely anomalous configurations

### 14.4 Hierarchy is not only decorative

Persistence and the hierarchy clarify which clusters are:
- broad and robust
- moderate
- weak
- tiny local pockets

This matters especially in MD and colloidal examples.

### 14.5 Parameter tuning is a scientific choice, not only a numerical one

The correct setting depends on what the user wants:

- discover small substructure?
- extract only large robust states?
- preserve borderline members?
- reject all uncertain material?

The project repeatedly shows that the parameter choice changes interpretation, not just score values.

---

## 15. Limitations and caveats revealed by the outcomes

The project is strong, but the results also reveal several important limitations.

### 15.1 Noise interpretation is domain-sensitive

Calling all noise “transition state” is too strong in a formal sense.  
Noise means low support under the clustering model, not necessarily kinetic transition-state membership.

### 15.2 Some anomaly pipelines use refit-on-all-data logic

In the anomaly-detection section, the scoring uses a model fit on training + test together.  
That is fine for didactic demonstration, but it is not a strict train-only novelty-detection deployment setting.

### 15.3 Custom condensed-tree plotting is approximate

The hierarchy is real, but the custom plot should not be over-read geometrically.

### 15.4 PCA visualization compresses feature structure

Many results are plotted in 2D PCA space.  
This is useful for intuition, but the clustering happens in standardized feature space, which may have higher dimension and richer separation than the 2D projection suggests.

---

## 16. Overall conclusion

This HDBSCAN project successfully demonstrates HDBSCAN as far more than a simple clustering algorithm.

It shows HDBSCAN as a full inference framework capable of:

- discovering clusters without specifying K
- handling non-convex and variable-density data
- measuring state confidence through soft membership
- identifying anomalous points via GLOSH
- revealing hierarchical stability via persistence
- separating robust cores from ambiguous fringe regions
- supporting physically interpretable analysis in soft matter and molecular simulation contexts

The strongest scientific conclusions from the project are:

1. **HDBSCAN handles variable-density structure far better than DBSCAN.**
2. **Soft membership is crucial for understanding cluster interiors and boundaries.**
3. **GLOSH detects genuine novelty well, but not all smooth transition states.**
4. **For colloidal phases, the coarse three-phase picture is robust, but diffuse interfacial material is substantial.**
5. **For MD conformations, the density landscape can be richer than coarse truth labels and may split broad manifolds into multiple basins.**
6. **A conservative HDBSCAN fit yields highly trustworthy state cores at the cost of rejecting many ambiguous points.**

In short, the project convincingly shows that HDBSCAN is not merely useful for assigning labels.  
It is useful for building a **scientifically nuanced interpretation of structured, ambiguous, multi-scale data**.
