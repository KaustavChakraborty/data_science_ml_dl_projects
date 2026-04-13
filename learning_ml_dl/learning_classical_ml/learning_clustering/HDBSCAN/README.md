# HDBSCAN Complete Tutorial Project

A deeply explained, research-oriented HDBSCAN project for learning **hierarchical density-based clustering**, **soft cluster membership**, **outlier scoring**, **parameter sensitivity**, **variable-density clustering**, and **scientific interpretation pipelines** on synthetic and soft-matter-inspired datasets.

This project is built around a single main script, `hdbscan_v1.py`, which is organized as a complete tutorial with sections **A through J** and is intended to take a reader from the fundamentals of HDBSCAN all the way to a full inference workflow for phase/state analysis and anomaly detection. The script expects datasets to be generated beforehand by `00_generate_datasets.py`, and it writes all figures to `plots/hdbscan/`. The code uses `numpy`, `scipy`, `scikit-learn`, `hdbscan`, `matplotlib`, `seaborn`, and `pandas`, sets Matplotlib to the non-interactive `Agg` backend, seeds randomness for reproducibility, and creates the output folder automatically on startup. It also includes custom helper logic for plotting a condensed tree because the script explicitly avoids the package's built-in `hdbscan.plot()` route due to a Matplotlib-version-related bug. fileciteturn20file2turn20file1turn20file4

---

## Table of contents

1. [Project purpose](#project-purpose)
2. [What HDBSCAN is and why this project exists](#what-hdbscan-is-and-why-this-project-exists)
3. [What this project teaches](#what-this-project-teaches)
4. [Repository / project layout](#repository--project-layout)
5. [Requirements](#requirements)
6. [How to run the project](#how-to-run-the-project)
7. [Expected inputs and generated outputs](#expected-inputs-and-generated-outputs)
8. [Core HDBSCAN concepts used throughout](#core-hdbscan-concepts-used-throughout)
9. [Understanding the two most important parameters](#understanding-the-two-most-important-parameters)
10. [How the helper utilities work](#how-the-helper-utilities-work)
11. [Detailed walkthrough of each tutorial section](#detailed-walkthrough-of-each-tutorial-section)
    - [A. HDBSCAN basics](#a-hdbscan-basics)
    - [B. Condensed tree analysis](#b-condensed-tree-analysis)
    - [C. Soft membership analysis](#c-soft-membership-analysis)
    - [D. GLOSH outlier analysis](#d-glosh-outlier-analysis)
    - [E. Parameter sensitivity](#e-parameter-sensitivity)
    - [F. DBSCAN vs HDBSCAN](#f-dbscan-vs-hdbscan)
    - [G. MD conformational state extraction](#g-md-conformational-state-extraction)
    - [H. Colloidal phases: EoM vs leaf selection](#h-colloidal-phases-eom-vs-leaf-selection)
    - [I. Anomaly detection pipeline](#i-anomaly-detection-pipeline)
    - [J. Full inference pipeline](#j-full-inference-pipeline)
12. [How to interpret the produced figures correctly](#how-to-interpret-the-produced-figures-correctly)
13. [Important caveats and subtle implementation details](#important-caveats-and-subtle-implementation-details)
14. [How to adapt this project to real data](#how-to-adapt-this-project-to-real-data)
15. [Suggested extensions](#suggested-extensions)
16. [Practical summary](#practical-summary)

---

## Project purpose

This project is not just a clustering demo. It is a **full educational and applied workflow** for understanding how HDBSCAN behaves on:

- clean synthetic clustering tasks,
- non-convex manifolds,
- variable-density datasets,
- soft-matter / colloidal phase descriptors,
- molecular-dynamics-like conformational state features,
- anomaly detection problems,
- and end-to-end scientific interpretation pipelines.

The script is intentionally structured as a sequence of increasingly sophisticated analyses. Early sections teach what HDBSCAN returns. Middle sections teach how to interpret those outputs. Final sections show how one might actually use those outputs in research.

The project therefore serves **three roles at once**:

1. a tutorial on HDBSCAN as an algorithm,
2. a practical plotting and analysis template,
3. a scientific interpretation guide for density-based state discovery.

---

## What HDBSCAN is and why this project exists

HDBSCAN stands for **Hierarchical Density-Based Spatial Clustering of Applications with Noise**. The script describes it in a very useful way: instead of choosing one DBSCAN radius `eps` and clustering at one density scale, HDBSCAN effectively explores **all density scales**, constructs a hierarchy of cluster births and deaths, and then extracts the most persistent clusters. It also returns soft membership strengths and outlier scores. The file explicitly highlights the main advantages over DBSCAN: no `eps` to tune, better handling of variable-density data, probabilities for membership, GLOSH anomaly scores, and direct access to the hierarchy through the condensed tree. The script also notes when HDBSCAN is the right tool and when DBSCAN may still be simpler or more reproducible. fileciteturn20file2

This project exists because many users first meet HDBSCAN only as a black-box clustering function. That is a loss. HDBSCAN is most useful when you understand:

- how it treats sparse points,
- why `min_cluster_size` and `min_samples` do different things,
- how probability and outlier scores complement the hard labels,
- how to interpret cluster persistence,
- and why the hierarchy matters.

This script teaches those ideas through carefully chosen examples.

---

## What this project teaches

By the end of the script, a reader should understand:

- what `.labels_`, `.probabilities_`, `.outlier_scores_`, `.cluster_persistence_`, `.condensed_tree_`, and `.minimum_spanning_tree_` are,
- why HDBSCAN can solve circles and moons where centroid methods struggle,
- why HDBSCAN is more appropriate than DBSCAN on variable-density data,
- how soft membership can be used to identify cluster cores versus boundary configurations,
- how GLOSH can rank atypical points,
- how to scan parameters sensibly,
- how to interpret multi-state landscapes such as colloidal phases and conformational basins,
- how to build a simple anomaly-detection workflow from HDBSCAN outputs,
- and how to convert raw clustering results into a physically interpretable summary table.

The script explicitly organizes these lessons into sections **A through J**, and the `main` block runs them in that order after loading `blobs_easy`, `colloidal_phases`, and `polymer_conf` datasets from `data/*.npz`. fileciteturn20file0turn20file1

---

## Repository / project layout

A typical working layout for this project looks like:

```text
project/
├── 00_generate_datasets.py
├── hdbscan_v1.py
├── data/
│   ├── blobs_easy.npz
│   ├── circles.npz
│   ├── moons.npz
│   ├── colloidal_phases.npz
│   └── polymer_conf.npz
└── plots/
    └── hdbscan/
        ├── A_basics.png
        ├── B_tree_isotropic_blobs.png
        ├── B_tree_colloidal_phases.png
        ├── C_soft_membership_colloidal_phases.png
        ├── D_outliers_colloidal_phases.png
        ├── E_sensitivity_colloidal_phases.png
        ├── E_sensitivity_blobs_easy.png
        ├── F_dbscan_vs_hdbscan.png
        ├── G_md_conformations.png
        ├── H_colloidal_phases.png
        ├── I_anomaly_detection.png
        └── J_full_pipeline.png
```

The exact plot filenames above are not guessed; they are produced directly by the named functions in the script and saved in `plots/hdbscan/`. The `main` block also prints a final message directing the user to that folder. fileciteturn20file0turn20file4

---

## Requirements

The file header states that the project requires:

- `numpy`
- `scipy`
- `scikit-learn`
- `hdbscan`
- `matplotlib`
- `seaborn`
- `pandas`

It also uses:

- `NearestNeighbors`, `PCA`, and clustering metrics from scikit-learn,
- `make_blobs` for one synthetic density-comparison showcase,
- and `matplotlib.gridspec` for multi-panel scientific figures. fileciteturn20file2turn20file1

A practical installation command is:

```bash
pip install numpy scipy scikit-learn hdbscan matplotlib seaborn pandas
```

Because the script forces Matplotlib to use `Agg`, it is designed to run safely in headless environments such as remote servers or compute nodes:

```python
import matplotlib
matplotlib.use("Agg")
```

So you do not need an interactive GUI backend to generate the figures. fileciteturn20file1

---

## How to run the project

The script header explicitly says to run `00_generate_datasets.py` first. That means the HDBSCAN tutorial assumes the data files already exist. The helper `load(name)` reads `data/{name}.npz` and returns arrays `X` and `y`. fileciteturn20file2turn20file1

A typical run sequence is:

```bash
python 00_generate_datasets.py
python hdbscan_v1.py
```

Or, if your environment uses a specific Python version:

```bash
python3.8 hdbscan_v1.py
```

When run, the script:

1. prints a banner,
2. runs all tutorial sections A through J in sequence,
3. writes the output figures to `plots/hdbscan/`,
4. prints summaries to the terminal,
5. ends with a final “see plots/hdbscan/” message. fileciteturn20file0

---

## Expected inputs and generated outputs

### Inputs

The script expects `.npz` files in the `data/` directory. Each file should contain:

- `X`: the feature matrix,
- `y`: the ground-truth labels for validation / illustration.

The helper is very simple:

```python
def load(name):
    d = np.load(f"data/{name}.npz")
    return d["X"], d["y"]
```

So your dataset generator must save those names exactly. fileciteturn20file1

### Outputs

The script generates a set of tutorial figures corresponding to the ten main sections:

- `A_basics.png`
- `B_tree_*.png`
- `C_soft_membership_*.png`
- `D_outliers_*.png`
- `E_sensitivity_*.png`
- `F_dbscan_vs_hdbscan.png`
- `G_md_conformations.png`
- `H_colloidal_phases.png`
- `I_anomaly_detection.png`
- `J_full_pipeline.png`

These images are the primary deliverables of the project. The console summaries complement them by printing cluster counts, ARI values, noise counts, persistence values, recovered anomalies, and state summaries. fileciteturn20file0turn20file4

---

## Core HDBSCAN concepts used throughout

This project repeatedly relies on a small number of HDBSCAN outputs. Understanding them is essential.

### 1. Hard labels: `clf.labels_`

Each point gets an integer cluster label or `-1` for noise.

Interpretation:

- `0, 1, 2, ...` mean the point belongs to one of the extracted stable clusters.
- `-1` means HDBSCAN does not trust assigning that point to any selected cluster.

This is **not** the same as “the point is bad.” It often means the point lies in a low-density, ambiguous, interfacial, or transitional region.

### 2. Membership probabilities: `clf.probabilities_`

The script describes these as confidence-like quantities in `[0, 1]`:

- near `1.0`: deep core member,
- near `0.0`: borderline member / weak support.

In the plotting helpers, these probabilities are often used as point opacity, which is a very intuitive way to visualize the difference between core and fringe structure. fileciteturn20file3

### 3. Outlier scores: `clf.outlier_scores_`

These are GLOSH scores in `[0, 1]` measuring how atypical a point is relative to the densest nearby branch in the hierarchy.

Interpretation:

- low score: typical cluster member,
- high score: sparse, edge-like, or anomalous point.

The script uses these scores both for educational plots and for practical anomaly flagging. fileciteturn20file3turn20file4

### 4. Persistence: `clf.cluster_persistence_`

Persistence measures how stable a cluster is across density scales. In the tutorial language, higher persistence means the cluster survives over a wider range of density thresholds and is therefore more “real” or robust. This is one of the most important advantages of HDBSCAN over flat clustering approaches. fileciteturn20file3

### 5. Condensed tree and minimum spanning tree

The script highlights `clf.condensed_tree_` and `clf.minimum_spanning_tree_` as core outputs, but because of plotting issues it uses a custom condensed-tree helper instead of the built-in plotting API. This is a useful reminder that the hierarchy itself is central to HDBSCAN, even if the plotting implementation has to be adapted. fileciteturn20file3turn20file1

---

## Understanding the two most important parameters

The header of the script already gives an excellent quick summary: `min_cluster_size` is the smallest group you would be willing to call a cluster, while `min_samples` controls density conservatism. If omitted, `min_samples` defaults to `min_cluster_size`. fileciteturn20file2

### `min_cluster_size`

This is the **minimum accepted cluster size**.

Effects of increasing it:

- fewer clusters,
- larger/coarser clusters,
- small branches are suppressed,
- fine substructure becomes harder to retain.

Scientific meaning:

- “How small a state/phase/sub-basin am I willing to recognize?”

### `min_samples`

This controls **local density strictness**.

Effects of increasing it:

- more conservative clustering,
- more points rejected as noise,
- cluster cores become denser,
- boundaries and bridges are more likely to be excluded.

Scientific meaning:

- “How cautious should the algorithm be about sparse or borderline points?”

### Why the default matters

If `min_samples=None`, HDBSCAN uses:

```text
min_samples = min_cluster_size
```

That means increasing `min_cluster_size` can simultaneously:

- demand larger final clusters,
- and demand stronger local density.

This coupling makes the fit much more conservative as `min_cluster_size` grows. That behavior is visible in the project’s colloidal-phase parameter scans. fileciteturn20file2turn20file4

---

## How the helper utilities work

The project includes a handful of compact but important helper functions.

### `load(name)`

Loads a dataset from `data/{name}.npz` and returns `(X, y)`. This function assumes the `.npz` contains keys `X` and `y`. fileciteturn20file1

### `plot_condensed_tree(clf, ax, title="Condensed Tree")`

This is a manual plotting routine using `clf.condensed_tree_.to_pandas()`. The reason for this custom logic is stated directly in the file: it avoids the built-in HDBSCAN plotting function because of a Matplotlib-version bug. Selected clusters are drawn as colored bars, and leaf/noise-falloff branches are shown in grey. fileciteturn20file1

**Important caution:** this custom plot is a simplified visualization, not a mathematically exact reproduction of the official HDBSCAN condensed-tree graphic. It is useful qualitatively, but the exact bar lengths should not be over-interpreted.

### `fit_hdbscan(...)`

This wrapper centralizes model construction. It sets:

- `metric="euclidean"`,
- `prediction_data=True`,
- `gen_min_span_tree=True` by default,
- and accepts `cluster_selection_method="eom"` or `"leaf"`.

The docstring explains the difference:

- `eom` tends to return fewer, larger, more stable clusters,
- `leaf` tends to return more fine-grained leaf clusters. fileciteturn20file1

### `summarise_hdbscan(clf, y_true=None, prefix="")`

Prints cluster count, noise fraction, and ARI on non-noise points when possible. This helper embodies a very important design choice in the project: many evaluations intentionally focus on the **purity of retained points** rather than forcing noise points into the ARI calculation. fileciteturn20file3

### `cluster_palette(labels, probs=None)`

Maps cluster labels to colors and optionally uses probability to scale alpha. Noise is always grey. This helper is used repeatedly across the project because opacity is a very intuitive visual representation of confidence. fileciteturn20file3

---

## Detailed walkthrough of each tutorial section

## A. HDBSCAN basics

This section fits HDBSCAN on three datasets:

- `blobs_easy`
- `circles`
- `moons`

and demonstrates the core output attributes:

- hard labels,
- probabilities,
- outlier scores,
- persistence. fileciteturn20file3

### Why this section exists

Before discussing sophisticated physical interpretations, the reader must see the simplest possible outputs:

- what a cluster label looks like,
- what “soft membership” means on a plot,
- how outlier scores light up edge points,
- and how HDBSCAN deals with non-convex geometry.

### What to look for

- On **easy blobs**, HDBSCAN should recover the obvious Gaussian groups cleanly.
- On **circles** and **moons**, it should still succeed despite the non-convex shape.
- Probability panels show core-versus-boundary structure.
- Outlier-score panels usually highlight sparse edge regions rather than only grossly isolated points.

### Educational takeaway

Section A teaches that HDBSCAN is not just a hard clustering method. It is a **state-structure detector** that also tells you how representative and how atypical each point is.

---

## B. Condensed tree analysis

This section focuses on the hierarchy itself, using the condensed tree, single-linkage view, true-label scatter, HDBSCAN-result scatter, and cluster-persistence bar chart. The function is run on isotropic blobs and colloidal phases in the `main` block. fileciteturn20file0

### Why the condensed tree matters

The condensed tree is the most HDBSCAN-specific object in the project. It tells you not just “what clusters were chosen,” but **which branches in density space survived long enough to be considered real**.

### How to read it conceptually

The script’s own explanation is very good:

- wide branches correspond to larger clusters,
- long survival across density scale corresponds to greater stability,
- selected colored branches are the final extracted clusters.

### What this teaches scientifically

For easy data, the tree is simple and the persistence values are usually all strong.
For more realistic data, the tree can show that some apparent groups are only weak or short-lived branches.

### Important caveat

Because the project uses a simplified custom condensed-tree plotter, the tree panels should be interpreted qualitatively. The **printed persistence values** are more trustworthy than the exact bar geometry. fileciteturn20file1

---

## C. Soft membership analysis

This section explains how to use `clf.probabilities_` in practice. It is run on the colloidal-phases dataset. The function prints the fraction of non-noise points above thresholds such as `0.8`, `0.5`, and `0.3`, and creates a six-panel figure showing all points, high-confidence members, transitional configurations, per-cluster histograms, centroid shifts, and cumulative probability coverage. fileciteturn20file0

### Why this section matters

Many clustering tutorials stop at hard labels. This section shows how HDBSCAN can distinguish:

- **deep-core representatives**,
- **moderate-confidence members**,
- **boundary-like assigned points**,
- **and fully rejected/noise points**.

### Key ideas demonstrated

1. **High-confidence cores** can be isolated with a threshold such as `prob > 0.8`.
2. **Transitional assigned points** can be approximated by intermediate probabilities.
3. **Probability-weighted centroids** can differ from ordinary means, especially in diffuse clusters.
4. **Probability histograms** reveal whether a cluster is compact or boundary-rich.

### Why this is useful in research

Probability is invaluable when you want to:

- choose representative structures,
- avoid contaminating centroids with fringe members,
- identify likely inter-state configurations,
- or quantify how sharp a phase/state really is.

---

## D. GLOSH outlier analysis

This section studies anomaly detection from HDBSCAN’s outlier scores. It optionally injects known outliers for benchmarking, computes GLOSH scores, chooses a threshold at the 90th percentile, and visualizes score maps, flagged points, score distributions, probability-vs-outlier relationships, threshold sensitivity, and anomaly ranking. fileciteturn20file0turn20file4

### Why this section exists

HDBSCAN is not only a clustering method. It is also a useful **anomaly-ranking method**. GLOSH makes it possible to ask:

- which points are most unusual,
- whether atypicality is localized in certain regions,
- and whether anomalies correspond to true foreign points or just fringe points.

### What to look for

- high-score points often accumulate on sparse fringes,
- truly injected outliers should appear in the extreme tail,
- the score distribution can reveal whether anomalies are clearly separated or only gradually ranked.

### Practical interpretation

A high GLOSH score does not automatically mean “bad data.” In many scientific settings it means:

- rare state,
- boundary configuration,
- low-density edge case,
- novel structure,
- or physical transient.

---

## E. Parameter sensitivity

This section scans `min_cluster_size` and `min_samples` on selected datasets and records:

- number of clusters found,
- noise fraction,
- ARI versus true labels.

It is run for both `colloidal_phases` and `blobs_easy`. fileciteturn20file0

### Why this section matters

This is where the project teaches one of HDBSCAN’s biggest practical strengths: compared with DBSCAN, it is often **much less fragile to parameter changes**.

### What the heatmaps tell you

- If cluster count stays constant over a broad region, the coarse answer is robust.
- If noise fraction changes strongly while ARI stays high, the main effect of tuning is how conservative you are about boundary points.
- If small-parameter settings create extra micro-clusters, that usually means you are allowing fine local substructure to survive.

### Main lesson

On easy blobs, everything should remain trivial across the grid. On realistic phase/state data, the **number of main basins may stay stable while the noise fraction changes a lot**. That is an extremely important real-world insight.

---

## F. DBSCAN vs HDBSCAN

This section builds a synthetic three-cluster dataset with deliberately different density scales using `make_blobs` and compares three DBSCAN settings with a single HDBSCAN run. The goal is to show the classic DBSCAN dilemma: one global `eps` cannot simultaneously respect tight, medium, and diffuse clusters. fileciteturn20file1turn20file2

### Why this section is central

This is arguably the conceptual heart of the whole tutorial.

It demonstrates that DBSCAN’s weakness is not just poor tuning by the user. The weakness is **structural**:

- `eps` too small → diffuse cluster is missed, many points become noise,
- `eps` medium → cluster fragmentation,
- `eps` too large → over-connection or merging.

HDBSCAN solves this by building a hierarchy across density scales and then extracting stable branches.

### Educational takeaway

If your data has significantly different density regimes, HDBSCAN is often the right default choice.

---

## G. MD conformational state extraction

This section treats a polymer / MD-like dataset as a conformational landscape in features such as `φ`, `ψ`, `Rg`, and `d_ee`. It shuffles the order, standardizes the features, fits HDBSCAN with `min_cluster_size=40` and `min_samples=10`, then reports:

- number of found states,
- noise count,
- ARI on non-noise frames,
- probability-weighted state populations,
- per-state centroids,
- core counts,
- transitional counts,
- persistence values,
- and top anomalous frames. fileciteturn20file0

### Why this section is important

This is where the project shifts from “clustering examples” to **landscape inference**.

Frames are interpreted as samples from metastable basins. HDBSCAN is used to discover those basins without pre-specifying the number of states.

### What this section teaches

- a true class may split into multiple density basins,
- a broad region may contain many noise/transition-like frames,
- persistence can distinguish broad stable basins from compact but weak niches,
- the top outliers are often fringe members of existing basins rather than completely foreign points.

### Research usefulness

This is directly relevant to:

- conformational clustering,
- metastable-state discovery,
- rare-event inspection,
- transition-region analysis,
- representative-structure selection.

---

## H. Colloidal phases: EoM vs leaf selection

This section compares the two extraction modes of HDBSCAN on colloidal phase data:

- `cluster_selection_method="eom"`
- `cluster_selection_method="leaf"`

Both are fit at `min_cluster_size=40`. The figure compares the true labels, EoM result, leaf result, probability-weighted centroids in original descriptor units, membership histograms, and outlier-score histograms. fileciteturn20file1turn20file0

### Why this matters

This section asks a subtle but very important scientific question:

> Is the system best understood as three main phases, or is there meaningful internal substructure inside those phases?

### Interpretation logic

- If leaf returns more clusters than EoM, that suggests real fine substructure may be present.
- If leaf collapses to the same answer as EoM, the hierarchy is effectively saying the coarse phase picture is sufficient at the chosen scale.

### Why the physical summary is valuable

The phase centroids are reported in original units such as `ψ₆`, `ρ_local`, and `Q_nematic`, making the output physically interpretable rather than purely geometric.

---

## I. Anomaly detection pipeline

This section builds an anomaly-detection experiment with a training set of normal colloidal-phase data and a mixed test set containing:

- transition points,
- artefacts,
- novel-phase points.

It computes a training-derived threshold from the 97th percentile of training GLOSH scores and then evaluates recovery by anomaly type. It also visualizes score maps, anomaly locations in feature space, and score distributions by type. The code later feeds the result into `I_anomaly_detection.png`. fileciteturn20file4

### What this section teaches

The detector is good at some anomaly notions and weak at others.

- **Novel phases** are often easy to detect because they form displaced density islands.
- **Random artefacts** are partly detectable because only some land in truly unsupported regions.
- **Transitions** can be hard to detect if they lie on bridges between known states rather than outside the known manifold.

### Why this is a powerful lesson

It shows that density-based anomaly detection is not the same thing as transition-state detection.
A point can be physically intermediate without being strongly out-of-distribution.

---

## J. Full inference pipeline

This final section is the most applied and most “research-like” part of the project. The function docstring explicitly describes it as a complete workflow from raw data to interpretation. It performs:

1. load and standardize,
2. parameter scan over `min_cluster_size`,
3. fit final HDBSCAN model,
4. validate with ARI and silhouette,
5. characterize clusters by centroid, spread, population,
6. identify core representatives,
7. identify transition-like assigned points,
8. flag anomalies by GLOSH,
9. summarize everything in a physical table,
10. visualize the entire result. fileciteturn20file4

### Why this section matters

Many HDBSCAN tutorials stop after plotting labels. This section goes further and asks:

- Which parameter should I choose?
- How many states/phases survive?
- What are their physical signatures?
- How large is the noise shell?
- Which points are core, transitional, or anomalous?
- How do I summarize this in one final figure/table?

### Why this is a realistic workflow

This is very close to how one would use HDBSCAN in actual scientific practice:

- scan plausible settings,
- choose a robust model,
- report cluster descriptors in domain-specific units,
- separate trusted core members from ambiguous regions,
- and explicitly acknowledge anomaly/noise populations.

---

## How to interpret the produced figures correctly

This project contains many visually rich plots. The following interpretation rules are essential.

### Grey points often mean “rejected” not “wrong”

In many panels, grey corresponds to `label = -1`, i.e. noise.
That usually means:

- interfacial region,
- diffuse state boundary,
- low-density bridge,
- or ambiguous assignment.

It is not simply “garbage.”

### High probability does not mean thermodynamically dominant

A tiny compact niche can have uniformly high probability but low persistence and small population. Probability is about **within-cluster support**, not global importance.

### High GLOSH does not necessarily mean foreign data

A point with high GLOSH may be:

- a true external anomaly,
- a fringe member of a known cluster,
- a transition-like point,
- or a rare internal variant.

### ARI may or may not include noise, depending on section

This is one of the most important subtle details in the project.
In several places the code computes ARI only on non-noise points. That means a high ARI can coexist with substantial noise if the retained points are very pure.

### The custom condensed tree is qualitative

The project’s condensed-tree panel is useful, but it is not the official HDBSCAN tree plot. Use it to understand the hierarchy qualitatively and use persistence values as the more trustworthy quantitative summary. fileciteturn20file1turn20file3

---

## Important caveats and subtle implementation details

This is a project worth reading carefully because some implementation choices strongly affect interpretation.

### 1. Standardization is used almost everywhere

Most analyses call `StandardScaler().fit_transform(X)` before clustering. This means the project assumes the feature dimensions should contribute on comparable scales.

That is usually the right choice for mixed descriptors, but if one feature’s absolute scale is physically meaningful, you may need a different preprocessing strategy.

### 2. Many evaluations focus on non-noise points

This is not wrong, but it must be remembered.
A model can have excellent non-noise ARI while still rejecting many points.
That means the extracted cluster cores are reliable, but the model may be low-coverage.

### 3. The anomaly pipeline is not a strict one-pass novelty detector

The code builds a threshold from the training subset of scores but computes scores after refitting on training + test combined. That is fine for a tutorial, but if you need strict deployment realism, you may want a train-only scoring strategy.

### 4. The script uses `Agg`

Plots are saved directly to files and not displayed interactively. That is ideal for remote execution.

### 5. Randomness is seeded

The file sets `RNG = np.random.default_rng(42)` and `SEED = 42`, which improves reproducibility for synthetic dataset generation, injected anomalies, and shuffled examples. fileciteturn20file1

### 6. Some imported symbols are not central

The script imports `KMeans`, `SK_HDBSCAN`, `mcolors`, and `pdist`, but the tutorial logic is driven mainly by the custom `fit_hdbscan` wrapper around the `hdbscan` library, plus scikit-learn utilities for metrics, PCA, nearest neighbors, and data generation. fileciteturn20file1

---

## How to adapt this project to real data

This project can be repurposed for real scientific or applied datasets with only moderate effort.

### For MD / conformational data

Replace the synthetic `polymer_conf` dataset with your own framewise features, for example:

- torsion angles,
- inter-residue distances,
- radius of gyration,
- contact fractions,
- principal components from structural embeddings.

Then keep sections G-like logic:

- fit HDBSCAN,
- inspect state counts and persistence,
- save representative core frames,
- use probability to define state centers,
- use GLOSH to rank rare frames.

### For colloidal / phase data

Replace `colloidal_phases` with per-particle or per-configuration descriptors such as:

- bond-order parameters,
- local density,
- orientational order,
- cluster coordination numbers,
- translational order measures.

Then keep sections H and J:

- compare EoM vs leaf,
- scan `min_cluster_size`,
- build physically interpretable centroid tables,
- inspect the noise population as an interfacial shell.

### For anomaly detection

Use section I as a starting point, but consider stricter deployment logic if needed:

- train on normal-only data,
- score new data without refitting on all points,
- compare score distributions by anomaly class,
- tune the threshold by acceptable false-positive rate.

---

## Suggested extensions

This tutorial is already rich, but several additions could make it even stronger.

### 1. Add confusion-style cluster-vs-truth matrices

This would make it easier to see exactly which true classes split or merge.

### 2. Save representative members per cluster

For scientific workflows, one often wants the top-N highest-probability members per state.

### 3. Add train-only anomaly scoring alternatives

For stricter novelty detection, you could compare HDBSCAN/GLOSH against:

- LOF,
- Isolation Forest,
- One-Class SVM,
- autoencoder reconstruction error.

### 4. Add UMAP/t-SNE visualization as optional extras

PCA is clean and interpretable, but nonlinear embeddings may reveal curved manifolds more clearly.

### 5. Add a proper command-line interface

For example:

```bash
python hdbscan_v1.py --section G --dataset polymer_conf --mcs 40 --ms 10
```

### 6. Add CSV exports

Exporting cluster assignments, probabilities, and outlier scores would make downstream analysis easier.

### 7. Compare `eom` and `leaf` more broadly

Running that comparison on multiple datasets would make the hierarchy story even more instructive.

---

## Practical summary

This project is best thought of as a **research-style HDBSCAN notebook converted into a script**.

It teaches that HDBSCAN is valuable because it can:

- discover clusters across density scales,
- leave ambiguous points unassigned instead of forcing them,
- quantify how strongly points belong to states,
- rank atypical points with outlier scores,
- expose cluster stability through persistence,
- and support physically interpretable end-to-end analysis pipelines.

The script’s ten sections are well chosen:

- **A** introduces the outputs,
- **B** introduces the hierarchy,
- **C** teaches soft membership,
- **D** teaches outlier scoring,
- **E** teaches robustness and parameter effects,
- **F** explains why HDBSCAN beats DBSCAN on variable-density data,
- **G** shows state discovery in conformational landscapes,
- **H** compares coarse and fine hierarchy extraction in colloidal phases,
- **I** demonstrates anomaly detection behavior,
- **J** ties everything together into a real inference pipeline. fileciteturn20file1turn20file0

If you are learning HDBSCAN, this project is a very strong conceptual foundation.
If you are doing scientific clustering, it is also a very useful template.
If you are building your own state/phase analysis pipeline, sections G through J are especially valuable because they show how to move from raw clustering output to interpretation, validation, and reporting.

