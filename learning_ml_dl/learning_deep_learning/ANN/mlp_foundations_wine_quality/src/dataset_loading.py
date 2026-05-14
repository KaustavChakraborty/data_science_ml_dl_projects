"""
============================================================
 02_dataset_loading.py
 Deep Learning for Soft Matter Physics
 ── Wine Quality Dataset: Load, Explore, Preprocess ─────────

 This script demonstrates a COMPLETE classical deep learning
 data pipeline using PyTorch.

 The goal is NOT just to load data.
 The goal is to understand:

   1. How ML datasets are structured
   2. How preprocessing works
   3. Why normalization matters
   4. How PyTorch datasets work internally
   5. How batching happens
   6. How train/validation/test splits are made
   7. How data flows into neural networks


 DATASET:
   UCI Wine Quality (Red Wine)
   URL: https://archive.ics.uci.edu/ml/machine-learning-databases/
        wine-quality/winequality-red.csv
   - 1599 samples (wine batches)
   - 11 physicochemical features (inputs)
   - 1 quality score 0-10 (target)
============================================================
"""
# ============================================================
# IMPORTS
# ============================================================
import torch
from torch.utils.data import Dataset, DataLoader, random_split

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("  DATASET LOADING AND PREPROCESSING")
print("=" * 60)


# ============================================================
# PART 1: LOAD DATASET
# ============================================================

print("\n── PART 1: Loading dataset ──────────────────────────────────")

url = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)

try:
    df = pd.read_csv(url, sep=";")
    print(f"  Downloaded from UCI repository")
except Exception:
    # Fallback: if no internet, try local file
    print("  No internet — attempting local file 'winequality-red.csv'")
    df = pd.read_csv("winequality-red.csv", sep=";")

# ============================================================
# BASIC DATASET INSPECTION
# ============================================================

print(f"\n  Dataset shape    : {df.shape}   ({df.shape[0]} samples, {df.shape[1]} columns)")
print(f"  Memory usage     : {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
print(f"\n  Columns:\n    {df.columns.tolist()}")

print(f"\n  First 3 rows:\n{df.head(3).to_string(index=False)}")

# ============================================================
# STATISTICAL SUMMARY
# ============================================================

print(f"\n  Statistical summary:")
print(df.describe().round(2).to_string())

# ============================================================
# QUALITY DISTRIBUTION
# ============================================================
print(f"\n  Quality score distribution:")
vc = df["quality"].value_counts().sort_index()

# Loop through each quality score.
for score, count in vc.items():
    bar = "-" * (count // 20)
    print(f"    Quality {score}: {count:4d} samples  {bar}")


# ============================================================
# PART 2: CREATE BINARY LABELS
# ============================================================
# We turn the regression target (quality 3-8) into a binary classification:
#   quality >= 7 => label = 1  ("good wine")
#   quality <  7 => label = 0  ("ordinary wine")
#
# This threshold (7) is a domain choice — wines rated 7+ are generally
# considered premium in sommelier literature.

print("\n── PART 2: Creating binary labels ──────────────────────────")

QUALITY_THRESHOLD = 7
df["label"] = (df["quality"] >= QUALITY_THRESHOLD).astype(int)

n_good = df["label"].sum()
n_bad  = len(df) - n_good
ratio  = n_good / len(df)

print(f"  Threshold: quality >= {QUALITY_THRESHOLD} => label=1")
print(f"  Class 0 (ordinary) : {n_bad}  samples  ({1-ratio:.1%})")
print(f"  Class 1 (good)     : {n_good}  samples  ({ratio:.1%})")
print(f"\n   Dataset is IMBALANCED — only {ratio:.1%} positive class")
print(f"    A naive classifier that always predicts 0 gets {1-ratio:.1%} accuracy.")
print(f"    => We must look at precision/recall, not just accuracy!")


# ============================================================
# PART 3: FEATURE/LABEL SEPARATION
# ============================================================

print("\n── PART 3: Separating features and labels ───────────────────")

feature_names = [c for c in df.columns if c not in ["quality", "label"]]
X = df[feature_names].values    # shape: (1599, 11),  dtype: float64
y = df["label"].values          # shape: (1599,),     dtype: int64

print(f"  Feature matrix X : shape {X.shape},  dtype {X.dtype}")
print(f"  Label vector   y : shape {y.shape},  dtype {y.dtype}")
print(f"\n  Feature value ranges (BEFORE normalization):")
for i, name in enumerate(feature_names):
    print(f"    {name:<25} min={X[:,i].min():.2f}  max={X[:,i].max():.2f}  std={X[:,i].std():.2f}")

# ============================================================
# PART 4: FEATURE NORMALIZATION
# ============================================================
# StandardScaler applies:  x_scaled = (x - mean) / std
# After scaling: every feature has mean=0 and std=1

print("\n── PART 4: Feature normalization ────────────────────────────")

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)  # shape: (1599, 11)

print(f"  Feature value ranges (AFTER normalization):")
for i, name in enumerate(feature_names):
    print(f"    {name:<25} mean={X_scaled[:,i].mean():.3f}  std={X_scaled[:,i].std():.3f}")

print(f"\n  Scaler mean (first 3 features): {scaler.mean_[:3].round(3)}")
print(f"  Scaler std  (first 3 features): {scaler.scale_[:3].round(3)}")
print(f"\n  To transform a new sample at inference time:")
print(f"    x_new_scaled = scaler.transform(x_new)")


# ============================================================
# PART 5: CONVERT TO PYTORCH TENSORS
# ============================================================

print("\n── PART 5: Converting to PyTorch tensors ────────────────────")

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)  # shape: (1599, 11)
y_tensor = torch.tensor(y,        dtype=torch.float32)  # shape: (1599,)
# Use dtype=torch.long (int64) if using nn.CrossEntropyLoss

print(f"  X_tensor  : shape={X_tensor.shape},  dtype={X_tensor.dtype}")
print(f"  y_tensor  : shape={y_tensor.shape},  dtype={y_tensor.dtype}")
# Show first sample.
print(f"\n  First sample features : {X_tensor[0]}")
print(f"  First sample label    : {y_tensor[0].item()}")


# ============================================================
# PART 6: CUSTOM DATASET CLASS
# ============================================================

print("\n── PART 6: Custom Dataset class ─────────────────────────────")

class WineDataset(Dataset):
    """
    PyTorch Dataset for the Wine Quality dataset.

    Args:
        X (Tensor): feature matrix, shape (N, 11)
        y (Tensor): label vector,   shape (N,)
    """

    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        # Sanity check.
        #
        # Features and labels MUST have same sample count.
        assert len(X) == len(y), "X and y must have same number of samples"
        # Store tensors inside object
        self.X = X
        self.y = y

    def __len__(self) -> int:
        """Total number of samples in the dataset."""
        return len(self.X)

    def __getitem__(self, idx: int):
        """Return one (feature_vector, label) pair."""
        return self.X[idx], self.y[idx]

# Create dataset object
dataset = WineDataset(X_tensor, y_tensor)
print(f"  Dataset total size    : {len(dataset)} samples")

# Test indexing
sample_x, sample_y = dataset[0]
print(f"  dataset[0] features   : shape {sample_x.shape}")
print(f"  dataset[0] label      : {sample_y.item()}")

# Slicing works too
batch_x, batch_y = dataset[10:15]
print(f"  dataset[10:15]        : features {batch_x.shape}, labels {batch_y.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# PART 7: TRAIN / VALIDATION / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
# Three-way split is standard practice:
#
#   Training set   (70%) => the model sees this data and learns from it
#   Validation set (15%) => used DURING training to tune hyperparameters
#                           (learning rate, layer sizes, etc.)
#                           model does NOT learn from this — only evaluated
#   Test set       (15%) => used ONLY ONCE at the end for final evaluation
#                           gives an unbiased estimate of real-world performance


print("\n── PART 7: Train / Val / Test split ─────────────────────────")

# Total dataset size
n_total = len(dataset)
# Training set
n_train = int(0.70 * n_total)    # 1119 samples
# Validation set
n_val   = int(0.15 * n_total)    # 239  samples
# Test set
n_test  = n_total - n_train - n_val  # 241 samples (remainder)

print(f"  Total  : {n_total}")
print(f"  Train  : {n_train}  ({n_train/n_total:.0%})")
print(f"  Val    : {n_val}    ({n_val/n_total:.0%})")
print(f"  Test   : {n_test}   ({n_test/n_total:.0%})")

# random_split returns Subset objects with random index assignments
# Generator with fixed seed → reproducible split every run
train_set, val_set, test_set = random_split(
    dataset,
    [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(42)
)

print(f"\n  train_set type : {type(train_set)}")
print(f"  val_set   size : {len(val_set)}")
print(f"  test_set  size : {len(test_set)}")


# ============================================================
# PART 8: DATALOADERS
# ============================================================
# DataLoader wraps a Dataset and provides an iterator over mini-batches.
#
# Key arguments:
#   batch_size : how many samples per batch
#                  - Smaller => noisier gradients, more updates per epoch
#                  - Larger  => smoother gradients, fewer updates per epoch
#                  - Common values: 16, 32, 64, 128
#   shuffle    : True for training (randomize order each epoch)
#                False for val/test (order doesn't matter, but reproducible)
#   num_workers: parallel data loading threads (use 0 on Windows)

print("\n── PART 8: Creating DataLoaders ─────────────────────────────")

# Number of samples per mini-batch
BATCH_SIZE = 32

# Training loader.
#
# shuffle=True:
#   randomize sample order each epoch.
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)

# Validation loader.
#
# shuffle=False because:
#   evaluation does not require randomness.
val_loader   = DataLoader(val_set,   batch_size=64,         shuffle=False, num_workers=0)

# Test loader.
test_loader  = DataLoader(test_set,  batch_size=64,         shuffle=False, num_workers=0)

# Number of batches.
n_train_batches = len(train_loader)


print(f"  Batch size           : {BATCH_SIZE}")
print(f"  Train batches/epoch  : {n_train_batches}  (= ceil({n_train}/{BATCH_SIZE}))")
print(f"  Val   batches        : {len(val_loader)}")
print(f"  Test  batches        : {len(test_loader)}")

# ============================================================
# INSPECT ONE MINI-BATCH
# ============================================================

print(f"\n  Inspecting first training batch:")

# Iterating over DataLoader automatically yields batches
for X_batch, y_batch in train_loader:
    print(f"    X_batch shape : {X_batch.shape}   (batch_size × n_features)")
    print(f"    y_batch shape : {y_batch.shape}   (batch_size,)")
    print(f"    X_batch dtype : {X_batch.dtype}")
    print(f"    y_batch dtype : {y_batch.dtype}")
    print(f"    Labels in batch: {y_batch[:10].tolist()}")
    print(f"    Positive rate  : {y_batch.mean().item():.2f}")
    break   # only look at first batch


# ============================================================
# PART 9: DATA VISUALIZATION
# ============================================================

print("\n── PART 9: Data visualization ───────────────────────────────")

fig, axes = plt.subplots(3, 4, figsize=(14, 9))
fig.suptitle("Wine Quality Dataset — Feature Distributions\n(Blue=ordinary, Orange=good)",
             fontsize=12)

df_good = df[df["label"] == 1]
df_bad  = df[df["label"] == 0]

for i, (name, ax) in enumerate(zip(feature_names, axes.flatten())):
    ax.hist(df_bad[name],  bins=25, alpha=0.6, color="#7F77DD", label="ordinary (0)", density=True)
    ax.hist(df_good[name], bins=25, alpha=0.6, color="#D85A30", label="good (1)",     density=True)
    ax.set_title(name, fontsize=9)
    ax.set_xlabel("Value", fontsize=8)
    ax.tick_params(labelsize=7)
    if i == 0:
        ax.legend(fontsize=7)

# Hide unused subplot
axes.flatten()[-1].set_visible(False)

plt.tight_layout()
plt.savefig("02_feature_distributions.png", dpi=150, bbox_inches="tight")
print("  Saved: 02_feature_distributions.png")
# plt.show()

# ── Correlation heatmap ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
corr = df[feature_names + ["label"]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax, annot_kws={"size": 8})
ax.set_title("Feature correlation matrix\n(last row = correlation with quality label)", fontsize=11)
plt.tight_layout()
plt.savefig("02_correlation_heatmap.png", dpi=150, bbox_inches="tight")
print("  Saved: 02_correlation_heatmap.png")
# plt.show()

print("\n[DONE] Script 02 complete.")
print("Key takeaways:")
print("  1. Normalization is essential — features on different scales hurt training")
print("  2. Always split BEFORE fitting the scaler (data leakage prevention)")
print("  3. Dataset class gives PyTorch a standard interface to your data")
print("  4. DataLoader handles batching + shuffling automatically")
print("  5. Check class balance — accuracy alone is misleading on imbalanced data")
print(f"\n  Saved objects for next scripts:")
print(f"    train_loader, val_loader, test_loader are ready")
