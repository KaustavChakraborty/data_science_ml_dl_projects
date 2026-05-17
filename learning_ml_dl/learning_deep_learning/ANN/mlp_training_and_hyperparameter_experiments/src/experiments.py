"""
============================================================

 GOAL:
   Isolate ONE variable at a time and observe its effect on
   the learning curves. This is the most important script
   for developing TRUE intuition about what each hyperparameter
   does to a neural network.

 EXPERIMENTS:
   A. Network depth   (1 to 6 hidden layers)
   B. Network width   (8 to 512 neurons per layer)
   C. Learning rate   (0.1, 0.01, 0.001, 0.0001)
   D. Activation fn   (ReLU, LeakyReLU, Tanh, Sigmoid)
   E. Batch size      (8, 32, 128, 512)

 WHAT TO OBSERVE:
   Each experiment isolates one factor. Study how the
   learning curves change — convergence speed, final accuracy,
   stability (oscillation), and overfitting.
============================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import time

# =============================================================================
# PRINT SCRIPT HEADER
# =============================================================================
print("=" * 60)
print("  HYPERPARAMETER EXPERIMENTS")
print("=" * 60)


# =============================================================================
# DATA SETUP — FIXED FOR ALL EXPERIMENTS
# =============================================================================

print("\nLoading data ...")

# -----------------------------------------------------------------------------
# Reproducibility settings
# -----------------------------------------------------------------------------

SEED   = 42
EPOCHS = 80
torch.manual_seed(SEED)
np.random.seed(SEED)

# -----------------------------------------------------------------------------
# Device selection
# -----------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
       "wine-quality/winequality-red.csv")
try:
    # First try loading directly from the UCI website
    df = pd.read_csv(url, sep=";")
except Exception:
    # If internet is unavailable, fall back to a local file with the same name
    df = pd.read_csv("winequality-red.csv", sep=";")

# -----------------------------------------------------------------------------
# Convert multiclass wine quality into binary classification
# -----------------------------------------------------------------------------

df["label"]  = (df["quality"] >= 7).astype(int)
# -----------------------------------------------------------------------------
# Separate input features from target labels
# -----------------------------------------------------------------------------
feat_cols    = [c for c in df.columns if c not in ["quality", "label"]]
# Extract input matrix X.
X_np         = df[feat_cols].values.astype(np.float32)
# Extract binary target vector y.
y_np         = df["label"].values.astype(np.float32)

# -----------------------------------------------------------------------------
# Create train/validation/test split
# -----------------------------------------------------------------------------
# Total number of samples.
n         = len(X_np)
# Use 70% for training.
n_train   = int(0.70 * n)
# Use 15% for validation.
n_val     = int(0.15 * n)
# Remaining 15% is test data.
n_test    = n - n_train - n_val
rng       = np.random.default_rng(SEED)
idx       = rng.permutation(n)
# First 70% of shuffled indices become training indices
tr_idx    = idx[:n_train]
# Next 15% become validation indices
vl_idx    = idx[n_train:n_train+n_val]
# Remaining samples become test indices
te_idx    = idx[n_train+n_val:]

# -----------------------------------------------------------------------------
# Standardize features using training statistics only
# -----------------------------------------------------------------------------
scaler      = StandardScaler().fit(X_np[tr_idx])
X_tr        = torch.tensor(scaler.transform(X_np[tr_idx]))
X_vl        = torch.tensor(scaler.transform(X_np[vl_idx]))
y_tr        = torch.tensor(y_np[tr_idx])
y_vl        = torch.tensor(y_np[vl_idx])

# -----------------------------------------------------------------------------
# Move tensors to selected device
# -----------------------------------------------------------------------------
X_tr, y_tr  = X_tr.to(device), y_tr.to(device)
X_vl, y_vl  = X_vl.to(device), y_vl.to(device)
print(f"Train: {len(X_tr)}  Val: {len(X_vl)}")


# =============================================================================
# UTILITY FUNCTION 1: BUILD AN MLP
# =============================================================================

def make_mlp(hidden_sizes, activation=nn.ReLU, input_size=11, output_size=1):
    """
    Build an MLP with given hidden layer sizes and activation function.
    Output: sigmoid for binary classification.
    """
    layers = []
    prev   = input_size
    # Loop over all requested hidden-layer sizes.
    # Example: hidden_sizes = [64, 32]
    # First iteration creates Linear(11, 64)
    # Second iteration creates Linear(64, 32)
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(activation())
        # The output size of this layer becomes the input size for the next layer.
        prev = h
    # Add the final Linear layer that maps from the last hidden representation to
    # a single scalar logit-like value.
    layers.append(nn.Linear(prev, output_size))
    # Add Sigmoid to convert the final scalar to a probability-like number.
    layers.append(nn.Sigmoid())
    # nn.Sequential chains all layers in order.
    return nn.Sequential(*layers).to(device)

# =============================================================================
# UTILITY FUNCTION 2: TRAIN ONE MODEL AND RETURN HISTORY
# =============================================================================

def train_and_eval(model, lr=1e-3, batch_size=32, epochs=EPOCHS, seed=SEED):
    """
    Train a model and return history dicts.
    Returns: (train_losses, val_losses, val_accs, final_val_acc, time_seconds)
    """
    torch.manual_seed(seed)   # reset for fair comparison

    # Re-initialize weights so all experiments start from same random state
    for m in model.modules():
        # Only initialize fully connected Linear layers.
        # Activation layers such as ReLU, Tanh, Sigmoid do not have weights.
        if isinstance(m, nn.Linear):
            # Kaiming/He initialization is designed for ReLU-like activations.
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            # Initialize biases to zero.
            nn.init.zeros_(m.bias)

    loader    = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    # Binary cross-entropy loss for probability outputs.
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Lists for storing training history.
    # These lists will later be plotted as learning curves.
    train_losses, val_losses, val_accs = [], [], []
    # Start timer to measure training cost.
    t0 = time.time()

    # -------------------------------------------------------------------------
    # Main epoch loop
    # -------------------------------------------------------------------------
    for epoch in range(epochs):
        # =====================================================================
        # TRAINING PHASE
        # =====================================================================
        model.train()
        # Accumulators for average training loss and accuracy
        ep_loss, ep_corr, ep_n = 0.0, 0, 0
        for xb, yb in loader:
            pred = model(xb).squeeze()
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * len(yb)
            ep_corr += ((pred >= 0.5).float() == yb).sum().item()
            ep_n    += len(yb)
        train_losses.append(ep_loss / ep_n)

        # =====================================================================
        # VALIDATION PHASE
        # =====================================================================
        model.eval()
        with torch.no_grad():
            # Predict probabilities for the full validation set at once.
            vp     = model(X_vl).squeeze()
            # Compute validation BCE loss.
            vloss  = criterion(vp, y_vl).item()
            # Compute validation accuracy using the same 0.5 threshold
            vacc   = ((vp >= 0.5).float() == y_vl).float().mean().item()
        # Store validation metrics for plotting
        val_losses.append(vloss)
        val_accs.append(vacc)

    elapsed = time.time() - t0
    # Return all histories and summary numbers
    return train_losses, val_losses, val_accs, val_accs[-1], elapsed

# =============================================================================
# UTILITY FUNCTION 3: PLOT EXPERIMENT RESULTS
# =============================================================================

def plot_experiment(results_dict, title, ylabel_loss="Val Loss",
                    save_name="exp.png", colors=None):
    """
    Plot loss and accuracy for multiple configs side by side.
    results_dict: { label: (train_losses, val_losses, val_accs, final_acc) }
    """
    if colors is None:
        colors = ["#7F77DD", "#1D9E75", "#D85A30", "#BA7517",
                  "#0F6E56", "#993C1D", "#185FA5", "#639922"][:len(results_dict)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=13)

    for (label, (tr_l, vl_l, vl_a, final_acc, elapsed)), color in zip(results_dict.items(), colors):
        ax1.plot(vl_l, color=color, lw=2, label=f"{label} ({elapsed:.0f}s)")
        ax2.plot(vl_a, color=color, lw=2, label=f"{label}  final={final_acc:.2%}")

    for ax, title_sub in zip([ax1, ax2], [ylabel_loss, "Val Accuracy"]):
        ax.set_xlabel("Epoch"); ax.set_title(title_sub)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax1.set_ylabel("Validation Loss")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0.75, 1.0)

    plt.tight_layout()
    plt.savefig(save_name, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_name}")
    # plt.show()


# =============================================================================
# EXPERIMENT A: NETWORK DEPTH
# =============================================================================
# Question: Does adding more layers improve performance?


print("\n" + "═"*60)
print("  EXPERIMENT A: Effect of Network Depth")
print("═"*60)

depth_configs = {
    "1 hidden  [64]":           [64],
    "2 hidden  [64,32]":        [64, 32],
    "3 hidden  [64,64,32]":     [64, 64, 32],
    "4 hidden  [64,64,32,16]":  [64, 64, 32, 16],
    "6 hidden  [64,64,32,32,16,8]": [64, 64, 32, 32, 16, 8],
}
# Store results for all depth configurations
results_depth = {}

# Loop over all depth settings.
for name, sizes in depth_configs.items():
    print(f"  Training: {name} ...", end=" ", flush=True)
    # Build model with the chosen hidden-layer sizes
    model = make_mlp(sizes)
    # Count model parameters.
    n_p   = sum(p.numel() for p in model.parameters())
    # Train and evaluate this model
    res   = train_and_eval(model)
    # Store result using a label that includes parameter count
    results_depth[f"{name} ({n_p:,}p)"] = res
    # res[3] is final validation accuracy
    print(f"final val acc = {res[3]:.2%}  ({res[4]:.1f}s)")

# Plot all depth results
plot_experiment(results_depth,
                "Experiment A: Network Depth\n(wider = more parameters)",
                save_name="07A_depth.png")



# =============================================================================
# EXPERIMENT B: NETWORK WIDTH
# =============================================================================
# Question: How does neuron count per layer affect learning?
#
# What changes?
# -------------
# Width of two hidden layers:
#     [8,8], [32,32], [64,64], [128,128], [512,512]
#
# What stays fixed?
# -----------------
#     - depth = 2 hidden layers
#     - learning rate = 0.001
#     - activation = ReLU
#     - batch size = 32

print("\n" + "═"*60)
print("  EXPERIMENT B: Effect of Network Width")
print("═"*60)

# Hidden-layer layouts for width experiment.
# Depth is fixed at two hidden layers.
width_configs = {
    "width=8    [8,8]":      [8,   8],
    "width=32   [32,32]":    [32,  32],
    "width=64   [64,64]":    [64,  64],
    "width=128  [128,128]":  [128, 128],
    "width=512  [512,512]":  [512, 512],
}

# Store width results
results_width = {}
# Train one model for each width setting
for name, sizes in width_configs.items():
    print(f"  Training: {name} ...", end=" ", flush=True)
    model = make_mlp(sizes)
    n_p   = sum(p.numel() for p in model.parameters())
    res   = train_and_eval(model)
    results_width[f"{name} ({n_p:,}p)"] = res
    print(f"final val acc = {res[3]:.2%}  ({res[4]:.1f}s)")

# Plot width comparison
plot_experiment(results_width,
                "Experiment B: Network Width\n(2 hidden layers, varying neuron count)",
                save_name="07B_width.png")



# =============================================================================
# EXPERIMENT C: LEARNING RATE
# =============================================================================
# Question: How does learning rate affect convergence?
#
# What changes?
# -------------
# Only the Adam learning rate changes.
#
# What stays fixed?
# -----------------
#     - architecture = [64, 32]
#     - activation = ReLU
#     - batch size = 32
#     - epochs = 80
# The learning rate is THE most important hyperparameter to tune.

print("\n" + "═"*60)
print("  EXPERIMENT C: Effect of Learning Rate")
print("═"*60)

# Candidate learning rates.
# The labels include interpretation hints.
lr_configs = {
    "lr=1.0   " :   1.0,
    "lr=0.1   " :   0.1,
    "lr=0.01  " :   0.01,
    "lr=0.001 " :   0.001,
    "lr=0.0001 ":   0.0001,
}

# Store learning-rate results
results_lr = {}
fixed_arch = [64, 32]   # keep architecture constant

# Train one model per learning-rate value
for name, lr in lr_configs.items():
    print(f"  Training: {name} ...", end=" ", flush=True)
    # Same model architecture for every learning rate
    model = make_mlp(fixed_arch)
    # Only lr changes here
    res   = train_and_eval(model, lr=lr)
    # Store result
    results_lr[name] = res
    print(f"final val acc = {res[3]:.2%}  ({res[4]:.1f}s)")

# -----------------------------------------------------------------------------
# Custom plot for learning-rate experiment
# -----------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Experiment C: Learning Rate Effect\n(same architecture [64,32], Adam optimizer)", fontsize=13)
colors = ["#888780", "#D85A30", "#BA7517", "#1D9E75", "#7F77DD"]

for (name, (tr_l, vl_l, vl_a, final_acc, elapsed)), color in zip(results_lr.items(), colors):
    ax1.plot(vl_l, lw=2, color=color, label=name)
    ax2.plot(vl_a, lw=2, color=color, label=f"{name}  final={final_acc:.2%}")

ax1.set_xlabel("Epoch"); ax1.set_ylabel("Val Loss"); ax1.set_title("Validation Loss")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Val Accuracy"); ax2.set_title("Val Accuracy")
ax1.set_ylim(0, 0.6); ax2.set_ylim(0.6, 1.0)
for ax in [ax1, ax2]:
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("07C_learning_rate.png", dpi=150, bbox_inches="tight")
print("  Saved: 07C_learning_rate.png")
# plt.show()



# =============================================================================
# EXPERIMENT D: ACTIVATION FUNCTIONS
# =============================================================================
# Question: How does the choice of activation affect learning?
#
# What changes?
# -------------
# Only the hidden-layer activation function changes.
#
# What stays fixed?
# -----------------
#     - architecture = [64, 32]
#     - learning rate = 0.001
#     - optimizer = Adam
#     - batch size = 32

print("\n" + "═"*60)
print("  EXPERIMENT D: Effect of Activation Function")
print("═"*60)

act_configs = {
    "ReLU":                nn.ReLU,
    "LeakyReLU (alpha=0.1)":  lambda: nn.LeakyReLU(0.1),
    "Tanh":                nn.Tanh,
    "Sigmoid":             nn.Sigmoid,
    "ELU":                 nn.ELU,
}

# Store activation results
results_act = {}

# Train one model for each activation function
for name, act_class in act_configs.items():
    print(f"  Training: {name} ...", end=" ", flush=True)
    # Architecture is fixed.
    # Only the activation constructor changes.
    model = make_mlp([64, 32], activation=act_class)
    # Train and validate.
    res   = train_and_eval(model)
    # Store results.
    results_act[name] = res
    print(f"final val acc = {res[3]:.2%}  ({res[4]:.1f}s)")

# Plot activation comparison.
plot_experiment(results_act,
                "Experiment D: Activation Function Effect\n(architecture [64,32], lr=0.001)",
                save_name="07D_activation.png")



# =============================================================================
# EXPERIMENT E: BATCH SIZE
# =============================================================================
# Question: How does batch size affect training dynamics?
#
# What changes?
# -------------
# Only batch size changes.
#
# What stays fixed?
# -----------------
#     - architecture = [64, 32]
#     - activation = ReLU
#     - learning rate = 0.001
#     - optimizer = Adam
#
#
# Key insight: small batches add NOISE to the gradient estimate.
# This noise is actually beneficial — acts as implicit regularization!

print("\n" + "═"*60)
print("  EXPERIMENT E: Effect of Batch Size")
print("═"*60)

# Batch-size candidates.
batch_configs = {
    "batch=8   (very noisy)":  8,
    "batch=32  (standard)":    32,
    "batch=128 (large)":       128,
    "batch=512 (near-full)":   512,
}

# Store batch-size results.
results_batch = {}

# Train one model for each batch size.
for name, bs in batch_configs.items():
    print(f"  Training: {name} ...", end=" ", flush=True)
    # Same model architecture for every batch size.
    model = make_mlp([64, 32])
    # Only batch_size changes here.
    res   = train_and_eval(model, batch_size=bs)
    # Store results.
    results_batch[name] = res
    # Approximate total number of optimizer updates
    n_updates = EPOCHS * (len(X_tr) // bs)

    print(f"final val acc = {res[3]:.2%}  ({n_updates} weight updates total)")

# Plot batch-size comparison
plot_experiment(results_batch,
                "Experiment E: Batch Size Effect\n(architecture [64,32], lr=0.001, Adam)",
                save_name="07E_batch_size.png")


# =============================================================================
# EXPERIMENT F: SUMMARY COMPARISON TABLE
# =============================================================================
#
# After running all experiments, this section prints a compact summary:
#     - best final validation accuracy in each experiment group
#     - worst final validation accuracy in each experiment group
#
# Important limitation
# --------------------
# This summary uses final validation accuracy, not best validation accuracy over
# all epochs. A model may have reached its best value earlier and then overfit.
#
# For a production workflow, one would usually save the model at the epoch with
# minimum validation loss or maximum validation F1-score.
# =============================================================================

print("\n" + "═"*60)
print("  EXPERIMENT SUMMARY")
print("═"*60)

print(f"\n  {'Experiment':<45} {'Best val acc':>12} {'Notes'}")
print(f"  {'-'*75}")

# Group all experiment result dictionaries
all_results = [
    ("A. Depth",         results_depth),
    ("B. Width",         results_width),
    ("C. Learning Rate", results_lr),
    ("D. Activation",    results_act),
    ("E. Batch Size",    results_batch),
]

# Print best and worst final validation accuracy for every experiment
for exp_name, results in all_results:
    # Find configuration with maximum final validation accuracy
    best_name  = max(results, key=lambda k: results[k][3])
    best_acc   = results[best_name][3]
    # Find configuration with minimum final validation accuracy
    worst_name = min(results, key=lambda k: results[k][3])
    worst_acc  = results[worst_name][3]
    # Print compact result
    print(f"  {exp_name:<45} best: {best_acc:.2%} ({best_name[:25]}...)")
    print(f"  {'':45} worst:{worst_acc:.2%} ({worst_name[:25]}...)")
    print()


print("[DONE] Script 07 complete.")
print("Generated plots:")
for name in ["07A_depth.png", "07B_width.png", "07C_learning_rate.png",
             "07D_activation.png", "07E_batch_size.png"]:
    print(f"  {name}")
