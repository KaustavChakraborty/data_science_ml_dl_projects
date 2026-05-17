"""
============================================================
 06_training_loop.py
 Deep Learning for Soft Matter Physics
 ── The Complete Training Pipeline ──────────────────────────

 GOAL:
   Put every piece together in a clean, production-ready
   training loop that you can use as a template for all
   future projects.

 PIPELINE:
   Data => Preprocessing => Model => Loss => Optimizer
   => Forward => Backward => Update => Validate => Plot => Evaluate

 CONCEPTS COVERED:
   - Full training loop with epoch structure
   - model.train() vs model.eval() mode
   - optimizer.zero_grad() => loss.backward() => optimizer.step()
   - Adam optimizer (adaptive learning rate)
   - Validation loss and accuracy at each epoch
   - Learning curve plotting
   - Final test evaluation
   - Sklearn classification report (precision, recall, F1)
   - Model saving and loading (state_dict)
============================================================
"""

# ==============================================================================
# 0. IMPORT LIBRARIES
# ==============================================================================

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ==============================================================================
# 1. PRINT SCRIPT HEADER
# ==============================================================================
print("=" * 60)
print("  COMPLETE TRAINING PIPELINE")
print("=" * 60)


# ==============================================================================
# 2. REPRODUCIBILITY SETUP
# ==============================================================================
# Setting seeds ensures identical results every run.
# Critical for debugging and fair comparison of experiments.

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")


# ==============================================================================
# 3. STEP 1: DATA LOADING AND PREPROCESSING
# ==============================================================================
# CORRECT pipeline order:
#   1. Split into train/val/test  (BEFORE any preprocessing statistics)
#   2. Fit scaler on TRAINING set only
#   3. Transform all three splits using training scaler
#
# This prevents data leakage: val/test statistics must not influence training.

print("\n── Step 1: Data loading and preprocessing ────────────────────")

# The red-wine-quality dataset is hosted by the UCI Machine Learning Repository.
# This URL points directly to the CSV file.
url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
       "wine-quality/winequality-red.csv")

# Try to load the dataset from the internet.
# If the internet is not available, fall back to a local CSV file named
# winequality-red.csv in the current working directory.
try:
    df = pd.read_csv(url, sep=";")
    print(f"  Loaded from URL: {df.shape}")
except Exception:
    df = pd.read_csv("winequality-red.csv", sep=";")
    print(f"  Loaded from local file: {df.shape}")

# ------------------------------------------------------------------------------
# 3.1 Convert the original wine-quality score into a binary label
# ------------------------------------------------------------------------------
df["label"] = (df["quality"] >= 7).astype(int)
# Feature columns are all physicochemical measurements.
# We exclude:
#   quality -> original target score
#   label   -> binary target created above
feature_cols = [c for c in df.columns if c not in ["quality", "label"]]

X_np = df[feature_cols].values.astype(np.float32)   # (1599, 11)
# Convert labels to NumPy float array
y_np = df["label"].values.astype(np.float32)         # (1599,)

# Split indices first (before scaling)
n      = len(X_np)
# Use 70% of samples for training
n_train = int(0.70 * n)
# Use 15% of samples for validation
n_val   = int(0.15 * n)
# Use the remaining 15% for final testing
n_test  = n - n_train - n_val
# Create a NumPy random generator with a fixed seed.
rng     = np.random.default_rng(SEED)
# permutation(n) returns a shuffled array containing
indices = rng.permutation(n)
# First 70% shuffled indices -> training set.
train_idx = indices[:n_train]
# Next 15% shuffled indices -> validation set.
val_idx   = indices[n_train:n_train + n_val]
# Remaining indices -> test set.
test_idx  = indices[n_train + n_val:]

# Slice the NumPy arrays into three independent splits.
X_train_np, y_train_np = X_np[train_idx], y_np[train_idx]
X_val_np,   y_val_np   = X_np[val_idx],   y_np[val_idx]
X_test_np,  y_test_np  = X_np[test_idx],  y_np[test_idx]

# ------------------------------------------------------------------------------
# 3.3 Fit StandardScaler on training data only
# ------------------------------------------------------------------------------
scaler = StandardScaler()
# Learn mean and standard deviation from training features only.
scaler.fit(X_train_np)                             # learn mean/std from train only

X_train = torch.tensor(scaler.transform(X_train_np))
X_val   = torch.tensor(scaler.transform(X_val_np))
X_test  = torch.tensor(scaler.transform(X_test_np))
y_train = torch.tensor(y_train_np)
y_val   = torch.tensor(y_val_np)
y_test  = torch.tensor(y_test_np)

# Move all tensors to the selected device.
# If device is cuda, tensors and model parameters must both be on the GPU.
# PyTorch will raise an error if model is on GPU but data is on CPU, or vice versa.
X_train, y_train = X_train.to(device), y_train.to(device)
X_val,   y_val   = X_val.to(device),   y_val.to(device)
X_test,  y_test  = X_test.to(device),  y_test.to(device)

# ------------------------------------------------------------------------------
# 3.4 Create DataLoaders
# ------------------------------------------------------------------------------
BATCH_SIZE = 32
# TensorDataset pairs each row of X_train with the corresponding y_train label.
# DataLoader then creates batches from this dataset.
# shuffle=True is used for training so that the model does not see samples in the same order every epoch.
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
# Validation loader does not need shuffling because no learning happens during validation. We only compute metrics.
val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=64,         shuffle=False)
# Test loader also does not need shuffling
test_loader  = DataLoader(TensorDataset(X_test,  y_test),  batch_size=64,         shuffle=False)

# Print split sizes and class balance.
# Positive rate is useful here because "good wine" is usually the minority class.
# If the positive rate is low, accuracy alone may be misleading.
print(f"  Train: {len(X_train):4d} samples  ({len(train_loader)} batches of {BATCH_SIZE})")
print(f"  Val  : {len(X_val):4d} samples")
print(f"  Test : {len(X_test):4d} samples")
print(f"  Positive rate — Train: {y_train.mean():.2%}  Val: {y_val.mean():.2%}  Test: {y_test.mean():.2%}")


# ==============================================================================
# 4. STEP 2: MODEL DEFINITION
# ==============================================================================

print("\n── Step 2: Model ─────────────────────────────────────────────")

class WineMLP(nn.Module):
    """
    MLP for wine quality binary classification.
    Architecture: 11 => 64 => ReLU => 32 => ReLU => 1 => Sigmoid
    """
    def __init__(self):
        # super().__init__() initializes the parent nn.Module class.
        # This is required so PyTorch can correctly register layers,
        # parameters, buffers, and submodules.
        super().__init__()
        # nn.Sequential chains layers in order.
        # The output of one layer becomes the input to the next layer.
        self.net = nn.Sequential(
            # First fully connected layer:
            #   input dimension  = 11 features
            #   output dimension = 64 hidden neurons
            # Each neuron learns a weighted combination of the 11 inputs.
            nn.Linear(11, 64),
            nn.ReLU(),
            # Second fully connected layer:
            #   maps 64 hidden features to 32 hidden features.
            nn.Linear(64, 32),
            nn.ReLU(),
            # Final linear layer:
            #   maps 32 hidden features to 1 output score.
            # The one output corresponds to binary classification.
            nn.Linear(32,  1),
            nn.Sigmoid()        # output belongs to (0,1) = P(good wine)
        )

    def forward(self, x):
        """
        Define the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input batch with shape (batch_size, 11).

        Returns
        -------
        torch.Tensor
            Output probabilities with shape (batch_size, 1).
        """
        return self.net(x)

# Create the model object and move it to the selected device.
model = WineMLP().to(device)
# Count trainable parameters.
# p.numel() gives the number of scalar values in a parameter tensor.
# requires_grad=True means the parameter will be updated during training.
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Model: {model}")
print(f"  Trainable parameters: {n_params:,}")


# ==============================================================================
# 5. STEP 3: LOSS FUNCTION AND OPTIMIZER
# ==============================================================================

print("\n── Step 3: Loss function and optimizer ──────────────────────")

# The model output already passes through Sigmoid, so nn.BCELoss() is suitable.
criterion = nn.BCELoss()                          # expects sigmoid output
# Adam optimizer updates model parameters using gradients from backpropagation.
#
# Compared with plain SGD, Adam adapts the effective learning rate for each
# parameter using estimates of first and second gradient moments.
#
# lr=1e-3 is a common starting value for Adam.
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print(f"  Loss     : nn.BCELoss()")
print(f"  Optimizer: Adam(lr=1e-3)")


# ==============================================================================
# 6. STEP 4: EVALUATION HELPER FUNCTION
# ==============================================================================

def evaluate(model, loader, criterion, device):
    """
    Evaluate the model on a validation or test DataLoader.

    Parameters
    ----------
    model : nn.Module
        The neural network to evaluate.

    loader : DataLoader
        DataLoader containing validation or test batches.

    criterion : loss function
        Here, nn.BCELoss(). Used to compute average loss.

    device : torch.device
        CPU or GPU device where tensors should be placed.

    Returns
    -------
    average_loss : float
        Mean BCE loss over all samples in the loader.

    accuracy : float
        Fraction of correctly classified samples.

    Why this helper is useful:
        Validation and test evaluation use the same logic. Placing it in a
        function avoids repeated code and reduces mistakes.
    """

    model.eval()            # switch OFF dropout, batchnorm uses running stats

    # Accumulators for total loss and accuracy.
    total_loss = 0.0
    correct    = 0
    total      = 0

    # torch.no_grad() disables gradient tracking.
    # Why?
    #   - Evaluation does not update weights.
    #   - No gradients are needed.
    #   - Saves memory and computation time.
    with torch.no_grad():   # no gradient computation — saves memory + time
        # Loop over validation/test batches.
        for X_b, y_b in loader:
            # Move batch data to the correct device.
            X_b, y_b  = X_b.to(device), y_b.to(device)
            # Forward pass.
            # model(X_b) returns shape (batch_size, 1).
            # squeeze() removes the last dimension, giving shape (batch_size,).
            # This matches y_b shape and avoids loss-shape mismatch.
            preds     = model(X_b).squeeze()           # (batch,)
            # Compute BCE loss for this batch.
            loss      = criterion(preds, y_b)
            # loss.item() gives the average loss for this batch.
            # Multiplying by len(y_b) converts it to total batch loss.
            # This is needed because the final batch may have fewer samples.
            total_loss += loss.item() * len(y_b)
            # Convert probabilities to hard class predictions using threshold 0.5.
            # If P(good wine) >= 0.5 -> class 1.
            # Otherwise -> class 0.
            predicted = (preds >= 0.5).float()
            # Count how many predictions match the true labels.
            correct   += (predicted == y_b).sum().item()
            # Count number of samples processed.
            total     += len(y_b)

    # Average loss over all samples, not over batches.,  Accuracy = correct predictions / total samples.
    return total_loss / total, correct / total


# ==============================================================================
# 7. STEP 5: TRAINING LOOP
# ==============================================================================
# Each epoch contains multiple BATCH iterations
#
# The inner loop structure per batch:
#   1. model(X_batch)       => forward pass => predictions
#   2. criterion(pred, y)   => compute scalar loss
#   3. optimizer.zero_grad() => clear accumulated gradients
#   4. loss.backward()       => backprop: compute delta(L)/delta(w) for ALL layers
#   5. optimizer.step()      => update ALL weights using gradients

print("\n── Step 4: Training loop ─────────────────────────────────────")

# One epoch means one full pass over the training set.
N_EPOCHS = 80
# Dictionary for storing learning history.
# These values will be plotted later.
history  = {"train_loss": [], "val_loss": [], "val_acc": [], "train_acc": []}

print(f"\n  {'Epoch':>6} {'Train Loss':>12} {'Val Loss':>10} {'Val Acc':>10} {'Train Acc':>10}")
print(f"  {'-'*52}")

# ------------------------------------------------------------------------------
# Outer loop: epoch loop
# ------------------------------------------------------------------------------
for epoch in range(1, N_EPOCHS + 1):

    # --------------------------------------------------------------------------
    # 7.1 TRAINING PHASE
    # --------------------------------------------------------------------------
    model.train()     # switches dropout/batchnorm to training mode
    # Reset epoch-level accumulators.
    epoch_loss  = 0.0
    epoch_corr  = 0
    epoch_total = 0

    # --------------------------------------------------------------------------
    # Inner loop: mini-batch loop
    # --------------------------------------------------------------------------
    # Each iteration receives one batch of samples and labels.
    for X_batch, y_batch in train_loader:
        # Move current batch to selected device.
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # ----------------------------------------------------------------------
        # Step A: Forward pass
        # ----------------------------------------------------------------------
        # The model maps input features to predicted probabilities.
        # Shape before squeeze: (batch_size, 1)
        # Shape after squeeze : (batch_size,)
        predictions = model(X_batch).squeeze()   # shape: (batch_size,)

        # ----------------------------------------------------------------------
        # Step B: Compute loss
        # ----------------------------------------------------------------------
        # The loss measures disagreement between predicted probabilities and true binary labels.
        loss = criterion(predictions, y_batch)   # scalar

        # ----------------------------------------------------------------------
        # Step C: Clear old gradients
        # ----------------------------------------------------------------------
        # MUST be done before backward() — gradients accumulate otherwise
        optimizer.zero_grad()

        # ----------------------------------------------------------------------
        # Step D: Backward pass / backpropagation
        # ----------------------------------------------------------------------
        # Computes delta(L)/delta(W^[l]) and delta(L)/delta(b^[l]) for ALL layers via chain rule
        loss.backward()

        # ----------------------------------------------------------------------
        # Step E: Optimizer update
        # ----------------------------------------------------------------------
        # For Adam, the update is approximately:
        #   parameter <- parameter - adaptive_step_size * gradient_direction
        optimizer.step()

        # ----------------------------------------------------------------------
        # Step F: Track training statistics
        # ----------------------------------------------------------------------
        # Accumulate total loss for this epoch.
        # loss.item() is average loss for the batch, so multiply by batch size.
        epoch_loss  += loss.item() * len(y_batch)
        # Convert probabilities to binary predictions.
        predicted    = (predictions >= 0.5).float()
        # Count correct predictions in this batch.
        epoch_corr  += (predicted == y_batch).sum().item()
        # Count total samples in this batch.
        epoch_total += len(y_batch)

    # Compute average training loss and accuracy over all training samples.
    avg_train_loss = epoch_loss  / epoch_total
    avg_train_acc  = epoch_corr  / epoch_total

    # --------------------------------------------------------------------------
    # 7.2 VALIDATION PHASE
    # --------------------------------------------------------------------------
    # Validation is done after each epoch.
    # It tells us how well the model performs on data not used for weight updates.
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    # Record history
    history["train_loss"].append(avg_train_loss)
    history["train_acc"].append(avg_train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    # Print progress every 10 epochs and also at epoch 1.
    # This keeps terminal output readable while still showing learning progress.
    if epoch % 10 == 0 or epoch == 1:
        print(f"  {epoch:>6} {avg_train_loss:>12.4f} {val_loss:>10.4f} {val_acc:>10.2%} {avg_train_acc:>10.2%}")


# ==============================================================================
# 8. STEP 6: PLOT LEARNING CURVES
# ==============================================================================
# The learning curve tells you the "story" of training:
#   - Decreasing train & val loss  => model is learning
#   - Train loss << val loss       => overfitting (memorizing, not generalizing)
#   - Both losses plateau          => converged or stuck
#   - Val loss increases           => overfitting beginning

print("\n── Step 5: Plotting learning curves ─────────────────────────")

# Create a figure with two panels:
#   left  -> train/validation loss
#   right -> train/validation accuracy
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Training History — Wine Quality MLP", fontsize=13)

# Epoch numbers for x-axis.
epochs = range(1, N_EPOCHS + 1)

# ------------------------------------------------------------------------------
# 8.1 Loss curves
# ------------------------------------------------------------------------------
# Training loss shows how well the model fits the training set.
axes[0].plot(epochs, history["train_loss"], color="#7F77DD", lw=2, label="Train loss")
# Validation loss shows how well the model generalizes to unseen validation data.
axes[0].plot(epochs, history["val_loss"],   color="#D85A30", lw=2, label="Val loss",   ls="--")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("BCE Loss")
axes[0].set_title("Loss curves")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# Find epoch with the lowest validation loss.
# This is often used as a simple model-selection criterion.
best_val_epoch = np.argmin(history["val_loss"]) + 1
# Draw a vertical line at the best validation epoch.
# This helps visually identify where validation performance was best.
axes[0].axvline(best_val_epoch, color="#1D9E75", lw=1.5, ls=":", label=f"Best val (ep {best_val_epoch})")
axes[0].legend()

# ------------------------------------------------------------------------------
# 8.2 Accuracy curves
# ------------------------------------------------------------------------------
# Training accuracy shows fraction of correct predictions on training data.
axes[1].plot(epochs, history["train_acc"], color="#7F77DD", lw=2, label="Train acc")
# Validation accuracy shows fraction of correct predictions on validation data.
axes[1].plot(epochs, history["val_acc"],   color="#D85A30", lw=2, label="Val acc",   ls="--")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
axes[1].set_title("Accuracy curves")
# The y-axis is restricted to 0.7--1.0 to focus on the useful accuracy range.
# If your model performs worse than 70%, adjust or remove this line.
axes[1].set_ylim(0.7, 1.0)
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("06_learning_curves.png", dpi=150, bbox_inches="tight")
print(f"  Saved: 06_learning_curves.png")
print(f"  Best validation epoch: {best_val_epoch}  (val_loss = {history['val_loss'][best_val_epoch-1]:.4f})")
# plt.show()


# ==============================================================================
# 9. STEP 7: FINAL TEST EVALUATION
# ==============================================================================

print("\n── Step 6: Final test evaluation ────────────────────────────")
# The test set should be used only after training decisions are complete.
# It estimates final generalization performance on unseen data.
test_loss, test_acc = evaluate(model, test_loader, criterion, device)

print(f"\n  Test Loss     : {test_loss:.4f}")
print(f"  Test Accuracy : {test_acc:.2%}")

# ------------------------------------------------------------------------------
# 9.1 Collect individual predictions for classification report
# ------------------------------------------------------------------------------
# classification_report needs arrays/lists of true class labels and predicted
# class labels. Therefore, we loop through the test set and collect all results.
all_preds, all_labels = [], []

# Evaluation mode again, for best practice.
model.eval()

# No gradients needed during test prediction collection.
with torch.no_grad():
    for X_b, y_b in test_loader:
        X_b = X_b.to(device)
        # Compute predicted probabilities.
        probs = model(X_b).squeeze()
        # Convert probabilities to class labels using threshold 0.5.
        preds = (probs >= 0.5).int().cpu().tolist()
        # Extend stores all batch predictions into the full list.
        all_preds.extend(preds)
        # Convert true labels to integer list.
        all_labels.extend(y_b.int().tolist())

print(f"\n  Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["ordinary", "good wine"]))

# ------------------------------------------------------------------------------
# 9.2 Confusion matrix
# ------------------------------------------------------------------------------
# Confusion matrix layout for binary classification:
#
#                 Predicted 0     Predicted 1
# Actual 0        true negative   false positive
# Actual 1        false negative  true positive
#
# It helps understand the types of mistakes the model makes.
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(6, 5))
# Plot confusion matrix as a heatmap.
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Predicted: ordinary", "Predicted: good"],
            yticklabels=["Actual: ordinary",    "Actual: good"],
            ax=ax)
ax.set_title("Confusion Matrix — Test Set")
plt.tight_layout()
plt.savefig("06_confusion_matrix.png", dpi=150, bbox_inches="tight")
print(f"  Saved: 06_confusion_matrix.png")
# plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: SAVE AND RELOAD THE MODEL
# ─────────────────────────────────────────────────────────────────────────────
# state_dict: a dictionary mapping layer names to parameter tensors
# This is the standard way to save PyTorch models.
# Save ONLY the state_dict (not the entire model object) for portability.

print("\n── Step 7: Saving and loading model ─────────────────────────")

torch.save(model.state_dict(), "wine_mlp.pth")
print(f"  Model saved to: wine_mlp.pth")

# Reload
model_loaded = WineMLP().to(device)
model_loaded.load_state_dict(torch.load("wine_mlp.pth", map_location=device))
model_loaded.eval()

# Verify identical predictions
with torch.no_grad():
    out_orig   = model(X_test[:5]).squeeze()
    out_loaded = model_loaded(X_test[:5]).squeeze()

print(f"  Original predictions : {out_orig.cpu().tolist()}")
print(f"  Reloaded predictions : {out_loaded.cpu().tolist()}")
print(f"  Identical            : {torch.allclose(out_orig, out_loaded)}")


# ==============================================================================
# 11. STEP 9: INFERENCE ON A NEW SAMPLE
# ==============================================================================
# How to use the trained model in production / on new data.

print("\n── Step 8: Inference on a new sample ────────────────────────")

# A new wine sample (raw, unscaled physicochemical measurements)
new_sample_raw = np.array([[7.8, 0.58, 0.02, 2.0, 0.073, 9.0, 18.0, 0.9968, 3.36, 0.57, 9.5]])
#                           acidity  vol.acid  cit.acid  sugar  chlor  free_SO2  tot_SO2  density  pH  sulph  alcohol

# Normalize using the SAME scaler fitted on training data
new_sample_scaled = scaler.transform(new_sample_raw)
new_sample_tensor = torch.tensor(new_sample_scaled, dtype=torch.float32).to(device)

model_loaded.eval()
with torch.no_grad():
    probability = model_loaded(new_sample_tensor).item()

print(f"\n  New sample (raw): {new_sample_raw.tolist()}")
print(f"  P(good wine)    : {probability:.4f}  ({probability*100:.1f}%)")
print(f"  Prediction      : {'GOOD WINE' if probability > 0.5 else 'ordinary wine'}")

print("\n[DONE] Script  complete.")

