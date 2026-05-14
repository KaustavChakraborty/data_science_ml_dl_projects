"""
============================================================
 05_loss_functions.py
 Deep Learning for Soft Matter Physics
 ── Loss Functions: Implement, Visualize, Compare ───────────

 GOAL:
   Understand loss functions as the "objective" that training
   minimizes. Different losses create different gradient
   landscapes — choosing the right one is crucial.

 CONCEPTS COVERED:
   - Binary Cross-Entropy (BCE): classification standard
   - Mean Squared Error (MSE): regression standard
   - Why BCE beats MSE for classification (gradient argument)
   - Manual BCE derivation vs PyTorch nn.BCELoss
   - Numerical stability: BCEWithLogitsLoss
   - Per-sample loss decomposition
   - Loss landscape visualization

============================================================
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


# ==============================================================================
# SCRIPT HEADER
# ==============================================================================

print("=" * 60)
print("  LOSS FUNCTIONS")
print("=" * 60)


# ==============================================================================
# PART 1: BINARY CROSS-ENTROPY — THE MATH
# ==============================================================================

print("\n── PART 1: Binary Cross-Entropy calculation ─────────────────")


# ------------------------------------------------------------------------------
# Create a small artificial batch of labels
# ------------------------------------------------------------------------------
#
# y_true contains the ground-truth labels.
#
# We intentionally choose three positive examples and three negative examples:
#
#     first 3 samples: y = 1
#     last  3 samples: y = 0
#
# This allows us to see how BCE behaves for correct, uncertain, and wrong
# predictions for both classes.
# ------------------------------------------------------------------------------


# A batch of 6 samples: true labels and model predictions
y_true = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
y_pred = torch.tensor([0.9, 0.5, 0.1, 0.1, 0.5, 0.9])   # predictions from sigmoid
# Human-readable descriptions for printing.
descriptions = [
    "y=1, y_cap=0.9 (correct, confident)",
    "y=1, y_cap=0.5 (uncertain)",
    "y=1, y_cap=0.1 (WRONG, confident)",
    "y=0, y_cap=0.1 (correct, confident)",
    "y=0, y_cap=0.5 (uncertain)",
    "y=0, y_cap=0.9 (WRONG, confident)",
]

# ------------------------------------------------------------------------------
# Manual BCE computation
# ------------------------------------------------------------------------------
# Why add 1e-8?
#
#     log(0) is undefined and gives -infinity.
#
#     Numerically, if ŷ is exactly 0 or exactly 1, log terms can explode.
#
#     Adding a tiny epsilon protects the manual computation.
#
# Important:
#
#     PyTorch's BCEWithLogitsLoss handles numerical stability better than this
#     manual epsilon approach.
# ------------------------------------------------------------------------------
loss_per_sample = -(y_true * torch.log(y_pred + 1e-8) +
                    (1 - y_true) * torch.log(1 - y_pred + 1e-8))

# ------------------------------------------------------------------------------
# Print per-sample BCE values
# ------------------------------------------------------------------------------
print(f"\n  {'Sample description':<38} {'y':>4} {'y_cap':>6} {'L_BCE':>8}")
print(f"  {'-'*60}")
for desc, y, yh, l in zip(descriptions, y_true.tolist(), y_pred.tolist(), loss_per_sample.tolist()):
    print(f"  {desc:<38} {y:>4.0f} {yh:>6.2f} {l:>8.4f}")

# PyTorch's BCELoss — identical result
bce_fn  = nn.BCELoss(reduction="none")    # reduction="none" → per-sample
bce_pt  = bce_fn(y_pred, y_true)
print(f"\n  PyTorch BCELoss matches manual: {torch.allclose(loss_per_sample, bce_pt, atol=1e-5)}")
print(f"  Mean BCE loss over batch      : {loss_per_sample.mean().item():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# PART 2: MSE LOSS — COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

print("\n── PART 2: MSE loss ──────────────────────────────────────────")

mse_fn           = nn.MSELoss(reduction="none")
mse_per_sample   = mse_fn(y_pred, y_true)

print(f"\n  {'Sample description':<38} {'BCE':>8} {'MSE':>8}")
print(f"  {'-'*60}")
for desc, b, m in zip(descriptions, loss_per_sample.tolist(), mse_per_sample.tolist()):
    print(f"  {desc:<38} {b:>8.4f} {m:>8.4f}")

print(f"\n  Ratio BCE/MSE for confident wrong answer (y=1, y_cap=0.1):")
print(f"    BCE = {loss_per_sample[2].item():.4f}")
print(f"    MSE = {mse_per_sample[2].item():.4f}")
print(f"    BCE is {loss_per_sample[2].item()/mse_per_sample[2].item():.1f} * larger")
print(f"  => BCE penalizes confident wrong answers MUCH more severely")
print(f"  => Larger loss = larger gradient = faster correction")


# ─────────────────────────────────────────────────────────────────────────────
# PART 3: LOSS LANDSCAPE VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

print("\n── PART 3: Visualizing loss landscapes ──────────────────────")

yhat_range = np.linspace(0.001, 0.999, 400)

# For TRUE POSITIVE (y=1)
bce_pos = -np.log(yhat_range)              # L = -log(y_cap)
mse_pos = (1.0 - yhat_range) ** 2         # L = (1-y_cap)^2

# Gradients w.r.t. ŷ (how strongly loss pushes ŷ upward)
grad_bce_pos = -1.0 / yhat_range           # ∂L_BCE/∂ŷ = -1/ŷ
grad_mse_pos = -2.0 * (1.0 - yhat_range)  # ∂L_MSE/∂ŷ = -2(1-ŷ)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Loss Functions: BCE vs MSE for Binary Classification", fontsize=13)

# ── Plot 1: Loss curve for y=1 ────────────────────────────────────────────────
ax = axes[0, 0]
ax.plot(yhat_range, bce_pos, color="#7F77DD", lw=2.5, label="BCE: -log(ŷ)")
ax.plot(yhat_range, mse_pos, color="#D85A30", lw=2.5, label="MSE: (1-ŷ)²")
ax.set_title("Loss for TRUE POSITIVE (y=1)")
ax.set_xlabel("Prediction ŷ")
ax.set_ylabel("Loss value")
ax.set_ylim(0, 5)
ax.axvline(0.5, color="gray", lw=1, ls="--", alpha=0.5, label="Decision boundary")
ax.legend(); ax.grid(True, alpha=0.3)
ax.annotate("BCE → ∞\n(huge gradient!)", xy=(0.05, 3.5), fontsize=9, color="#7F77DD")

# ── Plot 2: Gradient magnitude for y=1 ───────────────────────────────────────
ax = axes[0, 1]
ax.plot(yhat_range, np.abs(grad_bce_pos), color="#7F77DD", lw=2.5, label="|∂BCE/∂ŷ|")
ax.plot(yhat_range, np.abs(grad_mse_pos), color="#D85A30", lw=2.5, label="|∂MSE/∂ŷ|")
ax.set_title("Gradient magnitude for TRUE POSITIVE (y=1)")
ax.set_xlabel("Prediction ŷ")
ax.set_ylabel("|Gradient|")
ax.set_ylim(0, 15)
ax.legend(); ax.grid(True, alpha=0.3)
ax.annotate("BCE gradient is 50× larger\nwhen ŷ is near 0 (wrong!)",
            xy=(0.05, 9), fontsize=9, color="#7F77DD")

# ── Plot 3: Loss for both classes ─────────────────────────────────────────────
ax = axes[1, 0]
bce_neg = -np.log(1 - yhat_range)         # y=0: L = -log(1-ŷ)
mse_neg = yhat_range ** 2

ax.plot(yhat_range, bce_pos, color="#7F77DD", lw=2, ls="-",  label="BCE y=1")
ax.plot(yhat_range, bce_neg, color="#7F77DD", lw=2, ls="--", label="BCE y=0")
ax.plot(yhat_range, mse_pos, color="#D85A30", lw=2, ls="-",  label="MSE y=1")
ax.plot(yhat_range, mse_neg, color="#D85A30", lw=2, ls="--", label="MSE y=0")
ax.set_title("BCE vs MSE for both classes")
ax.set_xlabel("Prediction ŷ"); ax.set_ylabel("Loss")
ax.set_ylim(0, 4); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ── Plot 4: BCE saturation problem for sigmoid ────────────────────────────────
ax = axes[1, 1]
z_range   = np.linspace(-8, 8, 400)
sigma     = 1 / (1 + np.exp(-z_range))   # sigmoid output

# Gradient of BCE w.r.t. z (the logit), for y=1:
# ∂L/∂z = ŷ - y = σ(z) - 1
grad_bce_z_pos = sigma - 1.0             # for y=1
grad_mse_z_pos = -2 * (1 - sigma) * sigma * (1 - sigma)  # chain rule for MSE+sigmoid

ax.plot(z_range, np.abs(grad_bce_z_pos), color="#7F77DD", lw=2.5, label="|∂BCE/∂z|")
ax.plot(z_range, np.abs(grad_mse_z_pos), color="#D85A30", lw=2.5, label="|∂MSE/∂z|")
ax.set_title("Gradient w.r.t. logit z\n(why MSE+sigmoid has vanishing gradients)")
ax.set_xlabel("Logit z"); ax.set_ylabel("|Gradient|")
ax.axvline(0, color="gray", lw=0.8, ls="--")
ax.legend(); ax.grid(True, alpha=0.3)
ax.annotate("MSE+sigmoid gradient\nvanishes here!\n(sigmoid saturates)", 
            xy=(-7, 0.1), fontsize=8.5, color="#D85A30")
ax.annotate("BCE+sigmoid gradient\nis always nonzero",
            xy=(1, 0.7), fontsize=8.5, color="#7F77DD")

plt.tight_layout()
plt.savefig("05_loss_functions.png", dpi=150, bbox_inches="tight")
print("  Saved: 05_loss_functions.png")
# plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# PART 4: BCEWithLogitsLoss — NUMERICALLY STABLE VERSION
# ─────────────────────────────────────────────────────────────────────────────
# PREFERRED IN PRACTICE:
#   nn.BCELoss        : expects  ŷ = sigmoid(z)  as input  (TWO steps)
#   nn.BCEWithLogitsLoss : expects z (raw logit)  as input  (ONE step)
#
# BCEWithLogitsLoss is numerically more stable because it avoids computing
# sigmoid separately (can overflow for very large z).
# It fuses: loss = -[y·log(σ(z)) + (1-y)·log(1-σ(z))]
# into a single numerically stable formula.
#
# RULE: output layer should have NO activation when using BCEWithLogitsLoss.
# Let the loss function apply sigmoid internally.

print("\n── PART 4: Numerically stable BCEWithLogitsLoss ─────────────")

bce_logits_fn = nn.BCEWithLogitsLoss()

# Raw logits (before sigmoid) — this is what the output layer produces
z_logits = torch.tensor([-2.0, 0.0, 2.0, 5.0, -5.0])
y_labels  = torch.tensor([ 0.0, 1.0, 1.0, 0.0,  1.0])

# Method 1: sigmoid then BCELoss
y_preds       = torch.sigmoid(z_logits)
loss_two_step = nn.BCELoss()(y_preds, y_labels)

# Method 2: BCEWithLogitsLoss (preferred)
loss_one_step = bce_logits_fn(z_logits, y_labels)

print(f"\n  BCELoss(sigmoid(z), y)    : {loss_two_step.item():.6f}")
print(f"  BCEWithLogitsLoss(z, y)   : {loss_one_step.item():.6f}")
print(f"  Numerically identical     : {torch.isclose(loss_two_step, loss_one_step, atol=1e-5).item()}")
print(f"\n  Practical implication:")
print(f"  => Remove Sigmoid() from output layer")
print(f"  => Use nn.BCEWithLogitsLoss() instead of nn.BCELoss()")
print(f"  => At inference: ŷ = torch.sigmoid(model(x))")


# ─────────────────────────────────────────────────────────────────────────────
# PART 5: MSE FOR REGRESSION (CORRECT USE CASE)
# ─────────────────────────────────────────────────────────────────────────────
# MSE is the right choice when predicting CONTINUOUS values, not classes.
# Example: predicting the actual quality score (3–8) rather than good/bad.
# Soft matter analogy: predicting viscosity, G' (storage modulus), Rg, etc.

print("\n── PART 5: MSE for regression ────────────────────────────────")

# Simulate predicting continuous quality score
y_continuous = torch.tensor([5.0, 6.0, 7.0, 4.0, 8.0])  # true quality scores
y_predicted  = torch.tensor([5.3, 5.8, 6.5, 4.2, 7.8])  # model predictions

mse    = nn.MSELoss()(y_predicted, y_continuous)
mae    = nn.L1Loss()(y_predicted, y_continuous)
rmse   = torch.sqrt(mse)
errors = (y_predicted - y_continuous)

print(f"\n  True values      : {y_continuous.tolist()}")
print(f"  Predictions      : {y_predicted.tolist()}")
print(f"  Errors (ŷ - y)   : {errors.tolist()}")
print(f"\n  MSE  = (1/N)Σ(y-ŷ)² : {mse.item():.4f}")
print(f"  RMSE = √MSE          : {rmse.item():.4f}  (same units as y)")
print(f"  MAE  = (1/N)Σ|y-ŷ|  : {mae.item():.4f}  (more robust to outliers)")

print(f"\n  For REGRESSION tasks (predicting G', viscosity, Rg):")
print(f"    => Use nn.MSELoss() or nn.L1Loss()")
print(f"    => Output layer: nn.Linear(n, 1) with NO activation function")
print(f"    => Linear output can represent any continuous value")


# ─────────────────────────────────────────────────────────────────────────────
# PART 6: LOSS FUNCTION SELECTION GUIDE
# ─────────────────────────────────────────────────────────────────────────────

print("\n── PART 6: Loss function selection guide ─────────────────────")
print("""
  ┌─────────────────────────────────────────────────────────────────────┐
  │                   LOSS FUNCTION CHEAT SHEET                         │
  ├──────────────────┬───────────────────────┬──────────────────────────┤
  │ Task             │ Loss function          │ Output activation        │
  ├──────────────────┼───────────────────────┼──────────────────────────┤
  │ Binary classif.  │ nn.BCEWithLogitsLoss() │ None (raw logit)         │
  │                  │ nn.BCELoss()           │ nn.Sigmoid()             │
  ├──────────────────┼───────────────────────┼──────────────────────────┤
  │ Multi-class      │ nn.CrossEntropyLoss()  │ None (raw logits)        │
  │ classification   │ (fuses softmax+log)    │                          │
  ├──────────────────┼───────────────────────┼──────────────────────────┤
  │ Regression       │ nn.MSELoss()           │ None (linear output)     │
  │ (continuous y)   │ nn.L1Loss()            │                          │
  ├──────────────────┼───────────────────────┼──────────────────────────┤

""")

print("[DONE] Script 05 complete.")

