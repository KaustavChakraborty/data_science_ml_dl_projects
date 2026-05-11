"""
============================================================
 01_perceptron_scratch.py
 ── Single Perceptron Built from Raw PyTorch Tensors ────────

 GOAL:
   Understand the fundamental perceptron computation BEFORE
   using any of PyTorch's high-level abstractions (nn.Linear,
   nn.Module, etc.).

   Every weight, every multiply, every gradient is fully
   visible here. This is exactly what PyTorch does internally.

 CONCEPTS COVERED:
   - Weight vector and bias scalar
   - Pre-activation z = w*x + b
   - Activation function sigma(z)
   - Binary cross-entropy loss
   - Autograd: requires_grad, .backward()
   - Manual gradient descent weight update
   - Gradient zeroing

 HOW TO RUN:
   python perceptron_scratch_v1.py
============================================================
"""

# ==============================================================================
# SECTION 1 — IMPORT LIBRARIES
# ==============================================================================

import torch
# Contains functional implementations of neural-network operations.
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# SECTION 2 — SIMPLE PRINT HEADER
# ==============================================================================
print("=" * 60)
print("  PERCEPTRON FROM SCRATCH")
print("=" * 60)

# ==============================================================================
# PART 1 — SINGLE FORWARD PASS THROUGH ONE PERCEPTRON
# ==============================================================================

print("\n── PART 1: Single forward pass ──────────────────────────────")

x = torch.tensor([7.4, 0.70, 3.51])    # input feature vector, shape: (3,)
print(f"Input x (3 features)  : {x}")

# ── Initialize weights and bias ───────────────────────────────────────────────
# requires_grad=True is the critical flag — it tells PyTorch's autograd engine
# to record all operations involving these tensors, so gradients can be
# computed automatically when we call .backward().
#
# Without requires_grad=True: just a tensor, no gradient tracking.
# With  requires_grad=True: PyTorch builds a computational graph as you compute.

torch.manual_seed(42)                   # fix random seed for reproducibility
w = torch.randn(3, requires_grad=True)  # weight vector,  shape: (3,)
b = torch.zeros(1, requires_grad=True)  # bias scalar,    shape: (1,)

print(f"\nInitial weights w             : {w.detach()}")
print(f"Shape of w                    : {tuple(w.shape)}")
print(f"Initial bias b                : {b.detach()}")
print(f"Shape of b                    : {tuple(b.shape)}")

# ── Step 1: Linear combination  z = w*x + b ──────────────────────────────────
# In matrix notation (for batches later): z = Wx + b

z = torch.dot(w, x) + b                # shape: (1,)
print("\nLinear combination details:")
print(f"  torch.dot(w, x)             : {torch.dot(w, x).item():.6f}")
print(f"  bias b                      : {b.item():.6f}")
print(f"  pre-activation z = w·x + b  : {z.item():.6f}")
print(f"  Shape of z                  : {tuple(z.shape)}")

# ── Step 2: Activation function  y_cap = sigma(z) ────────────────────────────────────
# Sigmoid squashes any real number into the range (0, 1).
# Example:
#   y_hat = 0.80 means the model says "80% probability of class 1".

y_hat = torch.sigmoid(z)               # shape: (1,)
print("\nActivation details:")
print(f"  y_hat = sigmoid(z)          : {y_hat.item():.6f}")
print(f"  Interpreted probability     : {100.0 * y_hat.item():.2f}% for class 1")
print(f"  Shape of y_hat              : {tuple(y_hat.shape)}")

# ── Step 3: Compute loss ──────────────────────────────────────────────────────
# True label: 1 = good wine, 0 = bad wine
# Binary Cross-Entropy: L = -[y*log(y_cap) + (1-y)*log(1-y_cap)]


y_true = torch.tensor([1.0])           # ground truth
loss = F.binary_cross_entropy(y_hat, y_true)

print("\nLoss details:")
print(f"  True label y_true           : {y_true.item():.1f}")
print(f"  Binary cross-entropy loss   : {loss.item():.6f}")
print(f"  Shape of loss               : {tuple(loss.shape)}")

# ── Step 4: Backward pass  ────────────────────────────────────────────────────
# .backward() traverses the computational graph in REVERSE (chain rule)
# PyTorch walks backward through the computation graph:
#
#   loss -> y_hat -> z -> w and b
#
# and computes:
#
#   dL/dw[0], dL/dw[1], dL/dw[2]
#   dL/db
#
# These gradients are stored inside:
#
#   w.grad
#   b.grad
#
# For sigmoid + binary cross-entropy, the derivative has a simple form:
#
#   dL/dz = y_hat - y_true
#
# Since:
#
#   z = w · x + b
#
# we also get:
#
#   dL/dw = (y_hat - y_true) * x
#   dL/db = (y_hat - y_true)

loss.backward()

print("\nGradient details after loss.backward():")
print(f"  Gradient dL/dw              : {w.grad}")
print(f"  Gradient dL/db              : {b.grad}")

# ------------------------------------------------------------------------------
# Update weights using gradient descent
# ------------------------------------------------------------------------------
# Gradient descent update rule:
#
#   parameter_new = parameter_old - learning_rate * gradient
#
# For weights:
#
#   w_new = w_old - lr * dL/dw
#
# For bias:
#
#   b_new = b_old - lr * dL/db

lr = 0.01

old_w = w.detach().clone()
old_b = b.detach().clone()

print("\nGradient descent update:")
print(f"  Learning rate lr            : {lr}")
print(f"  Weights before update       : {old_w}")
print(f"  Bias before update          : {old_b}")

with torch.no_grad():
    w -= lr * w.grad    # gradient descent step
    b -= lr * b.grad

new_w = w.detach().clone()
new_b = b.detach().clone()

print(f"  Weights after update        : {new_w}")
print(f"  Bias after update           : {new_b}")
print(f"  Actual change Delta w       : {new_w - old_w}")
print(f"  Actual change Delta b       : {new_b - old_b}")

# ── CRITICAL: Zero the gradients ──────────────────────────────────────────────
# PyTorch ACCUMULATES gradients by default (+=).
# If you forget this, gradients from multiple backward() calls pile up.
# This is one of the most common bugs in PyTorch code!

w.grad.zero_()
b.grad.zero_()

print("\nGradients after zeroing:")
print(f"  w.grad                       : {w.grad}")
print(f"  b.grad                       : {b.grad}")

# ==============================================================================
# PART 2 — MULTIPLE GRADIENT DESCENT STEPS ON THE SAME SAMPLE
# ==============================================================================
# In Part 1, we performed exactly one learning step.
# Now we repeat the same process 50 times.
#
# Since the true label is y_true = 1, successful learning means:
#
#   y_hat should move closer to 1
#   BCE loss should move closer to 0
#
# Important limitation:
# ---------------------
# Training on one sample is only for education.
# A real model should be trained on many samples and tested on unseen data.

print("\n── PART 2: 50 gradient descent steps on 1 sample ──────────")
print(f"{'Step':>6}  {'z':>10}  {'y_hat':>10}  {'Loss':>10}  {'||grad_w||':>12}")
print("-" * 62)

# Re-initialize weights
torch.manual_seed(42)
w2 = torch.randn(3, requires_grad=True)
b2 = torch.zeros(1, requires_grad=True)

# These lists store values over training steps so we can plot them later.
loss_history = []
yhat_history = []
z_history = []

for step in range(1, 51):
    # --------------------------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------------------------
    # Compute raw score z2.
    z2 = torch.dot(w2, x) + b2

    # Convert raw score into probability.
    yhat2 = torch.sigmoid(z2)

    # Compare prediction against true label using binary cross-entropy.
    loss2 = F.binary_cross_entropy(yhat2, y_true)

    # --------------------------------------------------------------------------
    # Backward pass
    # --------------------------------------------------------------------------
    # Compute dL/dw2 and dL/db2.
    loss2.backward()

    # --------------------------------------------------------------------------
    # Store history before updating
    # --------------------------------------------------------------------------
    # .item() converts a one-element tensor to an ordinary Python float.
    # This is useful for printing and plotting.
    loss_history.append(loss2.item())
    yhat_history.append(yhat2.item())
    z_history.append(z2.item())

    # --------------------------------------------------------------------------
    # Parameter update
    # --------------------------------------------------------------------------
    # We calculate the norm of the gradient vector as a compact measure of how
    # large the current update direction is.
    #
    # For a vector g = [g1, g2, g3], the Euclidean norm is:
    #
    #   ||g|| = sqrt(g1^2 + g2^2 + g3^2)
    #
    # If training approaches a minimum, gradient norms often become smaller.
    with torch.no_grad():
        grad_norm = w2.grad.norm().item()
        w2 -= lr * w2.grad
        b2 -= lr * b2.grad

    # --------------------------------------------------------------------------
    # Clear gradients for next iteration
    # --------------------------------------------------------------------------
    # If this is omitted, the next backward() call will add new gradients on top
    # of the current ones, producing an incorrect basic gradient descent loop.
    w2.grad.zero_()
    b2.grad.zero_()

    # Print selected steps instead of printing all 50 lines.
    if step == 1 or step % 5 == 0:
        print(
            f"{step:>6}  "
            f"{z2.item():>10.5f}  "
            f"{yhat2.item():>10.5f}  "
            f"{loss2.item():>10.5f}  "
            f"{grad_norm:>12.6f}"
        )

print("\nFinal state after 50 steps:")
print(f"  Final prediction y_hat       : {yhat2.item():.6f}")
print(f"  Target y_true                : {y_true.item():.1f}")
print(f"  Final BCE loss               : {loss2.item():.6f}")
print(f"  Final weights w2             : {w2.detach()}")
print(f"  Final bias b2                : {b2.detach()}")


# ─────────────────────────────────────────────────────────────────────────────
# PART 3: VISUALIZE ACTIVATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
print("\n── PART 3: Visualizing activation functions ────────────────")

# Create 300 evenly spaced z-values from -6 to +6.
z_range = np.linspace(-6, 6, 300)
# Convert NumPy array to PyTorch tensor because the activation functions below
# are PyTorch operations.
z_t     = torch.tensor(z_range, dtype=torch.float32)

activations = {
    "Sigmoid":    torch.sigmoid(z_t).numpy(),
    "Tanh":       torch.tanh(z_t).numpy(),
    "ReLU":       torch.relu(z_t).numpy(),
    "Leaky ReLU": F.leaky_relu(z_t, negative_slope=0.1).numpy(),
}

# Create a 2 by 2 grid of subplots.
# axes is a 2D array of matplotlib Axes objects.
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
fig.suptitle("Activation Functions — outputs and gradients", fontsize=13, y=1.01)
colors = ["#7F77DD", "#1D9E75", "#D85A30", "#BA7517"]

for ax, (name, values), color in zip(axes.flatten(), activations.items(), colors):
    # Plot activation value f(z)
    ax.plot(z_range, values, color=color, lw=2.5, label="f(z)")
    # Add horizontal and vertical reference lines.
    # These help show where the function crosses zero and how it behaves near z=0.
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.axvline(0, color="gray", lw=0.7, ls="--")

    # Compute gradient numerically for illustration
    dz = z_range[1] - z_range[0]
    grad = np.gradient(values, dz)
    # Plot derivative as a dotted line.
    ax.plot(z_range, grad, color=color, lw=1.2, ls=":", alpha=0.7, label="f'(z)")
    # Make each panel readable.
    ax.set_title(name, fontsize=11)
    ax.set_xlabel("z (pre-activation)")
    ax.set_ylabel("Activation value")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.5, 1.5)

# Annotate key properties
axes[0,0].annotate("Saturates near 0 and 1\n => vanishing gradients",
                   xy=(4, 0.9), fontsize=8, color="#7F77DD")
axes[1,0].annotate("Dead zone: z<0 => output=0\ngrad=0 (neuron 'dies')",
                   xy=(-5.5, 0.9), fontsize=8, color="#D85A30")

plt.tight_layout()
plt.savefig("01_activation_functions.png", dpi=150, bbox_inches="tight")
print("  Saved: 01_activation_functions.png")
# plt.show()

# ==============================================================================
# PART 4 — PLOT LOSS AND PREDICTION HISTORY
# ==============================================================================

print("\n── PART 4: Plotting training history ─────────────────────────")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle("Single perceptron learning on 1 sample (y=1)", fontsize=12)

ax1.plot(loss_history, color="#7F77DD", lw=2)
ax1.set_xlabel("Gradient descent step")
ax1.set_ylabel("BCE Loss")
ax1.set_title("Loss decreasing toward 0")
ax1.grid(True, alpha=0.3)

ax2.plot(yhat_history, color="#1D9E75", lw=2)
ax2.axhline(1.0, color="#D85A30", lw=1.2, ls="--", label="Target y=1")
ax2.set_xlabel("Gradient descent step")
ax2.set_ylabel("Prediction ŷ")
ax2.set_title("Prediction converging to 1.0")
ax2.legend()
ax2.set_ylim(0, 1.1)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("01_perceptron_convergence.png", dpi=150, bbox_inches="tight")
print("  Saved: 01_perceptron_convergence.png")
# plt.show()

# ==============================================================================
# FINAL SUMMARY PRINTED BY THE SCRIPT
# ==============================================================================

print("\n[DONE] Script complete.")
print("\nKey takeaways:")
print("  1. A perceptron first computes z = w·x + b.")
print("  2. Sigmoid converts z into a probability between 0 and 1.")
print("  3. Binary cross-entropy measures how wrong the probability is.")
print("  4. loss.backward() computes dL/dw and dL/db automatically.")
print("  5. Gradient descent updates parameters opposite to the gradient direction.")
print("  6. torch.no_grad() is used during manual parameter updates.")
print("  7. Gradients must be zeroed after each update because PyTorch accumulates them.")
print("  8. Repeated updates make the prediction move toward the true label for this sample.")
