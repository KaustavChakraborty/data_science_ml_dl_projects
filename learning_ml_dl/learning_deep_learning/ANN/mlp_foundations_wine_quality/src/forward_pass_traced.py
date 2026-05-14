"""
============================================================
 04_forward_pass_traced.py
 Deep Learning for Soft Matter Physics
 ── Forward Propagation — Traced Layer by Layer ─────────────

 GOAL:
   Make the forward pass completely transparent.
   Print tensor shapes, value ranges, and statistics at
   every layer — both manually and using PyTorch hooks.

 CONCEPTS COVERED:
   - Layer-by-layer tensor shape tracking
   - Forward hooks: inspect tensors without modifying the model
   - Pre-activation z vs post-activation a
   - Effect of ReLU on activation statistics (dead neurons)
   - Gradient flow intuition from activation ranges
   - How batch processing parallelizes the computation

============================================================
"""


# ==============================================================================
# IMPORTS
# ==============================================================================

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("  FORWARD PASS — TRACED")
print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# PART 1: MANUAL LAYER-BY-LAYER TRACE (SINGLE SAMPLE)
# ─────────────────────────────────────────────────────────────────────────────
# We go step by step, computing z^[l] and a^[l] explicitly at each layer.
# This exactly matches the mathematical notation from the theory session.

print("\n── PART 1: Manual layer-by-layer trace (1 sample) ──────────")

torch.manual_seed(7)

# ------------------------------------------------------------------------------
# Define a simple MLP using nn.Sequential
# ------------------------------------------------------------------------------
model = nn.Sequential(
    nn.Linear(11, 64),    # layer 0: W^[1]
    nn.ReLU(),            # layer 1
    nn.Linear(64, 32),    # layer 2: W^[2]
    nn.ReLU(),            # layer 3
    nn.Linear(32,  1),    # layer 4: W^[3]
    nn.Sigmoid()          # layer 5
)

# One fake wine sample (normalized, 11 features)
x_single = torch.randn(11)   # shape: (11,)
print(f"\nInput x shape  : {x_single.shape}")
print(f"Input x values : {x_single[:5].tolist()} ...")

# ------------------------------------------------------------------------------
# Extract only the Linear layers from the Sequential model
# ------------------------------------------------------------------------------
#
L1, L2, L3 = model[0], model[2], model[4]  # nn.Linear modules

# ------------------------------------------------------------------------------
# Disable gradient tracking for tracing
# ------------------------------------------------------------------------------
#
# torch.no_grad() tells PyTorch:
#
#     Do not build the computational graph.
#
# Why?
#
#     We are only inspecting the forward pass.
#     We are not training.
#     We are not calling loss.backward().
#
# Benefits:
#
#     1. less memory use
#     2. faster execution
#     3. cleaner debugging
#
# In real training, we do NOT use torch.no_grad() during the training forward pass.
# We use it only during evaluation, inference, visualization, or debugging.
# ------------------------------------------------------------------------------

with torch.no_grad():

    # ── Layer 1 ──────────────────────────────────────────────────────────────
    # z^[1] = W^[1] * x + b^[1]
    # This is a matrix-vector product: (64, 11) * (11,) = (64,)

    z1 = L1(x_single)                     # shape: (64,)
    a1 = torch.relu(z1)                   # shape: (64,)

    # "Dead" ReLU neurons: those with z < 0 output 0 and have zero gradient
    n_active   = (a1 > 0).sum().item()
    frac_active = n_active / len(a1)

    print(f"\n── Layer 1 (Linear 11 => 64) ───────────────────────────────")
    print(f"  z^[1] shape          : {z1.shape}")
    print(f"  z^[1] range          : [{z1.min().item():.3f}, {z1.max().item():.3f}]")
    print(f"  z^[1] mean / std     : {z1.mean().item():.3f} / {z1.std().item():.3f}")
    print(f"  a^[1] = ReLU(z^[1])")
    print(f"  a^[1] active neurons : {n_active}/{len(a1)}  ({frac_active:.0%})")
    print(f"  a^[1] mean (excl. 0) : {a1[a1>0].mean().item():.3f}")

    # ── Layer 2 ──────────────────────────────────────────────────────────────
    z2 = L2(a1)                           # (64,) => (32,)
    a2 = torch.relu(z2)

    n_active2 = (a2 > 0).sum().item()
    print(f"\n── Layer 2 (Linear 64 => 32) ───────────────────────────────")
    print(f"  z^[2] shape          : {z2.shape}")
    print(f"  z^[2] range          : [{z2.min().item():.3f}, {z2.max().item():.3f}]")
    print(f"  a^[2] active neurons : {n_active2}/{len(a2)}  ({n_active2/len(a2):.0%})")

    # ── Output layer ─────────────────────────────────────────────────────────
    z3   = L3(a2)                         # (32,) => (1,)
    yhat = torch.sigmoid(z3)

    print(f"\n── Output layer (Linear 32 => 1 + Sigmoid) ─────────────────")
    print(f"  z^[3] (logit)        : {z3.item():.4f}")
    print(f"  ŷ = sigma(z^[3])         : {yhat.item():.4f}")
    print(f"  Prediction           : {'GOOD wine' if yhat.item() > 0.5 else 'ordinary wine'}  (threshold 0.5)")


# ─────────────────────────────────────────────────────────────────────────────
# PART 2: BATCH FORWARD PASS (MULTIPLE SAMPLES IN PARALLEL)
# ─────────────────────────────────────────────────────────────────────────────
# In real training, we process a BATCH of N samples simultaneously.
# The computation is identical, but x is now a matrix: (N, 11)
# This enables GPU parallelism and more stable gradient estimates.

print("\n── PART 2: Batch forward pass (32 samples) ─────────────────")

x_batch = torch.randn(32, 11)   # 32 fake wine samples

with torch.no_grad():
    out = model(x_batch)        # shape: (32, 1)

print(f"\n  Input  shape : {x_batch.shape}  (batch_size * n_features)")
print(f"  Output shape : {out.shape}       (batch_size * 1)")
print(f"  All outputs in (0,1)? {bool((out > 0).all() and (out < 1).all())}")
print(f"  Batch predictions: {out.squeeze()[:8].tolist()}")
print(f"  Predicted positives: {(out.squeeze() > 0.5).sum().item()}/32")


# ─────────────────────────────────────────────────────────────────────────────
# PART 3: FORWARD HOOKS — INSPECT EVERY LAYER AUTOMATICALLY
# ─────────────────────────────────────────────────────────────────────────────
# PyTorch hooks let you "tap into" any module during forward/backward pass
# without modifying the model's source code.
#
# register_forward_hook(fn):
#   fn is called as fn(module, input, output) after each module's forward()
#   - module : the layer itself
#   - input  : tuple of inputs to this layer
#   - output : output of this layer
#
# This is extremely useful for:
#   - Debugging activation statistics
#   - Detecting dying ReLU or exploding activations
#   - Building visualization tools

print("\n── PART 3: Forward hooks — automatic layer tracing ─────────")

# Storage for recorded activations
activation_records = {}

def make_forward_hook(layer_name):
    """Factory function — creates a hook that records stats for layer_name."""
    def hook_fn(module, input, output):
        out = output.detach()   # detach from computation graph (no grad needed)
        activation_records[layer_name] = {
            "shape":    tuple(out.shape),
            "min":      out.min().item(),
            "max":      out.max().item(),
            "mean":     out.mean().item(),
            "std":      out.std().item(),
            "fraction_positive": (out > 0).float().mean().item(),
        }
    return hook_fn

# Register hooks on every named module
hooks = []
layer_map = {
    "0_Linear_1":   model[0],
    "1_ReLU_1":     model[1],
    "2_Linear_2":   model[2],
    "3_ReLU_2":     model[3],
    "4_Linear_out": model[4],
    "5_Sigmoid":    model[5],
}

for name, layer in layer_map.items():
    h = layer.register_forward_hook(make_forward_hook(name))
    hooks.append(h)

# Run a forward pass — hooks fire automatically
x_trace = torch.randn(64, 11)  # batch of 64 samples
with torch.no_grad():
    _ = model(x_trace)

# Print collected stats
print(f"\n  {'Layer':<22} {'Shape':<15} {'Min':>8} {'Max':>8} {'Mean':>8} {'Std':>8} {'%>0':>8}")
print(f"  {'-'*80}")
print(f"  {'0_Input':<22} {str((64,11)):<15} {'—':>8} {'—':>8} {x_trace.mean().item():>8.3f} {x_trace.std().item():>8.3f} {'—':>8}")
for name, stats in activation_records.items():
    frac_str = f"{stats['fraction_positive']:.0%}" if "ReLU" in name or "Sigmoid" in name else "—"
    print(f"  {name:<22} {str(stats['shape']):<15} {stats['min']:>8.3f} {stats['max']:>8.3f} {stats['mean']:>8.3f} {stats['std']:>8.3f} {frac_str:>8}")

# IMPORTANT: Always remove hooks after use — they persist in memory
for h in hooks:
    h.remove()
activation_records.clear()
print("\n  Hooks removed.")


# ─────────────────────────────────────────────────────────────────────────────
# PART 4: EFFECT OF INITIALIZATION ON ACTIVATIONS
# ─────────────────────────────────────────────────────────────────────────────
# The initial weight distribution strongly affects how activations flow
# through a deep network at the very beginning of training.
# Bad init → vanishing or exploding activations → slow or no learning.

print("\n── PART 4: Effect of initialization on deep activations ─────")

def check_activations_deep(init_type, depth=8, width=64):
    """
    Build a deep network with specified init and measure activation std per layer.
    If std explodes (>>1) or vanishes (<<1), training will struggle.
    """
    layers = []
    for i in range(depth):
        in_size = 11 if i == 0 else width
        lin = nn.Linear(in_size, width)

        if init_type == "zeros":
            nn.init.zeros_(lin.weight)
        elif init_type == "too_large":
            nn.init.normal_(lin.weight, std=5.0)
        elif init_type == "kaiming":
            nn.init.kaiming_uniform_(lin.weight, nonlinearity="relu")
        elif init_type == "xavier":
            nn.init.xavier_uniform_(lin.weight)
        nn.init.zeros_(lin.bias)

        layers += [lin, nn.ReLU()]

    model_deep = nn.Sequential(*layers)

    # Measure std at each layer
    std_per_layer = []
    hooks_deep = []
    records = {}

    for i, layer in enumerate(model_deep):
        if isinstance(layer, nn.ReLU):
            def h(m, inp, out, idx=i):
                records[idx] = out.detach().std().item()
            hooks_deep.append(layer.register_forward_hook(h))

    x_in = torch.randn(32, 11)
    with torch.no_grad():
        model_deep(x_in)

    for h in hooks_deep:
        h.remove()

    return [records[k] for k in sorted(records.keys())]

inits = {
    "Kaiming (correct)":  "kaiming",
    "Xavier":             "xavier",
    "Too large (std=5)":  "too_large",
    "Zero weights (bad)": "zeros",
}

fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#1D9E75", "#7F77DD", "#D85A30", "#888780"]

for (label, init), color in zip(inits.items(), colors):
    stds = check_activations_deep(init)
    ax.plot(range(1, len(stds)+1), stds, marker="o", lw=2, color=color, label=label, markersize=5)

ax.axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.5, label="std=1 (ideal)")
ax.set_xlabel("Layer (after ReLU)")
ax.set_ylabel("Activation std")
ax.set_title("Activation standard deviation across layers\n(8-layer network, batch=32)")
ax.legend(fontsize=9)
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("04_activation_flow.png", dpi=150, bbox_inches="tight")
print(f"\n  Saved: 04_activation_flow.png")
# plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# PART 5: VISUALIZE PRE vs POST ACTIVATION DISTRIBUTIONS
# ─────────────────────────────────────────────────────────────────────────────

print("\n── PART 5: Pre vs post-activation distributions ─────────────")

torch.manual_seed(42)
x_vis = torch.randn(512, 11)

with torch.no_grad():
    z1_dist = model[0](x_vis)            # pre-activation (layer 1)
    a1_dist = torch.relu(z1_dist)        # post-ReLU

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle("Layer 1: Pre-activation z^[1] vs Post-activation a^[1] = ReLU(z^[1])", fontsize=11)

axes[0].hist(z1_dist.flatten().numpy(), bins=60, color="#7F77DD", edgecolor="white", lw=0.2)
axes[0].axvline(0, color="red", lw=1.5, ls="--", label="z=0")
axes[0].set_title(f"z^[1] (pre-ReLU)\nmean={z1_dist.mean().item():.2f}, std={z1_dist.std().item():.2f}")
axes[0].set_xlabel("Activation value"); axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].hist(a1_dist.flatten().numpy(), bins=60, color="#1D9E75", edgecolor="white", lw=0.2)
frac_zero = (a1_dist == 0).float().mean().item()
axes[1].set_title(f"a^[1] (post-ReLU)\n{frac_zero:.0%} values are 0 (dead neurons)")
axes[1].set_xlabel("Activation value")
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(left=-0.5)

plt.tight_layout()
plt.savefig("04_pre_post_activation.png", dpi=150, bbox_inches="tight")
print("  Saved: 04_pre_post_activation.png")
# plt.show()

print("\n[DONE] Script 04 complete.")

