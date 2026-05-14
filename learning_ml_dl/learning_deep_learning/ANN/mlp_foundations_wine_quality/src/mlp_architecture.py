"""
============================================================
 03_mlp_architecture.py
 Deep Learning for Soft Matter Physics
 ── MLP Architecture with nn.Module ─────────────────────────

 GOAL:
   Build MLPs the "proper" PyTorch way using nn.Module.
   Understand how weight matrices map to the mathematical
   notation W^[l], b^[l] from theory.

 TWO VERSIONS:
   SimpleMLP  — hardcoded 3-layer network (explicit, transparent)
   DeepMLP    — configurable network (takes hidden_sizes as list)

 CONCEPTS COVERED:
   - nn.Module base class and why we inherit from it
   - nn.Linear: W^[l] and b^[l] stored and initialized automatically
   - forward() method: defines the computation graph
   - nn.Sequential: chains layers in order
   - named_parameters(): inspect all weight matrices
   - Weight initialization: Kaiming uniform (default for ReLU nets)
   - Counting trainable parameters

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

print("=" * 60)
print("  MLP ARCHITECTURE — nn.Module")
print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# VERSION 1: SimpleMLP — explicit, transparent
# ─────────────────────────────────────────────────────────────────────────────
# Architecture:
#   Input(11) => Linear(11=>64) => ReLU => Linear(64=>32) => ReLU => Linear(32=>1) => Sigmoid
#
# Why inherit from nn.Module?
#   - Automatically registers sub-modules (nn.Linear layers) as parameters
#   - model.parameters() returns all learnable weights => given to optimizer
#   - model.train() / model.eval() switches behavior of Dropout, BatchNorm
#   - .to(device) moves all parameters to GPU/CPU in one call
#   - .state_dict() / .load_state_dict() for saving/loading

print("\n── VERSION 1: SimpleMLP ─────────────────────────────────────")

class SimpleMLP(nn.Module):
    """
    A transparent 2-hidden-layer MLP for binary wine quality classification.

    Architecture:  11 => 64 => 32 => 1
    Activations:   ReLU (hidden), Sigmoid (output)
    Task:          Binary classification (good vs ordinary wine)

    Input  shape:  (batch_size, 11)
    Output shape:  (batch_size, 1)  — value in (0, 1) = P(good wine)
    """

    def __init__(self):
        """
        Define all layers that contain parameters.

        This function is called once when the model object is created:

            model = SimpleMLP()

        The layers created here will be reused every time we call:

            output = model(x)

        Important:
            We define layers in __init__.
            We define the data flow in forward().
        """

        # This initializes the parent nn.Module class.
        #
        # It is absolutely necessary.
        #
        # Without this line, PyTorch will not properly track submodules,
        # parameters, buffers, and internal states.
        super().__init__()

        # ----------------------------------------------------------------------
        # First fully connected layer
        # ----------------------------------------------------------------------
        #
        # nn.Linear(in_features, out_features)
        #
        # Here:
        #     in_features  = 11
        #     out_features = 64
        #
        # This means:
        #     Each wine sample has 11 input values.
        #     The layer produces 64 new learned features.
        #
        # Mathematical operation:
        #
        #     z1 = x @ W1.T + b1
        #
        # Shapes:
        #
        #     x      : (batch_size, 11)
        #     W1     : (64, 11)
        #     W1.T   : (11, 64)
        #     b1     : (64,)
        #     z1     : (batch_size, 64)
        #
        # Note:
        #     PyTorch stores Linear weights as:
        #
        #         (out_features, in_features)
        #
        #     Therefore W1 has shape:
        #
        #         (64, 11)
        #
        # ----------------------------------------------------------------------

        self.layer1 = nn.Linear(11, 64)   # W^[1]: (64, 11),   b^[1]: (64,)
        self.layer2 = nn.Linear(64, 32)   # W^[2]: (32, 64),   b^[2]: (32,)
        self.layer3 = nn.Linear(32,  1)   # W^[3]: (1,  32),   b^[3]: (1,)

        # Activation functions (these have no learnable parameters)
        self.relu    = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define the forward pass computation.
        PyTorch calls this automatically when you do: output = model(input)

        The two-step pattern per layer:
          z^[l] = W^[l] · a^[l-1] + b^[l]    (linear transform)
          a^[l] = activation(z^[l])             (non-linearity)

        Args:
            x: input tensor, shape (batch_size, 11)

        Returns:
            output: tensor of shape (batch_size, 1), values in (0, 1)
        """

        # Layer 1: (batch, 11) => (batch, 64)
        # nn.Linear handles the matrix multiply and bias add:
        #   z = x @ W.T + b   (PyTorch broadcasts bias over batch dimension)
        x = self.layer1(x)    # z^[1]: pre-activation
        x = self.relu(x)      # a^[1]: post-activation

        # Layer 2: (batch, 64) => (batch, 32)
        x = self.layer2(x)    # z^[2]
        x = self.relu(x)      # a^[2]

        # Output layer: (batch, 32) => (batch, 1)
        # No ReLU here — sigmoid maps the logit to a probability
        x = self.layer3(x)    # z^[3] = logit
        x = self.sigmoid(x)   # ŷ = σ(z^[3]) ∈ (0, 1)

        return x

# Create one instance of SimpleMLP.
# This initializes all weights and biases automatically.
model_simple = SimpleMLP()
# Print the model structure.
# This shows the layers, but not all individual weights.
print(model_simple)

# ─────────────────────────────────────────────────────────────────────────────
# VERSION 2: DeepMLP — configurable, for rapid experimentation
# ─────────────────────────────────────────────────────────────────────────────

print("\n── VERSION 2: DeepMLP (configurable) ────────────────────────")

class DeepMLP(nn.Module):
    """
    Configurable MLP — architecture defined entirely by arguments.
    Pass any list of hidden layer sizes and get a working network.

    Args:
        input_size   (int)        : number of input features
        hidden_sizes (list[int])  : widths of hidden layers
                                    e.g. [64, 32, 16] => 3 hidden layers
        output_size  (int)        : output neurons (1 for binary classification)
        activation   (nn.Module)  : activation class (not instance!) for hidden layers
        dropout_p    (float)      : dropout probability 0=off, 0.5=drop half
        use_batchnorm(bool)       : apply BatchNorm1d after each linear layer

    Example:
        model = DeepMLP(11, [128, 64, 32], dropout_p=0.3)
    """

    def __init__(
        self,
        input_size:    int,
        hidden_sizes:  list,
        output_size:   int   = 1,
        activation           = nn.ReLU,
        dropout_p:     float = 0.0,
        use_batchnorm: bool  = False,
    ):

        # Initialize nn.Module internals.
        super().__init__()

        # This Python list will temporarily store all layers.
        #
        # Later, we convert this list into nn.Sequential.
        layers = []
        # prev_size keeps track of the number of features entering
        # the next Linear layer.
        #
        # At the beginning, the first Linear layer receives raw input features.
        prev_size = input_size

        for h_size in hidden_sizes:
            # Linear layer: z = Wx + b
            layers.append(nn.Linear(prev_size, h_size))

            # Optional: BatchNorm normalizes activations within a batch
            # Helps with training stability, especially for deeper networks
            # Applied BEFORE activation (pre-activation BatchNorm)
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_size))

            # Activation: introduce non-linearity
            layers.append(activation())

            # Optional: Dropout randomly zeros some neurons during training
            # Acts as regularization — prevents overfitting
            if dropout_p > 0.0:
                layers.append(nn.Dropout(p=dropout_p))

            prev_size = h_size

        # Output layer — no dropout, sigmoid for binary classification
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Sigmoid())

        # nn.Sequential chains all modules — forward() applies them in order
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the configurable MLP.

        Because all layers are stored inside self.network,
        the forward pass is very compact.

        Args:
            x:
                Tensor of shape:

                    (batch_size, input_size)

        Returns:
            Tensor of shape:

                    (batch_size, output_size)

            For binary wine classification:
                    (batch_size, 1)
        """

        # Pass input through the full Sequential model.
        return self.network(x)

# ==============================================================================
# CREATE MULTIPLE ARCHITECTURES
# ==============================================================================

architectures = {
    "tiny    (1 hidden, 16)":       DeepMLP(11, [16]),
    "small   (2 hidden, 64-32)":    DeepMLP(11, [64, 32]),
    "medium  (3 hidden, 128-64-32)":DeepMLP(11, [128, 64, 32]),
    "large   (4 hidden, 256-128-64-32)": DeepMLP(11, [256, 128, 64, 32]),
    "deep    (6 hidden)":           DeepMLP(11, [128, 128, 64, 64, 32, 16]),
    "dropout (3 hidden + p=0.3)":   DeepMLP(11, [128, 64, 32], dropout_p=0.3),
    "batchnorm":                    DeepMLP(11, [128, 64, 32], use_batchnorm=True),
}


# ==============================================================================
# PART 3: PARAMETER COUNTING
# ==============================================================================

print("\n── PART 3: Parameter counts ──────────────────────────────────")

def count_parameters(model):
    """Count total trainable parameters."""
    # model.parameters() returns all parameter tensors in the model.
    #
    # For each parameter p:
    #
    #     p.numel()
    #
    # gives the number of scalar values in that tensor.
    #
    # Example:
    #     a weight matrix of shape (128, 11)
    #     has 128 * 11 = 1408 scalar parameters.
    #
    # We only count parameters where p.requires_grad is True.
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def parameter_breakdown(model):
    """Show parameter count per layer."""
    total = 0
    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        print(f"    {name:<35} shape={str(list(param.shape)):<15}  params={n:,}")
    print(f"    {'TOTAL':<35} {'':15}  params={total:,}")

print(f"\n  Architecture comparison:")
print(f"  {'Name':<38} {'Parameters':>12}")
print(f"  {'-'*52}")
for name, model in architectures.items():
    n = count_parameters(model)
    print(f"  {name:<38} {n:>12,}")

# Detailed breakdown for medium model
print(f"\n  Detailed parameter breakdown (medium 128-64-32):")
parameter_breakdown(architectures["medium  (3 hidden, 128-64-32)"])


# ==============================================================================
# PART 4: WEIGHT INITIALIZATION
# ==============================================================================

# Default: Kaiming uniform — designed for ReLU networks.
# Variance of weights scales as 2/n_in → prevents vanishing/exploding activations.
#
# Alternative initializations you can apply manually:

print("\n── PART 4: Weight initialization ────────────────────────────")

def apply_init(model, init_type="kaiming"):
    """
    Apply custom weight initialization to all Linear layers.
    init_type: 'kaiming' | 'xavier' | 'normal' | 'zeros_test'
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            if init_type == "kaiming":
                # Best for ReLU — variance = 2/fan_in
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            elif init_type == "xavier":
                # Best for tanh/sigmoid — variance = 2/(fan_in + fan_out)
                nn.init.xavier_uniform_(module.weight)
            elif init_type == "normal":
                # Small random normal — simple baseline
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
            elif init_type == "zeros_test":
                # BAD: all zeros => symmetric, gradients identical, network can't learn
                nn.init.zeros_(module.weight)

            # Bias is usually initialized to zeros
            nn.init.zeros_(module.bias)
    return model

# ------------------------------------------------------------------------------
# Compare Kaiming and Xavier initialization distributions
# ------------------------------------------------------------------------------

# Inspect distribution of initialized weights
model_test = DeepMLP(11, [128, 64, 32])
weights_kaiming = model_test.network[0].weight.data.numpy().flatten()

model_test2 = apply_init(DeepMLP(11, [128, 64, 32]), init_type="xavier")
weights_xavier = model_test2.network[0].weight.data.numpy().flatten()

# ------------------------------------------------------------------------------
# Plot histograms of initial weights
# ------------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("Weight Initialization Distributions (layer 1 weights)", fontsize=11)

axes[0].hist(weights_kaiming, bins=50, color="#7F77DD", edgecolor="white", lw=0.3)
axes[0].set_title(f"Kaiming Uniform  (std={weights_kaiming.std():.4f})")
axes[0].set_xlabel("Weight value"); axes[0].grid(True, alpha=0.3)

axes[1].hist(weights_xavier, bins=50, color="#1D9E75", edgecolor="white", lw=0.3)
axes[1].set_title(f"Xavier Uniform  (std={weights_xavier.std():.4f})")
axes[1].set_xlabel("Weight value"); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("03_weight_init.png", dpi=150, bbox_inches="tight")
print(f"  Saved: 03_weight_init.png")
# plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# PART 5: VERIFY SHAPES WITH A DUMMY FORWARD PASS
# ─────────────────────────────────────────────────────────────────────────────

print("\n── PART 5: Shape verification with dummy data ───────────────")

torch.manual_seed(0)
batch_sizes = [1, 8, 32]

for bs in batch_sizes:
    x_dummy = torch.randn(bs, 11)   # fake batch
    with torch.no_grad():
        out = model_simple(x_dummy)
    print(f"  Input ({bs}, 11) → Output {tuple(out.shape)}  — all values in (0,1)? {bool((out > 0).all() and (out < 1).all())}")

# Test all architectures
print(f"\n  Testing all architectures with batch size 16:")
x_test = torch.randn(16, 11)
for name, model in architectures.items():
    model.eval()
    with torch.no_grad():
        out = model(x_test)
    print(f"    {name:<38}  output shape: {tuple(out.shape)}")


# ─────────────────────────────────────────────────────────────────────────────
# PART 6: ARCHITECTURE DIAGRAM (text-based)
# ─────────────────────────────────────────────────────────────────────────────

print("\n── PART 6: Architecture diagram (medium MLP) ────────────────")
print("""
  Input x ∈ R11
      │
      
  ┌──────────────────────────────────────┐
  │  nn.Linear(11, 128)                  │
  │  z^[1] = W^[1]·x + b^[1]            │  W^[1] ∈ R^(128*11)
  │  a^[1] = ReLU(z^[1])                │  1408 + 128 = 1536 params
  └──────────────────────────────────────┘
      │
      
  ┌──────────────────────────────────────┐
  │  nn.Linear(128, 64)                  │
  │  z^[2] = W^[2]·a^[1] + b^[2]        │  W^[2] ∈ R^(64*128)
  │  a^[2] = ReLU(z^[2])                │  8192 + 64 = 8256 params
  └──────────────────────────────────────┘
      │
      
  ┌──────────────────────────────────────┐
  │  nn.Linear(64, 32)                   │
  │  z^[3] = W^[3]·a^[2] + b^[3]        │  W^[3] ∈ R^(32*64)
  │  a^[3] = ReLU(z^[3])                │  2048 + 32 = 2080 params
  └──────────────────────────────────────┘
      │
      
  ┌──────────────────────────────────────┐
  │  nn.Linear(32, 1) + Sigmoid          │
  │  z^[4] = W^[4]·a^[3] + b^[4]        │  W^[4] ∈ R^(1*32)
  │  ŷ     = sigma(z^[4]) ∈ (0, 1)          │  32 + 1 = 33 params
  └──────────────────────────────────────┘
      │
      
  y_cap ∈ (0, 1) — P(good wine)
""")

print("[DONE] Script 03 complete.")
print("Key takeaways:")
print("  1. nn.Module is the base class for ALL PyTorch models")
print("  2. nn.Linear(m, n) stores W belongs to R^(n*m) and b belongs to R^n, init'd automatically")
print("  3. forward() defines computation — never call it directly, use model(x)")
print("  4. Kaiming init is best for ReLU; Xavier for sigmoid/tanh")
print("  5. More parameters not equals better — need enough data to support model size")
