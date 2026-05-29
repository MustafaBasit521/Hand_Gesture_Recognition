"""
cnn_model.py — Defines the MNISTNet Convolutional Neural Network architecture.

Architecture overview:
  Input  : (batch, 1, 28, 28)  — single-channel 28×28 grayscale image
  Conv1  : 1  → 32 feature maps, 3×3 kernel, same padding
  Pool1  : 2×2 Max Pooling  →  (batch, 32, 14, 14)
  Conv2  : 32 → 64 feature maps, 3×3 kernel, same padding
  Pool2  : 2×2 Max Pooling  →  (batch, 64, 7, 7)
  Flatten: (batch, 64×7×7) = (batch, 3136)
  FC1    : 3136 → 128  (with ReLU + Dropout)
  FC2    : 128  → 10   (raw logits for digits 0–9)
"""

import torch          # PyTorch core library for tensor operations and autograd
import torch.nn as nn # torch.nn contains building blocks for neural networks (layers, activations, etc.)


class MNISTNet(nn.Module):
    """
    A Convolutional Neural Network (CNN) designed to classify handwritten digits (0–9).

    This model is trained on the MNIST dataset. It uses two convolutional blocks
    (Conv → ReLU → MaxPool) followed by two fully connected layers for classification.
    """

    def __init__(self):
        """
        Constructor: define and register all the learnable layers of the network.
        These layers are initialized with random weights and will be trained later.
        """

        # Call the parent class (nn.Module) constructor — required for PyTorch to properly
        # register all layers and enable features like .parameters() and .state_dict()
        super().__init__()

        # ── CONVOLUTIONAL LAYERS ──────────────────────────────────────────────

        # First convolutional layer:
        #   in_channels  = 1  (grayscale image has 1 color channel)
        #   out_channels = 32 (learns 32 different feature detectors / filters)
        #   kernel_size  = 3  (each filter looks at a 3×3 patch of pixels)
        #   padding      = 1  (adds 1 pixel of zero-padding around the border so the
        #                       output spatial size stays the same as the input: 28×28)
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)

        # Second convolutional layer:
        #   in_channels  = 32 (takes the 32 feature maps output by conv1)
        #   out_channels = 64 (learns 64 higher-level feature detectors)
        #   kernel_size  = 3  (3×3 filter)
        #   padding      = 1  (preserves spatial dimensions before pooling)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        # ── POOLING LAYER ─────────────────────────────────────────────────────

        # Max Pooling layer (shared/reused for both conv blocks):
        #   kernel_size = 2  (looks at 2×2 regions)
        #   stride      = 2  (moves 2 pixels at a time — no overlap)
        # Effect: halves both the width and height of the feature maps
        #   After pool1: 28×28 → 14×14
        #   After pool2: 14×14 → 7×7
        self.pool = nn.MaxPool2d(2, 2)

        # ── ACTIVATION FUNCTION ───────────────────────────────────────────────

        # ReLU (Rectified Linear Unit): introduces non-linearity into the network
        # Formula: f(x) = max(0, x)  — zeros out all negative values
        # This allows the network to learn complex, non-linear patterns
        self.relu = nn.ReLU()

        # ── REGULARIZATION ────────────────────────────────────────────────────

        # Dropout layer with 50% probability (p=0.5):
        # During training, randomly sets 50% of neuron outputs to zero in each forward pass.
        # This prevents overfitting by forcing the network not to rely on specific neurons.
        # During evaluation/inference (.eval() mode), dropout is automatically disabled.
        self.dropout = nn.Dropout(0.5)

        # ── FULLY CONNECTED LAYERS ────────────────────────────────────────────

        # First fully connected (linear) layer:
        #   in_features  = 64 * 7 * 7 = 3136
        #     (64 feature maps, each of size 7×7, after two rounds of Conv + Pool)
        #   out_features = 128 (compressed hidden representation)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)

        # Second fully connected layer (output layer):
        #   in_features  = 128 (output of fc1)
        #   out_features = 10  (one raw score / logit per digit class: 0, 1, 2, ..., 9)
        # No activation here — raw logits are passed to CrossEntropyLoss during training
        # or to Softmax during inference to get probabilities
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """
        Define the forward pass — how data flows through the network layer by layer.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, 1, 28, 28).

        Returns:
            torch.Tensor: Raw output logits of shape (batch_size, 10),
                          one score per digit class (0–9).
        """

        # ── BLOCK 1: Conv1 → ReLU → MaxPool ──────────────────────────────────

        # 1. Apply conv1: learns low-level features (edges, corners)
        #    Shape: (batch, 1, 28, 28) → (batch, 32, 28, 28)
        # 2. Apply ReLU: zero out negatives to add non-linearity
        # 3. Apply pool: halve spatial dimensions via 2×2 max pooling
        #    Shape: (batch, 32, 28, 28) → (batch, 32, 14, 14)
        x = self.pool(self.relu(self.conv1(x)))

        # ── BLOCK 2: Conv2 → ReLU → MaxPool ──────────────────────────────────

        # 1. Apply conv2: learns higher-level features (curves, digit parts)
        #    Shape: (batch, 32, 14, 14) → (batch, 64, 14, 14)
        # 2. Apply ReLU: zero out negatives
        # 3. Apply pool: halve spatial dimensions again
        #    Shape: (batch, 64, 14, 14) → (batch, 64, 7, 7)
        x = self.pool(self.relu(self.conv2(x)))

        # ── FLATTEN ───────────────────────────────────────────────────────────

        # Flatten all dimensions except the batch dimension (dim 0) into a 1D vector
        # Shape: (batch, 64, 7, 7) → (batch, 3136)
        # This is required to feed the spatial feature maps into a fully connected layer
        x = torch.flatten(x, 1)

        # ── FULLY CONNECTED BLOCK ─────────────────────────────────────────────

        # Apply fc1 followed by ReLU: compress 3136 features down to 128
        # Shape: (batch, 3136) → (batch, 128)
        x = self.relu(self.fc1(x))

        # Apply dropout: randomly zero 50% of the 128 activations during training
        # Helps regularize the model to reduce overfitting
        x = self.dropout(x)

        # Apply fc2 (output layer): map 128 features to 10 class scores (logits)
        # Shape: (batch, 128) → (batch, 10)
        x = self.fc2(x)

        # Return the raw logits — no softmax here because:
        # - During training: nn.CrossEntropyLoss applies log-softmax internally
        # - During inference: we apply torch.softmax() manually in app.py
        return x