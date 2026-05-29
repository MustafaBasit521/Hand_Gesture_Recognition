"""
train.py — Training script for the MNISTNet CNN model.

This script:
  1. Sets up data transforms (with augmentation for training)
  2. Downloads and splits the MNIST dataset into train/validation sets
  3. Defines an EarlyStopping callback to prevent overfitting
  4. Trains the MNISTNet model using Adam optimizer and CrossEntropyLoss
  5. Saves the best model weights whenever validation loss improves
  6. Plots and saves the training vs. validation loss curve
"""

# ── IMPORTS ────────────────────────────────────────────────────────────────────

import os   # For file path construction and directory creation
import sys  # For modifying Python's module search path

# Add the project root directory to sys.path so we can import from model/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch                          # PyTorch core: tensors, device management
import torch.nn as nn                 # Neural network building blocks (layers, loss functions)
import torchvision.transforms as transforms  # Image transformation pipeline utilities

from torchvision import datasets           # Pre-built datasets including MNIST
from torch.utils.data import DataLoader, random_split  # Data loading and splitting utilities

import matplotlib.pyplot as plt  # For plotting and saving the loss curve graph

from model.cnn_model import MNISTNet  # Our custom CNN architecture

# =========================================================
# PATHS — Resolve directory locations relative to this file
# =========================================================

# Compute the absolute path to the project root directory
# os.path.dirname(__file__) → directory containing train.py (i.e., training/)
# '..'                       → one level up → project root (Hand_Gesture_Recognition/)
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Create the model/ directory if it doesn't already exist
# exist_ok=True prevents an error if the directory is already there
os.makedirs(f'{BASE_PATH}/model', exist_ok=True)

# Create the results/ directory for saving the loss curve plot
os.makedirs(f'{BASE_PATH}/results', exist_ok=True)

# =========================================================
# DEVICE — Use GPU if available, otherwise fall back to CPU
# =========================================================

# torch.cuda.is_available() returns True if a CUDA-capable GPU is detected
# "cuda" enables GPU acceleration; "cpu" runs on the processor (slower but always available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Print which device will be used so the user knows what hardware is active
print(f"Using device: {device}")

# =========================================================
# TRANSFORMS — Define image preprocessing pipelines
# =========================================================

# Training transform: includes data augmentation to make the model more robust
train_transform = transforms.Compose([

    # Convert PIL Image or numpy array to a float32 PyTorch tensor
    # Also scales pixel values from [0, 255] → [0.0, 1.0]
    transforms.ToTensor(),

    # Randomly rotate the image by up to ±10 degrees
    # Helps model handle slightly tilted handwriting
    transforms.RandomRotation(degrees=10),

    # Randomly apply affine transformations (without rotation):
    #   translate=(0.1, 0.1) → shift image up to 10% of width/height in any direction
    #   scale=(0.9, 1.1)     → randomly zoom in/out between 90% and 110%
    # Helps model handle digits drawn at different sizes and positions
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1)
    ),

    # Randomly erase a small rectangular patch from the image
    #   p=0.1       → 10% chance of applying this augmentation each time
    #   scale=(0.02, 0.08) → erased patch covers 2–8% of the image area
    # Simulates occlusions and teaches the model to be robust to missing parts
    transforms.RandomErasing(
        p=0.1,
        scale=(0.02, 0.08)
    ),

    # Normalize pixel values: (pixel - mean) / std = (pixel - 0.5) / 0.5
    # Maps [0, 1] → [-1, 1] to center the data around 0
    # Consistent normalization helps gradient descent converge faster
    transforms.Normalize((0.5,), (0.5,))
])

# Validation/test transform: no augmentation, just convert and normalize
# Augmentation is only used during training, not evaluation
test_transform = transforms.Compose([

    # Convert to float tensor and scale to [0, 1]
    transforms.ToTensor(),

    # Same normalization as training so model sees consistent input distribution
    transforms.Normalize((0.5,), (0.5,))
])

# =========================================================
# DATASETS — Download and load MNIST
# =========================================================

# Load the full MNIST training set (60,000 images of handwritten digits 0–9)
# root='./data' → download/cache the dataset in a local 'data/' folder
# train=True    → load the training split (not the test split)
# download=True → automatically download if not already cached
# transform     → apply the augmented pipeline to every training image
full_train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=train_transform
)

# =========================================================
# TRAIN / VALIDATION SPLIT — Split training data 80/20
# =========================================================

# Calculate how many samples go into training (80%) and validation (20%)
train_size = int(0.8 * len(full_train_dataset))  # e.g., 48,000 samples

# Remaining samples go to validation
val_size = len(full_train_dataset) - train_size   # e.g., 12,000 samples

# Randomly split the full training dataset into train and validation subsets
# random_split returns two Dataset objects with non-overlapping indices
train_dataset, val_dataset = random_split(
    full_train_dataset,
    [train_size, val_size]
)

# Override the transform for validation data to use the clean (no augmentation) transform
# The validation set shares the same underlying dataset object, so we replace its transform
# This ensures validation is evaluated on clean images, not randomly augmented ones
val_dataset.dataset.transform = test_transform

# =========================================================
# DATALOADERS — Wrap datasets for batched iteration
# =========================================================

# Training DataLoader: feeds batches of 64 images to the model during training
# shuffle=True → randomly shuffles the order of training samples each epoch
#   (prevents the model from memorizing the order of training examples)
train_loader = DataLoader(
    train_dataset,
    batch_size=64,   # Process 64 images at a time
    shuffle=True     # Randomly shuffle for each new epoch
)

# Validation DataLoader: same batch size but no shuffling needed
# shuffle=False → consistent order makes validation results reproducible
val_loader = DataLoader(
    val_dataset,
    batch_size=64,   # Same batch size as training
    shuffle=False    # No need to shuffle validation data
)

# Print dataset sizes so the user can verify the split looks correct
print(f"Train: {len(train_dataset)}")       # Should be ~48,000
print(f"Validation: {len(val_dataset)}")    # Should be ~12,000

# =========================================================
# EARLY STOPPING — Stop training when validation loss stops improving
# =========================================================

class EarlyStopping:
    """
    Monitors validation loss and signals when training should stop early.

    If the validation loss does not improve by at least `min_delta` for
    `patience` consecutive epochs, the `stop` flag is set to True.

    This prevents wasting time on epochs that no longer improve the model
    and helps prevent overfitting.
    """

    def __init__(self, patience=5, min_delta=0.001):
        """
        Initialize the early stopping tracker.

        Args:
            patience  (int):   Number of epochs without improvement before stopping.
            min_delta (float): Minimum amount of improvement required to be counted.
        """

        # How many consecutive epochs of no improvement to tolerate before stopping
        self.patience = patience

        # Minimum change in loss to count as an improvement
        self.min_delta = min_delta

        # Counter: how many consecutive epochs have passed without improvement
        self.counter = 0

        # Tracks the best (lowest) validation loss seen so far
        self.best_loss = None

        # Flag: when True, the training loop should break
        self.stop = False

    def check(self, loss):
        """
        Check whether the current validation loss constitutes an improvement.

        Args:
            loss (float): Validation loss for the current epoch.
        """

        if self.best_loss is None:
            # First epoch: set the baseline best loss
            self.best_loss = loss

        elif loss < self.best_loss - self.min_delta:
            # Validation loss improved by more than min_delta → record new best
            self.best_loss = loss

            # Reset the counter since we just saw an improvement
            self.counter = 0

            # Inform the user that validation loss got better
            print(f"✅ Validation improved: {loss:.4f}")

        else:
            # No meaningful improvement → increment the patience counter
            self.counter += 1

            # Show the user how many epochs without improvement have passed
            print(f"⚠️ No improvement {self.counter}/{self.patience}")

            # If we've waited `patience` epochs with no improvement → trigger early stop
            if self.counter >= self.patience:
                self.stop = True  # Signal the training loop to break
                print("🛑 Early stopping triggered")

# =========================================================
# INITIALIZE MODEL, LOSS, OPTIMIZER, SCHEDULER
# =========================================================

# Instantiate the CNN and move it to the selected device (GPU or CPU)
model = MNISTNet().to(device)

# Loss function: CrossEntropyLoss is the standard for multi-class classification
# It internally applies log-softmax, so the model just needs to output raw logits
criterion = nn.CrossEntropyLoss()

# Optimizer: Adam (Adaptive Moment Estimation)
# Adjusts learning rates per-parameter using first and second moment estimates
# lr=0.001 is the standard starting learning rate for Adam
optimizer = torch.optim.Adam(
    model.parameters(),  # All trainable weights and biases in the network
    lr=0.001             # Initial learning rate
)

# Learning rate scheduler: ReduceLROnPlateau
# Automatically reduces the learning rate when validation loss stops decreasing
# mode='min'    → we want the monitored metric (val_loss) to decrease
# factor=0.5    → multiply learning rate by 0.5 when reducing (halve it)
# patience=2    → wait 2 epochs of no improvement before reducing the LR
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=2
)

# Instantiate early stopping with 5 epochs of patience and 0.001 minimum improvement
early_stop = EarlyStopping(
    patience=5,
    min_delta=0.001
)

# =========================================================
# TRAINING LOOP — Train for up to EPOCHS epochs
# =========================================================

EPOCHS = 20  # Maximum number of training epochs (may stop earlier via early stopping)

train_losses = []     # Accumulates average training loss per epoch for plotting
val_losses = []       # Accumulates average validation loss per epoch for plotting
best_val_loss = float('inf')  # Track the best validation loss seen (start at infinity)

for epoch in range(EPOCHS):  # Iterate over each epoch from 0 to EPOCHS-1

    # =====================================================
    # TRAINING PHASE — Update model weights
    # =====================================================

    # Set model to training mode:
    # - Enables dropout (randomly zeros activations)
    # - Uses batch statistics for batch normalization (if any)
    model.train()

    running_loss = 0.0  # Accumulates the total loss over all training batches this epoch

    for images, labels in train_loader:  # Iterate over all batches in the training set

        # Move images and labels to the device (GPU/CPU) for computation
        images = images.to(device)  # Tensor of shape (64, 1, 28, 28)
        labels = labels.to(device)  # Tensor of shape (64,) with digit class indices 0–9

        # Zero out gradients from the previous batch
        # If not cleared, gradients accumulate across batches (usually undesired)
        optimizer.zero_grad()

        # Forward pass: compute predictions by passing images through the network
        # outputs shape: (64, 10) — raw logits for each of the 10 digit classes
        outputs = model(images)

        # Compute the cross-entropy loss between predictions and true labels
        loss = criterion(outputs, labels)

        # Backward pass: compute gradients of loss w.r.t. all model parameters
        loss.backward()

        # Update model weights by taking a step in the direction of the negative gradient
        optimizer.step()

        # Accumulate this batch's loss for computing the epoch average
        running_loss += loss.item()  # .item() converts tensor to Python float

    # Calculate average training loss for this epoch (divide total by number of batches)
    train_loss = running_loss / len(train_loader)

    # Append the average training loss to the list for later plotting
    train_losses.append(train_loss)

    # =====================================================
    # VALIDATION PHASE — Evaluate on unseen data
    # =====================================================

    # Set model to evaluation mode:
    # - Disables dropout (uses all neurons for prediction)
    # - Uses running statistics for batch normalization (if any)
    model.eval()

    val_running_loss = 0.0  # Accumulates total validation loss
    correct = 0             # Counts correctly classified validation images
    total = 0               # Counts total validation images processed

    # Disable gradient computation during validation:
    # - We don't need gradients for evaluation (saves memory and speeds up inference)
    with torch.no_grad():

        for images, labels in val_loader:  # Iterate over all validation batches

            # Move validation batch to device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass through the model (no gradient tracking)
            outputs = model(images)

            # Compute validation loss for this batch
            loss = criterion(outputs, labels)

            # Accumulate batch loss
            val_running_loss += loss.item()

            # torch.max returns (max_values, indices); we only need indices (predicted class)
            # dim=1 → take the max across the 10 class logits for each sample
            _, predicted = torch.max(outputs, 1)

            # Add the number of samples in this batch to the total count
            total += labels.size(0)

            # Count how many predictions exactly matched the true labels
            # (predicted == labels) is a boolean tensor; .sum() counts Trues; .item() → Python int
            correct += (predicted == labels).sum().item()

    # Calculate average validation loss across all validation batches
    val_loss = val_running_loss / len(val_loader)

    # Append to the list for plotting
    val_losses.append(val_loss)

    # Calculate validation accuracy as a percentage
    accuracy = 100 * correct / total

    # =====================================================
    # SAVE BEST MODEL — Only when validation improves
    # =====================================================

    if val_loss < best_val_loss:  # This epoch has the best (lowest) validation loss so far

        # Update the record of the best validation loss
        best_val_loss = val_loss

        # Save the model's learned parameters (weights + biases) to disk
        # state_dict() returns an ordered dictionary of all parameter tensors
        torch.save(
            model.state_dict(),
            f'{BASE_PATH}/model/airdraw_model.pth'  # File path for the saved model weights
        )

        print("✅ Best model saved!")

    # =====================================================
    # SCHEDULER + EARLY STOPPING
    # =====================================================

    # Inform the learning rate scheduler of the current validation loss
    # If loss hasn't improved for `patience` epochs, it will reduce the learning rate
    scheduler.step(val_loss)

    # Check if training should stop early based on validation loss stagnation
    early_stop.check(val_loss)

    # Print a summary of this epoch's results
    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")           # e.g., "Epoch [3/20]"
    print(f"Train Loss: {train_loss:.4f}")            # Average training loss, 4 decimal places
    print(f"Validation Loss: {val_loss:.4f}")         # Average validation loss
    print(f"Validation Accuracy: {accuracy:.2f}%")   # Accuracy on unseen data

    # If early stopping was triggered → exit the training loop early
    if early_stop.stop:
        break

# Print a final message when training is complete (either all epochs or early stopped)
print("\n✅ Training Complete!")

# =========================================================
# LOSS CURVES — Plot and save training vs. validation loss
# =========================================================

# Create a new matplotlib figure with a specified size (width=8 inches, height=5 inches)
plt.figure(figsize=(8, 5))

# Plot the training loss over epochs as a line
# Each element in train_losses is the average loss for one epoch
plt.plot(train_losses, label='Train Loss')

# Plot the validation loss over epochs as a separate line
plt.plot(val_losses, label='Validation Loss')

# Label the x-axis
plt.xlabel('Epoch')

# Label the y-axis
plt.ylabel('Loss')

# Set the chart title
plt.title('Training vs Validation Loss')

# Add a legend to identify which line is training vs. validation
plt.legend()

# Add a grid to make the plot easier to read
plt.grid(True)

# Adjust layout so labels and title don't get clipped
plt.tight_layout()

# Save the figure as a PNG image to the results/ directory
plt.savefig(f'{BASE_PATH}/results/loss_curve.png')

# Display the plot interactively (opens a window if running locally)
plt.show()