"""
utils.py — Image preprocessing pipeline for the Air-Draw digit recognizer.

Converts a raw uploaded (or captured) handwritten digit image into a normalized
28×28 PyTorch tensor that matches the format expected by the MNISTNet CNN model.

Pipeline steps:
  1.  Convert to grayscale
  2.  Boost contrast
  3.  Convert to NumPy array
  4.  Invert if background is white (MNIST style = white digit on black bg)
  5.  Apply binary threshold
  6.  Find bounding box of the digit
  7.  Convert cropped region back to PIL Image
  8.  Resize to 20×20 preserving aspect ratio
  9.  Paste onto a 28×28 black canvas (centered)
  10. Apply slight Gaussian blur
  11. Normalize pixel values to match training normalization
  12. Convert to PyTorch tensor shaped (1, 1, 28, 28)
"""

import torch          # PyTorch: used to create the output tensor
import numpy as np    # NumPy: used for array math and pixel operations

from PIL import Image         # Pillow: base image class for opening/creating images
from PIL import ImageOps      # Pillow: helper for image operations like inversion (not used directly here)
from PIL import ImageFilter   # Pillow: provides convolution filters like GaussianBlur
from PIL import ImageEnhance  # Pillow: provides enhancement tools like Contrast


# =========================================================
# PREPROCESS IMAGE
# =========================================================

def preprocess_image(image):
    """
    Transform a raw PIL Image of a handwritten digit into a model-ready tensor.

    This function mirrors the preprocessing that was applied to MNIST images
    during training, so the model receives input in a consistent format.

    Args:
        image (PIL.Image): Raw image uploaded by the user (any size, any mode).

    Returns:
        tensor (torch.Tensor): Normalized tensor of shape (1, 1, 28, 28).
                               dtype = float32, values in range roughly [-1, 1].
    """

    # =====================================================
    # STEP 1 — Convert to grayscale
    # =====================================================

    # "L" mode = 8-bit luminance (grayscale), one channel per pixel
    # Drops color information; MNIST is grayscale so the model expects 1-channel input
    image = image.convert("L")

    # =====================================================
    # STEP 2 — Increase contrast
    # =====================================================

    # Create a Contrast enhancer object for the grayscale image
    enhancer = ImageEnhance.Contrast(image)

    # Boost contrast by a factor of 2.5
    # Factor > 1 increases contrast (makes dark pixels darker, light pixels lighter)
    # This helps separate the digit from a noisy or lightly-drawn background
    image = enhancer.enhance(2.5)

    # =====================================================
    # STEP 3 — Convert to numpy
    # =====================================================

    # Convert the PIL Image to a 2D NumPy array for pixel-level manipulation
    # Shape: (height, width), dtype: uint8, values: 0–255
    img_array = np.array(image)

    # =====================================================
    # STEP 4 — Invert image if needed
    # MNIST = white digit on black background
    # =====================================================

    # Calculate the average brightness of all pixels
    # If mean > 127, the image has a mostly bright (white) background
    if img_array.mean() > 127:
        # Invert the pixel values: new_pixel = 255 - old_pixel
        # This turns a white background → black and dark digit → bright digit
        # Required because MNIST uses white digits on black backgrounds
        img_array = 255 - img_array

    # =====================================================
    # STEP 5 — Threshold
    # =====================================================

    # Define a threshold value: pixels brighter than this are "digit", rest are "background"
    threshold = 30

    # Apply binary threshold: every pixel is set to either 255 (digit) or 0 (background)
    # np.where(condition, value_if_true, value_if_false)
    # Pixels > 30 → 255 (white, part of the digit)
    # Pixels ≤ 30 → 0   (black, background)
    # .astype(np.uint8) ensures values stay in valid 0–255 range for image operations
    img_array = np.where(
        img_array > threshold,
        255,
        0
    ).astype(np.uint8)

    # =====================================================
    # STEP 6 — Find bounding box of the digit
    # =====================================================

    # Find all pixel positions where the value is > 0 (part of the digit)
    # np.argwhere returns an array of [row, col] pairs for all non-zero pixels
    coords = np.argwhere(img_array > 0)

    # If no white pixels were found → the image contains no digit
    if len(coords) == 0:

        # Return a blank 28×28 tensor of zeros (model input = no digit)
        blank = np.zeros((28, 28), dtype=np.float32)

        # unsqueeze(0) adds a dimension: (28,28) → (1,28,28) [channel dim]
        # unsqueeze(0) again: (1,28,28) → (1,1,28,28) [batch dim]
        tensor = torch.tensor(blank).unsqueeze(0).unsqueeze(0)

        return tensor  # Return early with a blank tensor

    # Get the topmost and leftmost pixel positions (minimum row and column)
    y_min, x_min = coords.min(axis=0)

    # Get the bottommost and rightmost pixel positions (maximum row and column)
    y_max, x_max = coords.max(axis=0)

    # Crop the array to just the bounding box containing the digit
    # +1 on max indices to include the last row/column (Python slice is exclusive at end)
    cropped = img_array[
        y_min:y_max+1,
        x_min:x_max+1
    ]

    # =====================================================
    # STEP 7 — Convert back to PIL
    # =====================================================

    # Convert the cropped NumPy array back to a PIL Image for resizing
    cropped_img = Image.fromarray(cropped)

    # =====================================================
    # STEP 8 — Resize while preserving aspect ratio
    # =====================================================

    # Get the current width and height of the cropped digit image
    width, height = cropped_img.size

    # Scale the longer dimension to 20 pixels, and compute the shorter dimension
    # proportionally so the aspect ratio (shape) of the digit is preserved
    if width > height:
        # Digit is wider than tall → scale width to 20 and compute proportional height
        new_width = 20
        new_height = int((height / width) * 20)  # Proportional height

    else:
        # Digit is taller than wide (or square) → scale height to 20 and compute proportional width
        new_height = 20
        new_width = int((width / height) * 20)   # Proportional width

    # Resize the cropped digit to the calculated dimensions
    # Image.Resampling.LANCZOS = high-quality downsampling filter (anti-aliased)
    resized = cropped_img.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    # =====================================================
    # STEP 9 — Create 28x28 black canvas and center the digit
    # =====================================================

    # Create a blank 28×28 grayscale canvas filled with black (0)
    # "L" mode = grayscale, 0 = black pixel value
    canvas = Image.new("L", (28, 28), 0)

    # Calculate where to paste the resized digit so it is centered on the 28×28 canvas
    # Integer division (//) gives the top-left corner position for centering
    paste_x = (28 - new_width) // 2   # Horizontal offset from the left edge
    paste_y = (28 - new_height) // 2  # Vertical offset from the top edge

    # Paste the resized digit onto the canvas at the calculated centered position
    canvas.paste(resized, (paste_x, paste_y))

    # =====================================================
    # STEP 10 — Slight Gaussian blur
    # Makes the digit look closer to MNIST style
    # =====================================================

    # Apply a very light Gaussian blur (radius=0.2) to slightly smooth sharp pixel edges
    # MNIST digits are slightly smoothed, so this helps bridge the domain gap
    canvas = canvas.filter(
        ImageFilter.GaussianBlur(radius=0.2)
    )

    # =====================================================
    # STEP 11 — Normalize pixel values
    # Must match the training normalization exactly
    # =====================================================

    # Convert the canvas to a float32 NumPy array (required for math operations below)
    # Shape: (28, 28), dtype: float32
    final_array = np.array(canvas).astype(np.float32)

    # Scale pixel values from [0, 255] down to [0.0, 1.0]
    final_array = final_array / 255.0

    # Apply the same normalization used during training:
    # transforms.Normalize((0.5,), (0.5,)) → (pixel - mean) / std = (pixel - 0.5) / 0.5
    # This maps [0, 1] → [-1, 1], centering data around 0 for better model convergence
    final_array = (final_array - 0.5) / 0.5

    # =====================================================
    # STEP 12 — Convert to PyTorch tensor
    # Final shape must be (1, 1, 28, 28)
    # =====================================================

    # Convert the 2D NumPy array to a PyTorch float tensor
    # Shape at this point: (28, 28)
    tensor = torch.tensor(final_array)

    # Add channel dimension: (28, 28) → (1, 28, 28)   [single grayscale channel]
    # Add batch dimension:   (1, 28, 28) → (1, 1, 28, 28) [single image in batch]
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    # Return the final tensor ready for the CNN model
    return tensor