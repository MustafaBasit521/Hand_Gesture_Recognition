import torch
import numpy as np

from PIL import Image
from PIL import ImageOps
from PIL import ImageFilter
from PIL import ImageEnhance

# =========================================================
# PREPROCESS IMAGE
# =========================================================

def preprocess_image(image):

    # =====================================================
    # STEP 1 — Convert to grayscale
    # =====================================================

    image = image.convert("L")

    # =====================================================
    # STEP 2 — Increase contrast
    # =====================================================

    enhancer = ImageEnhance.Contrast(image)

    image = enhancer.enhance(2.5)

    # =====================================================
    # STEP 3 — Convert to numpy
    # =====================================================

    img_array = np.array(image)

    # =====================================================
    # STEP 4 — Invert image
    # MNIST = white digit on black background
    # =====================================================

    # If image background is mostly white
    # then invert it

    if img_array.mean() > 127:
        img_array = 255 - img_array

    # =====================================================
    # STEP 5 — Threshold
    # =====================================================
    threshold = 30
    img_array = np.where(
        img_array > threshold,
        255,
        0
    ).astype(np.uint8)

    # =====================================================
    # STEP 6 — Find bounding box
    # =====================================================

    coords = np.argwhere(img_array > 0)

    # If no digit found
    if len(coords) == 0:

        blank = np.zeros((28, 28), dtype=np.float32)

        tensor = torch.tensor(blank).unsqueeze(0).unsqueeze(0)

        return tensor

    y_min, x_min = coords.min(axis=0)

    y_max, x_max = coords.max(axis=0)

    cropped = img_array[
        y_min:y_max+1,
        x_min:x_max+1
    ]

    # =====================================================
    # STEP 7 — Convert back to PIL
    # =====================================================

    cropped_img = Image.fromarray(cropped)

    # =====================================================
    # STEP 8 — Resize while preserving aspect ratio
    # =====================================================

    width, height = cropped_img.size

    # longest side becomes 20
    if width > height:

        new_width = 20

        new_height = int((height / width) * 20)

    else:

        new_height = 20

        new_width = int((width / height) * 20)

    resized = cropped_img.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    # =====================================================
    # STEP 9 — Create 28x28 black canvas
    # =====================================================

    canvas = Image.new("L", (28, 28), 0)

    # center digit
    paste_x = (28 - new_width) // 2

    paste_y = (28 - new_height) // 2

    canvas.paste(resized, (paste_x, paste_y))

    # =====================================================
    # STEP 10 — Slight blur
    # makes digit closer to MNIST style
    # =====================================================

    canvas = canvas.filter(
        ImageFilter.GaussianBlur(radius=0.2)
    )

    # =====================================================
    # STEP 11 — Normalize
    # Match training normalization
    # =====================================================

    final_array = np.array(canvas).astype(np.float32)

    final_array = final_array / 255.0

    # Normalize same as training:
    # Normalize((0.5,), (0.5,))
    final_array = (final_array - 0.5) / 0.5

    # =====================================================
    # STEP 12 — Convert to tensor
    # Shape: (1,1,28,28)
    # =====================================================

    tensor = torch.tensor(final_array)

    tensor = tensor.unsqueeze(0).unsqueeze(0)

    return tensor