import torch
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import cv2

def preprocess_image(image):

    # Convert to RGB to handle any format (PNG, JPG, colored)
    image = image.convert('RGB')
    img_array = np.array(image)

    # Convert to grayscale using OpenCV
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Histogram equalization — fixes colored digit problem
    # Orange/red digits become clearly visible after this
    # Maximizes contrast between digit and background
    gray = cv2.equalizeHist(gray)

    # Gaussian blur — removes noise and paper texture
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Otsu thresholding — converts to pure black and white
    # Automatically finds best threshold value
    # THRESH_BINARY_INV inverts so digit = white, background = black
    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Morphological closing with 5x5 kernel
    # Fills gaps between digital/LCD style segments
    # Fixes: gap in 9 circle, broken strokes, segment gaps
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Dilation — thickens thin strokes to match MNIST style
    kernel_dilate = np.ones((2, 2), np.uint8)
    thresh = cv2.dilate(thresh, kernel_dilate, iterations=1)

    # Crop tightly around digit and add padding
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(thresh.shape[1] - x, w + 2 * padding)
        h = min(thresh.shape[0] - y, h + 2 * padding)
        thresh = thresh[y:y+h, x:x+w]

    # Resize to 28x28 — CNN expects exactly this size
    img_pil = Image.fromarray(thresh)
    img_pil = img_pil.resize((28, 28), Image.Resampling.LANCZOS)

    # Normalize 0-1
    final_array = np.array(img_pil) / 255.0

    # Convert to tensor shape (1, 1, 28, 28)
    tensor = torch.tensor(final_array, dtype=torch.float32)
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    return tensor