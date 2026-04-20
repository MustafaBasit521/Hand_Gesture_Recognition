import torch
import numpy as np
from PIL import Image, ImageOps
import cv2

def preprocess_image(image):
    # Convert to grayscale
    img = ImageOps.grayscale(image)
    img = np.array(img)

    # Blur (reduce noise)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Threshold (make digit white on black)
    _, img = cv2.threshold(
        img, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Crop digit using bounding box
    coords = cv2.findNonZero(img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img = img[y:y+h, x:x+w]

    # Resize to 28x28
    img = cv2.resize(img, (28, 28))

    # Normalize
    img = img / 255.0

    # Convert to tensor (1,1,28,28)
    tensor = torch.tensor(img, dtype=torch.float32)
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    return tensor