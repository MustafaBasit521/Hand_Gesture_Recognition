import torch
import numpy as np
from PIL import Image, ImageOps, ImageFilter

def preprocess_image(image):

    # Convert to RGB first
    image = image.convert('RGB')
    
    # Convert to grayscale
    gray = ImageOps.grayscale(image)
    
    # Enhance contrast
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Contrast(gray)
    gray = enhancer.enhance(2.0)
    
    # Convert to numpy for thresholding
    img_array = np.array(gray)
    
    # Otsu-style threshold
    threshold = np.mean(img_array)
    binary = np.where(img_array < threshold, 255, 0).astype(np.uint8)
    
    # Convert back to PIL
    thresh = Image.fromarray(binary)
    
    # Resize to 28x28
    thresh = thresh.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Normalize 0-1
    final_array = np.array(thresh) / 255.0
    
    # Convert to tensor shape (1, 1, 28, 28)
    tensor = torch.tensor(final_array, dtype=torch.float32)
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    
    return tensor