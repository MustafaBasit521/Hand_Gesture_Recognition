# Project Documentation

## 1. Overview

Air-Draw is a handwritten digit recognition project designed to classify digits from `0` to `9`. The current version is based on a Convolutional Neural Network trained on the MNIST dataset and deployed through a Streamlit app.

At present, the system works on uploaded images rather than live webcam or air-drawn input. The main goal of the project is to provide a complete classification pipeline:

- train a CNN on MNIST
- save the trained model
- preprocess new images
- run inference
- display prediction confidence

## 2. Objective

The objective of the project is to recognize a single handwritten digit from an image and classify it correctly into one of `10` classes.

This project also serves as a foundation for future work in:

- real-world handwritten digit capture
- webcam-based drawing recognition
- air-draw interaction using computer vision

## 3. Current Implementation

### 3.1 Application Layer

The Streamlit app in `app/app.py`:

- loads the trained model from `model/airdraw_model.pth`
- accepts uploaded image files
- preprocesses the image through `app/utils.py`
- runs model inference
- shows the predicted digit
- shows confidence percentage
- shows the class probability chart

### 3.2 Model Architecture

The model in `model/cnn_model.py` is a lightweight CNN suitable for MNIST:

- two convolution layers
- ReLU activation
- max pooling
- dropout regularization
- two fully connected layers

Architecture summary:

- Input: `1 x 28 x 28`
- Conv1: `1 -> 32`
- Conv2: `32 -> 64`
- Flatten
- FC1: `64 * 7 * 7 -> 128`
- FC2: `128 -> 10`

This architecture is appropriate for MNIST and explains the strong baseline performance.

### 3.3 Training Pipeline

Training was performed in `notebooks/model_implementation.ipynb`.

Configuration used:

- Dataset: MNIST
- Train set: `48,000`
- Validation set: `12,000`
- Test set: `10,000`
- Batch size: `64`
- Optimizer: `Adam`
- Learning rate: `0.001`
- Loss: `CrossEntropyLoss`
- Scheduler: `ReduceLROnPlateau`
- Early stopping enabled
- Planned epochs: `20`

### 3.4 Data Augmentation

Training augmentation includes:

- random rotation
- random affine translation and scaling
- random erasing
- normalization to match model input distribution

This helps the model generalize better than training only on untouched MNIST digits.

## 4. Current Preprocessing Logic

The current preprocessing in `app/utils.py` is designed to convert uploaded images into an MNIST-like format.

Steps:

1. Convert image to grayscale
2. Increase contrast
3. Convert to NumPy array
4. Invert image so the foreground resembles MNIST
5. Apply a fixed threshold
6. Detect nonzero foreground pixels
7. Crop to the digit bounding box
8. Resize while preserving aspect ratio
9. Place the digit on a centered `28x28` black canvas
10. Apply slight Gaussian blur
11. Normalize to the same range used in training
12. Convert to tensor shape `(1, 1, 28, 28)`

## 5. Model Performance

Latest stored evaluation results:

- Test Accuracy: `99.39%`
- Macro F1-score: `0.99`
- Weighted F1-score: `0.99`

The classification report shows very strong precision and recall across all digit classes, which confirms that the model performs well on MNIST-style inputs.

## 6. What the Current System Does Well

- It provides a complete end-to-end classification pipeline.
- It performs strongly on clean, single-digit images.
- It uses sensible preprocessing for MNIST-like inference.
- It includes regularization and augmentation during training.
- It presents predictions clearly in the user interface.

## 7. Current Limitations

- The system is not yet a real air-draw or webcam-based recognizer.
- It is optimized for MNIST-like digits, not uncontrolled real-world photos.
- It assumes a single digit is present.
- It uses a fixed threshold, which can fail under uneven lighting.
- It does not reject blank or low-quality inputs explicitly.
- It does not yet include test scripts, packaging, or deployment-oriented structure.

## 8. Stronger Preprocessing Upgrades for Real-World Photos

If the goal is to make the model work better on mobile photos, notebook-paper images, or webcam snapshots, preprocessing should be strengthened before touching the model architecture.

### 8.1 Replace Fixed Thresholding

Current logic uses a fixed threshold:

- this works for clean images
- it fails when brightness changes across the image

Recommended upgrade:

- use Otsu thresholding or adaptive thresholding
- this allows the foreground digit to separate more reliably from the background

### 8.2 Add Denoising

Real images often contain:

- paper texture
- camera noise
- shadows
- compression artifacts

Recommended upgrade:

- median blur or Gaussian blur before thresholding
- optionally bilateral filtering if you want to preserve edges

### 8.3 Detect the Largest Foreground Contour

The current method uses all nonzero pixels after thresholding. In real images, background noise can interfere with the bounding box.

Recommended upgrade:

- find connected components or contours
- keep the largest contour only
- crop around that contour instead of all foreground pixels

### 8.4 Add Morphological Cleanup

Recommended operations:

- erosion or opening to remove isolated noise
- dilation or closing to strengthen broken strokes

This is especially useful for faint pen writing or rough webcam input.

### 8.5 Add Deskewing

MNIST digits are mostly centered and aligned. Real images may be tilted.

Recommended upgrade:

- compute image moments
- estimate skew angle
- rotate the digit to a more upright pose before resizing

### 8.6 Better Centering

Current centering uses the bounding box midpoint. A more robust approach is:

- center using the digit's center of mass
- pad to a square canvas before final resize

This better matches the way MNIST digits are positioned.

### 8.7 Add Input Rejection Rules

A production-style app should not confidently classify every image.

Recommended checks:

- if foreground area is too small, reject the image
- if the largest contour is missing, show "No digit detected"
- if model confidence is below a threshold, show "Uncertain prediction"

### 8.8 Align Training With Real Inference

If real-world photos are expected, the model should be exposed to similar distortions during training:

- stronger translation
- thickness variation
- background noise
- blur
- brightness shifts
- contrast changes

The better the training data matches deployment data, the better the results will be.

## 9. Suggested Improved Real-World Preprocessing Flow

A stronger preprocessing pipeline could follow this order:

1. Convert RGB to grayscale
2. Resize large images to a manageable working size
3. Denoise
4. Apply adaptive or Otsu threshold
5. Invert if needed
6. Use contour detection to isolate the digit
7. Apply morphological cleanup
8. Deskew
9. Center by mass
10. Resize and pad to `28x28`
11. Normalize using the training mean and standard deviation

## 10. Future Goals

### 10.1 Interface Goals

- add an in-browser drawing canvas
- let users draw directly instead of uploading files
- provide a reset button and live prediction preview

### 10.2 Computer Vision Goals

- integrate webcam capture
- detect hand or fingertip movement
- convert air-drawn motion into a digit trace
- preprocess that trace into model input

### 10.3 Data Goals

- collect custom user-drawn digits
- include photos taken in real environments
- expand beyond MNIST to more realistic samples

### 10.4 Model Goals

- compare the current CNN with deeper CNNs
- evaluate lightweight modern architectures
- calibrate prediction confidence
- explore ensemble or transfer-learning approaches if the dataset grows

### 10.5 Engineering Goals

- move training from notebooks into reusable scripts
- add unit tests for preprocessing
- add inference validation and error handling
- version the trained model and experiment settings
- prepare deployment documentation

## 11. Conclusion

The current project is already a strong academic implementation of handwritten digit classification. It has a working app, a trained CNN, a preprocessing pipeline, and strong MNIST accuracy.

The next important step is not necessarily a bigger model. The biggest gain for real-world performance will come from stronger preprocessing and better real-world training data. That is the natural bridge from a good student project to a more industry-ready system.
