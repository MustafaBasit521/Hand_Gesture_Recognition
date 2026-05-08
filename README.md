# Air-Draw: Handwritten Digit Recognizer

Air-Draw is a handwritten digit classification project built with PyTorch and Streamlit. The current system loads a trained CNN model, preprocesses an uploaded image into an MNIST-like format, and predicts the digit from `0` to `9`.

The app is currently upload-based. Real-time webcam or air-draw input is a future extension, not part of the current implementation.

## Project Info

- Students: Mustafa Basit, Izma Qamar, Manal Hameed
- Dataset: MNIST (`70,000` grayscale digit images)
- Framework: PyTorch
- Interface: Streamlit
- Task: Single-digit handwritten classification

## Current Features

- Upload a handwritten digit image in `PNG`, `JPG`, or `JPEG`
- Automatically preprocess the image before inference
- Predict the digit using a trained CNN
- Show confidence score and class probability distribution
- Save training artifacts such as confusion matrix and loss curve

## Project Structure

```text
Air-Draw/
├── app/
│   ├── app.py
│   └── utils.py
├── model/
│   ├── cnn_model.py
│   └── airdraw_model.pth
├── notebooks/
│   ├── MNIST_Preprocessing.ipynb
│   └── model_implementation.ipynb
├── results/
│   ├── confusion_matrix.png
│   └── loss_curve.png
├── requirements.txt
├── README.md
└── PROJECT_DOCUMENTATION.md
```

## Model Summary

The CNN defined in `model/cnn_model.py` uses:

- `Conv2d(1, 32, kernel_size=3, padding=1)`
- `Conv2d(32, 64, kernel_size=3, padding=1)`
- `MaxPool2d(2, 2)`
- `ReLU`
- `Dropout(0.5)`
- `Linear(64 * 7 * 7, 128)`
- `Linear(128, 10)`

## Training Configuration

Training details from `notebooks/model_implementation.ipynb`:

- Train: `48,000`
- Validation: `12,000`
- Test: `10,000`
- Batch size: `64`
- Optimizer: `Adam`
- Learning rate: `0.001`
- Loss function: `CrossEntropyLoss`
- Scheduler: `ReduceLROnPlateau(factor=0.5, patience=2)`
- Early stopping: `patience=5`, `min_delta=0.001`
- Planned epochs: `20`

### Training Augmentation

- `RandomRotation(degrees=10)`
- `RandomAffine(translate=(0.1, 0.1), scale=(0.9, 1.1))`
- `RandomErasing(p=0.1, scale=(0.02, 0.08))`
- `Normalize((0.5,), (0.5,))`

Validation and test data use:

- `ToTensor()`
- `Normalize((0.5,), (0.5,))`

## Inference Preprocessing

The upload preprocessing in `app/utils.py` currently:

1. Converts the image to grayscale
2. Increases contrast
3. Inverts the image to match MNIST style
4. Applies a fixed threshold
5. Detects the digit bounding box
6. Crops the foreground digit
7. Resizes while preserving aspect ratio
8. Centers it on a `28x28` black canvas
9. Applies slight Gaussian blur
10. Normalizes the tensor for the model

## Results

Latest evaluation from `notebooks/model_implementation.ipynb`:

- Test Accuracy: `99.39%`
- Macro F1-score: `0.99`
- Weighted F1-score: `0.99`

Per-class precision and recall are also around `0.99` to `1.00` across the MNIST test set.

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit app

```bash
streamlit run app/app.py
```

## Input Recommendations

Best results come from:

- A single handwritten digit only
- Clear foreground and plain background
- Large, centered digit
- Dark writing on a light background or the reverse
- Minimal shadows, blur, or background clutter

Likely poor inputs:

- Multiple digits in one image
- Full notebook pages
- Very noisy or low-light photos
- Digits mixed with text or symbols
- Small or faint digits

RGB images are accepted, but color is ignored because the preprocessing converts them to grayscale before inference.

## Future Goals

- Add a browser drawing canvas
- Add webcam or air-draw capture with computer vision
- Improve preprocessing for real-world photos
- Train on custom user-drawn data beyond MNIST
- Add better validation, testing, and deployment structure

For a full implementation write-up and preprocessing upgrade plan, see [PROJECT_DOCUMENTATION.md](/c:/Air-Draw/PROJECT_DOCUMENTATION.md).
