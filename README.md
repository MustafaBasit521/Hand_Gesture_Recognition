# Air-Draw: Real-Time Handwritten Digit Recognizer

Air-Draw is a real-time computer vision project built with PyTorch, MediaPipe, OpenCV, and Streamlit. It allows users to write digits in the air using their index finger. A trained Convolutional Neural Network (CNN) then predicts the drawn digit from `0` to `9` in real-time.

## Project Info

- **Students**: Mustafa Basit, Izma Qamar, Manal Hameed
- **Model**: CNN trained on the MNIST dataset (`70,000` grayscale images)
- **Computer Vision**: MediaPipe for hand tracking, OpenCV for virtual canvas
- **Interface**: Streamlit Web App
- **Task**: Real-time single-digit handwritten classification

## ✨ Features

- **Real-Time Air-Drawing**: Use your webcam and finger to draw digits in the air.
- **Gesture Controls**: 
  - ☝️ **Index Finger Up**: Draw mode.
  - ✌️ **Two Fingers Up**: Hover / Stop drawing.
  - 🖐️ **All Fingers Up**: Clear the canvas.
- **Live Prediction**: Predicts the drawn digit instantly using a trained CNN.
- **Confidence Heatmap**: Shows a real-time probability bar chart for all digits (0-9).
- **Dataset Collector**: Save your drawn digits locally with a click of a button to build your own dataset for future training.
- **Image Upload (Legacy)**: Still supports uploading a static image of a handwritten digit for classification.

## 🚀 How to Run

### Easiest Way (Using the automated script)
We've added a shortcut script so you don't have to manually activate the virtual environment or remember the Streamlit command.

1. Open your terminal in the `Hand_Gesture_Recognition` folder.
2. Run the script:
   ```bash
   ./run.sh
   ```
*(Note: If you get a permission error, run `chmod +x run.sh` first).*

### Manual Way
1. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```
2. Run the Real-Time Camera App:
   ```bash
   streamlit run app/air_draw_app.py
   ```
*(To run the original image upload version, use `streamlit run app/app.py`)*

## 📂 Project Structure

```text
Air-Draw/
├── app/
│   ├── air_draw_app.py      # Main real-time webcam Streamlit app
│   └── app.py               # Legacy image upload Streamlit app
├── cv/
│   ├── hand_tracking.py     # MediaPipe hand gesture tracking logic
│   └── virtual_canvas.py    # OpenCV logic to draw lines on screen
├── model/
│   ├── cnn_model.py         # PyTorch CNN architecture
│   └── airdraw_model.pth    # Saved trained weights
├── preprocessing/
│   └── utils.py             # Transforms camera/canvas images to MNIST format
├── captured_data/           # Directory where user-saved digits are stored
├── requirements.txt         # Project dependencies
├── run.sh                   # Shortcut script to start the app
├── README.md                # This file
└── PROJECT_DOCUMENTATION.md # Detailed academic report
```

## 🧠 Under the Hood

### 1. Computer Vision (The "Eyes")
We use **MediaPipe Solutions** to detect hand landmarks. By tracking the coordinates of Landmark #8 (Index Finger Tip), we map its movement onto a black OpenCV `VirtualCanvas`. Gestures (number of fingers raised) control the state between drawing, hovering, and clearing.

### 2. Preprocessing (The "Bridge")
Real-world drawings are messy. Before passing the canvas to the model, `preprocessing/utils.py` applies steps to make it look exactly like MNIST training data:
- Bounding box extraction
- Resizing while preserving aspect ratio
- Centering on a 28x28 canvas
- Applying a slight Gaussian blur and normalization

### 3. CNN Model (The "Brain")
The PyTorch model features 2 Convolutional Layers, Max Pooling, Dropout (0.5), and 2 Fully Connected Layers. It achieves **99.39% accuracy** on the MNIST test set.

## ⚠️ Troubleshooting

**Error: "ModuleNotFoundError: No module named 'mediapipe.solutions'"**
This happens if you accidentally installed MediaPipe v0.10.21 or newer, which removed the `solutions` API.
*Fix*: Open terminal, activate your venv, and run:
`pip uninstall mediapipe -y && pip install mediapipe==0.10.14`
