"""
app.py — Main Streamlit web application for the Air-Draw Digit Recognizer.

This file sets up the user interface, loads the trained CNN model,
accepts an uploaded handwritten digit image, preprocesses it, runs
inference through the model, and displays the predicted digit along
with a confidence score and probability bar chart.
"""

# ── IMPORTS ────────────────────────────────────────────────────────────────────

import sys   # Provides access to the Python interpreter's runtime environment
import os    # Provides functions for interacting with the operating system (file paths, directories)

# ── PATH SETUP ─────────────────────────────────────────────────────────────────

# Get the absolute path of the directory where this file (app.py) lives
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go one level up from app/ to reach the project root (Hand_Gesture_Recognition/)
project_root = os.path.abspath(os.path.join(current_dir, ".."))

# Insert the project root at the beginning of sys.path so Python can find
# our custom modules like 'model' and 'preprocessing' packages
sys.path.insert(0, project_root)

# ── MODULE IMPORTS ─────────────────────────────────────────────────────────────

from model.cnn_model import MNISTNet              # Our custom CNN architecture defined in model/cnn_model.py
from preprocessing.utils import preprocess_image  # Image preprocessing pipeline from preprocessing/utils.py
import streamlit as st                            # Streamlit: framework for building interactive web apps in Python
import torch                                      # PyTorch: deep learning library used for model loading and inference
from PIL import Image                             # Pillow: Python Imaging Library for opening and handling image files

# ── PAGE SETUP ─────────────────────────────────────────────────────────────────

# Configure the Streamlit page with a browser tab title and an emoji icon
st.set_page_config(page_title="Air-Draw Digit Recognizer", page_icon="✏️")

# Display the main heading at the top of the web page
st.title("✏️ Air-Draw: Handwritten Digit Recognizer")

# Display a short description below the title to guide the user
st.write("Upload a handwritten digit image and CNN will predict it.")

# ── LOAD MODEL ─────────────────────────────────────────────────────────────────

@st.cache_resource   # Streamlit decorator: caches the returned model object so it is only loaded once across reruns
def load_model():
    """
    Load the trained MNISTNet model from disk and prepare it for inference.

    Returns:
        model (MNISTNet): The trained CNN model in evaluation mode.
    """

    # Create a new instance of our CNN architecture (random weights at this point)
    model = MNISTNet()

    # Build the absolute path to the saved model weights file (.pth)
    # os.path.dirname(__file__) gives the directory of app.py (i.e., app/)
    # '..' moves up to the project root, then into model/
    MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'airdraw_model.pth')

    # Load the saved weight dictionary (state_dict) into the model
    # map_location='cpu' ensures the model works even without a GPU
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    )

    # Switch the model to evaluation mode:
    # - Disables dropout layers (used only during training)
    # - Makes batch normalization use running statistics instead of batch statistics
    model.eval()

    return model  # Return the fully loaded, evaluation-ready model

# Call load_model() once; thanks to @st.cache_resource it won't reload on every user interaction
model = load_model()

# ── FILE UPLOADER ──────────────────────────────────────────────────────────────

# Render a file upload widget that accepts PNG, JPG, and JPEG image formats
uploaded_file = st.file_uploader(
    "Upload a digit image (PNG or JPG)",  # Label shown above the upload button
    type=["png", "jpg", "jpeg"]           # Restrict accepted file types to these image formats
)

# ── INFERENCE SECTION ──────────────────────────────────────────────────────────

# Only execute the block below if the user has uploaded a file
if uploaded_file is not None:

    # Open the uploaded file as a PIL Image object for manipulation
    image = Image.open(uploaded_file)

    # Display the uploaded image in the Streamlit app with a caption and fixed width
    st.image(image, caption="Uploaded Image", width=150)

    # Show the original pixel dimensions (width x height) of the uploaded image
    st.write("Image size:", image.size)

    # Render a clickable button; the indented block only runs when the button is pressed
    if st.button("🔍 Predict Digit"):

        # ── PREPROCESSING ──────────────────────────────────────────────────────

        # Pass the PIL image through the full preprocessing pipeline:
        # grayscale → contrast boost → invert → threshold → crop → resize → normalize → tensor
        # Returns a PyTorch tensor of shape (1, 1, 28, 28)
        tensor = preprocess_image(image)

        # ── DISPLAY PROCESSED IMAGE ────────────────────────────────────────────

        # Remove the batch and channel dimensions to get a 2D (28×28) numpy array
        # squeeze() removes all dimensions of size 1 → shape goes from (1,1,28,28) to (28,28)
        processed = tensor.squeeze().numpy()

        # Undo the normalization that was applied during preprocessing:
        # Original: (pixel - 0.5) / 0.5  →  Reverse: pixel * 0.5 + 0.5
        # This maps values back to [0, 1] so the image displays correctly
        processed = (processed * 0.5) + 0.5

        # Show the processed 28×28 grayscale image so the user can verify it looks correct
        st.image(
            processed,                     # The 2D numpy array to display as an image
            caption="Processed 28x28 Image",  # Label shown below the image
            width=150                      # Display size in pixels
        )

        # ── MODEL INFERENCE ────────────────────────────────────────────────────

        # Disable gradient computation during inference:
        # - Not needed for prediction (saves memory and speeds up computation)
        with torch.no_grad():

            # Forward pass: feed the preprocessed tensor through the CNN
            # output shape: (1, 10) — raw scores (logits) for each digit class 0–9
            output = model(tensor)

            # Convert raw logits to probabilities using softmax along the class dimension (dim=1)
            # Each value represents the probability that the image is that digit
            probabilities = torch.softmax(output, dim=1)

            # Find the index (0–9) with the highest probability — this is the predicted digit
            predicted = torch.argmax(probabilities)

            # Extract the confidence value (probability of the predicted digit) as a Python float
            # Multiply by 100 to convert from a decimal (e.g., 0.97) to a percentage (e.g., 97%)
            confidence = probabilities[0][predicted].item() * 100

        # ── DISPLAY RESULTS ────────────────────────────────────────────────────

        # Show the predicted digit in a green success box with large heading style
        st.success(f"### Predicted Digit: {predicted.item()}")

        # Show the confidence percentage in a blue information box, rounded to 2 decimal places
        st.info(f"Confidence: {confidence:.2f}%")

        # Add a subheading above the bar chart
        st.subheader("Probability Distribution")

        # Render a horizontal bar chart showing the probability for each digit 0–9
        # detach() removes the tensor from the computation graph before converting to numpy
        st.bar_chart(probabilities[0].detach().numpy())

        # Add a small caption below the chart to explain what it shows
        st.caption("Probability for each digit 0 through 9")