import sys
import os
import cv2
import numpy as np
import torch
import streamlit as st
from PIL import Image

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from cv.hand_tracking import HandDetector
from cv.virtual_canvas import VirtualCanvas
from preprocessing.utils import preprocess_image
from model.cnn_model import MNISTNet

# ── PAGE SETUP ─────────────────────────────────────────────
st.set_page_config(page_title="Air-Draw AI", page_icon="🎨", layout="wide")
st.title("🎨 Air-Draw: Real-Time Digit Recognition")
st.markdown("""
    Write digits in the air using your **index finger**!
    *   **Index Finger Up**: Draw
    *   **Two Fingers Up**: Hover / Stop Drawing
    *   **All Fingers Up**: Clear Canvas
""")

# ── LOAD MODEL ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = MNISTNet()
    MODEL_PATH = os.path.join(project_root, 'model', 'airdraw_model.pth')
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# ── INITIALIZE CV COMPONENTS ───────────────────────────────
detector = HandDetector(max_hands=1, detection_con=0.8)
# We'll initialize the canvas inside the loop to match camera resolution

# ── STREAMLIT UI LAYOUT ────────────────────────────────────
st.sidebar.header("Controls")
run_app = st.sidebar.checkbox("Run Camera", value=True)

if st.sidebar.button("🗑️ Clear Canvas"):
    st.session_state.clear_requested = True

save_button = st.sidebar.button("💾 Save Current Digit")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🎮 Controls Guide
*   **Index Finger Up**: Draw
*   **Two Fingers Up**: Hover / Stop Drawing
*   **All Fingers Up**: Clear Canvas
""")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Webcam Feed")
    frame_placeholder = st.empty()

with col2:
    st.subheader("Prediction Results")
    result_placeholder = st.empty()
    chart_placeholder = st.empty()
    st.markdown("---")
    st.subheader("What the model sees")
    canvas_placeholder = st.empty()

if 'clear_requested' not in st.session_state:
    st.session_state.clear_requested = False

# ── MAIN CV LOOP ───────────────────────────────────────────
cap = cv2.VideoCapture(0)
canvas = None

while run_app:
    success, img = cap.read()
    if not success:
        st.error("Failed to access webcam. Please check your camera connection.")
        break

    img = cv2.flip(img, 1) # Flip for mirror effect
    h, w, c = img.shape

    if canvas is None:
        canvas = VirtualCanvas(w, h)

    if st.session_state.clear_requested:
        canvas.clear_canvas()
        st.session_state.clear_requested = False

    # Find Hands
    img = detector.find_hands(img)
    lm_list = detector.find_position(img, draw=False)

    if len(lm_list) != 0:
        x1, y1 = lm_list[8][1:] # Index finger tip
        fingers = detector.fingers_up()

        # Drawing Mode: Only index finger is up
        if fingers[1] == 1 and fingers[2] == 0:
            cv2.circle(img, (x1, y1), 10, (0, 255, 0), cv2.FILLED)
            img = canvas.draw(img, x1, y1, is_drawing=True)
        
        # Hover Mode: Reset previous points
        else:
            img = canvas.draw(img, x1, y1, is_drawing=False)

        # Clear Gesture: All fingers up (except thumb maybe)
        if all(f == 1 for f in fingers[1:]):
            canvas.clear_canvas()

    # Combine image and canvas
    img_gray = cv2.cvtColor(canvas.get_canvas(), cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, canvas.get_canvas())

    # Display in Streamlit
    frame_placeholder.image(img, channels="BGR")
    
    # Real-time Prediction
    raw_canvas = canvas.get_canvas()
    if np.sum(raw_canvas) > 0: # If something is drawn
        # Convert canvas to PIL for preprocessing
        canvas_pil = Image.fromarray(cv2.cvtColor(raw_canvas, cv2.COLOR_BGR2RGB))
        tensor = preprocess_image(canvas_pil)
        
        with torch.no_grad():
            output = model(tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted = torch.argmax(probabilities).item()
            confidence = probabilities[0][predicted].item() * 100
        
        # Update results in UI
        result_placeholder.markdown(f"## Predicted: {predicted} ({confidence:.1f}%)")
        
        # Feature C: Real-time probability chart
        probs_numpy = probabilities[0].numpy()
        chart_placeholder.bar_chart(probs_numpy)
        
        # Feature D: Dataset Collector
        processed_img = (tensor.squeeze().numpy() * 0.5) + 0.5
        canvas_placeholder.image(processed_img, caption="28x28 Model Input", width=150)

        # Handle Save Button (State persists between loop iterations if not careful)
        # In this simple loop, we check the button state from the last script run
        if save_button:
            import time
            timestamp = int(time.time())
            save_path = os.path.join(project_root, 'captured_data', str(predicted), f"digit_{timestamp}.png")
            # Convert back to 0-255 range for saving
            img_to_save = (processed_img * 255).astype(np.uint8)
            cv2.imwrite(save_path, img_to_save)
            st.sidebar.success(f"Saved as {predicted}!")
            save_button = False # Reset for this loop iteration

    if not run_app:
        break

cap.release()
