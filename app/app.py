import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.cnn_model import MNISTNet
from utils import preprocess_image
import streamlit as st
import torch

from PIL import Image

# ── PAGE SETUP ─────────────────────────────────────────────
st.set_page_config(page_title="Air-Draw Digit Recognizer", page_icon="✏️")
st.title("✏️ Air-Draw: Handwritten Digit Recognizer")
st.write("Upload a handwritten digit image and CNN will predict it.")

# ── LOAD MODEL ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = MNISTNet()

    MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'airdraw_model.pth')

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    )

    model.eval()
    return model

model = load_model()

# ── FILE UPLOADER ──────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a digit image (PNG or JPG)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=150)
    
    if st.button("🔍 Predict Digit"):
        
        tensor = preprocess_image(image)
        
        with torch.no_grad():
            output = model(tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted = torch.argmax(probabilities)
            confidence = probabilities[0][predicted].item() * 100
        
        st.success(f"### Predicted Digit: {predicted.item()}")
        st.info(f"Confidence: {confidence:.2f}%")
        
        st.subheader("Probability Distribution")
        st.bar_chart(probabilities[0].detach().numpy())
        st.caption("Probability for each digit 0 through 9")