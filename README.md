# ✏️ Air-Draw: Handwritten Digit Recognizer

A deep learning project that recognizes handwritten digits using a 
Convolutional Neural Network (CNN) trained on the MNIST dataset, 
with a real-time drawing interface.

---

## 👨‍💻 Project Info
- **Student:** Mustafa Basit
- **Dataset:** MNIST (70,000 handwritten digit images)
- **Model:** Convolutional Neural Network (CNN) — PyTorch
- **Interface:** Streamlit Web App

---

## 📁 Project Structure
Air-Draw/
├── notebooks/
│   ├── phase1_preprocessing.ipynb
│   └── phase2_training_evaluation.ipynb
│
├── model/
│   ├── cnn_model.py
│   └── airdraw_model.pth   ✅ (VERY IMPORTANT)
│
├── app/
│   ├── app.py
│   └── utils.py
│
├── results/
│   ├── confusion_matrix.png
│   └── accuracy_plot.png
│
├── requirements.txt
└── README.md

---

## 🔄 Project Phases

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Data Preprocessing | ✅ Complete |
| Phase 2 | CNN Model + Evaluation + Streamlit App | 🔄 In Progress |
| Phase 3 | Real-time Webcam Integration | 🔜 Upcoming |

---

## ⚙️ How to Run

### 1 — Install Dependencies
pip install -r requirements.txt

### 2 — Run the Streamlit App
streamlit run app/app.py

---

## 📊 Results (Phase 2)
- **Accuracy:** XX%
- **F1 Score:** XX%
- **Loss:** XX

*(Will be updated after Phase 2 completion)*

---

## 📚 Libraries Used
- PyTorch
- Torchvision
- Streamlit
- NumPy
- Pillow
- OpenCV