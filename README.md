# ASL Alphabet Recognition Using Deep Learning

---

## 📌 Overview

This project uses deep learning and video input to recognize the American Sign Language (ASL) alphabet and display the letters and words from that input to english text. It supports both **static hand gestures (A–Z)** and **dynamic motion gestures (J & Z)**.

It was built to demonstrate my skills in AI model training, data preprocessing, real-time inference, and the application of machine learning to real-world accessibility challenges.

---

## 🌍 Real-World Impact

Millions of people globally rely on sign language to communicate. However, language barriers often persist between Deaf and hard of hearing (Hoh) communities. This project aims to:

- Bridge that communication gap using AI
- Support assistive technology tools
- Enable more inclusive and accessible communication interfaces
- Serve as a foundation for educational or accessibility-focused applications

---

## 💻 Tech Stack

- **Language:** Python  
- **Frameworks:** TensorFlow, Keras, OpenCV  
- **Data Handling:** Pandas, NumPy  
- **Interface:** OpenCV video feed

---

## 📁 Project Structure

```bash
asl_recognition/
│
├── data/ # Downloaded datasets
│ ├── asl_alphabet_train/ # A–Z static image dataset
│ └── jz_videos/ # J and Z dynamic gesture videos
│
├── models/ # Trained models and encoder
│ ├── asl_dynamic_ml.keras # J and Z motion model
│ ├── asl_letter_mlp.keras # A–Z static model
│ └── label_encoder.pkl 
│
├── scripts
│ ├── extract_dynamic_landmarks.py
│ ├── extract_landmarks.py
│ ├── real_time_inference.py # Program Entry Point
│ ├── recorded_dynamic.py 
│ ├── train_dynamic_model.py # Model training script (J & Z)
│ ├── train_model.py # Model training script (A–Z)
│
├── requirements.txt 
└── README.md
```

---

## 📥 Dataset Instructions

### 1. Download Datasets

- **Static A–Z images:**  
  [ASL Alphabet Dataset (Kaggle)](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

- **Dynamic J & Z videos:**  
  [ASL Motion Dataset (Kaggle)](https://www.kaggle.com/datasets/signnteam/asl-sign-language-alphabet-videos-j-z)

> *Make sure you’re logged in to Kaggle and have set up the Kaggle API.*

---

### 2. Download via Kaggle CLI (Recommended)
```bash
pip install kaggle
```
Place your kaggle.json in ~/.kaggle. and run:
```bash
# Static images
kaggle datasets download -d grassknoted/asl-alphabet
unzip asl-alphabet.zip -d data/

# Dynamic gestures (J & Z)
kaggle datasets download -d signnteam/asl-sign-language-alphabet-videos-j-z
unzip asl-sign-language-alphabet-videos-j-z.zip -d data/jz_videos/
```

---

### ⚙️ Setup & Usage
1. Clone the repository
```bash
git clone https://github.com/RyanChessell/asl_recognition.git
cd asl_recognition
```
2. Create & Active Virtual Enviroment
On WSL / Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```
On Windows PowerShell:
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

4. Install Dependencies
```bash
pip install -r requirements.txt
```

4. Train the Models
```bash
python scripts/train_dynamic_model.py
```
5. Run Real-Time ASL Recognition
```bash
python .\scripts\real_time_inference.py
```

---

### Future Improvements
- Support full ASL word recognition and sentence generation

- Add voice feedback for real-time spoken output

- Deploy as a web/mobile application (via TensorFlow Lite or TF.js)

- Train on more diverse hand shapes and lighting conditions
  
---

### Contact
Ryan Chessell
[GitHub](https://github.com/RyanChessell)
[LinkedIn](www.linkedin.com/in/ryanchessell)
Email: ryans.chessell@gmail.com




