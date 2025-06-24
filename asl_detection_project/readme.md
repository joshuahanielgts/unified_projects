# 🤟 ASL Detection using Deep Learning

A deep learning-based system that detects American Sign Language (ASL) letters from an input image and outputs the corresponding alphabet character (A–Z, space, nothing).

---

## 🎯 Objective

> Build a system that can detect a given ASL input image and output what the sign represents (what letter of the alphabet is the sign).

---

## 📁 Project Structure

asl_detection_project/

├── dataset/

│ └── asl_alphabet_train/ # Training dataset

├── test_images/

│ └── B_test.jpg # Example test input

├── train.py # CNN model training script

├── predict.py # Predicts ASL letter from test image

├── asl_model.h5 # Trained model

└── README.md # Project documentation

---

## 📦 Dataset

- Source: [Kaggle - ASL Alphabet by Grassknoted](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- Contains over 87,000+ images across 29 classes: `A–Z`, `space`, and `nothing`
- Each image: 200x200 resolution (resized to 64x64 during training)

---

## 🧠 Model

- **Architecture**: Convolutional Neural Network (CNN)
- **Input**: RGB image of ASL hand sign (resized to 64x64)
- **Output**: Predicted alphabet letter (or space/nothing)

---
