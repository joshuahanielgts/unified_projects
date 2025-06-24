# 🗂 Dataset Setup Instructions – ASL Detection Project

This guide will help you download and correctly place the dataset required to train and test the ASL Detection model.

---

## 📦 Step 1: Download the Dataset

The dataset is too large to upload directly to GitHub. Please download it manually from Kaggle:

🔗 [ASL Alphabet Dataset (by grassknoted)](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

---

## 📁 Step 2: Folder Structure After Extraction

Once downloaded and extracted:
You will see a folder named: `asl_alphabet_train`


✅ Make sure the folder `asl_alphabet_train` is directly inside the `dataset/` folder.

---

## 🧠 Step 3: Train the Model

Make sure you have TensorFlow installed:

```bash
pip install tensorflow
```
Now run the training script:
```
python train.py
```

This will:

-Train a CNN model on the ASL dataset

-Save the trained model as asl_model.h5

🧪 Step 4: Run Predictions
Place your test images (like A_test.jpg, B_test.jpg) in the test_images/ folder.

In predict.py, update the image path:

```
TEST_IMAGE_PATH = 'test_images/B_test.jpg'
```
Run the prediction script:
```
python predict.py
```
You will see output like:

✅ Prediction: The ASL sign is 'B'

🗂 Final Project Structure

asl_detection_project/

├── dataset/

│   └── asl_alphabet_train/   # <-- Place downloaded dataset here

├── test_images/                  # <-- Place test images here

├── train.py                      # CNN model training script

├── predict.py                    # Predicts ASL sign from image

├── asl_model.h5                  # Trained model (after running train.py)

└── DATASET_SETUP.md              # You're reading this file!

🙌 Author

J Joshua Haniel  
[GitHub](https://github.com/joshuahanielgts)  
[LinkedIn](https://www.linkedin.com/in/joshuahanielgts)
