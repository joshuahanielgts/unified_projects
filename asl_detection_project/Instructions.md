====================================================
🖐 ASL Detection Project – Setup Instructions
====================================================

This project builds a system that detects ASL (American Sign Language) signs and predicts the corresponding alphabet using a deep learning model (CNN).

----------------------------------------------------
📦 Step 1: Download Dataset
----------------------------------------------------

The dataset is too large to include in this repository.

Please manually download it from Kaggle:
🔗 https://www.kaggle.com/datasets/grassknoted/asl-alphabet

Once downloaded:

1. Extract the ZIP file.
2. Copy the folder `asl_alphabet_train` into the following directory:
   => asl_detection_project/dataset/

So it should look like this:
📁 asl_detection_project/
├── dataset/
│   └── asl_alphabet_train/
│       ├── A/
│       ├── B/
│       ├── ...
│       └── nothing/

----------------------------------------------------
🧠 Step 2: Train the Model
----------------------------------------------------

Install TensorFlow:

    pip install tensorflow

Then run:

    python train.py

This will:
- Train a CNN model on the ASL dataset
- Save it as `asl_model.h5` inside the project directory

----------------------------------------------------
🧪 Step 3: Predict from a Test Image
----------------------------------------------------

1. Add your test images to the `test_images/` folder.
   Example:
   => asl_detection_project/test_images/B_test.jpg

2. Open `predict.py` and change the image path:

   TEST_IMAGE_PATH = 'test_images/B_test.jpg'

3. Run the script:

    python predict.py

It will print the detected ASL letter like this:
✅ Prediction: The ASL sign is 'B'

----------------------------------------------------
📄 File Structure Overview
----------------------------------------------------

asl_detection_project/
├── dataset/
│   └── asl_alphabet_train/      <-- Paste downloaded dataset here
├── test_images/                 <-- Put your test images here
├── train.py                     <-- CNN model training script
├── predict.py                   <-- Run predictions on test images
├── asl_model.h5                 <-- Model saved after training
└── INSTRUCTIONS.txt             <-- You're reading it!

----------------------------------------------------
🙌 Credits
----------------------------------------------------

J Joshua Haniel  
[GitHub](https://github.com/joshuahanielgts)  
[LinkedIn](https://www.linkedin.com/in/joshuahanielgts)
