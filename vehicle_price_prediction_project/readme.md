# 🚗 Vehicle Price Prediction

This project uses machine learning to predict the selling price of vehicles based on features like brand, model, mileage, fuel type, transmission, and more.

---

## 📂 Project Structure

vehicle_price_prediction_project/

├── dataset.csv # Dataset with vehicle listings and features

└── train.py # Python script to train and evaluate the ML model
## 📊 Dataset

- `dataset.csv` contains 1000+ rows of vehicle data
- Key features: `make`, `model`, `year`, `mileage`, `fuel`, `transmission`, `body`, `drivetrain`
- Target variable: `price`

---
📈 Model

-Model Used: RandomForestRegressor (from sklearn)
-Handles both numeric and categorical data
-Outputs performance metrics: RMSE and R² score

Output Example
✅ Vehicle Price Prediction Model Trained Successfully!
✅ RMSE: $4589.42
✅ R² Score: 0.87
