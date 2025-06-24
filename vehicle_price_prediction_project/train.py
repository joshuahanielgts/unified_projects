import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("dataset.csv")

# Drop irrelevant or high-cardinality columns
df.drop(columns=['name', 'description', 'engine', 'trim', 'exterior_color', 'interior_color'], inplace=True)

# Drop rows with missing target
df.dropna(subset=['price'], inplace=True)

# Fill remaining missing values
df.fillna({
    'cylinders': df['cylinders'].median(),
    'mileage': df['mileage'].median(),
    'fuel': df['fuel'].mode()[0],
    'transmission': df['transmission'].mode()[0],
    'body': df['body'].mode()[0],
    'doors': df['doors'].mode()[0],
}, inplace=True)

# Encode categorical columns
categorical_cols = ['make', 'model', 'fuel', 'transmission', 'body', 'drivetrain']
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# Features and target
X = df.drop('price', axis=1)
y = df['price']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # ✅ FIXED
r2 = r2_score(y_test, y_pred)

# Results
print("✅ Vehicle Price Prediction Model Trained Successfully!")
print(f"✅ RMSE: ${rmse:.2f}")
print(f"✅ R² Score: {r2:.2f}")
