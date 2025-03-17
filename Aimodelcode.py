import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================
# 1. Load & Preprocess Data
# ==============================
file_path = "data_from_March_April_2024.csv"  # Update with the correct path
df = pd.read_csv(file_path)

# Drop unnecessary columns
df = df.drop(columns=["Unnamed: 8"], errors='ignore')

# Define features (X) and target (Y)
X = df.drop(columns=["Potassium", "currentMillis"])  # Features
y = df["Potassium"]  # Target variable

# Train-Test Split (85% Training - 15% Testing, No Shuffling)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)

# Standardize features (only for ANN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# 2. Train Random Forest Model
# ==============================
rf_model = RandomForestRegressor(
    n_estimators=200,  # Increased trees for better generalization
    max_depth=25,  # Optimal depth to prevent overfitting
    min_samples_split=4,  # Prevents overfitting
    min_samples_leaf=2,  # Ensures multiple samples per leaf
    random_state=42,  # Reproducibility
    n_jobs=-1  # Uses all CPU cores
)

rf_model.fit(X_train, y_train)  # Train the model
y_pred_rf = rf_model.predict(X_test)  # Predict on test data

# Evaluate Random Forest Model
rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_mse = mean_squared_error(y_test, y_pred_rf)
rf_r2 = r2_score(y_test, y_pred_rf)

# ==============================
# 3. Train ANN Model
# ==============================
ann_model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer
    layers.Dense(128, activation='relu'),  # Hidden layer 1
    layers.Dense(64, activation='relu'),  # Hidden layer 2
    layers.Dense(1)  # Output layer
])

ann_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
                  loss='mean_squared_error', 
                  metrics=['mae'])

# Early Stopping to prevent overfitting
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = ann_model.fit(
    X_train_scaled, y_train, 
    validation_data=(X_test_scaled, y_test), 
    epochs=100,  # No fixed iterations
    batch_size=32,  # Custom batch size
    verbose=1,
    callbacks=[early_stopping]
)

# Predict on Test Set using ANN
y_pred_ann = ann_model.predict(X_test_scaled).flatten()

# Evaluate ANN Model
ann_mae = mean_absolute_error(y_test, y_pred_ann)
ann_mse = mean_squared_error(y_test, y_pred_ann)
ann_r2 = r2_score(y_test, y_pred_ann)

# ==============================
# 4. Display Results
# ==============================
print("\n=== Random Forest Results ===")
print(f"Mean Absolute Error (MAE): {rf_mae:.4f}")
print(f"Mean Squared Error (MSE): {rf_mse:.4f}")
print(f"R² Score: {rf_r2:.4f}")

print("\n=== Artificial Neural Network Results ===")
print(f"Mean Absolute Error (MAE): {ann_mae:.4f}")
print(f"Mean Squared Error (MSE): {ann_mse:.4f}")
print(f"R² Score: {ann_r2:.4f}")
