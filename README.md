import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
file_path = "/mnt/data/lettuce_dataset_updated.csv"  
df = pd.read_csv(file_path, encoding='latin1')

# Drop non-numeric columns and check for missing values
df_numeric = df.drop(columns=['Plant_ID', 'Date']).dropna()

# Define features and target variable
X = df_numeric.drop(columns=['Growth Days'])  # Features
y = df_numeric['Growth Days']  # Target variable

# Set global random seed for reproducibility
np.random.seed(42)

# Split data into training and testing sets (shuffle=False for consistency)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Standardize the features for ANN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# FINAL Random Forest settings to ensure full stability
rf_model = RandomForestRegressor(
    n_estimators=500,  # More trees for fully stable results
    random_state=42,
    bootstrap=False  # Remove randomness from bootstrapping
)

# FINAL ANN settings for full stability and convergence
ann_model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=5000,  # More iterations for full training
    alpha=0.0001,  # Weight regularization
    tol=1e-6,  # Better convergence criteria
    random_state=42,
    early_stopping=True,  # Stops training only when fully converged
    learning_rate_init=0.001  # Smoother training process
)

# Train models
rf_model.fit(X_train, y_train)
ann_model.fit(X_train_scaled, y_train)

# Predict using the models
rf_predictions = rf_model.predict(X_test)
ann_predictions = ann_model.predict(X_test_scaled)

# Evaluate models
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"Model": model_name, "MAE": mae, "MSE": mse, "R² Score": r2}

rf_results = evaluate_model(y_test, rf_predictions, "Random Forest")
ann_results = evaluate_model(y_test, ann_predictions, "Artificial Neural Network")

# Convert results to DataFrame
final_results_df = pd.DataFrame([rf_results, ann_results])

# Save the results to an Excel file
final_verified_excel_path = "/mnt/data/final_verified_model_evaluation_results.xlsx"
final_results_df.to_excel(final_verified_excel_path, index=False)

# Print results
print(final_results_df)

# Provide the download link
print(f"Download the results: {final_verified_excel_path}")
