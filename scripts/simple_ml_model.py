#!/usr/bin/env python3
"""
Simplified ML Model for Bengaluru Groundwater Level Prediction
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("Starting simplified ML pipeline...")

# Load data
print("Loading data...")
df = pd.read_csv('bengaluru_groundwater_merged.csv')
print(f"Dataset shape: {df.shape}")

# Create datetime column
df['datetime'] = pd.to_datetime(
    df['dataTime_year'].astype(str) + '-' + 
    df['dataTime_monthValue'].astype(str) + '-' + 
    df['dataTime_dayOfMonth'].astype(str)
)

# Extract time-based features
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['day_of_year'] = df['datetime'].dt.dayofyear

# Handle categorical variables
categorical_cols = ['wellType', 'wellAquiferType', 'tehsil']
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col].fillna('Unknown'))

# Select features
feature_cols = [
    'latitude', 'longitude', 'wellDepth',
    'year', 'month', 'day', 'day_of_year',
    'wellType_encoded', 'wellAquiferType_encoded', 'tehsil_encoded'
]

# Remove rows with missing values
df = df.dropna(subset=feature_cols + ['dataValue'])
print(f"Final dataset shape: {df.shape}")

# Prepare X and y
X = df[feature_cols]
y = df['dataValue']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set: {X_train.shape[0]}, Test set: {X_test.shape[0]}")

# Train Random Forest
print("Training Random Forest model...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

# Create visualization
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted')

plt.subplot(1, 3, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals Plot')

plt.subplot(1, 3, 3)
feature_importance = rf_model.feature_importances_
feature_names = X_train.columns
indices = np.argsort(feature_importance)[::-1]
plt.bar(range(len(feature_importance)), feature_importance[indices])
plt.xticks(range(len(feature_importance)), [feature_names[i] for i in indices], rotation=45)
plt.title('Feature Importance')
plt.ylabel('Importance')

plt.tight_layout()
plt.savefig('simple_ml_results.png', dpi=300, bbox_inches='tight')
plt.close()

# Save results
with open('simple_ml_report.txt', 'w') as f:
    f.write(f"""SIMPLE ML MODEL RESULTS
{'='*40}
Dataset: {df.shape[0]} records
Features: {len(feature_cols)}
Training samples: {X_train.shape[0]}
Test samples: {X_test.shape[0]}

Model Performance:
RMSE: {rmse:.4f}
MAE: {mae:.4f}
R²: {r2:.4f}

Top 5 Most Important Features:
""")
    for i, idx in enumerate(indices[:5]):
        f.write(f"{i+1}. {feature_names[idx]}: {feature_importance[idx]:.4f}\n")

print("✓ Saved simple_ml_results.png")
print("✓ Saved simple_ml_report.txt")
print("Simple ML pipeline completed!") 