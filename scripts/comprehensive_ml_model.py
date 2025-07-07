#!/usr/bin/env python3
"""
Comprehensive ML Model for Bengaluru Groundwater Level Prediction
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("GROUNDWATER LEVEL PREDICTION - COMPREHENSIVE ML PIPELINE")
print("=" * 60)

# Load and preprocess data
print("Loading and preprocessing data...")
df = pd.read_csv('bengaluru_groundwater_merged.csv')
print(f"Dataset shape: {df.shape}")

# Create datetime column
df['datetime'] = pd.to_datetime(
    df['dataTime_year'].astype(str) + '-' + 
    df['dataTime_monthValue'].astype(str) + '-' + 
    df['dataTime_dayOfMonth'].astype(str) + ' ' +
    df['dataTime_hour'].astype(str) + ':' + 
    df['dataTime_minute'].astype(str) + ':' + 
    df['dataTime_second'].astype(str)
)

# Extract time-based features
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['day_of_year'] = df['datetime'].dt.dayofyear
df['quarter'] = df['datetime'].dt.quarter
df['is_weekend'] = df['datetime'].dt.weekday >= 5

# Create seasonal features
df['season'] = pd.cut(df['month'], bins=[0, 3, 6, 9, 12], labels=[0, 1, 2, 3])

# Handle categorical variables
categorical_cols = ['wellType', 'wellAquiferType', 'tehsil', 'stationCode']
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col].fillna('Unknown'))

# Select features for modeling
feature_cols = [
    'latitude', 'longitude', 'wellDepth',
    'year', 'month', 'day', 'day_of_year', 'quarter', 'is_weekend',
    'season', 'dataTime_hour',
    'wellType_encoded', 'wellAquiferType_encoded', 'tehsil_encoded'
]

# Remove rows with missing values
df = df.dropna(subset=feature_cols + ['dataValue'])
print(f"Final dataset shape: {df.shape}")

# Prepare X and y
X = df[feature_cols]
y = df['dataValue']

print(f"Features used: {feature_cols}")
print(f"Target variable range: {y.min():.2f} to {y.max():.2f}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create models
print("\nCreating ML models...")
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42),
    'SVR': SVR(kernel='rbf', C=1.0, gamma='scale')
}

# Train and evaluate models
print("\nTraining and evaluating models...")
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train model
    if name == 'SVR':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Store results
    results[name] = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'predictions': y_pred
    }
    
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")

# Hyperparameter tuning for Random Forest
print(f"\nPerforming hyperparameter tuning for Random Forest...")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# Evaluate tuned model
y_pred_tuned = best_rf_model.predict(X_test)
mse_tuned = mean_squared_error(y_test, y_pred_tuned)
rmse_tuned = np.sqrt(mse_tuned)
r2_tuned = r2_score(y_test, y_pred_tuned)
print(f"Tuned Random Forest RMSE: {rmse_tuned:.4f}, R²: {r2_tuned:.4f}")

# Add tuned model to results
results['Random Forest (Tuned)'] = {
    'MSE': mse_tuned,
    'RMSE': rmse_tuned,
    'MAE': mean_absolute_error(y_test, y_pred_tuned),
    'R2': r2_tuned,
    'predictions': y_pred_tuned
}

# Create visualizations
print("\nCreating visualizations...")
plt.figure(figsize=(15, 10))

# Model Performance Comparison
model_names = list(results.keys())
rmse_values = [results[name]['RMSE'] for name in model_names]
r2_values = [results[name]['R2'] for name in model_names]
mae_values = [results[name]['MAE'] for name in model_names]

plt.subplot(2, 3, 1)
plt.bar(model_names, rmse_values, color='skyblue')
plt.title('Model RMSE Comparison')
plt.ylabel('RMSE')
plt.xticks(rotation=45)

plt.subplot(2, 3, 2)
plt.bar(model_names, r2_values, color='lightgreen')
plt.title('Model R² Comparison')
plt.ylabel('R² Score')
plt.xticks(rotation=45)

plt.subplot(2, 3, 3)
plt.bar(model_names, mae_values, color='lightcoral')
plt.title('Model MAE Comparison')
plt.ylabel('MAE')
plt.xticks(rotation=45)

# Actual vs Predicted for best model
best_model_name = min(results.keys(), key=lambda x: results[x]['RMSE'])
best_predictions = results[best_model_name]['predictions']

plt.subplot(2, 3, 4)
plt.scatter(y_test, best_predictions, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'Actual vs Predicted ({best_model_name})')

# Residuals plot
plt.subplot(2, 3, 5)
residuals = y_test - best_predictions
plt.scatter(best_predictions, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title(f'Residuals Plot ({best_model_name})')

# Feature importance
plt.subplot(2, 3, 6)
feature_importance = best_rf_model.feature_importances_
feature_names = X_train.columns
indices = np.argsort(feature_importance)[::-1]
plt.bar(range(len(feature_importance)), feature_importance[indices])
plt.xticks(range(len(feature_importance)), [feature_names[i] for i in indices], rotation=45)
plt.title('Feature Importance (Tuned Random Forest)')
plt.ylabel('Importance')

plt.tight_layout()
plt.savefig('comprehensive_ml_performance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved comprehensive_ml_performance.png")

# Generate comprehensive report
print("\nGenerating performance report...")
best_model_name = min(results.keys(), key=lambda x: results[x]['RMSE'])
best_results = results[best_model_name]

report = f"""
COMPREHENSIVE GROUNDWATER LEVEL PREDICTION - ML MODEL PERFORMANCE REPORT
{'='*70}

DATASET SUMMARY:
- Total records: {len(df)}
- Training samples: {len(X_train)}
- Test samples: {len(X_test)}
- Features used: {len(X_train.columns)}

MODEL PERFORMANCE COMPARISON:
{'='*70}
"""

for name, result in results.items():
    report += f"""
{name}:
  RMSE: {result['RMSE']:.4f}
  MAE:  {result['MAE']:.4f}
  R²:   {result['R2']:.4f}
"""

report += f"""
{'='*70}
BEST MODEL: {best_model_name}
- RMSE: {best_results['RMSE']:.4f}
- MAE:  {best_results['MAE']:.4f}
- R²:   {best_results['R2']:.4f}

FEATURE IMPORTANCE (Tuned Random Forest):
"""

# Sort features by importance
feature_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

for _, row in feature_imp_df.head(10).iterrows():
    report += f"  {row['Feature']}: {row['Importance']:.4f}\n"

# Save report
with open('comprehensive_ml_report.txt', 'w') as f:
    f.write(report)

print("✓ Saved comprehensive_ml_report.txt")
print(f"\nBest model: {best_model_name}")
print(f"Best RMSE: {best_results['RMSE']:.4f}")
print(f"Best R²: {best_results['R2']:.4f}")

# Save best model
import joblib
joblib.dump(best_rf_model, f'best_groundwater_model_{best_model_name.replace(" ", "_")}.pkl')
joblib.dump(scaler, 'groundwater_scaler.pkl')

print(f"✓ Saved best model: best_groundwater_model_{best_model_name.replace(' ', '_')}.pkl")
print("✓ Saved scaler: groundwater_scaler.pkl")

print("\n" + "="*60)
print("COMPREHENSIVE ML PIPELINE COMPLETE!")
print("="*60)
print("Generated files:")
print("- comprehensive_ml_performance.png")
print("- comprehensive_ml_report.txt")
print("- best_groundwater_model_*.pkl")
print("- groundwater_scaler.pkl") 