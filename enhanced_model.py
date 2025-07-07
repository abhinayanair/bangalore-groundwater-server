#!/usr/bin/env python3
"""
Enhanced Groundwater ML Model with XGBoost and LightGBM
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

def load_and_preprocess_data(data_path="data/bengaluru_groundwater_merged.csv"):
    """Enhanced data preprocessing"""
    print("Loading and preprocessing data...")
    
    df = pd.read_csv(data_path)
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
    
    # Enhanced temporal features
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['day_of_year'] = df['datetime'].dt.dayofyear
    df['quarter'] = df['datetime'].dt.quarter
    df['is_weekend'] = df['datetime'].dt.weekday >= 5
    
    # Cyclical encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # Spatial features
    df['distance_from_center'] = np.sqrt(
        (df['latitude'] - 12.9716)**2 + (df['longitude'] - 77.5946)**2
    )
    
    # Seasonal features
    df['season'] = pd.cut(df['month'], 
                         bins=[0, 3, 6, 9, 12], 
                         labels=[0, 1, 2, 3]).astype(int)
    
    # Handle categorical variables
    categorical_cols = ['wellType', 'wellAquiferType', 'tehsil']
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].fillna('Unknown'))
    
    # Interaction features
    df['lat_lon_interaction'] = df['latitude'] * df['longitude']
    df['depth_year_interaction'] = df['wellDepth'] * df['year']
    
    # Select features
    feature_cols = [
        'latitude', 'longitude', 'wellDepth',
        'year', 'month', 'day', 'day_of_year', 'quarter', 'is_weekend',
        'season', 'dataTime_hour',
        'month_sin', 'month_cos', 'day_sin', 'day_cos',
        'distance_from_center', 'lat_lon_interaction', 'depth_year_interaction',
        'wellType_encoded', 'wellAquiferType_encoded', 'tehsil_encoded'
    ]
    
    df = df.dropna(subset=feature_cols + ['dataValue'])
    
    X = df[feature_cols]
    y = df['dataValue']
    
    print(f"Enhanced features used: {len(feature_cols)}")
    print(f"Final dataset shape: {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, feature_cols

def create_models():
    """Create advanced models"""
    models = {}
    
    # Baseline models
    models['Random Forest'] = RandomForestRegressor(
        n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
    )
    
    models['Gradient Boosting'] = GradientBoostingRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42
    )
    
    # Advanced models
    models['XGBoost'] = xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    )
    
    models['LightGBM'] = lgb.LGBMRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42, 
        n_jobs=-1, verbose=-1
    )
    
    return models

def evaluate_models(models, X_train, X_test, y_train, y_test):
    """Evaluate all models"""
    results = {}
    feature_importance = {}
    
    print("\nTraining and evaluating models...")
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
        
        results[name] = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'CV_R2_mean': cv_scores.mean(),
            'CV_R2_std': cv_scores.std(),
            'predictions': y_pred
        }
        
        # Store feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance[name] = model.feature_importances_
        
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return results, feature_importance

def generate_report(results, feature_importance, feature_cols):
    """Generate performance report"""
    print("\nGenerating enhanced performance report...")
    
    best_model_name = max(results.keys(), key=lambda x: results[x]['R2'])
    best_results = results[best_model_name]
    
    report = f"""
ENHANCED GROUNDWATER LEVEL PREDICTION - ML MODEL PERFORMANCE REPORT
{'='*70}

MODEL PERFORMANCE COMPARISON:
{'='*70}
"""
    
    for name, result in results.items():
        report += f"""
{name}:
  RMSE: {result['RMSE']:.4f}
  MAE:  {result['MAE']:.4f}
  R²:   {result['R2']:.4f}
  CV R²: {result['CV_R2_mean']:.4f} (+/- {result['CV_R2_std'] * 2:.4f})
"""
    
    report += f"""
{'='*70}
BEST MODEL: {best_model_name}
- RMSE: {best_results['RMSE']:.4f}
- MAE:  {best_results['MAE']:.4f}
- R²:   {best_results['R2']:.4f}
- CV R²: {best_results['CV_R2_mean']:.4f}

IMPROVEMENTS OVER BASELINE:
- Enhanced feature engineering (cyclical encoding, spatial features)
- Advanced algorithms (XGBoost, LightGBM)
- Cross-validation for robust evaluation
"""
    
    if feature_importance:
        report += "\nFEATURE IMPORTANCE (Best Model):\n"
        best_tree_model = None
        for name in ['XGBoost', 'LightGBM', 'Random Forest', 'Gradient Boosting']:
            if name in feature_importance:
                best_tree_model = name
                break
        
        if best_tree_model:
            importance = feature_importance[best_tree_model]
            feature_imp_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            for _, row in feature_imp_df.head(10).iterrows():
                report += f"  {row['Feature']}: {row['Importance']:.4f}\n"
    
    with open('enhanced_ml_report.txt', 'w') as f:
        f.write(report)
    
    print("✓ Saved enhanced_ml_report.txt")
    print(f"\nBest model: {best_model_name}")
    print(f"Best RMSE: {best_results['RMSE']:.4f}")
    print(f"Best R²: {best_results['R2']:.4f}")
    
    return report

def main():
    """Main pipeline"""
    print("ENHANCED GROUNDWATER LEVEL PREDICTION - ML PIPELINE")
    print("=" * 70)
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, feature_cols = load_and_preprocess_data()
    
    # Create models
    models = create_models()
    
    # Evaluate models
    results, feature_importance = evaluate_models(models, X_train, X_test, y_train, y_test)
    
    # Generate report
    generate_report(results, feature_importance, feature_cols)
    
    print("\n" + "="*70)
    print("ENHANCED ML PIPELINE COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main() 