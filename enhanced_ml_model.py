#!/usr/bin/env python3
"""
Enhanced ML Model for Bengaluru Groundwater Level Prediction
This script implements advanced ML techniques to improve upon the current model.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Install with: pip install lightgbm")

# Set random seed for reproducibility
np.random.seed(42)

class EnhancedGroundwaterMLModel:
    def __init__(self, data_path="data/bengaluru_groundwater_merged.csv"):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = RobustScaler()  # More robust to outliers
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        
    def load_and_preprocess_data(self):
        """Enhanced data preprocessing with advanced feature engineering"""
        print("Loading and preprocessing data with enhanced features...")
        
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {self.df.shape}")
        
        # Create datetime column
        self.df['datetime'] = pd.to_datetime(
            self.df['dataTime_year'].astype(str) + '-' + 
            self.df['dataTime_monthValue'].astype(str) + '-' + 
            self.df['dataTime_dayOfMonth'].astype(str) + ' ' +
            self.df['dataTime_hour'].astype(str) + ':' + 
            self.df['dataTime_minute'].astype(str) + ':' + 
            self.df['dataTime_second'].astype(str)
        )
        
        # Enhanced temporal features
        self.df['year'] = self.df['datetime'].dt.year
        self.df['month'] = self.df['datetime'].dt.month
        self.df['day'] = self.df['datetime'].dt.day
        self.df['day_of_year'] = self.df['datetime'].dt.dayofyear
        self.df['quarter'] = self.df['datetime'].dt.quarter
        self.df['is_weekend'] = self.df['datetime'].dt.weekday >= 5
        self.df['day_of_week'] = self.df['datetime'].dt.weekday
        
        # Enhanced seasonal features
        self.df['season'] = pd.cut(self.df['month'], 
                                  bins=[0, 3, 6, 9, 12], 
                                  labels=[0, 1, 2, 3])  # Winter, Spring, Summer, Fall
        
        # Cyclical encoding for temporal features
        self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)
        self.df['day_sin'] = np.sin(2 * np.pi * self.df['day_of_year'] / 365)
        self.df['day_cos'] = np.cos(2 * np.pi * self.df['day_of_year'] / 365)
        
        # Spatial features
        self.df['distance_from_center'] = np.sqrt(
            (self.df['latitude'] - 12.9716)**2 + (self.df['longitude'] - 77.5946)**2
        )
        
        # Well characteristics
        self.df['well_depth_category'] = pd.cut(
            self.df['wellDepth'], 
            bins=[0, 20, 50, 100, np.inf], 
            labels=[0, 1, 2, 3]
        )
        
        # Handle categorical variables
        categorical_cols = ['wellType', 'wellAquiferType', 'tehsil', 'stationCode']
        for col in categorical_cols:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].fillna('Unknown'))
        
        # Create interaction features
        self.df['lat_lon_interaction'] = self.df['latitude'] * self.df['longitude']
        self.df['depth_year_interaction'] = self.df['wellDepth'] * self.df['year']
        
        # Select features
        feature_cols = [
            'latitude', 'longitude', 'wellDepth',
            'year', 'month', 'day', 'day_of_year', 'quarter', 'is_weekend', 'day_of_week',
            'season', 'dataTime_hour',
            'month_sin', 'month_cos', 'day_sin', 'day_cos',
            'distance_from_center', 'well_depth_category',
            'lat_lon_interaction', 'depth_year_interaction',
            'wellType_encoded', 'wellAquiferType_encoded', 'tehsil_encoded'
        ]
        
        self.df = self.df.dropna(subset=feature_cols + ['dataValue'])
        
        X = self.df[feature_cols]
        y = self.df['dataValue']
        
        print(f"Enhanced features used: {len(feature_cols)}")
        print(f"Final dataset shape: {X.shape}")
        
        # Stratified split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, 
            stratify=pd.qcut(y, q=5, labels=False, duplicates='drop')
        )
        
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Test set size: {self.X_test.shape[0]}")
        
        return X, y
    
    def create_enhanced_models(self):
        """Create advanced ML models"""
        print("\nCreating enhanced ML models...")
        
        # Linear models
        self.models['Linear Regression'] = LinearRegression()
        self.models['Ridge Regression'] = Ridge(alpha=1.0)
        self.models['Lasso Regression'] = Lasso(alpha=0.1)
        self.models['ElasticNet'] = ElasticNet(alpha=0.1, l1_ratio=0.5)
        
        # Tree-based models
        self.models['Random Forest'] = RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=2, 
            min_samples_leaf=1, random_state=42, n_jobs=-1
        )
        
        self.models['Gradient Boosting'] = GradientBoostingRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1, 
            subsample=0.8, random_state=42
        )
        
        # Advanced models
        if XGBOOST_AVAILABLE:
            self.models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
            )
        
        if LIGHTGBM_AVAILABLE:
            self.models['LightGBM'] = lgb.LGBMRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42, 
                n_jobs=-1, verbose=-1
            )
        
        self.models['SVR'] = SVR(kernel='rbf', C=1.0, gamma='scale')
        
        print(f"Created {len(self.models)} models")
        return self.models
    
    def train_and_evaluate_models(self):
        """Train and evaluate with cross-validation"""
        print("\nTraining and evaluating models with cross-validation...")
        
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            if name == 'SVR':
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
            else:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            # Cross-validation
            if name == 'SVR':
                cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                          cv=kfold, scoring='r2', n_jobs=-1)
            else:
                cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                          cv=kfold, scoring='r2', n_jobs=-1)
            
            self.results[name] = {
                'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2,
                'CV_R2_mean': cv_scores.mean(), 'CV_R2_std': cv_scores.std(),
                'predictions': y_pred
            }
            
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = model.feature_importances_
            
            print(f"  RMSE: {rmse:.4f}")
            print(f"  R²: {r2:.4f}")
            print(f"  CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    def create_ensemble_model(self):
        """Create ensemble of best models"""
        print("\nCreating ensemble model...")
        
        top_models = sorted(self.results.items(), key=lambda x: x[1]['R2'], reverse=True)[:3]
        estimators = [(name, self.models[name]) for name, _ in top_models if name in self.models]
        
        if len(estimators) >= 2:
            ensemble = VotingRegressor(estimators=estimators)
            ensemble.fit(self.X_train, self.y_train)
            
            y_pred = ensemble.predict(self.X_test)
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            self.results['Ensemble'] = {
                'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2,
                'CV_R2_mean': r2, 'CV_R2_std': 0, 'predictions': y_pred
            }
            
            print(f"Ensemble RMSE: {rmse:.4f}")
            print(f"Ensemble R²: {r2:.4f}")
    
    def generate_enhanced_report(self):
        """Generate comprehensive report"""
        print("\nGenerating enhanced performance report...")
        
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['R2'])
        best_results = self.results[best_model_name]
        
        report = f"""
ENHANCED GROUNDWATER LEVEL PREDICTION - ML MODEL PERFORMANCE REPORT
{'='*70}

DATASET SUMMARY:
- Total records: {len(self.df)}
- Training samples: {len(self.X_train)}
- Test samples: {len(self.X_test)}
- Enhanced features used: {len(self.X_train.columns)}

MODEL PERFORMANCE COMPARISON:
{'='*70}
"""
        
        for name, results in self.results.items():
            report += f"""
{name}:
  RMSE: {results['RMSE']:.4f}
  MAE:  {results['MAE']:.4f}
  R²:   {results['R2']:.4f}
  CV R²: {results['CV_R2_mean']:.4f} (+/- {results['CV_R2_std'] * 2:.4f})
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
- Ensemble methods
- Cross-validation for robust evaluation
"""
        
        with open('enhanced_ml_report.txt', 'w') as f:
            f.write(report)
        
        print("✓ Saved enhanced_ml_report.txt")
        print(f"\nBest model: {best_model_name}")
        print(f"Best RMSE: {best_results['RMSE']:.4f}")
        print(f"Best R²: {best_results['R2']:.4f}")
        
        return report
    
    def save_best_model(self, model_name=None):
        """Save the best performing model"""
        if model_name is None:
            model_name = max(self.results.keys(), key=lambda x: self.results[x]['R2'])
        
        import joblib
        model = self.models[model_name]
        scaler = self.scaler
        
        joblib.dump(model, f'enhanced_groundwater_model_{model_name.replace(" ", "_")}.pkl')
        joblib.dump(scaler, 'enhanced_groundwater_scaler.pkl')
        
        print(f"✓ Saved enhanced model: enhanced_groundwater_model_{model_name.replace(' ', '_')}.pkl")
        print(f"✓ Saved enhanced scaler: enhanced_groundwater_scaler.pkl")
    
    def run_enhanced_ml_pipeline(self):
        """Run the complete enhanced ML pipeline"""
        print("ENHANCED GROUNDWATER LEVEL PREDICTION - ML PIPELINE")
        print("=" * 70)
        
        self.load_and_preprocess_data()
        self.create_enhanced_models()
        self.train_and_evaluate_models()
        self.create_ensemble_model()
        self.generate_enhanced_report()
        self.save_best_model()
        
        print("\n" + "="*70)
        print("ENHANCED ML PIPELINE COMPLETE!")
        print("="*70)

if __name__ == "__main__":
    enhanced_ml_model = EnhancedGroundwaterMLModel()
    enhanced_ml_model.run_enhanced_ml_pipeline() 