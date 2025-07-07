#!/usr/bin/env python3
"""
Comprehensive ML Model for Bengaluru Groundwater Level Prediction
This script builds and evaluates multiple ML models for predicting groundwater levels.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class GroundwaterMLModel:
    def __init__(self, data_path="data/bengaluru_groundwater_merged.csv"):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_and_preprocess_data(self):
        """Load and preprocess the merged dataset"""
        print("Loading and preprocessing data...")
        
        # Load data
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
        
        # Extract time-based features
        self.df['year'] = self.df['datetime'].dt.year
        self.df['month'] = self.df['datetime'].dt.month
        self.df['day'] = self.df['datetime'].dt.day
        self.df['day_of_year'] = self.df['datetime'].dt.dayofyear
        self.df['quarter'] = self.df['datetime'].dt.quarter
        self.df['is_weekend'] = self.df['datetime'].dt.weekday >= 5
        
        # Create seasonal features
        self.df['season'] = pd.cut(self.df['month'], 
                                  bins=[0, 3, 6, 9, 12], 
                                  labels=[0, 1, 2, 3])  # Winter, Spring, Summer, Fall
        
        # Handle categorical variables
        categorical_cols = ['wellType', 'wellAquiferType', 'tehsil', 'stationCode']
        label_encoders = {}
        
        for col in categorical_cols:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].fillna('Unknown'))
                label_encoders[col] = le
        
        # Select features for modeling
        feature_cols = [
            'latitude', 'longitude', 'wellDepth',
            'year', 'month', 'day', 'day_of_year', 'quarter', 'is_weekend',
            'season', 'dataTime_hour',
            'wellType_encoded', 'wellAquiferType_encoded', 'tehsil_encoded'
        ]
        
        # Remove rows with missing values in features
        self.df = self.df.dropna(subset=feature_cols + ['dataValue'])
        
        # Prepare X and y
        X = self.df[feature_cols]
        y = self.df['dataValue']
        
        print(f"Features used: {feature_cols}")
        print(f"Final dataset shape: {X.shape}")
        print(f"Target variable range: {y.min():.2f} to {y.max():.2f}")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Test set size: {self.X_test.shape[0]}")
        
        return X, y
    
    def create_models(self):
        """Create and configure multiple ML models"""
        print("\nCreating ML models...")
        
        # 1. Linear Regression
        self.models['Linear Regression'] = LinearRegression()
        
        # 2. Ridge Regression
        self.models['Ridge Regression'] = Ridge(alpha=1.0)
        
        # 3. Lasso Regression
        self.models['Lasso Regression'] = Lasso(alpha=0.1)
        
        # 4. Random Forest
        self.models['Random Forest'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # 5. Gradient Boosting
        self.models['Gradient Boosting'] = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        # 6. Support Vector Regression
        self.models['SVR'] = SVR(kernel='rbf', C=1.0, gamma='scale')
        
        print(f"Created {len(self.models)} models")
        return self.models
    
    def train_and_evaluate_models(self):
        """Train and evaluate all models"""
        print("\nTraining and evaluating models...")
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train model
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
            
            # Store results
            self.results[name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'predictions': y_pred
            }
            
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  R²: {r2:.4f}")
    
    def hyperparameter_tuning(self, model_name='Random Forest'):
        """Perform hyperparameter tuning for the best model"""
        print(f"\nPerforming hyperparameter tuning for {model_name}...")
        
        if model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestRegressor(random_state=42)
        
        elif model_name == 'Gradient Boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
            model = GradientBoostingRegressor(random_state=42)
        
        else:
            print("Hyperparameter tuning not implemented for this model")
            return
        
        # Perform grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
        )
        grid_search.fit(self.X_train, self.y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        print(f"Best parameters: {best_params}")
        
        # Evaluate best model
        y_pred_best = best_model.predict(self.X_test)
        mse_best = mean_squared_error(self.y_test, y_pred_best)
        rmse_best = np.sqrt(mse_best)
        r2_best = r2_score(self.y_test, y_pred_best)
        
        print(f"Best model RMSE: {rmse_best:.4f}")
        print(f"Best model R²: {r2_best:.4f}")
        
        # Update the model
        self.models[f'{model_name} (Tuned)'] = best_model
        
        return best_model, best_params
    
    def create_visualizations(self):
        """Create visualizations for model performance"""
        print("\nCreating visualizations...")
        
        try:
            # 1. Model Performance Comparison
            plt.figure(figsize=(15, 10))
            
            # RMSE comparison
            plt.subplot(2, 3, 1)
            model_names = list(self.results.keys())
            rmse_values = [self.results[name]['RMSE'] for name in model_names]
            plt.bar(model_names, rmse_values, color='skyblue')
            plt.title('Model RMSE Comparison')
            plt.ylabel('RMSE')
            plt.xticks(rotation=45)
            
            # R² comparison
            plt.subplot(2, 3, 2)
            r2_values = [self.results[name]['R2'] for name in model_names]
            plt.bar(model_names, r2_values, color='lightgreen')
            plt.title('Model R² Comparison')
            plt.ylabel('R² Score')
            plt.xticks(rotation=45)
            
            # MAE comparison
            plt.subplot(2, 3, 3)
            mae_values = [self.results[name]['MAE'] for name in model_names]
            plt.bar(model_names, mae_values, color='lightcoral')
            plt.title('Model MAE Comparison')
            plt.ylabel('MAE')
            plt.xticks(rotation=45)
            
            # 2. Actual vs Predicted for best model
            best_model_name = min(self.results.keys(), key=lambda x: self.results[x]['RMSE'])
            best_predictions = self.results[best_model_name]['predictions']
            
            plt.subplot(2, 3, 4)
            plt.scatter(self.y_test, best_predictions, alpha=0.6)
            plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title(f'Actual vs Predicted ({best_model_name})')
            
            # 3. Residuals plot
            plt.subplot(2, 3, 5)
            residuals = self.y_test - best_predictions
            plt.scatter(best_predictions, residuals, alpha=0.6)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title(f'Residuals Plot ({best_model_name})')
            
            # 4. Feature importance (for tree-based models)
            if 'Random Forest' in self.models:
                rf_model = self.models['Random Forest']
                feature_importance = rf_model.feature_importances_
                feature_names = self.X_train.columns
                
                plt.subplot(2, 3, 6)
                indices = np.argsort(feature_importance)[::-1]
                plt.bar(range(len(feature_importance)), feature_importance[indices])
                plt.xticks(range(len(feature_importance)), [feature_names[i] for i in indices], rotation=45)
                plt.title('Feature Importance (Random Forest)')
                plt.ylabel('Importance')
            
            plt.tight_layout()
            plt.savefig('ml_model_performance.png', dpi=300, bbox_inches='tight')
            print("✓ Saved ml_model_performance.png")
            plt.close()
            
        except Exception as e:
            print(f"Error creating visualizations: {e}")
    
    def generate_report(self):
        """Generate a comprehensive model performance report"""
        print("\nGenerating performance report...")
        
        # Find best model
        best_model_name = min(self.results.keys(), key=lambda x: self.results[x]['RMSE'])
        best_results = self.results[best_model_name]
        
        # Create report
        report = f"""
GROUNDWATER LEVEL PREDICTION - ML MODEL PERFORMANCE REPORT
{'='*60}

DATASET SUMMARY:
- Total records: {len(self.df)}
- Training samples: {len(self.X_train)}
- Test samples: {len(self.X_test)}
- Features used: {len(self.X_train.columns)}

MODEL PERFORMANCE COMPARISON:
{'='*60}
"""
        
        # Add results for each model
        for name, results in self.results.items():
            report += f"""
{name}:
  RMSE: {results['RMSE']:.4f}
  MAE:  {results['MAE']:.4f}
  R²:   {results['R2']:.4f}
"""
        
        report += f"""
{'='*60}
BEST MODEL: {best_model_name}
- RMSE: {best_results['RMSE']:.4f}
- MAE:  {best_results['MAE']:.4f}
- R²:   {best_results['R2']:.4f}

FEATURE IMPORTANCE (Random Forest):
"""
        
        if 'Random Forest' in self.models:
            rf_model = self.models['Random Forest']
            feature_importance = rf_model.feature_importances_
            feature_names = self.X_train.columns
            
            # Sort features by importance
            feature_imp_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False)
            
            for _, row in feature_imp_df.head(10).iterrows():
                report += f"  {row['Feature']}: {row['Importance']:.4f}\n"
        
        # Save report
        with open('ml_model_report.txt', 'w') as f:
            f.write(report)
        
        print("✓ Saved ml_model_report.txt")
        print(f"\nBest model: {best_model_name}")
        print(f"Best RMSE: {best_results['RMSE']:.4f}")
        print(f"Best R²: {best_results['R2']:.4f}")
        
        return report
    
    def save_best_model(self, model_name=None):
        """Save the best performing model"""
        if model_name is None:
            model_name = min(self.results.keys(), key=lambda x: self.results[x]['RMSE'])
        
        import joblib
        
        model = self.models[model_name]
        scaler = self.scaler
        
        # Save model and scaler
        joblib.dump(model, f'best_groundwater_model_{model_name.replace(" ", "_")}.pkl')
        joblib.dump(scaler, 'groundwater_scaler.pkl')
        
        print(f"✓ Saved best model: best_groundwater_model_{model_name.replace(' ', '_')}.pkl")
        print(f"✓ Saved scaler: groundwater_scaler.pkl")
    
    def run_complete_ml_pipeline(self):
        """Run the complete ML pipeline"""
        print("GROUNDWATER LEVEL PREDICTION - ML PIPELINE")
        print("=" * 60)
        
        # Load and preprocess data
        self.load_and_preprocess_data()
        
        # Create models
        self.create_models()
        
        # Train and evaluate
        self.train_and_evaluate_models()
        
        # Hyperparameter tuning for best model
        self.hyperparameter_tuning('Random Forest')
        
        # Create visualizations
        self.create_visualizations()
        
        # Generate report
        self.generate_report()
        
        # Save best model
        self.save_best_model()
        
        print("\n" + "="*60)
        print("ML PIPELINE COMPLETE!")
        print("="*60)
        print("Generated files:")
        print("- ml_model_performance.png")
        print("- ml_model_report.txt")
        print("- best_groundwater_model_*.pkl")
        print("- groundwater_scaler.pkl")

if __name__ == "__main__":
    # Initialize and run ML pipeline
    ml_model = GroundwaterMLModel()
    ml_model.run_complete_ml_pipeline() 