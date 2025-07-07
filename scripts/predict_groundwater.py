#!/usr/bin/env python3
"""
Groundwater Level Prediction Script
This script loads the trained model and makes predictions on new data.
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_model_and_scaler():
    """Load the trained model and scaler"""
    try:
        # Load the best model (Gradient Boosting)
        model = joblib.load('best_groundwater_model_Gradient_Boosting.pkl')
        scaler = joblib.load('groundwater_scaler.pkl')
        print("✓ Model and scaler loaded successfully")
        return model, scaler
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def prepare_features(data):
    """Prepare features for prediction"""
    # Create datetime column if not exists
    if 'datetime' not in data.columns:
        if all(col in data.columns for col in ['dataTime_year', 'dataTime_monthValue', 'dataTime_dayOfMonth']):
            data['datetime'] = pd.to_datetime(
                data['dataTime_year'].astype(str) + '-' + 
                data['dataTime_monthValue'].astype(str) + '-' + 
                data['dataTime_dayOfMonth'].astype(str)
            )
        else:
            data['datetime'] = pd.to_datetime('now')
    
    # Extract time-based features
    data['year'] = data['datetime'].dt.year
    data['month'] = data['datetime'].dt.month
    data['day'] = data['datetime'].dt.day
    data['day_of_year'] = data['datetime'].dt.dayofyear
    data['quarter'] = data['datetime'].dt.quarter
    data['is_weekend'] = data['datetime'].dt.weekday >= 5
    
    # Create seasonal features
    data['season'] = pd.cut(data['month'], bins=[0, 3, 6, 9, 12], labels=[0, 1, 2, 3])
    
    # Handle categorical variables
    categorical_cols = ['wellType', 'wellAquiferType', 'tehsil']
    for col in categorical_cols:
        if col in data.columns:
            # Use simple encoding for new data
            data[f'{col}_encoded'] = pd.Categorical(data[col]).codes
        else:
            # Default values if column doesn't exist
            data[f'{col}_encoded'] = 0
    
    # Select features for modeling
    feature_cols = [
        'latitude', 'longitude', 'wellDepth',
        'year', 'month', 'day', 'day_of_year', 'quarter', 'is_weekend',
        'season', 'dataTime_hour',
        'wellType_encoded', 'wellAquiferType_encoded', 'tehsil_encoded'
    ]
    
    # Fill missing values with defaults
    for col in feature_cols:
        if col not in data.columns:
            if col in ['latitude', 'longitude']:
                data[col] = 12.9716  # Default Bengaluru coordinates
            elif col in ['wellDepth']:
                data[col] = 100  # Default well depth
            elif col in ['dataTime_hour']:
                data[col] = 12  # Default hour
            else:
                data[col] = 0
    
    return data[feature_cols]

def predict_groundwater_levels(input_data, model, scaler):
    """Make predictions on input data"""
    try:
        # Prepare features
        features = prepare_features(input_data.copy())
        
        # Make predictions
        predictions = model.predict(features)
        
        # Add predictions to the input data
        result = input_data.copy()
        result['predicted_groundwater_level'] = predictions
        
        return result, predictions
        
    except Exception as e:
        print(f"Error making predictions: {e}")
        return None, None

def create_sample_data():
    """Create sample data for demonstration"""
    sample_data = pd.DataFrame({
        'latitude': [12.9716, 12.9789, 12.9655],
        'longitude': [77.5946, 77.5917, 77.5875],
        'wellDepth': [100, 120, 80],
        'wellType': ['Dug Well', 'Tube Well', 'Dug Well'],
        'wellAquiferType': ['Unconfined', 'Confined', 'Unconfined'],
        'tehsil': ['Bengaluru North', 'Bengaluru South', 'Bengaluru East'],
        'dataTime_year': [2024, 2024, 2024],
        'dataTime_monthValue': [7, 7, 7],
        'dataTime_dayOfMonth': [3, 3, 3],
        'dataTime_hour': [12, 14, 10]
    })
    return sample_data

def main():
    """Main function to demonstrate prediction"""
    print("GROUNDWATER LEVEL PREDICTION")
    print("=" * 40)
    
    # Load model and scaler
    model, scaler = load_model_and_scaler()
    if model is None:
        return
    
    # Create sample data
    print("\nCreating sample data for prediction...")
    sample_data = create_sample_data()
    print("Sample data:")
    print(sample_data[['latitude', 'longitude', 'wellDepth', 'wellType', 'tehsil']])
    
    # Make predictions
    print("\nMaking predictions...")
    result, predictions = predict_groundwater_levels(sample_data, model, scaler)
    
    if result is not None:
        print("\nPrediction Results:")
        print("=" * 40)
        for i, (_, row) in enumerate(result.iterrows()):
            print(f"Location {i+1}:")
            print(f"  Coordinates: ({row['latitude']:.4f}, {row['longitude']:.4f})")
            print(f"  Well Type: {row['wellType']}")
            print(f"  Tehsil: {row['tehsil']}")
            print(f"  Predicted Groundwater Level: {row['predicted_groundwater_level']:.2f} meters")
            print()
        
        # Save results
        result.to_csv('prediction_results.csv', index=False)
        print("✓ Saved prediction results to 'prediction_results.csv'")
    
    # Interactive prediction
    print("\n" + "=" * 40)
    print("INTERACTIVE PREDICTION")
    print("=" * 40)
    
    try:
        print("\nEnter location details for prediction:")
        lat = float(input("Latitude (e.g., 12.9716): "))
        lon = float(input("Longitude (e.g., 77.5946): "))
        well_depth = float(input("Well Depth (meters): "))
        well_type = input("Well Type (Dug Well/Tube Well): ")
        tehsil = input("Tehsil: ")
        
        # Create single prediction data
        single_data = pd.DataFrame({
            'latitude': [lat],
            'longitude': [lon],
            'wellDepth': [well_depth],
            'wellType': [well_type],
            'wellAquiferType': ['Unconfined'],
            'tehsil': [tehsil],
            'dataTime_year': [datetime.now().year],
            'dataTime_monthValue': [datetime.now().month],
            'dataTime_dayOfMonth': [datetime.now().day],
            'dataTime_hour': [datetime.now().hour]
        })
        
        # Make prediction
        result_single, pred_single = predict_groundwater_levels(single_data, model, scaler)
        
        if result_single is not None:
            prediction = result_single.iloc[0]['predicted_groundwater_level']
            print(f"\nPredicted Groundwater Level: {prediction:.2f} meters")
            
            # Interpretation
            if prediction < 0:
                print("Interpretation: Water level is below ground surface (artesian condition)")
            elif prediction < 5:
                print("Interpretation: Very shallow water level - high risk of depletion")
            elif prediction < 15:
                print("Interpretation: Moderate water level - monitor closely")
            else:
                print("Interpretation: Deep water level - relatively stable")
    
    except ValueError:
        print("Invalid input. Please enter numeric values for coordinates and well depth.")
    except KeyboardInterrupt:
        print("\nPrediction cancelled.")

if __name__ == "__main__":
    main() 