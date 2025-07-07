#!/usr/bin/env python3
"""
Groundwater Level Prediction API Server
Flask REST API for predicting groundwater levels
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Set base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for model and scaler
model = None
scaler = None

def load_model():
    """Load the trained model and scaler"""
    global model, scaler
    try:
        model = joblib.load(os.path.join(MODEL_DIR, 'best_groundwater_model_Gradient_Boosting.pkl'))
        scaler = joblib.load(os.path.join(MODEL_DIR, 'groundwater_scaler.pkl'))
        print("✓ Model and scaler loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

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

def interpret_prediction(prediction):
    """Interpret the prediction result"""
    if prediction < 0:
        return "Water level is below ground surface (artesian condition)"
    elif prediction < 5:
        return "Very shallow water level - high risk of depletion"
    elif prediction < 15:
        return "Moderate water level - monitor closely"
    else:
        return "Deep water level - relatively stable"

@app.route('/')
def home():
    """Home endpoint with API information"""
    return jsonify({
        "message": "Groundwater Level Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/": "API information",
            "/health": "Health check",
            "/predict": "Single prediction",
            "/predict_batch": "Batch predictions",
            "/model_info": "Model information"
        },
        "usage": {
            "single_prediction": "POST /predict with JSON body",
            "batch_prediction": "POST /predict_batch with JSON array"
        }
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    if model is not None and scaler is not None:
        return jsonify({
            "status": "healthy",
            "model_loaded": True,
            "timestamp": datetime.now().isoformat()
        })
    else:
        return jsonify({
            "status": "unhealthy",
            "model_loaded": False,
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/model_info')
def model_info():
    """Get model information"""
    return jsonify({
        "model_type": "Gradient Boosting Regressor",
        "features": [
            "latitude", "longitude", "wellDepth",
            "year", "month", "day", "day_of_year", "quarter", "is_weekend",
            "season", "dataTime_hour",
            "wellType_encoded", "wellAquiferType_encoded", "tehsil_encoded"
        ],
        "performance": {
            "r2_score": 0.9285,
            "rmse": 3.6213,
            "mae": 2.1004
        },
        "feature_importance": {
            "tehsil_encoded": 0.6169,
            "wellDepth": 0.1730,
            "longitude": 0.0912,
            "latitude": 0.0527,
            "year": 0.0313
        }
    })

@app.route('/predict', methods=['POST'])
def predict_single():
    """Single prediction endpoint"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Validate required fields
        required_fields = ['latitude', 'longitude', 'wellDepth']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                "error": f"Missing required fields: {missing_fields}",
                "required_fields": required_fields
            }), 400
        
        # Create DataFrame for prediction
        prediction_data = pd.DataFrame([data])
        
        # Add default values for missing optional fields
        defaults = {
            'wellType': 'Dug Well',
            'wellAquiferType': 'Unconfined',
            'tehsil': 'Bengaluru North',
            'dataTime_year': datetime.now().year,
            'dataTime_monthValue': datetime.now().month,
            'dataTime_dayOfMonth': datetime.now().day,
            'dataTime_hour': datetime.now().hour
        }
        
        for key, value in defaults.items():
            if key not in prediction_data.columns:
                prediction_data[key] = value
        
        # Prepare features
        features = prepare_features(prediction_data)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Prepare response
        response = {
            "prediction": float(prediction),
            "prediction_meters": float(prediction),
            "interpretation": interpret_prediction(prediction),
            "input_data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            "error": f"Prediction failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data or not isinstance(data, list):
            return jsonify({"error": "Data must be a list of prediction requests"}), 400
        
        if len(data) > 100:
            return jsonify({"error": "Maximum 100 predictions per batch"}), 400
        
        # Validate each item
        for i, item in enumerate(data):
            required_fields = ['latitude', 'longitude', 'wellDepth']
            missing_fields = [field for field in required_fields if field not in item]
            
            if missing_fields:
                return jsonify({
                    "error": f"Item {i}: Missing required fields: {missing_fields}",
                    "required_fields": required_fields
                }), 400
        
        # Create DataFrame for predictions
        prediction_data = pd.DataFrame(data)
        
        # Add default values for missing optional fields
        defaults = {
            'wellType': 'Dug Well',
            'wellAquiferType': 'Unconfined',
            'tehsil': 'Bengaluru North',
            'dataTime_year': datetime.now().year,
            'dataTime_monthValue': datetime.now().month,
            'dataTime_dayOfMonth': datetime.now().day,
            'dataTime_hour': datetime.now().hour
        }
        
        for key, value in defaults.items():
            if key not in prediction_data.columns:
                prediction_data[key] = value
        
        # Prepare features
        features = prepare_features(prediction_data)
        
        # Make predictions
        predictions = model.predict(features)
        
        # Prepare response
        results = []
        for i, (_, row) in enumerate(prediction_data.iterrows()):
            result = {
                "id": i,
                "prediction": float(predictions[i]),
                "prediction_meters": float(predictions[i]),
                "interpretation": interpret_prediction(predictions[i]),
                "input_data": row.to_dict()
            }
            results.append(result)
        
        response = {
            "predictions": results,
            "total_predictions": len(results),
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            "error": f"Batch prediction failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": ["/", "/health", "/predict", "/predict_batch", "/model_info"]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        "error": "Internal server error",
        "timestamp": datetime.now().isoformat()
    }), 500

if __name__ == '__main__':
    # Load model on startup
    if load_model():
        print("Starting Groundwater Prediction API Server...")
        print("Available endpoints:")
        print("  GET  / - API information")
        print("  GET  /health - Health check")
        print("  GET  /model_info - Model information")
        print("  POST /predict - Single prediction")
        print("  POST /predict_batch - Batch predictions")
        print("\nServer starting on http://localhost:5000")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("Failed to load model. Please check if model files exist.")
        exit(1) 