#!/usr/bin/env python3
"""
Test script for Groundwater Prediction API
Demonstrates how to use the API endpoints
"""

import requests
import json
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:5000"

def test_health():
    """Test health check endpoint"""
    print("Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_model_info():
    """Test model info endpoint"""
    print("Testing model info...")
    response = requests.get(f"{BASE_URL}/model_info")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_single_prediction():
    """Test single prediction endpoint"""
    print("Testing single prediction...")
    
    # Sample data for prediction
    data = {
        "latitude": 12.9716,
        "longitude": 77.5946,
        "wellDepth": 100,
        "wellType": "Dug Well",
        "wellAquiferType": "Unconfined",
        "tehsil": "Bengaluru North"
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print(f"Status: {response.status_code}")
    print(f"Request data: {json.dumps(data, indent=2)}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("Testing batch prediction...")
    
    # Sample batch data
    batch_data = [
        {
            "latitude": 12.9716,
            "longitude": 77.5946,
            "wellDepth": 100,
            "wellType": "Dug Well",
            "tehsil": "Bengaluru North"
        },
        {
            "latitude": 12.9789,
            "longitude": 77.5917,
            "wellDepth": 120,
            "wellType": "Tube Well",
            "tehsil": "Bengaluru South"
        },
        {
            "latitude": 12.9655,
            "longitude": 77.5875,
            "wellDepth": 80,
            "wellType": "Dug Well",
            "tehsil": "Bengaluru East"
        }
    ]
    
    response = requests.post(f"{BASE_URL}/predict_batch", json=batch_data)
    print(f"Status: {response.status_code}")
    print(f"Request data: {json.dumps(batch_data, indent=2)}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_error_handling():
    """Test error handling"""
    print("Testing error handling...")
    
    # Test missing required fields
    invalid_data = {
        "latitude": 12.9716
        # Missing longitude and wellDepth
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=invalid_data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def main():
    """Run all tests"""
    print("GROUNDWATER PREDICTION API TESTS")
    print("=" * 50)
    
    try:
        # Test all endpoints
        test_health()
        test_model_info()
        test_single_prediction()
        test_batch_prediction()
        test_error_handling()
        
        print("All tests completed!")
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API server.")
        print("Make sure the server is running on http://localhost:5000")
        print("Run: python app.py")
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    main() 