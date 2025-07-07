# Groundwater Level Prediction API

A REST API for predicting groundwater levels in Bengaluru using machine learning.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Server
```bash
python app.py
```

The server will start on `http://localhost:5000`

### 3. Test the API
```bash
python test_api.py
```

## API Endpoints

### 1. GET `/` - API Information
Returns information about the API and available endpoints.

**Response:**
```json
{
  "message": "Groundwater Level Prediction API",
  "version": "1.0.0",
  "endpoints": {
    "/": "API information",
    "/health": "Health check",
    "/predict": "Single prediction",
    "/predict_batch": "Batch predictions",
    "/model_info": "Model information"
  }
}
```

### 2. GET `/health` - Health Check
Check if the API is running and the model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-07-03T23:50:00.000000"
}
```

### 3. GET `/model_info` - Model Information
Get information about the trained model.

**Response:**
```json
{
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
}
```

### 4. POST `/predict` - Single Prediction
Make a single groundwater level prediction.

**Request Body:**
```json
{
  "latitude": 12.9716,
  "longitude": 77.5946,
  "wellDepth": 100,
  "wellType": "Dug Well",
  "wellAquiferType": "Unconfined",
  "tehsil": "Bengaluru North"
}
```

**Required Fields:**
- `latitude` (float): Latitude coordinate
- `longitude` (float): Longitude coordinate
- `wellDepth` (float): Depth of the well in meters

**Optional Fields:**
- `wellType` (string): Type of well (default: "Dug Well")
- `wellAquiferType` (string): Aquifer type (default: "Unconfined")
- `tehsil` (string): Administrative region (default: "Bengaluru North")

**Response:**
```json
{
  "prediction": 11.85,
  "prediction_meters": 11.85,
  "interpretation": "Moderate water level - monitor closely",
  "input_data": {
    "latitude": 12.9716,
    "longitude": 77.5946,
    "wellDepth": 100,
    "wellType": "Dug Well",
    "wellAquiferType": "Unconfined",
    "tehsil": "Bengaluru North"
  },
  "timestamp": "2024-07-03T23:50:00.000000"
}
```

### 5. POST `/predict_batch` - Batch Predictions
Make multiple groundwater level predictions at once (max 100).

**Request Body:**
```json
[
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
  }
]
```

**Response:**
```json
{
  "predictions": [
    {
      "id": 0,
      "prediction": 11.85,
      "prediction_meters": 11.85,
      "interpretation": "Moderate water level - monitor closely",
      "input_data": {
        "latitude": 12.9716,
        "longitude": 77.5946,
        "wellDepth": 100,
        "wellType": "Dug Well",
        "tehsil": "Bengaluru North"
      }
    },
    {
      "id": 1,
      "prediction": 8.16,
      "prediction_meters": 8.16,
      "interpretation": "Moderate water level - monitor closely",
      "input_data": {
        "latitude": 12.9789,
        "longitude": 77.5917,
        "wellDepth": 120,
        "wellType": "Tube Well",
        "tehsil": "Bengaluru South"
      }
    }
  ],
  "total_predictions": 2,
  "timestamp": "2024-07-03T23:50:00.000000"
}
```

## Usage Examples

### Python Example
```python
import requests
import json

# Single prediction
data = {
    "latitude": 12.9716,
    "longitude": 77.5946,
    "wellDepth": 100,
    "wellType": "Dug Well",
    "tehsil": "Bengaluru North"
}

response = requests.post("http://localhost:5000/predict", json=data)
result = response.json()
print(f"Predicted groundwater level: {result['prediction']} meters")
print(f"Interpretation: {result['interpretation']}")
```

### JavaScript Example
```javascript
// Single prediction
const data = {
    latitude: 12.9716,
    longitude: 77.5946,
    wellDepth: 100,
    wellType: "Dug Well",
    tehsil: "Bengaluru North"
};

fetch('http://localhost:5000/predict', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify(data)
})
.then(response => response.json())
.then(result => {
    console.log(`Predicted groundwater level: ${result.prediction} meters`);
    console.log(`Interpretation: ${result.interpretation}`);
});
```

### cURL Example
```bash
# Single prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 12.9716,
    "longitude": 77.5946,
    "wellDepth": 100,
    "wellType": "Dug Well",
    "tehsil": "Bengaluru North"
  }'

# Health check
curl http://localhost:5000/health

# Model info
curl http://localhost:5000/model_info
```

## Error Handling

### Missing Required Fields
```json
{
  "error": "Missing required fields: ['longitude', 'wellDepth']",
  "required_fields": ["latitude", "longitude", "wellDepth"]
}
```

### Invalid Data Format
```json
{
  "error": "Data must be a list of prediction requests"
}
```

### Server Error
```json
{
  "error": "Internal server error",
  "timestamp": "2024-07-03T23:50:00.000000"
}
```

## Model Performance

- **R² Score**: 92.85% (excellent predictive power)
- **RMSE**: 3.62 meters (low prediction error)
- **MAE**: 2.10 meters (high accuracy)

## Interpretation Guide

| Prediction Range | Interpretation |
|------------------|----------------|
| < 0 meters | Water level is below ground surface (artesian condition) |
| 0-5 meters | Very shallow water level - high risk of depletion |
| 5-15 meters | Moderate water level - monitor closely |
| > 15 meters | Deep water level - relatively stable |

## Deployment

### Local Development
```bash
python app.py
```

### Production Deployment
For production deployment, consider using:
- Gunicorn: `gunicorn -w 4 -b 0.0.0.0:5000 app:app`
- Docker containerization
- Cloud platforms (AWS, Google Cloud, Azure)

### Environment Variables
- `PORT`: Server port (default: 5000)
- `HOST`: Server host (default: 0.0.0.0)

## Security Considerations

- Enable HTTPS in production
- Implement authentication if needed
- Add rate limiting for API endpoints
- Validate and sanitize input data
- Monitor API usage and performance

## Support

For issues or questions:
1. Check the health endpoint: `GET /health`
2. Review the model info: `GET /model_info`
3. Test with the provided test script: `python test_api.py` 