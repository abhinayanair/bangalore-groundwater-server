#!/usr/bin/env python3
import sys
import traceback

try:
    print("Starting ML pipeline...")
    from groundwater_ml_model import GroundwaterMLModel
    
    print("Creating ML model instance...")
    ml_model = GroundwaterMLModel()
    
    print("Running complete pipeline...")
    ml_model.run_complete_ml_pipeline()
    
    print("Pipeline completed successfully!")
    
except Exception as e:
    print(f"Error occurred: {e}")
    print("Traceback:")
    traceback.print_exc() 