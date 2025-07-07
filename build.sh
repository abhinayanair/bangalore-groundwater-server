#!/bin/bash

# Groundwater Microservice Build Script for Vercel
echo "Starting build process..."

# Use standard Python commands available in Vercel
python --version
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Build completed successfully!" 