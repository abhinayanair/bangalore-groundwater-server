#!/bin/bash

# Groundwater Microservice Build Script for Vercel
echo "Starting build process..."

# Set timeout for pip install
export PIP_DEFAULT_TIMEOUT=300

# Use standard Python commands available in Vercel
python --version
python -m pip install --upgrade pip --timeout 300
python -m pip install -r requirements.txt --timeout 300

echo "Build completed successfully!" 