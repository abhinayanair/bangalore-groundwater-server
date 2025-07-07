#!/bin/bash

# Groundwater Microservice Build Script for Vercel
echo "Starting build process..."

# Ensure we're using Python 3.11
python3.11 --version

# Install dependencies with specific Python version
python3.11 -m pip install --upgrade pip
python3.11 -m pip install -r requirements.txt

echo "Build completed successfully!" 