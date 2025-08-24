#!/bin/bash
# Memristor-NN Production Deployment Script
# Generated on 2025-08-24 13:40:12

set -e

echo "🚀 Deploying Memristor-NN Simulator..."

# Build container
echo "🐳 Building container..."
docker build -t memristor-nn-sim:latest .

# Run tests
echo "🧪 Running tests..."
docker run --rm memristor-nn-sim:latest python3 test_core_functionality.py

echo "✅ Deployment complete!"
