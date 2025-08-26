#!/bin/bash
set -e

# Health Check Script
ENVIRONMENT=${1:-development}
TIMEOUT=${2:-300}

case $ENVIRONMENT in
  development)
    NAMESPACE="memristor-nn-dev"
    ;;
  staging)
    NAMESPACE="memristor-nn-staging"
    ;;
  production)
    NAMESPACE="memristor-nn-prod"
    ;;
  *)
    echo "❌ Invalid environment: $ENVIRONMENT"
    exit 1
    ;;
esac

echo "🏥 Running health checks for $ENVIRONMENT environment"

# Check if pods are running
echo "📋 Checking pod status..."
kubectl get pods -n $NAMESPACE -l app=memristor-nn

# Wait for pods to be ready
echo "⏳ Waiting for pods to be ready..."
kubectl wait --for=condition=ready pod -l app=memristor-nn -n $NAMESPACE --timeout=${TIMEOUT}s

# Check service endpoints
echo "📋 Checking service endpoints..."
kubectl get endpoints -n $NAMESPACE memristor-nn-service

# Test health endpoint
echo "🔍 Testing health endpoint..."
SERVICE_IP=$(kubectl get service memristor-nn-service -n $NAMESPACE -o jsonpath='{.spec.clusterIP}')
kubectl run health-check-pod --rm -i --restart=Never --image=curlimages/curl -- \
  curl -f "http://$SERVICE_IP/health" --max-time 10

echo "✅ All health checks passed!"
