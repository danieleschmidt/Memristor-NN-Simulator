#!/bin/bash
set -e

# Memristor NN Deployment Script
ENVIRONMENT=${1:-development}
IMAGE_TAG=${2:-latest}

echo "🚀 Deploying Memristor NN to $ENVIRONMENT environment"

# Validate environment
case $ENVIRONMENT in
  development|staging|production)
    echo "✅ Valid environment: $ENVIRONMENT"
    ;;
  *)
    echo "❌ Invalid environment: $ENVIRONMENT"
    echo "Valid options: development, staging, production"
    exit 1
    ;;
esac

# Check prerequisites
command -v kubectl >/dev/null 2>&1 || { echo "❌ kubectl is required but not installed"; exit 1; }
command -v helm >/dev/null 2>&1 || { echo "❌ helm is required but not installed"; exit 1; }

# Set cluster context
case $ENVIRONMENT in
  development)
    CLUSTER_NAME="memristor-dev"
    NAMESPACE="memristor-nn-dev"
    ;;
  staging)
    CLUSTER_NAME="memristor-staging"
    NAMESPACE="memristor-nn-staging"
    ;;
  production)
    CLUSTER_NAME="memristor-prod"
    NAMESPACE="memristor-nn-prod"
    ;;
esac

echo "📋 Updating kubeconfig for cluster: $CLUSTER_NAME"
aws eks update-kubeconfig --name $CLUSTER_NAME --region us-west-2

# Create namespace if it doesn't exist
echo "📋 Ensuring namespace exists: $NAMESPACE"
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Deploy using Helm
echo "🚀 Deploying with Helm"
helm upgrade --install memristor-nn ./helm/memristor-nn \
  --namespace $NAMESPACE \
  --values ./helm/memristor-nn/values-$ENVIRONMENT.yaml \
  --set image.tag=$IMAGE_TAG \
  --wait --timeout 600s

# Verify deployment
echo "✅ Verifying deployment"
kubectl rollout status deployment/memristor-nn -n $NAMESPACE --timeout=300s

# Run health checks
echo "🏥 Running health checks"
kubectl wait --for=condition=ready pod -l app=memristor-nn -n $NAMESPACE --timeout=300s

# Get service information
echo "📊 Deployment information:"
kubectl get pods,svc,hpa -n $NAMESPACE -l app=memristor-nn

echo "✅ Deployment completed successfully!"
