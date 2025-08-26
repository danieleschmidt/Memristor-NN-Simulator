#!/bin/bash
set -e

# Rollback Script
ENVIRONMENT=${1:-development}
REVISION=${2:-0}

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

echo "🔄 Rolling back Memristor NN in $ENVIRONMENT environment"

if [ "$REVISION" -eq 0 ]; then
  echo "📋 Rolling back to previous revision"
  helm rollback memristor-nn -n $NAMESPACE
else
  echo "📋 Rolling back to revision $REVISION"
  helm rollback memristor-nn $REVISION -n $NAMESPACE
fi

# Wait for rollback to complete
echo "⏳ Waiting for rollback to complete..."
kubectl rollout status deployment/memristor-nn -n $NAMESPACE --timeout=300s

# Verify rollback
echo "✅ Verifying rollback..."
kubectl get pods -n $NAMESPACE -l app=memristor-nn

echo "✅ Rollback completed successfully!"
