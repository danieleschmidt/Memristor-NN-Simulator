# Memristor-NN Production Deployment

Generated on 2025-08-24 13:40:12

## Quick Start

```bash
# Deploy
./deployment/deploy.sh
```

## Manual Deployment

```bash
# Build container
docker build -t memristor-nn-sim:latest .

# Run tests
docker run --rm memristor-nn-sim:latest python3 test_core_functionality.py
```

## Configuration

Copy `deployment/.env.template` to `.env` and customize.
