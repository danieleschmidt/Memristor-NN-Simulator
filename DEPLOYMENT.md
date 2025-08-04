# Deployment Guide

This guide covers deploying memristor-nn-simulator in various environments from development to production.

## Table of Contents

1. [Local Development](#local-development)
2. [Docker Deployment](#docker-deployment)
3. [Cloud Deployment](#cloud-deployment)
4. [Production Considerations](#production-considerations)
5. [Monitoring and Observability](#monitoring-and-observability)
6. [Scaling](#scaling)

## Local Development

### Quick Start

```bash
# Clone repository
git clone https://github.com/danieleschmidt/photonic-mlir-synth-bridge.git
cd photonic-mlir-synth-bridge

# Setup environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e ".[dev,rtl]"

# Run example
python examples/basic_usage.py
```

### Development with Hot Reload

For active development with automatic reloading:

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Start development server (if building web interface)
python -m memristor_nn.server --reload --debug
```

## Docker Deployment

### Development Environment

```bash
# Build and run development container
docker-compose up memristor-dev

# Interactive development
docker-compose run memristor-dev bash

# Run tests in container
docker-compose run memristor-dev pytest tests/
```

### Production Environment

```bash
# Build production image
docker build --target production -t memristor-nn:latest .

# Run production container
docker run -d \
  --name memristor-nn-prod \
  -e MEMRISTOR_LOG_LEVEL=INFO \
  -v memristor-cache:/app/cache \
  --restart unless-stopped \
  memristor-nn:latest
```

### Docker Compose Production

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  memristor-nn:
    build:
      context: .
      target: production
    environment:
      - MEMRISTOR_LOG_LEVEL=INFO
      - MEMRISTOR_CACHE_DIR=/app/cache
      - MEMRISTOR_MAX_WORKERS=4
    volumes:
      - memristor-cache:/app/cache
      - ./results:/app/results
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import memristor_nn; print('OK')"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '2.0'
        reservations:
          memory: 1G
          cpus: '1.0'

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    volumes:
      - redis-data:/data

volumes:
  memristor-cache:
  redis-data:
```

## Cloud Deployment

### AWS ECS Deployment

#### Task Definition

```json
{
  "family": "memristor-nn",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "memristor-nn",
      "image": "your-repo/memristor-nn:latest",
      "essential": true,
      "environment": [
        {"name": "MEMRISTOR_LOG_LEVEL", "value": "INFO"},
        {"name": "MEMRISTOR_CACHE_DIR", "value": "/app/cache"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/memristor-nn",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "python -c 'import memristor_nn; print(\"OK\")'"],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ]
}
```

#### ECS Service

```bash
# Create ECS service
aws ecs create-service \
  --cluster memristor-cluster \
  --service-name memristor-nn-service \
  --task-definition memristor-nn:1 \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-12345],securityGroups=[sg-12345],assignPublicIp=ENABLED}"
```

### Kubernetes Deployment

#### Deployment Manifest

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: memristor-nn
  labels:
    app: memristor-nn
spec:
  replicas: 3
  selector:
    matchLabels:
      app: memristor-nn
  template:
    metadata:
      labels:
        app: memristor-nn
    spec:
      containers:
      - name: memristor-nn
        image: memristor-nn:latest
        ports:
        - containerPort: 8000
        env:
        - name: MEMRISTOR_LOG_LEVEL
          value: "INFO"
        - name: MEMRISTOR_CACHE_DIR
          value: "/app/cache"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          exec:
            command:
            - python
            - -c
            - "import memristor_nn; print('OK')"
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          exec:
            command:
            - python
            - -c
            - "import memristor_nn; print('OK')"
          initialDelaySeconds: 5
          periodSeconds: 10
        volumeMounts:
        - name: cache-volume
          mountPath: /app/cache
      volumes:
      - name: cache-volume
        persistentVolumeClaim:
          claimName: memristor-cache-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: memristor-nn-service
spec:
  selector:
    app: memristor-nn
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

#### Persistent Volume

```yaml
# k8s/pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: memristor-cache-pvc
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
```

### Google Cloud Run

```bash
# Build and push to Google Container Registry
docker build -t gcr.io/PROJECT-ID/memristor-nn:latest .
docker push gcr.io/PROJECT-ID/memristor-nn:latest

# Deploy to Cloud Run
gcloud run deploy memristor-nn \
  --image gcr.io/PROJECT-ID/memristor-nn:latest \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --max-instances 10 \
  --set-env-vars MEMRISTOR_LOG_LEVEL=INFO
```

## Production Considerations

### Environment Configuration

```bash
# Production environment variables
export MEMRISTOR_LOG_LEVEL=INFO
export MEMRISTOR_CACHE_DIR=/app/cache
export MEMRISTOR_MAX_WORKERS=4
export MEMRISTOR_MEMORY_LIMIT=2048  # MB
export MEMRISTOR_ENABLE_METRICS=true
export MEMRISTOR_METRICS_PORT=9090
```

### Security Configuration

```bash
# Security settings
export MEMRISTOR_ENABLE_AUTH=true
export MEMRISTOR_SECRET_KEY=your-secret-key
export MEMRISTOR_MAX_REQUEST_SIZE=100MB
export MEMRISTOR_RATE_LIMIT=100  # requests per minute
```

### Database Configuration

For persistent storage of results and metadata:

```bash
# PostgreSQL connection
export DATABASE_URL=postgresql://user:pass@host:5432/memristor_db

# Redis for caching
export REDIS_URL=redis://host:6379/0
```

### Performance Tuning

```python
# config/production.py
PERFORMANCE_CONFIG = {
    "max_workers": 8,
    "memory_limit_mb": 4096,
    "cache_size": 10000,
    "cache_ttl": 7200,
    "enable_gpu": True,
    "batch_size": 64,
    "max_concurrent_simulations": 4
}
```

## Monitoring and Observability

### Health Checks

```python
# health_check.py
import memristor_nn as mn
from memristor_nn.utils.logger import get_logger

def health_check():
    """Comprehensive health check."""
    logger = get_logger("health_check")
    
    try:
        # Test basic functionality
        crossbar = mn.CrossbarArray(rows=4, cols=4)
        assert crossbar.rows == 4
        
        # Test memory usage
        from memristor_nn.optimization import MemoryOptimizer
        optimizer = MemoryOptimizer()
        stats = optimizer.get_memory_usage()
        
        if stats["system_usage_fraction"] > 0.9:
            logger.warning("High memory usage detected")
            
        return {"status": "healthy", "memory_usage": stats}
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}

if __name__ == "__main__":
    result = health_check()
    print(result)
    exit(0 if result["status"] == "healthy" else 1)
```

### Metrics Collection

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from memristor_nn.utils.logger import get_logger

# Metrics
SIMULATION_COUNT = Counter('memristor_simulations_total', 'Total simulations run')
SIMULATION_DURATION = Histogram('memristor_simulation_duration_seconds', 'Simulation duration')
ACTIVE_SIMULATIONS = Gauge('memristor_active_simulations', 'Currently active simulations')
MEMORY_USAGE = Gauge('memristor_memory_usage_bytes', 'Memory usage in bytes')

def start_metrics_server(port=9090):
    """Start metrics server."""
    start_http_server(port)
    logger = get_logger("metrics")
    logger.info(f"Metrics server started on port {port}")
```

### Logging Configuration

```python
# logging_config.py
import logging.config

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
        'json': {
            'format': '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
        }
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'json',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'DEBUG',
            'formatter': 'standard',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/app/logs/memristor.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
        }
    },
    'loggers': {
        'memristor_nn': {
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': False
        }
    }
}

logging.config.dictConfig(LOGGING_CONFIG)
```

## Scaling

### Horizontal Scaling

#### Load Balancer Configuration (nginx)

```nginx
# nginx.conf
upstream memristor_backend {
    server memristor-nn-1:8000;
    server memristor-nn-2:8000;
    server memristor-nn-3:8000;
}

server {
    listen 80;
    server_name memristor-nn.example.com;
    
    location / {
        proxy_pass http://memristor_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Increase timeout for long simulations
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
    
    location /health {
        proxy_pass http://memristor_backend/health;
        access_log off;
    }
}
```

### Vertical Scaling

#### Resource Limits

```yaml
# High-performance configuration
resources:
  requests:
    memory: "4Gi"
    cpu: "2000m"
  limits:
    memory: "8Gi"
    cpu: "4000m"
```

### Auto Scaling

#### Kubernetes HPA

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: memristor-nn-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: memristor-nn
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Backup and Recovery

### Data Backup

```bash
#!/bin/bash
# backup.sh

# Backup cache and results
tar -czf memristor-backup-$(date +%Y%m%d).tar.gz \
  /app/cache \
  /app/results \
  /app/logs

# Upload to S3
aws s3 cp memristor-backup-$(date +%Y%m%d).tar.gz \
  s3://memristor-backups/
```

### Disaster Recovery

```yaml
# disaster-recovery.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: disaster-recovery-plan
data:
  recovery-steps: |
    1. Restore from latest backup
    2. Verify data integrity
    3. Restart services
    4. Run health checks
    5. Monitor for issues
```

## Troubleshooting

### Common Issues

#### Memory Issues
```bash
# Check memory usage
docker stats memristor-nn
kubectl top pods

# Increase memory limits
docker run --memory=4g memristor-nn:latest
```

#### Performance Issues
```bash
# Enable profiling
export MEMRISTOR_ENABLE_PROFILING=true

# Check metrics
curl http://localhost:9090/metrics
```

#### Connection Issues
```bash
# Test health endpoint
curl -f http://localhost:8000/health || exit 1

# Check logs
docker logs memristor-nn
kubectl logs deployment/memristor-nn
```

This deployment guide provides comprehensive coverage for deploying memristor-nn-simulator across different environments while maintaining security, performance, and reliability standards.