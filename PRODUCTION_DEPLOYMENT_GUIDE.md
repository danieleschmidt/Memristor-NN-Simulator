# Production Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the Memristor Neural Network Simulator in production environments with global-first architecture.

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- 2GB RAM minimum (8GB recommended)
- 1GB disk space
- Docker (optional but recommended)

### Basic Installation

```bash
# Install from source
git clone <repository-url>
cd Memristor-NN-Simulator
pip install -e ".[dev,rtl]"

# Verify installation
python -c "import memristor_nn; print('✓ Installation successful')"
```

### Docker Deployment (Recommended)

```bash
# Build container
docker build -t memristor-nn-sim:latest .

# Run single instance
docker run -d \
  --name memristor-sim \
  -p 8080:8080 \
  -v $(pwd)/data:/app/data \
  memristor-nn-sim:latest

# Verify deployment
curl http://localhost:8080/health
```

## 🌍 Global Production Architecture

### Multi-Region Deployment

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  simulator:
    image: memristor-nn-sim:latest
    deploy:
      replicas: 3
      placement:
        constraints:
          - node.labels.region == us-east
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 512M
          cpus: '0.5'
  
  load-balancer:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: memristor-nn-simulator
  labels:
    app: memristor-nn-sim
spec:
  replicas: 3
  selector:
    matchLabels:
      app: memristor-nn-sim
  template:
    metadata:
      labels:
        app: memristor-nn-sim
    spec:
      containers:
      - name: simulator
        image: memristor-nn-sim:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "512Mi"
            cpu: "0.5"
          limits:
            memory: "2Gi"
            cpu: "1.0"
        env:
        - name: LOG_LEVEL
          value: "INFO"
        - name: CACHE_SIZE
          value: "1000"
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: memristor-nn-service
spec:
  selector:
    app: memristor-nn-sim
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer
```

## ⚙️ Configuration Management

### Environment Variables

```bash
# Core Configuration
MEMRISTOR_LOG_LEVEL=INFO
MEMRISTOR_CACHE_SIZE=1000
MEMRISTOR_MAX_CROSSBAR_SIZE=512
MEMRISTOR_ENABLE_GPU=false

# Performance Tuning
MEMRISTOR_WORKER_THREADS=4
MEMRISTOR_BATCH_SIZE=32
MEMRISTOR_MEMORY_LIMIT_MB=2048

# Security Settings
MEMRISTOR_RATE_LIMIT_CALLS=1000
MEMRISTOR_RATE_LIMIT_WINDOW=3600
MEMRISTOR_ENABLE_AUTH=true

# Global Settings
MEMRISTOR_TIMEZONE=UTC
MEMRISTOR_LOCALE=en_US
MEMRISTOR_REGION=us-east-1
```

### Configuration File

```yaml
# config/production.yml
server:
  host: "0.0.0.0"
  port: 8080
  workers: 4

logging:
  level: INFO
  format: json
  file: "/var/log/memristor-nn.log"

cache:
  type: "memory"
  size: 1000
  ttl: 3600

security:
  rate_limit:
    calls: 1000
    window: 3600
  memory_limit_mb: 2048
  input_validation: strict

performance:
  batch_size: 32
  parallel_workers: 4
  cache_enabled: true
  profiling_enabled: false

global:
  timezone: "UTC"
  locale: "en_US"
  region: "us-east-1"
  compliance:
    gdpr: true
    ccpa: true
    pdpa: true
```

## 📊 Monitoring & Observability

### Health Checks

```python
# health_check.py
import requests
import sys

def check_health():
    try:
        response = requests.get("http://localhost:8080/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "healthy":
                print("✅ Service healthy")
                return 0
        print("❌ Service unhealthy")
        return 1
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(check_health())
```

### Monitoring Endpoints

```bash
# Service health
curl http://localhost:8080/health

# Performance metrics
curl http://localhost:8080/metrics

# System information
curl http://localhost:8080/info

# Cache statistics
curl http://localhost:8080/cache/stats
```

### Prometheus Integration

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'memristor-nn-sim'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
    scrape_interval: 15s
    scrape_timeout: 10s
```

## 🔒 Security Configuration

### SSL/TLS Setup

```nginx
# nginx.conf
server {
    listen 443 ssl http2;
    server_name memristor-nn.example.com;

    ssl_certificate /etc/ssl/memristor-nn.crt;
    ssl_certificate_key /etc/ssl/memristor-nn.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;

    location / {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

upstream backend {
    least_conn;
    server memristor-sim-1:8080 max_fails=3 fail_timeout=30s;
    server memristor-sim-2:8080 max_fails=3 fail_timeout=30s;
    server memristor-sim-3:8080 max_fails=3 fail_timeout=30s;
}
```

### Firewall Rules

```bash
# UFW configuration
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw --force enable

# Docker-specific rules
sudo ufw allow from 172.16.0.0/12 to any port 8080
```

## 📈 Scaling & Performance

### Horizontal Scaling

```bash
# Docker Swarm
docker service create \
  --name memristor-nn-sim \
  --replicas 5 \
  --publish 8080:8080 \
  --env MEMRISTOR_WORKER_THREADS=2 \
  memristor-nn-sim:latest

# Scale up
docker service scale memristor-nn-sim=10

# Scale down
docker service scale memristor-nn-sim=3
```

### Auto-scaling (Kubernetes)

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: memristor-nn-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: memristor-nn-simulator
  minReplicas: 3
  maxReplicas: 20
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

### Performance Tuning

```bash
# System-level optimizations
echo 'net.core.somaxconn = 65535' >> /etc/sysctl.conf
echo 'net.core.netdev_max_backlog = 5000' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_max_syn_backlog = 65535' >> /etc/sysctl.conf
sysctl -p

# Memory optimization
echo 'vm.swappiness = 1' >> /etc/sysctl.conf
echo 'vm.overcommit_memory = 1' >> /etc/sysctl.conf
```

## 🗄️ Data Management

### Persistent Storage

```yaml
# docker-compose with persistent volumes
version: '3.8'
services:
  simulator:
    image: memristor-nn-sim:latest
    volumes:
      - sim_data:/app/data
      - sim_cache:/app/cache
      - sim_logs:/var/log
    environment:
      - MEMRISTOR_DATA_PATH=/app/data
      - MEMRISTOR_CACHE_PATH=/app/cache

volumes:
  sim_data:
    driver: local
    driver_opts:
      type: nfs
      o: addr=nfs-server.example.com,nolock,soft,rw
      device: ":/path/to/memristor/data"
  sim_cache:
    driver: local
  sim_logs:
    driver: local
```

### Backup Strategy

```bash
#!/bin/bash
# backup.sh
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup/memristor-nn"

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup application data
docker run --rm \
  -v memristor_sim_data:/source:ro \
  -v $BACKUP_DIR:/backup \
  alpine:latest \
  tar czf /backup/data_$DATE.tar.gz -C /source .

# Backup configuration
tar czf $BACKUP_DIR/config_$DATE.tar.gz config/

# Backup logs (last 7 days)
docker run --rm \
  -v memristor_sim_logs:/logs:ro \
  -v $BACKUP_DIR:/backup \
  alpine:latest \
  find /logs -mtime -7 -type f -exec tar czf /backup/logs_$DATE.tar.gz {} +

# Cleanup old backups (keep 30 days)
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete

echo "Backup completed: $DATE"
```

## 🌐 Global Compliance

### GDPR Compliance

```python
# gdpr_config.py
GDPR_SETTINGS = {
    'data_retention_days': 90,
    'anonymization_enabled': True,
    'consent_required': True,
    'right_to_deletion': True,
    'data_portability': True,
    'audit_logging': True
}
```

### Multi-language Support

```yaml
# i18n/en.yml
simulation:
  started: "Simulation started"
  completed: "Simulation completed successfully"
  failed: "Simulation failed: {error}"
  
validation:
  invalid_input: "Invalid input provided"
  out_of_range: "Value out of acceptable range"

# i18n/es.yml
simulation:
  started: "Simulación iniciada"
  completed: "Simulación completada exitosamente"
  failed: "Simulación falló: {error}"
```

## 🚨 Disaster Recovery

### Backup and Restore

```bash
# Full system backup
#!/bin/bash
kubectl create namespace backup
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: CronJob
metadata:
  name: memristor-backup
  namespace: backup
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: velero/velero:latest
            command:
            - velero
            - backup
            - create
            - memristor-daily-backup
            - --include-namespaces
            - production
          restartPolicy: OnFailure
EOF
```

### Failover Procedures

```bash
# Automated failover script
#!/bin/bash
PRIMARY_REGION="us-east-1"
SECONDARY_REGION="eu-west-1"

# Check primary health
if ! curl -sf http://$PRIMARY_REGION.memristor-nn.com/health; then
    echo "Primary region down, initiating failover..."
    
    # Update DNS to point to secondary
    aws route53 change-resource-record-sets \
        --hosted-zone-id Z123456789 \
        --change-batch '{
            "Changes": [{
                "Action": "UPSERT",
                "ResourceRecordSet": {
                    "Name": "memristor-nn.com",
                    "Type": "CNAME",
                    "TTL": 60,
                    "ResourceRecords": [{"Value": "'$SECONDARY_REGION'.memristor-nn.com"}]
                }
            }]
        }'
    
    # Scale up secondary region
    kubectl --context=$SECONDARY_REGION scale deployment memristor-nn-simulator --replicas=10
    
    echo "Failover completed"
fi
```

## 📋 Deployment Checklist

### Pre-deployment
- [ ] Environment variables configured
- [ ] SSL certificates installed
- [ ] Database connections tested
- [ ] Firewall rules applied
- [ ] Monitoring configured
- [ ] Backup system tested

### Deployment
- [ ] Application deployed
- [ ] Health checks passing
- [ ] Performance tests completed
- [ ] Security scan passed
- [ ] Load balancer configured
- [ ] DNS updated

### Post-deployment
- [ ] Monitoring alerts configured
- [ ] Log aggregation working
- [ ] Backup jobs scheduled
- [ ] Documentation updated
- [ ] Team notifications sent
- [ ] Rollback plan confirmed

## 📞 Support & Maintenance

### Log Analysis

```bash
# View application logs
docker logs memristor-sim --tail=100 -f

# Search for errors
docker logs memristor-sim 2>&1 | grep -i error

# Performance monitoring
docker stats memristor-sim
```

### Troubleshooting

| Issue | Symptoms | Solution |
|-------|----------|----------|
| High Memory Usage | >2GB RAM usage | Restart container, check for memory leaks |
| Slow Response | >5s response time | Check cache, scale horizontally |
| Connection Errors | 502/503 errors | Check upstream services, restart load balancer |
| SSL Issues | Certificate errors | Renew certificates, check nginx config |

### Maintenance Windows

```bash
# Rolling update (zero-downtime)
kubectl set image deployment/memristor-nn-simulator \
    simulator=memristor-nn-sim:v2.0.0

# Verify rollout
kubectl rollout status deployment/memristor-nn-simulator

# Rollback if needed
kubectl rollout undo deployment/memristor-nn-simulator
```

---

**Production Deployment Complete** ✅

For additional support or questions, contact the Terragon Labs team.