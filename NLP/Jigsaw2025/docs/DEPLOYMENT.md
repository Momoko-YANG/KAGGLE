# Deployment Guide

## 🚀 Deployment Options

### 1. Local Deployment

#### Development Setup
```bash
# Clone repository
git clone https://github.com/Momoko-YANG/kaggle-Jigsaw2025.git
cd kaggle-Jigsaw2025

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirement.txt
pip install -e .

# Setup directories
make setup-dirs

# Configure environment
cp .env.example .env
# Edit .env with your settings
```

#### Run Training
```bash
# Using make
make train

# Or directly
python scripts/train.py
```

#### Run Inference
```bash
make inference
# Or
python scripts/inference.py
```

---

### 2. Docker Deployment

#### Build and Run
```bash
# Build Docker image
docker build -t rule-violation-detection:latest .

# Run training
docker run -v $(pwd)/data:/app/data \
           -v $(pwd)/models:/app/models \
           -v $(pwd)/logs:/app/logs \
           rule-violation-detection:latest \
           python scripts/train.py

# Run inference
docker run -v $(pwd)/data:/app/data \
           -v $(pwd)/models:/app/models \
           rule-violation-detection:latest \
           python scripts/inference.py
```

#### Using Docker Compose
```bash
# Start all services
docker-compose up

# Run specific service
docker-compose run train
docker-compose run inference

# View logs
docker-compose logs -f
```

---

### 3. API Deployment

#### Local API Server
```bash
# Install FastAPI dependencies
pip install fastapi uvicorn

# Start API server
python api/app.py

# Or using uvicorn directly
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

#### API Usage Examples
```bash
# Health check
curl http://localhost:8000/

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a test comment",
    "rule": "Be respectful",
    "positive_examples": ["Violation example 1", "Violation example 2"],
    "negative_examples": ["Compliant example 1", "Compliant example 2"]
  }'

# Batch prediction
curl -X POST http://localhost:8000/batch_predict \
  -H "Content-Type: application/json" \
  -d '[{"text": "...", "rule": "...", ...}, ...]'
```

#### Production API Deployment
```bash
# Using Gunicorn
pip install gunicorn
gunicorn api.app:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120
```

---

### 4. Cloud Deployment

#### AWS (EC2/ECS)
```bash
# 1. Create EC2 instance or ECS cluster
# 2. Install Docker
# 3. Pull and run container

# Example EC2 deployment
ssh ubuntu@your-ec2-instance
git clone https://github.com/Momoko-YANG/kaggle-Jigsaw2025.git
cd kaggle-Jigsaw2025
docker-compose up -d
```

#### Google Cloud Platform (GCP)
```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/rule-violation-detection

# Deploy to Cloud Run
gcloud run deploy rule-violation-api \
  --image gcr.io/PROJECT_ID/rule-violation-detection \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --timeout 300
```

#### Azure
```bash
# Login to Azure
az login

# Create container registry
az acr create --resource-group myResourceGroup \
              --name myContainerRegistry \
              --sku Basic

# Build and push image
az acr build --registry myContainerRegistry \
             --image rule-violation-detection:v1 .

# Deploy to Azure Container Instances
az container create \
  --resource-group myResourceGroup \
  --name rule-violation-app \
  --image myContainerRegistry.azurecr.io/rule-violation-detection:v1 \
  --cpu 2 --memory 4
```

---

### 5. Kubernetes Deployment

#### Create Kubernetes Manifests

**deployment.yaml**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rule-violation-detection
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rule-violation
  template:
    metadata:
      labels:
        app: rule-violation
    spec:
      containers:
      - name: api
        image: rule-violation-detection:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        env:
        - name: LOG_LEVEL
          value: "INFO"
```

**service.yaml**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: rule-violation-service
spec:
  selector:
    app: rule-violation
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

Deploy:
```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

---

## 📊 Monitoring

### Health Checks
```bash
# API health
curl http://localhost:8000/health

# Metrics
curl http://localhost:8000/metrics
```

### Logging
Logs are stored in `logs/` directory:
```bash
# View logs
tail -f logs/rule_violation_*.log

# Search logs
grep "ERROR" logs/rule_violation_*.log
```

### Performance Monitoring
```python
from src.utils.monitoring import PerformanceMonitor

monitor = PerformanceMonitor()
# ... run your code ...
print(monitor.get_summary())
```

---

## 🔒 Security Considerations

1. **API Keys**: Store in environment variables, never in code
2. **HTTPS**: Use SSL/TLS for production APIs
3. **Rate Limiting**: Implement rate limiting for public APIs
4. **Input Validation**: Validate all user inputs
5. **Model Access**: Restrict model file access

---

## 🔄 CI/CD Pipeline

### GitHub Actions
The project includes automated CI/CD:
- Runs tests on push/PR
- Checks code formatting
- Generates coverage reports
- Builds Docker images on main branch

### Manual Deployment Workflow
```bash
# 1. Run tests
make test

# 2. Format code
make format

# 3. Build Docker image
make docker-build

# 4. Deploy
docker push your-registry/rule-violation-detection:latest
kubectl set image deployment/rule-violation-detection \
  api=your-registry/rule-violation-detection:latest
```

---

## 📈 Scaling

### Horizontal Scaling
```bash
# Kubernetes
kubectl scale deployment rule-violation-detection --replicas=5

# Docker Compose
docker-compose up --scale api=5
```

### Vertical Scaling
Adjust resource limits in:
- Docker: `docker run --memory=8g --cpus=4`
- Kubernetes: Update `resources` in deployment.yaml
- Cloud: Adjust instance size

---

## 🐛 Troubleshooting

### Common Issues

**Issue: Out of Memory**
```bash
# Solution: Reduce batch size in config.yaml
inference:
  batch_size: 32  # Reduce this
```

**Issue: Slow Inference**
```bash
# Solution: Enable FP16 and GPU
model:
  use_fp16: true
```

**Issue: Model Not Found**
```bash
# Solution: Check model path
ls -la models/
# Ensure model files exist in expected location
```

---

## 📞 Support

For deployment issues:
1. Check logs in `logs/` directory
2. Review configuration in `config/config.yaml`
3. Open an issue on GitHub
4. Contact: yangmy1215@gmail.com