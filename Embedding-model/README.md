# 🚀 Embedding Service API

A production-ready FastAPI application that provides text embeddings using open-source Sentence Transformers models. Fast, efficient, and cost-effective alternative to commercial embedding APIs.

## ✨ Features

- **🚀 Fast Inference** - Generate embeddings in milliseconds
- **📦 Batch Processing** - Process up to 100 texts in a single request
- **🔧 Highly Configurable** - Environment-based configuration
- **📊 Health Monitoring** - Built-in health checks and metrics
- **🌐 CORS Enabled** - Ready for web applications
- **📝 Auto Documentation** - Interactive API docs (Swagger/ReDoc)
- **🔒 Input Validation** - Robust error handling with Pydantic
- **💾 Model Caching** - Automatic model download and caching
- **🎯 Multiple Models** - Support for any Sentence Transformer model

---

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [API Endpoints](#-api-endpoints)
- [Usage Examples](#-usage-examples)
- [Available Models](#-available-models)
- [Configuration](#-configuration)
- [Production Deployment](#-production-deployment)
- [Integration Guide](#-integration-guide)
- [Troubleshooting](#-troubleshooting)

---

## 🏁 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) GPU for faster inference

### Installation

```bash
# Navigate to the Embedding-model directory
cd Embedding-model

# Create a virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Start the Server

```bash
# Development mode with auto-reload
python main.py

# Or using uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production mode (no reload)
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at `http://localhost:8000`

### Verify Installation

```bash
# Check health status
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy","model":"all-MiniLM-L6-v2","model_loaded":true}
```

---

## 📡 API Endpoints

### Overview

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information and links |
| `/health` | GET | Health check and status |
| `/model-info` | GET | Model specifications |
| `/embed` | POST | Generate embeddings |
| `/docs` | GET | Interactive API documentation (Swagger UI) |
| `/redoc` | GET | Alternative API documentation (ReDoc) |

---

### 📍 GET `/` - Root Endpoint

Returns basic API information.

**Response:**
```json
{
  "message": "Embedding Service API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health"
}
```

**Status Codes:**
- `200 OK` - Success

---

### 🏥 GET `/health` - Health Check

Check if the service is running and the model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "model": "all-MiniLM-L6-v2",
  "model_loaded": true
}
```

**Status Codes:**
- `200 OK` - Service is healthy

**Use Case:** Monitor service availability in production

---

### ℹ️ GET `/model-info` - Model Information

Get detailed information about the loaded embedding model.

**Response:**
```json
{
  "model_name": "all-MiniLM-L6-v2",
  "max_seq_length": 256,
  "embedding_dimension": 384,
  "max_batch_size": 100
}
```

**Status Codes:**
- `200 OK` - Success
- `503 Service Unavailable` - Model not loaded

**Use Case:** Verify model configuration and capabilities

---

### 🎯 POST `/embed` - Generate Embeddings

Generate embeddings for one or more texts.

#### Request Body

```json
{
  "texts": "string or array of strings",
  "normalize": true
}
```

**Parameters:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `texts` | `string` or `string[]` | ✅ Yes | - | Text(s) to embed |
| `normalize` | `boolean` | ❌ No | `true` | Normalize to unit length |

**Constraints:**
- Maximum batch size: 100 texts
- Maximum text length: 256 tokens (model-dependent)
- Texts cannot be empty

#### Response

```json
{
  "embeddings": [[0.123, -0.456, ...]],
  "model": "all-MiniLM-L6-v2",
  "dimension": 384,
  "num_embeddings": 1,
  "processing_time": 0.028
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `embeddings` | `float[][]` | List of embedding vectors |
| `model` | `string` | Model name used |
| `dimension` | `integer` | Embedding dimension |
| `num_embeddings` | `integer` | Number of embeddings generated |
| `processing_time` | `float` | Processing time in seconds |

**Status Codes:**
- `200 OK` - Success
- `422 Unprocessable Entity` - Invalid input (empty text, batch too large, etc.)
- `500 Internal Server Error` - Server error during embedding generation
- `503 Service Unavailable` - Model not loaded

#### Example: Single Text

**Request:**
```bash
curl -X POST "http://localhost:8000/embed" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": "Machine learning is transforming technology",
    "normalize": true
  }'
```

**Response:**
```json
{
  "embeddings": [[0.0234, -0.0891, 0.1234, ...]],
  "model": "all-MiniLM-L6-v2",
  "dimension": 384,
  "num_embeddings": 1,
  "processing_time": 0.025
}
```

#### Example: Batch Processing

**Request:**
```bash
curl -X POST "http://localhost:8000/embed" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "First document about AI",
      "Second document about ML",
      "Third document about NLP"
    ],
    "normalize": true
  }'
```

**Response:**
```json
{
  "embeddings": [
    [0.0234, -0.0891, ...],
    [0.0456, -0.0123, ...],
    [0.0789, -0.0456, ...]
  ],
  "model": "all-MiniLM-L6-v2",
  "dimension": 384,
  "num_embeddings": 3,
  "processing_time": 0.045
}
```

#### Error Responses

**Empty Text:**
```json
{
  "detail": [
    {
      "type": "value_error",
      "msg": "Text cannot be empty"
    }
  ]
}
```

**Batch Too Large:**
```json
{
  "detail": [
    {
      "type": "value_error",
      "msg": "Batch size cannot exceed 100"
    }
  ]
}
```

---

## 💻 Usage Examples

### Using cURL (Linux/Mac)

```bash
# Single text embedding
curl -X POST "http://localhost:8000/embed" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": "Hello, world!",
    "normalize": true
  }'

# Batch embedding
curl -X POST "http://localhost:8000/embed" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "First text",
      "Second text",
      "Third text"
    ],
    "normalize": true
  }'
```

### Using PowerShell (Windows)

```powershell
# Single text embedding
$body = @{
    texts = "Hello, world!"
    normalize = $true
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/embed" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body

# Batch embedding
$body = @{
    texts = @(
        "First text",
        "Second text",
        "Third text"
    )
    normalize = $true
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/embed" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

### Using Python

#### Simple Request

```python
import requests

# API endpoint
url = "http://localhost:8000/embed"

# Single text
response = requests.post(
    url,
    json={"texts": "Hello, world!", "normalize": True}
)

if response.status_code == 200:
    result = response.json()
    print(f"Embedding dimension: {result['dimension']}")
    print(f"Processing time: {result['processing_time']}s")
    embedding = result['embeddings'][0]
    print(f"Embedding vector (first 5 values): {embedding[:5]}")
else:
    print(f"Error: {response.status_code}")
    print(response.json())
```

#### Batch Processing

```python
import requests

url = "http://localhost:8000/embed"

# Multiple texts
texts = [
    "Machine learning is a subset of artificial intelligence",
    "Natural language processing enables computers to understand text",
    "Deep learning uses neural networks with multiple layers"
]

response = requests.post(
    url,
    json={"texts": texts, "normalize": True}
)

result = response.json()
embeddings = result['embeddings']

print(f"Generated {len(embeddings)} embeddings")
print(f"Each embedding has {result['dimension']} dimensions")
print(f"Total processing time: {result['processing_time']}s")

# Use embeddings for similarity search, clustering, etc.
for i, embedding in enumerate(embeddings):
    print(f"Text {i+1}: {len(embedding)} dimensions")
```

#### Python Client Class

```python
import requests
from typing import List, Union

class EmbeddingClient:
    """Client for the Embedding Service API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def embed(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True
    ) -> dict:
        """Generate embeddings for text(s)"""
        response = requests.post(
            f"{self.base_url}/embed",
            json={"texts": texts, "normalize": normalize}
        )
        response.raise_for_status()
        return response.json()
    
    def health(self) -> dict:
        """Check service health"""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def model_info(self) -> dict:
        """Get model information"""
        response = requests.get(f"{self.base_url}/model-info")
        response.raise_for_status()
        return response.json()

# Usage
client = EmbeddingClient()

# Check if service is ready
if client.health()['status'] == 'healthy':
    # Get embeddings
    result = client.embed("This is a test sentence")
    print(f"Embedding: {result['embeddings'][0][:5]}...")
```

### Using JavaScript/TypeScript

#### Fetch API

```javascript
// Single text embedding
async function getEmbedding(text) {
  const response = await fetch('http://localhost:8000/embed', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      texts: text,
      normalize: true
    })
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const result = await response.json();
  return result.embeddings[0];
}

// Usage
getEmbedding('Hello, world!')
  .then(embedding => {
    console.log('Embedding dimension:', embedding.length);
    console.log('First 5 values:', embedding.slice(0, 5));
  })
  .catch(error => console.error('Error:', error));
```

#### Batch Processing

```javascript
async function getBatchEmbeddings(texts) {
  const response = await fetch('http://localhost:8000/embed', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      texts: texts,
      normalize: true
    })
  });

  const result = await response.json();
  return result;
}

// Usage
const texts = [
  'First document',
  'Second document',
  'Third document'
];

getBatchEmbeddings(texts)
  .then(result => {
    console.log(`Generated ${result.num_embeddings} embeddings`);
    console.log(`Dimension: ${result.dimension}`);
    console.log(`Processing time: ${result.processing_time}s`);
  });
```

#### TypeScript Client

```typescript
interface EmbedRequest {
  texts: string | string[];
  normalize?: boolean;
}

interface EmbedResponse {
  embeddings: number[][];
  model: string;
  dimension: number;
  num_embeddings: number;
  processing_time: number;
}

class EmbeddingClient {
  constructor(private baseUrl: string = 'http://localhost:8000') {}

  async embed(
    texts: string | string[],
    normalize: boolean = true
  ): Promise<EmbedResponse> {
    const response = await fetch(`${this.baseUrl}/embed`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ texts, normalize })
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    return response.json();
  }

  async health(): Promise<{ status: string; model: string; model_loaded: boolean }> {
    const response = await fetch(`${this.baseUrl}/health`);
    return response.json();
  }
}

// Usage
const client = new EmbeddingClient();
const result = await client.embed('Test sentence');
console.log(result.embeddings[0]);
```

### Real-World Use Cases

#### Semantic Search

```python
import requests
import numpy as np

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Documents to search
documents = [
    "Python is a high-level programming language",
    "JavaScript is used for web development",
    "Machine learning is a branch of AI",
    "React is a JavaScript library for building UIs"
]

# Get embeddings for all documents
response = requests.post(
    "http://localhost:8000/embed",
    json={"texts": documents, "normalize": True}
)
doc_embeddings = response.json()['embeddings']

# Search query
query = "What is a programming language?"
query_response = requests.post(
    "http://localhost:8000/embed",
    json={"texts": query, "normalize": True}
)
query_embedding = query_response.json()['embeddings'][0]

# Find most similar document
similarities = [
    cosine_similarity(query_embedding, doc_emb)
    for doc_emb in doc_embeddings
]

best_match_idx = np.argmax(similarities)
print(f"Query: {query}")
print(f"Best match: {documents[best_match_idx]}")
print(f"Similarity: {similarities[best_match_idx]:.4f}")
```

#### Document Clustering

```python
import requests
from sklearn.cluster import KMeans
import numpy as np

# Get embeddings for documents
documents = [
    "Python programming tutorial",
    "JavaScript web development",
    "Machine learning basics",
    "Deep learning with PyTorch",
    "React component design",
    "Neural networks explained"
]

response = requests.post(
    "http://localhost:8000/embed",
    json={"texts": documents, "normalize": True}
)
embeddings = np.array(response.json()['embeddings'])

# Cluster documents
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# Print clusters
for i in range(2):
    print(f"\nCluster {i+1}:")
    cluster_docs = [doc for doc, cluster in zip(documents, clusters) if cluster == i]
    for doc in cluster_docs:
        print(f"  - {doc}")
```

---

---

## ⚙️ Configuration

The service can be configured using environment variables. Create a `.env` file in the project root or set environment variables directly.

### Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MODEL_NAME` | string | `all-MiniLM-L6-v2` | Sentence Transformer model to use |
| `CACHE_DIR` | string | `./model_cache` | Directory to cache downloaded models |
| `HOST` | string | `0.0.0.0` | Server host address |
| `PORT` | integer | `8000` | Server port number |
| `RELOAD` | boolean | `true` | Enable auto-reload in development |
| `MAX_BATCH_SIZE` | integer | `100` | Maximum texts per batch request |
| `CORS_ORIGINS` | string | `*` | Comma-separated allowed CORS origins |

### Configuration File

Create a `.env` file:

```bash
# Copy the example file
cp .env.example .env

# Edit with your preferred settings
nano .env
```

Example `.env` file:

```bash
# Model Configuration
MODEL_NAME=all-MiniLM-L6-v2
CACHE_DIR=./model_cache

# Server Configuration
HOST=0.0.0.0
PORT=8000
RELOAD=true

# API Configuration
MAX_BATCH_SIZE=100

# CORS Configuration
CORS_ORIGINS=http://localhost:3000,https://myapp.com
```

### Runtime Configuration

Override settings at runtime:

```bash
# Change model
MODEL_NAME=all-mpnet-base-v2 python main.py

# Change port
PORT=8080 python main.py

# Multiple variables
MODEL_NAME=all-mpnet-base-v2 PORT=8080 python main.py
```

### Model Selection Guide

Choose a model based on your requirements:

**For Speed** (Low latency, real-time applications):
```bash
MODEL_NAME=all-MiniLM-L6-v2  # 384 dimensions, fastest
```

**For Quality** (Best accuracy, offline processing):
```bash
MODEL_NAME=all-mpnet-base-v2  # 768 dimensions, best quality
```

**For Multilingual** (Non-English text):
```bash
MODEL_NAME=paraphrase-multilingual-MiniLM-L12-v2  # 50+ languages
```

**For Balance** (Good speed and quality):
```bash
MODEL_NAME=all-distilroberta-v1  # 768 dimensions, balanced
```

---

## 📦 Available Models

You can use any Sentence Transformer model from the [Hugging Face Hub](https://huggingface.co/models?library=sentence-transformers). Popular options:

| Model | Dimensions | Speed | Quality | Use Case |
|-------|-----------|-------|---------|----------|
| `all-MiniLM-L6-v2` | 384 | ⚡⚡⚡ | ⭐⭐ | Fast, general purpose |
| `all-mpnet-base-v2` | 768 | ⚡⚡ | ⭐⭐⭐ | Best quality, slower |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | ⚡⚡ | ⭐⭐ | Multilingual support |
| `all-distilroberta-v1` | 768 | ⚡⚡ | ⭐⭐⭐ | Good balance |

To change the model, set the `MODEL_NAME` environment variable:

```bash
MODEL_NAME=all-mpnet-base-v2 python main.py
```

## Interactive API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Integration with Your RAG Pipeline

To integrate this embedding service with your existing Firebase Cloud Function:

```python
# In your functions/main.py
import requests

EMBEDDING_SERVICE_URL = "http://your-embedding-service:8000/embed"

def get_embeddings_local(texts):
    """Get embeddings from local Sentence Transformer service"""
    response = requests.post(
        EMBEDDING_SERVICE_URL,
        json={"texts": texts, "normalize": True}
    )
    response.raise_for_status()
    data = response.json()
    return data["embeddings"]

# Replace in your main function:
# embeddings = get_embeddings_openai(batch_chunks)
# With:
embeddings = get_embeddings_local(batch_chunks)
```

## Performance Considerations

- **First Request**: The first request will be slower as the model loads into memory (~5-10 seconds)
- **Batch Processing**: Use batch requests for better throughput (up to 100 texts)
- **GPU Support**: If you have a GPU, PyTorch will automatically use it for faster inference
- **Model Caching**: Models are cached locally after first download
- **Memory Usage**: Smaller models (384d) use ~500MB RAM, larger models (768d) use ~1-2GB RAM

---

## 🐳 Docker Deployment

### Build Docker Image

```bash
# Build the image
docker build -t embedding-service .

# Run the container
docker run -p 8000:8000 embedding-service

# Run with custom environment variables
docker run -p 8000:8000 \
  -e MODEL_NAME=all-mpnet-base-v2 \
  -e MAX_BATCH_SIZE=50 \
  embedding-service
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  embedding-service:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_NAME=all-MiniLM-L6-v2
      - CACHE_DIR=/app/model_cache
      - MAX_BATCH_SIZE=100
    volumes:
      - model-cache:/app/model_cache
    restart: unless-stopped

volumes:
  model-cache:
```

Run with Docker Compose:

```bash
docker-compose up -d
```

---

## ☁️ Cloud Deployment

### Deploy to Render.com (Recommended)

Render offers easy deployment with automatic HTTPS and scaling.

**Quick Deploy:**

1. Push code to GitHub
2. Connect repository to Render
3. Select "Docker" runtime
4. Deploy!

**📖 See detailed guide:** [RENDER_DEPLOY.md](./RENDER_DEPLOY.md)

**Features:**
- ✅ Free tier available
- ✅ Automatic HTTPS
- ✅ Auto-deploy from Git
- ✅ Built-in monitoring
- ✅ Easy scaling

### Other Cloud Platforms

<details>
<summary><b>Deploy to Railway</b></summary>

1. Install Railway CLI: `npm install -g @railway/cli`
2. Login: `railway login`
3. Initialize: `railway init`
4. Deploy: `railway up`

Railway will auto-detect the Dockerfile and deploy.
</details>

<details>
<summary><b>Deploy to Google Cloud Run</b></summary>

```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/embedding-service

# Deploy to Cloud Run
gcloud run deploy embedding-service \
  --image gcr.io/PROJECT_ID/embedding-service \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi
```
</details>

<details>
<summary><b>Deploy to AWS (ECS/Fargate)</b></summary>

1. Push Docker image to ECR
2. Create ECS task definition
3. Create ECS service
4. Configure load balancer

See [AWS ECS documentation](https://docs.aws.amazon.com/ecs/) for details.
</details>

<details>
<summary><b>Deploy to Azure Container Instances</b></summary>

```bash
# Create resource group
az group create --name embedding-service-rg --location eastus

# Deploy container
az container create \
  --resource-group embedding-service-rg \
  --name embedding-service \
  --image your-registry/embedding-service \
  --dns-name-label embedding-service \
  --ports 8000
```
</details>

---

## 🔧 Troubleshooting

### Common Issues

#### Model Download Fails

**Problem:** Model fails to download or times out

**Solutions:**
```bash
# Check internet connectivity
ping huggingface.co

# Manually download model first
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Check disk space
df -h
```

#### Out of Memory Errors

**Problem:** Service crashes with OOM errors

**Solutions:**
1. Use a smaller model:
   ```bash
   MODEL_NAME=all-MiniLM-L6-v2  # 384d, ~500MB RAM
   ```

2. Reduce batch size:
   ```bash
   MAX_BATCH_SIZE=50
   ```

3. Increase available RAM (Docker):
   ```bash
   docker run -m 2g -p 8000:8000 embedding-service
   ```

#### Port Already in Use

**Problem:** `Address already in use` error

**Solutions:**
```bash
# Check what's using port 8000
# Windows:
netstat -ano | findstr :8000

# Linux/Mac:
lsof -i :8000

# Use different port
PORT=8001 python main.py
```

#### Slow First Request

**Problem:** First request takes 30+ seconds

**Explanation:** This is normal! The model needs to:
1. Download (first time only)
2. Load into memory (~5-10 seconds)

**Solutions:**
- Pre-download model during build/startup
- Use smaller model for faster loading
- Keep service running (don't restart frequently)

#### CORS Errors

**Problem:** Browser requests blocked by CORS

**Solutions:**
```bash
# Allow specific origins
CORS_ORIGINS=https://myapp.com,https://api.myapp.com python main.py

# Allow all origins (development only)
CORS_ORIGINS=* python main.py
```

#### Service Unreachable in Docker

**Problem:** Can't connect to service in Docker container

**Solutions:**
```bash
# Ensure correct host binding
docker run -p 8000:8000 -e HOST=0.0.0.0 embedding-service

# Check container logs
docker logs <container-id>

# Test from inside container
docker exec -it <container-id> curl localhost:8000/health
```

### Performance Issues

#### Slow Embedding Generation

**Check:**
1. **Model size:** Larger models are slower
2. **Batch size:** Larger batches are more efficient
3. **Hardware:** GPU vs CPU makes huge difference

**Optimize:**
```python
# Use batch processing
texts = ["text1", "text2", "text3", ...]
response = requests.post(url, json={"texts": texts})  # Faster

# Instead of individual requests
for text in texts:
    response = requests.post(url, json={"texts": text})  # Slower
```

#### High Memory Usage

**Monitor:**
```bash
# Check memory usage
docker stats

# Or inside container
free -h
```

**Reduce:**
- Use smaller model
- Lower MAX_BATCH_SIZE
- Process in smaller batches

### Getting Help

**Check logs:**
```bash
# Local
python main.py  # See console output

# Docker
docker logs <container-id>

# Render
View logs in dashboard
```

**Enable debug mode:**
```python
# In main.py, add:
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Common log messages:**
- `Loading model: ...` - Model is loading (normal)
- `Model loaded successfully` - Ready to serve requests
- `Generated X embeddings in Ys` - Request completed successfully

---

## 📊 Monitoring & Observability

### Health Checks

```bash
# Basic health check
curl http://localhost:8000/health

# Monitor continuously
watch -n 5 'curl -s http://localhost:8000/health | jq'
```

### Metrics

The API returns processing time with each request:

```json
{
  "processing_time": 0.028  // seconds
}
```

### Logging

All requests are logged with:
- Timestamp
- Number of embeddings generated
- Processing time

Example log:
```
2025-12-04 01:23:15 - main - INFO - Generated 3 embeddings in 0.045s
```

---

## License

This project uses open-source libraries. Please check individual library licenses for details.
