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

## API Endpoints

### 📍 Root Endpoint
```bash
GET /
```
Returns API information and links to documentation.

### 🏥 Health Check
```bash
GET /health
```
Returns the health status and model information.

**Response:**
```json
{
  "status": "healthy",
  "model": "all-MiniLM-L6-v2",
  "model_loaded": true
}
```

### ℹ️ Model Information
```bash
GET /model-info
```
Returns detailed information about the loaded model.

**Response:**
```json
{
  "model_name": "all-MiniLM-L6-v2",
  "max_seq_length": 256,
  "embedding_dimension": 384,
  "max_batch_size": 100
}
```

### 🎯 Generate Embeddings
```bash
POST /embed
```

Generate embeddings for single or multiple texts.

**Request Body:**
```json
{
  "texts": "This is a sample text",
  "normalize": true
}
```

Or for batch processing:
```json
{
  "texts": [
    "First text to embed",
    "Second text to embed",
    "Third text to embed"
  ],
  "normalize": true
}
```

**Response:**
```json
{
  "embeddings": [[0.123, -0.456, ...], [0.789, -0.012, ...]],
  "model": "all-MiniLM-L6-v2",
  "dimension": 384,
  "num_embeddings": 2,
  "processing_time": 0.045
}
```

## Usage Examples

### Using cURL

```bash
# Single text embedding
curl -X POST "http://localhost:8000/embed" \
  -H "Content-Type: application/json" \
  -d '{"texts": "Hello, world!", "normalize": true}'

# Batch embedding
curl -X POST "http://localhost:8000/embed" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["First text", "Second text"], "normalize": true}'
```

### Using Python

```python
import requests

# API endpoint
url = "http://localhost:8000/embed"

# Single text
response = requests.post(
    url,
    json={"texts": "Hello, world!", "normalize": True}
)
result = response.json()
print(f"Embedding dimension: {result['dimension']}")
print(f"Processing time: {result['processing_time']}s")

# Batch processing
texts = [
    "First document to embed",
    "Second document to embed",
    "Third document to embed"
]
response = requests.post(
    url,
    json={"texts": texts, "normalize": True}
)
result = response.json()
embeddings = result['embeddings']
print(f"Generated {len(embeddings)} embeddings")
```

### Using JavaScript/TypeScript

```javascript
// Single text embedding
const response = await fetch('http://localhost:8000/embed', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    texts: 'Hello, world!',
    normalize: true
  })
});

const result = await response.json();
console.log(`Embedding dimension: ${result.dimension}`);
console.log(`Processing time: ${result.processing_time}s`);
```

## Available Models

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

- **First Request**: The first request will be slower as the model loads into memory
- **Batch Processing**: Use batch requests for better throughput
- **GPU Support**: If you have a GPU, PyTorch will automatically use it for faster inference
- **Model Caching**: Models are cached locally after first download

## Deployment

### Docker (Recommended)

Create a `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t embedding-service .
docker run -p 8000:8000 embedding-service
```

### Production Server

For production, use Gunicorn with Uvicorn workers:

```bash
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Troubleshooting

### Model Download Issues
If the model fails to download, ensure you have internet connectivity and sufficient disk space.

### Memory Issues
If you encounter memory errors, try:
- Using a smaller model (e.g., `all-MiniLM-L6-v2`)
- Reducing `MAX_BATCH_SIZE`
- Increasing available RAM

### Port Already in Use
Change the port in `.env` or run with a different port:
```bash
PORT=8001 python main.py
```

## License

This project uses open-source libraries. Please check individual library licenses for details.
