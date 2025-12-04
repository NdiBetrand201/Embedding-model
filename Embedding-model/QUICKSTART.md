# 🚀 Quick Start Guide

## What You Have

A production-ready FastAPI embedding service using Sentence Transformers, fully dockerized and ready to deploy to Render.

## Files Created

```
Embedding-model/
├── main.py                 # FastAPI application
├── config.py               # Configuration management
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker container definition
├── docker-compose.yml     # Docker Compose setup
├── .dockerignore          # Docker build exclusions
├── .env.example           # Environment template
├── .gitignore             # Git exclusions
├── render.yaml            # Render configuration
├── README.md              # Full documentation
├── RENDER_DEPLOY.md       # Render deployment guide
└── test_payload.json      # Test data
```

## Run Locally

```bash
# Option 1: Python (with virtual environment)
cd Embedding-model
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python main.py

# Option 2: Docker
docker build -t embedding-service .
docker run -p 8000:8000 embedding-service

# Option 3: Docker Compose
docker-compose up -d
```

Access at: http://localhost:8000

## Deploy to Render

1. Push code to GitHub
2. Go to [Render Dashboard](https://dashboard.render.com/)
3. New Web Service → Connect GitHub repo
4. Select "Docker" runtime
5. Deploy!

**See [RENDER_DEPLOY.md](./RENDER_DEPLOY.md) for detailed instructions**

## Test the API

```bash
# Health check
curl http://localhost:8000/health

# Get embedding
curl -X POST "http://localhost:8000/embed" \
  -H "Content-Type: application/json" \
  -d '{"texts": "Hello, world!", "normalize": true}'
```

## API Endpoints

- `GET /` - API info
- `GET /health` - Health check
- `GET /model-info` - Model details
- `POST /embed` - Generate embeddings
- `GET /docs` - Interactive API docs

## Configuration

Set environment variables in `.env` file:

```bash
MODEL_NAME=all-MiniLM-L6-v2
CACHE_DIR=./model_cache
HOST=0.0.0.0
PORT=8000
MAX_BATCH_SIZE=100
CORS_ORIGINS=*
```

## Documentation

- **README.md** - Complete API documentation
- **RENDER_DEPLOY.md** - Render deployment guide
- **API Docs** - http://localhost:8000/docs

## Performance

- Single text: ~28ms
- Batch (3 texts): ~30ms
- Embedding dimension: 384
- Model: all-MiniLM-L6-v2

## Next Steps

1. ✅ Test locally
2. ✅ Deploy to Render
3. ✅ Integrate with your RAG pipeline
4. ✅ Monitor with `/health` endpoint

---

**Everything is ready to go! 🎉**
