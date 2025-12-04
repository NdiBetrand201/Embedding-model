# 🚀 Deploying to Render

This guide will help you deploy the Embedding Service API to Render.com.

## Prerequisites

- A [Render](https://render.com) account (free tier available)
- Git repository with your code (GitHub, GitLab, or Bitbucket)

## Quick Deploy

### Option 1: Deploy from GitHub (Recommended)

1. **Push your code to GitHub:**
   ```bash
   cd Embedding-model
   git init
   git add .
   git commit -m "Initial commit: Embedding service"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/embedding-service.git
   git push -u origin main
   ```

2. **Create a new Web Service on Render:**
   - Go to [Render Dashboard](https://dashboard.render.com/)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the repository containing your embedding service

3. **Configure the service:**
   - **Name:** `embedding-service` (or your preferred name)
   - **Region:** Choose closest to your users
   - **Branch:** `main`
   - **Root Directory:** `Embedding-model` (if in subdirectory) or leave blank
   - **Runtime:** `Docker`
   - **Instance Type:** Choose based on your needs
     - Free tier: Limited resources, sleeps after inactivity
     - Starter ($7/month): 512 MB RAM, no sleep
     - Standard ($25/month): 2 GB RAM, better performance

4. **Set Environment Variables:**
   Click "Advanced" and add:
   ```
   MODEL_NAME=all-MiniLM-L6-v2
   CACHE_DIR=/app/model_cache
   HOST=0.0.0.0
   PORT=8000
   MAX_BATCH_SIZE=100
   CORS_ORIGINS=*
   ```

5. **Deploy:**
   - Click "Create Web Service"
   - Render will automatically build and deploy your service
   - First deployment takes 5-10 minutes (downloading model)

### Option 2: Manual Dockerfile Deploy

If you prefer not to use Git:

1. Create a new Web Service on Render
2. Choose "Deploy an existing image from a registry"
3. Build and push your Docker image to Docker Hub first
4. Enter your image URL

## Configuration

### Environment Variables

Set these in the Render dashboard under "Environment":

| Variable | Value | Description |
|----------|-------|-------------|
| `MODEL_NAME` | `all-MiniLM-L6-v2` | Embedding model to use |
| `CACHE_DIR` | `/app/model_cache` | Model cache directory |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port (Render auto-assigns) |
| `MAX_BATCH_SIZE` | `100` | Max texts per request |
| `CORS_ORIGINS` | `*` | Allowed CORS origins |

### Choosing the Right Instance Type

**Free Tier:**
- ✅ Good for: Testing, development, low-traffic apps
- ❌ Limitations: Sleeps after 15 min inactivity, 512 MB RAM
- 💰 Cost: Free

**Starter ($7/month):**
- ✅ Good for: Small production apps, personal projects
- ✅ Features: No sleep, 512 MB RAM, always available
- 💰 Cost: $7/month

**Standard ($25/month):**
- ✅ Good for: Production apps with moderate traffic
- ✅ Features: 2 GB RAM, better performance
- 💰 Cost: $25/month

### Model Selection for Render

Choose based on your instance type:

**Free/Starter (512 MB RAM):**
```
MODEL_NAME=all-MiniLM-L6-v2  # Smallest, fastest
```

**Standard (2 GB RAM):**
```
MODEL_NAME=all-mpnet-base-v2  # Better quality
```

## Accessing Your Service

Once deployed, Render provides a URL like:
```
https://embedding-service-xxxx.onrender.com
```

Test your deployment:
```bash
# Health check
curl https://your-service.onrender.com/health

# Get embeddings
curl -X POST "https://your-service.onrender.com/embed" \
  -H "Content-Type: application/json" \
  -d '{"texts": "Hello from Render!", "normalize": true}'
```

## Monitoring

### View Logs

1. Go to your service in Render dashboard
2. Click "Logs" tab
3. Monitor startup and requests in real-time

### Health Checks

Render automatically monitors your service health:
- Health check path: `/health`
- Render pings this endpoint regularly
- Service restarts automatically if unhealthy

## Troubleshooting

### Service Won't Start

**Check logs for:**
```
Model loading errors → Increase instance size
Port binding errors → Ensure PORT=8000
Memory errors → Upgrade to larger instance
```

### Slow First Request

The first request after deployment or sleep is slow because:
1. Model needs to load into memory (~5-10 seconds)
2. Solution: Use Starter tier or higher (no sleep)

### Out of Memory

If you see OOM errors:
1. Use a smaller model: `MODEL_NAME=all-MiniLM-L6-v2`
2. Reduce batch size: `MAX_BATCH_SIZE=50`
3. Upgrade to Standard instance (2 GB RAM)

### Service Sleeps (Free Tier)

Free tier sleeps after 15 minutes of inactivity:
- First request after sleep takes ~30 seconds
- Solution: Upgrade to Starter ($7/month) for always-on service

## Updating Your Service

### Automatic Deploys

Enable auto-deploy in Render:
1. Go to service settings
2. Enable "Auto-Deploy"
3. Every push to `main` branch triggers deployment

### Manual Deploy

1. Make changes to your code
2. Push to GitHub
3. Go to Render dashboard
4. Click "Manual Deploy" → "Deploy latest commit"

## Cost Optimization

### Tips to Reduce Costs

1. **Use Free Tier for Development:**
   - Test and develop on free tier
   - Deploy to paid tier for production

2. **Choose Right Model:**
   - Smaller models = less memory = cheaper instance
   - `all-MiniLM-L6-v2` works well on Starter tier

3. **Optimize Batch Size:**
   - Lower `MAX_BATCH_SIZE` reduces memory usage
   - Balance between throughput and memory

## Integration Example

Update your Firebase Cloud Function to use Render deployment:

```python
# In your functions/main.py
import requests

# Replace with your Render URL
EMBEDDING_SERVICE_URL = "https://your-service.onrender.com/embed"

def get_embeddings_render(texts):
    """Get embeddings from Render-hosted service"""
    response = requests.post(
        EMBEDDING_SERVICE_URL,
        json={"texts": texts, "normalize": True},
        timeout=30  # Increase timeout for first request
    )
    response.raise_for_status()
    return response.json()["embeddings"]

# Use in your pipeline
embeddings = get_embeddings_render(batch_chunks)
```

## Security Best Practices

### 1. Restrict CORS Origins

Instead of `CORS_ORIGINS=*`, specify your domains:
```
CORS_ORIGINS=https://myapp.com,https://api.myapp.com
```

### 2. Add Authentication (Optional)

For production, consider adding API key authentication:
- Use Render environment variables for API keys
- Implement middleware in FastAPI
- Validate keys on each request

### 3. Rate Limiting

Consider adding rate limiting to prevent abuse:
- Use FastAPI middleware
- Limit requests per IP/API key
- Protect against DDoS

## Support

- **Render Docs:** https://render.com/docs
- **Render Community:** https://community.render.com
- **Service Issues:** Check Render status page

## Next Steps

1. ✅ Deploy to Render
2. ✅ Test all endpoints
3. ✅ Monitor logs and performance
4. ✅ Set up auto-deploy
5. ✅ Configure custom domain (optional)
6. ✅ Add monitoring/alerting
7. ✅ Implement authentication (production)

---

**Your embedding service is now live and ready to use! 🎉**
