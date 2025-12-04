"""
FastAPI Embedding Service using Sentence Transformers
Provides REST API endpoints for generating text embeddings
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from sentence_transformers import SentenceTransformer
from typing import List, Union
import logging
import time
from config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Embedding Service API",
    description="Generate text embeddings using Sentence Transformers",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model: SentenceTransformer = None


# Pydantic models for request/response
class EmbedRequest(BaseModel):
    """Request model for embedding generation"""
    texts: Union[str, List[str]] = Field(
        ...,
        description="Single text string or list of texts to embed"
    )
    normalize: bool = Field(
        default=True,
        description="Whether to normalize embeddings to unit length"
    )
    
    @field_validator('texts')
    @classmethod
    def validate_texts(cls, v):
        """Validate texts input"""
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("Text cannot be empty")
            return [v]
        elif isinstance(v, list):
            if not v:
                raise ValueError("Text list cannot be empty")
            if len(v) > settings.MAX_BATCH_SIZE:
                raise ValueError(f"Batch size cannot exceed {settings.MAX_BATCH_SIZE}")
            for text in v:
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("All texts must be non-empty strings")
            return v
        else:
            raise ValueError("Texts must be a string or list of strings")


class EmbedResponse(BaseModel):
    """Response model for embedding generation"""
    embeddings: List[List[float]] = Field(
        ...,
        description="List of embedding vectors"
    )
    model: str = Field(
        ...,
        description="Name of the model used"
    )
    dimension: int = Field(
        ...,
        description="Dimension of each embedding vector"
    )
    num_embeddings: int = Field(
        ...,
        description="Number of embeddings generated"
    )
    processing_time: float = Field(
        ...,
        description="Processing time in seconds"
    )


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    """Response model for model information"""
    model_name: str
    max_seq_length: int
    embedding_dimension: int
    max_batch_size: int


@app.on_event("startup")
async def startup_event():
    """Load the model on startup"""
    global model
    try:
        logger.info(f"Loading model: {settings.MODEL_NAME}")
        start_time = time.time()
        
        model = SentenceTransformer(
            settings.MODEL_NAME,
            cache_folder=settings.CACHE_DIR
        )
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        logger.info(f"Model dimension: {model.get_sentence_embedding_dimension()}")
        logger.info(f"Max sequence length: {model.max_seq_length}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Embedding Service API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model=settings.MODEL_NAME,
        model_loaded=model is not None
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Info"])
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfoResponse(
        model_name=settings.MODEL_NAME,
        max_seq_length=model.max_seq_length,
        embedding_dimension=model.get_sentence_embedding_dimension(),
        max_batch_size=settings.MAX_BATCH_SIZE
    )


@app.post("/embed", response_model=EmbedResponse, tags=["Embeddings"])
async def generate_embeddings(request: EmbedRequest):
    """
    Generate embeddings for input text(s)
    
    Args:
        request: EmbedRequest containing texts and options
        
    Returns:
        EmbedResponse with embeddings and metadata
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = time.time()
        
        # Generate embeddings
        embeddings = model.encode(
            request.texts,
            normalize_embeddings=request.normalize,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # Convert to list format
        embeddings_list = embeddings.tolist()
        
        processing_time = time.time() - start_time
        
        logger.info(
            f"Generated {len(embeddings_list)} embeddings in {processing_time:.3f}s"
        )
        
        return EmbedResponse(
            embeddings=embeddings_list,
            model=settings.MODEL_NAME,
            dimension=len(embeddings_list[0]),
            num_embeddings=len(embeddings_list),
            processing_time=round(processing_time, 3)
        )
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD
    )
