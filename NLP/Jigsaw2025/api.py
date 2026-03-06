"""
FastAPI service for rule violation detection.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import numpy as np
import logging

from src.config.model_config import Config
from src.models.embedding_model import EmbeddingModel
from src.inference.predictor import ViolationPredictor
from src.data.preprocessor import TextPreprocessor
from src.utils.logging_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Rule Violation Detection API",
    description="API for detecting rule violations in text",
    version="0.1.0"
)

# Global model and predictor (loaded on startup)
model_wrapper = None
predictor = None
preprocessor = None
config = None


class PredictionRequest(BaseModel):
    """Request model for predictions."""
    text: str = Field(..., description="Text to analyze")
    rule: str = Field(..., description="Rule to check against")
    positive_examples: List[str] = Field(..., description="Examples of violations")
    negative_examples: List[str] = Field(..., description="Examples of compliance")


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    text: str
    rule: str
    violation_score: float
    is_violation: bool
    confidence: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    version: str


@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global model_wrapper, predictor, preprocessor, config
    
    try:
        logger.info("Loading model and initializing components...")
        config = Config()
        
        # Load model
        model_wrapper = EmbeddingModel(
            model_path=f"{config.data.output_dir}/final",
            max_seq_length=config.model.max_seq_length,
            use_fp16=config.model.use_fp16
        )
        model_wrapper.load_model()
        
        # Initialize predictor and preprocessor
        predictor = ViolationPredictor(config.inference.distance_metric)
        preprocessor = TextPreprocessor()
        
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model_wrapper else "unhealthy",
        model_loaded=model_wrapper is not None,
        version="0.1.0"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict rule violation."""
    if not model_wrapper:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Preprocess text
        clean_text = preprocessor.clean_text(request.text)
        clean_positives = [preprocessor.clean_text(ex) for ex in request.positive_examples]
        clean_negatives = [preprocessor.clean_text(ex) for ex in request.negative_examples]
        
        # Get embeddings
        all_texts = [clean_text] + clean_positives + clean_negatives
        embeddings = model_wrapper.encode(all_texts, batch_size=32)
        
        # Split embeddings
        text_emb = embeddings[0]
        pos_embs = embeddings[1:1+len(clean_positives)]
        neg_embs = embeddings[1+len(clean_positives):]
        
        # Calculate centroids
        pos_centroid = np.mean(pos_embs, axis=0)
        neg_centroid = np.mean(neg_embs, axis=0)
        
        # Normalize
        pos_centroid /= np.linalg.norm(pos_centroid)
        neg_centroid /= np.linalg.norm(neg_centroid)
        
        # Calculate distances
        pos_dist = np.linalg.norm(text_emb - pos_centroid)
        neg_dist = np.linalg.norm(text_emb - neg_centroid)
        
        # Score: positive if closer to violation examples
        violation_score = float(neg_dist - pos_dist)
        is_violation = violation_score > 0
        
        # Confidence: normalized distance difference
        max_dist = max(pos_dist, neg_dist)
        confidence = float(abs(violation_score) / max_dist) if max_dist > 0 else 0.0
        
        return PredictionResponse(
            text=request.text,
            rule=request.rule,
            violation_score=violation_score,
            is_violation=is_violation,
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict")
async def batch_predict(requests: List[PredictionRequest]):
    """Batch prediction endpoint."""
    if not model_wrapper:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    for req in requests:
        try:
            result = await predict(req)
            results.append(result)
        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            results.append({"error": str(e)})
    
    return results


@app.get("/metrics")
async def get_metrics():
    """Get model metrics and statistics."""
    if not model_wrapper:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_path": config.data.output_dir,
        "max_seq_length": config.model.max_seq_length,
        "embedding_dim": config.model.embedding_dim,
        "distance_metric": config.inference.distance_metric,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)