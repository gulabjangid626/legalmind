from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["embeddings"])

# Load the same model used for Pinecone indexing
_model = None
_cross_encoder = None

def get_model():
    global _model
    if _model is None:
        logger.info("Loading embedding model: all-MiniLM-L6-v2")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedding model loaded successfully")
    return _model

def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        logger.info("Loading Cross-Encoder model: cross-encoder/ms-marco-MiniLM-L-6-v2")
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("Cross-Encoder model loaded successfully")
    return _cross_encoder

class EmbeddingRequest(BaseModel):
    text: str

class EmbeddingResponse(BaseModel):
    embedding: list[float]
    dimension: int

class RerankRequest(BaseModel):
    query: str
    documents: list[str]
    top_k: int = 5

class RerankResponse(BaseModel):
    documents: list[str]

@router.post("/embed", response_model=EmbeddingResponse)
async def generate_embedding(req: EmbeddingRequest):
    """Generate embedding using the same model used for Pinecone indexing"""
    model = get_model()
    embedding = model.encode([req.text], convert_to_numpy=True)[0]
    return EmbeddingResponse(
        embedding=embedding.tolist(),
        dimension=len(embedding)
    )

@router.post("/rerank", response_model=RerankResponse)
async def rerank_documents(req: RerankRequest):
    """Rerank documents against a query using a Cross-Encoder"""
    if not req.documents:
        return RerankResponse(documents=[])
        
    try:
        model = get_cross_encoder()
        # Create (query, document) pairs
        pairs = [[req.query, doc] for doc in req.documents]
        
        # Predict relevance scores
        scores = model.predict(pairs)
        
        # Sort documents by score descending
        doc_score_pairs = list(zip(req.documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Select top_k documents
        top_docs = [doc for doc, score in doc_score_pairs[:req.top_k]]
        
        return RerankResponse(documents=top_docs)
    except Exception as e:
        logger.error(f"Reranking failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Reranking failed")
