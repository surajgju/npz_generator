import os
import logging
import numpy as np
from typing import Optional, List
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from admin_server.db_models import ServingConfig, KnowledgeChunk
from google.genai import Client

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "mysql+pymysql://root@localhost/mhmtrag")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

def resolve_tenant_config(servingid: str):
    """Resolves system prompt and other settings for a specific SID."""
    db = SessionLocal()
    try:
        config = db.query(ServingConfig).filter(ServingConfig.servingid == servingid).first()
        if config:
            return {
                "system_prompt": config.system_prompt,
                "tenantid": config.tenantid,
                "siteid": config.siteid
            }
        return None
    finally:
        db.close()

async def get_rag_context(servingid: str, query: str, top_k: int = 3):
    """Retrieves relevant knowledge chunks for a query (RAG)."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return ""

    client = Client(api_key=api_key)
    
    # 1. Embed query
    try:
        response = client.models.embed_content(
            model="models/text-embedding-004",
            contents=query
        )
        query_embedding = np.array(response.embeddings[0].values, dtype=np.float32)
    except Exception as e:
        logger.error(f"Failed to embed query: {e}")
        return ""

    # 2. Vector search (Simple Cosine Similarity in SQL/Memory)
    db = SessionLocal()
    try:
        chunks = db.query(KnowledgeChunk).filter(KnowledgeChunk.servingid == servingid).all()
        if not chunks:
            return ""

        # Score chunks
        results = []
        for chunk in chunks:
            chunk_emb = np.frombuffer(chunk.embedding, dtype=np.float32)
            # Cosine similarity
            score = np.dot(query_embedding, chunk_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(chunk_emb))
            results.append((score, chunk.content))
        
        # Sort and take top_k
        results.sort(key=lambda x: x[0], reverse=True)
        top_chunks = [c[1] for c in results[:top_k]]
        
        return "\n---\n".join(top_chunks)
    finally:
        db.close()
