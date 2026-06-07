import os
import logging
import asyncio
import numpy as np
from typing import List
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .db_models import KnowledgeDocument, KnowledgeChunk
from google.genai import Client

logger = logging.getLogger(__name__)

async def run_rag_pipeline(db_url: str):
    """Background worker that chunks documents and generates embeddings."""
    engine = create_engine(db_url)
    SessionLocal = sessionmaker(bind=engine)
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not set. RAG pipeline disabled.")
        return

    client = Client(api_key=api_key)

    while True:
        db = SessionLocal()
        try:
            # Find documents that don't have chunks yet
            # (Simple check: no KnowledgeChunk with document_id exists)
            doc = db.query(KnowledgeDocument).outerjoin(
                KnowledgeChunk, KnowledgeDocument.id == KnowledgeChunk.document_id
            ).filter(KnowledgeChunk.id == None, KnowledgeDocument.status == "active").first()

            if doc:
                logger.info(f"Processing RAG for document {doc.id}")
                await _process_document(db, doc, client)
        except Exception as e:
            logger.error(f"RAG pipeline error: {e}")
        finally:
            db.close()
        
        await asyncio.sleep(10)

async def _process_document(db, doc: KnowledgeDocument, client: Client):
    """Chunks text and generates embeddings using Gemini."""
    content = doc.content
    # Simple chunking by length (approx 1000 chars)
    chunk_size = 1000
    chunks = [content[i : i + chunk_size] for i in range(0, len(content), chunk_size)]
    
    model = "models/text-embedding-004"
    
    for i, text in enumerate(chunks):
        try:
            # Generate embedding
            response = client.models.embed_content(
                model=model,
                contents=text
            )
            embedding = response.embeddings[0].values
            
            # Save chunk
            chunk = KnowledgeChunk(
                document_id=doc.id,
                tenantid=doc.tenantid,
                siteid=doc.siteid,
                servingid=doc.servingid,
                chunk_index=i,
                content=text,
                embedding_model=model,
                embedding_dim=len(embedding),
                embedding=np.array(embedding, dtype=np.float32).tobytes()
            )
            db.add(chunk)
            db.commit()
        except Exception as e:
            logger.error(f"Failed to embed chunk {i} for doc {doc.id}: {e}")
            continue

    logger.info(f"RAG processing complete for doc {doc.id}. Total chunks: {len(chunks)}")
