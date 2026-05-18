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

