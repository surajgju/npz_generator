import os
import uuid
import asyncio
from typing import List, Optional
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
from .db_models import Base, Tenant, Site, ServingConfig, CrawlJob, KnowledgeSource
from .crawler_engine import run_crawler
from .rag_pipeline import run_rag_pipeline

# --- Configuration ---
DATABASE_URL = os.getenv("DATABASE_URL", "mysql+pymysql://root@localhost/mhmtrag")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

app = FastAPI(title="MHMTRAG Admin API")

@app.on_event("startup")
async def startup_event():
    # Start background workers
    asyncio.create_task(run_crawler(DATABASE_URL))
    asyncio.create_task(run_rag_pipeline(DATABASE_URL))

# --- Dependency ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Pydantic Schemas ---
class TenantCreate(BaseModel):
    tenantid: str
    name: str

class SiteCreate(BaseModel):
    siteid: str
    tenantid: str
    name: str
    domain: str = ""

class ConfigCreate(BaseModel):
    servingid: str
    tenantid: str
    siteid: str
    system_prompt: Optional[str] = "You are a concise helpful voice assistant."

class CrawlRequest(BaseModel):
    servingid: str
    start_url: str
    max_depth: int = 1
    max_pages: int = 10

# --- Routes ---

@app.post("/tenants")
def create_tenant(data: TenantCreate, db: Session = Depends(get_db)):
    db_tenant = Tenant(tenantid=data.tenantid, name=data.name)
    db.add(db_tenant)
    db.commit()
    db.refresh(db_tenant)
    return db_tenant

@app.get("/tenants")
def list_tenants(db: Session = Depends(get_db)):
    return db.query(Tenant).all()

@app.post("/sites")
def create_site(data: SiteCreate, db: Session = Depends(get_db)):
    db_site = Site(siteid=data.siteid, tenantid=data.tenantid, name=data.name, domain=data.domain)
    db.add(db_site)
    db.commit()
    db.refresh(db_site)
    return db_site

@app.post("/configs")
def create_config(data: ConfigCreate, db: Session = Depends(get_db)):
    db_config = ServingConfig(
        servingid=data.servingid,
        tenantid=data.tenantid,
        siteid=data.siteid,
        system_prompt=data.system_prompt
    )
    db.add(db_config)
    db.commit()
    db.refresh(db_config)
    return db_config

@app.get("/configs/{servingid}")
def get_config(servingid: str, db: Session = Depends(get_db)):
    config = db.query(ServingConfig).filter(ServingConfig.servingid == servingid).first()
    if not config:
        raise HTTPException(status_code=404, detail="Config not found")
    return config

@app.post("/crawl")
async def start_crawl(data: CrawlRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    # Verify SID exists
    config = db.query(ServingConfig).filter(ServingConfig.servingid == data.servingid).first()
    if not config:
        raise HTTPException(status_code=404, detail="Serving ID not found")

    # Create job
    job = CrawlJob(
        tenantid=config.tenantid,
        siteid=config.siteid,
        servingid=data.servingid,
        start_url=data.start_url,
        max_depth=data.max_depth,
        max_pages=data.max_pages,
        status="queued"
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # In a real app, a background worker would pick this up
    # For now, we'll just return the job ID
    return {"job_id": job.id, "status": job.status}

@app.get("/jobs/{job_id}")
def get_job_status(job_id: int, db: Session = Depends(get_db)):
    job = db.query(CrawlJob).filter(CrawlJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
