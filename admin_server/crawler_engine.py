import asyncio
import logging
import requests
from bs4 import BeautifulSoup
from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import sessionmaker
from .db_models import CrawlJob, KnowledgeDocument, KnowledgeSource, ServingConfig

logger = logging.getLogger(__name__)

async def run_crawler(db_url: str):
    """Background worker that polls for queued crawl jobs."""
    engine = create_engine(db_url)
    SessionLocal = sessionmaker(bind=engine)

    while True:
        db = SessionLocal()
        try:
            # Pick a queued job
            job = db.query(CrawlJob).filter(CrawlJob.status == "queued").first()
            if job:
                logger.info(f"Starting crawl job {job.id} for SID {job.servingid}")
                job.status = "processing"
                db.commit()

                try:
                    await _perform_crawl(db, job)
                    job.status = "completed"
                except Exception as e:
                    logger.error(f"Crawl job {job.id} failed: {e}")
                    job.status = "failed"
                    job.error = str(e)
                
                db.commit()
        except Exception as e:
            logger.error(f"Crawler engine error: {e}")
        finally:
            db.close()
        
        await asyncio.sleep(5)

async def _perform_crawl(db, job: CrawlJob):
    """Simple single-page crawler (can be expanded to depth later)."""
    # 1. Ensure a KnowledgeSource exists for this SID/URL
    source = db.query(KnowledgeSource).filter(
        KnowledgeSource.servingid == job.servingid,
        KnowledgeSource.url == job.start_url
    ).first()
    
    if not source:
        source = KnowledgeSource(
            tenantid=job.tenantid,
            siteid=job.siteid,
            servingid=job.servingid,
            source_type="website",
            title=f"Crawl: {job.start_url}",
            url=job.start_url
        )
        db.add(source)
        db.commit()
        db.refresh(source)

    # 2. Fetch and parse
    response = requests.get(job.start_url, timeout=10)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Remove script and style elements
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()

    text = soup.get_text(separator=" ", strip=True)
    title = soup.title.string if soup.title else job.start_url

    # 3. Save as KnowledgeDocument
    doc = KnowledgeDocument(
        source_id=source.id,
        tenantid=job.tenantid,
        siteid=job.siteid,
        servingid=job.servingid,
        title=title,
        url=job.start_url,
        content=text
    )
    db.add(doc)
    job.pages_indexed = 1
    db.commit()
    
    logger.info(f"Crawl completed for {job.start_url}. Document ID: {doc.id}")
