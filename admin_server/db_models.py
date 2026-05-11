from sqlalchemy import Column, String, Integer, Text, DateTime, TIMESTAMP, ForeignKey, BigInteger, VARBINARY, func
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime

Base = declarative_base()

class Tenant(Base):
    __tablename__ = "tenants"
    tenantid = Column(String(128), primary_key=True)
    name = Column(String(255), nullable=False)
    status = Column(String(32), default="active", nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp(), nullable=False)
    updated_at = Column(TIMESTAMP, server_default=func.current_timestamp(), onupdate=func.current_timestamp(), nullable=False)

    sites = relationship("Site", back_populates="tenant")

class Site(Base):
    __tablename__ = "sites"
    siteid = Column(String(128), primary_key=True)
    tenantid = Column(String(128), ForeignKey("tenants.tenantid"), nullable=False)
    name = Column(String(255), nullable=False)
    domain = Column(String(255), default="")
    status = Column(String(32), default="active", nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp(), nullable=False)
    updated_at = Column(TIMESTAMP, server_default=func.current_timestamp(), onupdate=func.current_timestamp(), nullable=False)

    tenant = relationship("Tenant", back_populates="sites")
    configs = relationship("ServingConfig", back_populates="site")

class ServingConfig(Base):
    __tablename__ = "serving_configs"
    servingid = Column(String(128), primary_key=True)
    tenantid = Column(String(128), ForeignKey("tenants.tenantid"), nullable=False)
    siteid = Column(String(128), ForeignKey("sites.siteid"), nullable=False)
    status = Column(String(32), default="active", nullable=False)
    origin_allowlist = Column(Text, default="*")
    system_prompt = Column(Text)
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp(), nullable=False)
    updated_at = Column(TIMESTAMP, server_default=func.current_timestamp(), onupdate=func.current_timestamp(), nullable=False)

    site = relationship("Site", back_populates="configs")

class KnowledgeSource(Base):
    __tablename__ = "knowledge_sources"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    tenantid = Column(String(128), nullable=False)
    siteid = Column(String(128), nullable=False)
    servingid = Column(String(128), nullable=False)
    source_type = Column(String(32), nullable=False)
    title = Column(String(255), nullable=False)
    url = Column(Text)
    status = Column(String(32), default="active", nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp(), nullable=False)
    updated_at = Column(TIMESTAMP, server_default=func.current_timestamp(), onupdate=func.current_timestamp(), nullable=False)

class KnowledgeDocument(Base):
    __tablename__ = "knowledge_documents"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    source_id = Column(BigInteger, ForeignKey("knowledge_sources.id"), nullable=False)
    tenantid = Column(String(128), nullable=False)
    siteid = Column(String(128), nullable=False)
    servingid = Column(String(128), nullable=False)
    title = Column(String(512), nullable=False)
    url = Column(Text)
    content = Column(Text, nullable=False)  # mediumtext in SQL, Text is fine here
    status = Column(String(32), default="active", nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp(), nullable=False)
    updated_at = Column(TIMESTAMP, server_default=func.current_timestamp(), onupdate=func.current_timestamp(), nullable=False)

class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    document_id = Column(BigInteger, ForeignKey("knowledge_documents.id"), nullable=False)
    tenantid = Column(String(128), nullable=False)
    siteid = Column(String(128), nullable=False)
    servingid = Column(String(128), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding_model = Column(String(128), nullable=False)
    embedding_dim = Column(Integer, nullable=False)
    embedding = Column(VARBINARY(4096), nullable=False)
    status = Column(String(32), default="active", nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp(), nullable=False)

class CrawlJob(Base):
    __tablename__ = "crawl_jobs"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    tenantid = Column(String(128), nullable=False)
    siteid = Column(String(128), nullable=False)
    servingid = Column(String(128), nullable=False)
    start_url = Column(Text, nullable=False)
    status = Column(String(32), default="queued", nullable=False)
    max_depth = Column(Integer, default=1, nullable=False)
    max_pages = Column(Integer, default=10, nullable=False)
    pages_indexed = Column(Integer, default=0, nullable=False)
    error = Column(Text)
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp(), nullable=False)
    finished_at = Column(DateTime)
