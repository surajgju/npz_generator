import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")

# Engine/session are created lazily on first call so the server starts
# successfully even when no database is available.
_engine = None
_SessionLocal = None


def _get_session():
    """Return a SQLAlchemy session, creating the engine lazily on first call.

    Returns None if DATABASE_URL is not configured or the DB is unreachable.
    """
    global _engine, _SessionLocal
    if DATABASE_URL is None:
        return None
    if _engine is None:
        try:
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            _engine = create_engine(
                DATABASE_URL,
                pool_pre_ping=True,       # detect stale connections
                connect_args={"connect_timeout": 3},
            )
            _SessionLocal = sessionmaker(bind=_engine)
        except Exception as exc:
            logger.warning("tenant_service: could not initialise DB engine: %s", exc)
            return None
    try:
        return _SessionLocal()
    except Exception as exc:
        logger.warning("tenant_service: could not open DB session: %s", exc)
        return None


def resolve_tenant_config(servingid: str) -> Optional[Dict[str, Any]]:
    """Resolve system prompt and other settings for a specific SID.

    Returns None if the database is not configured, unreachable, or if the
    given servingid does not exist.  The caller should fall back to a safe
    default prompt in that case.
    """
    db = _get_session()
    if db is None:
        logger.debug(
            "tenant_service: no DB session available, returning None for sid=%s", servingid
        )
        return None
    try:
        from admin_server.db_models import ServingConfig
        config = db.query(ServingConfig).filter(ServingConfig.servingid == servingid).first()
        if config:
            return {
                "system_prompt": config.system_prompt,
                "tenantid": config.tenantid,
                "siteid": config.siteid,
            }
        return None
    except Exception as exc:
        logger.warning(
            "tenant_service: DB query failed for sid=%s: %s", servingid, exc
        )
        return None
    finally:
        try:
            db.close()
        except Exception:
            pass
