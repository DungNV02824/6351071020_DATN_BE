from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from db.base import engine
from typing import Optional

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
    
)

def set_tenant_context(db_session, tenant_id: int):
    try:
        if tenant_id is None:
            raise ValueError("tenant_id cannot be None")

        tenant_id_int = int(tenant_id)  # ép kiểu chắc chắn

        db_session.execute(
            text("SELECT set_config('app.current_tenant', :tenant_id, false)"),
            {"tenant_id": str(tenant_id_int)},
        )
        db_session.execute(
            text("SELECT set_config('app.current_tenant_id', :tenant_id, false)"),
            {"tenant_id": str(tenant_id_int)},
        )

        print(f"✓ PostgreSQL RLS context set: app.current_tenant = {tenant_id_int}")

    except Exception as e:
        print(f"⚠️ Failed to set RLS context: {e}")
        raise  # ❗ QUAN TRỌNG: phải raise để tránh chạy sai tenant

def get_db():
    """
    Get database session.
    Note: Call set_tenant_context() immediately after getting this session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_db_with_tenant(tenant_id: int):
    """
    Get database session with RLS context pre-configured.
    
    Args:
        tenant_id: The tenant ID for RLS context
    
    Returns:
        Configured SQLAlchemy session
    """
    db = SessionLocal()
    set_tenant_context(db, tenant_id)
    return db
