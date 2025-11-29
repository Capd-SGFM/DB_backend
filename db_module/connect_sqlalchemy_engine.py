import os
from urllib.parse import quote_plus
from typing import AsyncIterator, Iterator

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

load_dotenv()

# .env 값 로드
DB_USER = os.getenv("DB_USER", "")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "")
_escaped_pw = quote_plus(DB_PASSWORD or "")

SYNC_URL = (
    f"postgresql+psycopg2://{DB_USER}:{_escaped_pw}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)
ASYNC_URL = (
    f"postgresql+asyncpg://{DB_USER}:{_escaped_pw}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


class DBConnectionManager:
    """싱글톤 성격의 엔진 보관소"""

    _sync_engine = None
    _async_engine = None

    @classmethod
    def get_sync_engine(cls):
        if cls._sync_engine is None:
            cls._sync_engine = create_engine(
                SYNC_URL,
                pool_pre_ping=True,      # Connection health check
                pool_size=20,             # 🚀 Optimized: Base pool size
                max_overflow=40,          # 🚀 Optimized: Max extra connections
                pool_recycle=1800,        # Recycle connections every 30 min
                echo=False,               # Disable SQL logging for performance
            )
        return cls._sync_engine

    @classmethod
    def get_async_engine(cls):
        if cls._async_engine is None:
            cls._async_engine = create_async_engine(
                ASYNC_URL,
                pool_pre_ping=True,
            )
        return cls._async_engine


# 세션팩토리
SyncSessionLocal = sessionmaker(
    bind=DBConnectionManager.get_sync_engine(),
    autocommit=False,
    autoflush=False,
)

AsyncSessionLocal = sessionmaker(
    bind=DBConnectionManager.get_async_engine(),
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


# FastAPI 의존성
def get_sync_db() -> Iterator:
    db = SyncSessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_async_db() -> AsyncIterator[AsyncSession]:
    async with AsyncSessionLocal() as session:
        yield session
