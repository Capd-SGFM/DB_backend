from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
from pydantic import BaseModel
from loguru import logger

try:
    from models import CryptoInfo
except ImportError:
    logger.error("CryptoInfo 모델을 models.py에서 찾을 수 없습니다.")
    CryptoInfo = None

from db_module.connect_sqlalchemy_engine import get_async_db

router = APIRouter(prefix="/symbols", tags=["Symbols"])


class SymbolsResponse(BaseModel):
    """심볼 목록 응답 스키마"""

    symbols: List[str]


@router.get("/all", response_model=SymbolsResponse)
async def get_all_symbols(db: AsyncSession = Depends(get_async_db)):
    """
    metadata.crypto_info 테이블에 등록된 모든 심볼 목록을 반환합니다.
    """
    if CryptoInfo is None:
        raise HTTPException(
            status_code=500, detail="CryptoInfo 모델이 서버에 로드되지 않았습니다."
        )

    try:
        # pair 정보가 있는 심볼만 조회
        stmt = (
            select(CryptoInfo.symbol)
            .where(CryptoInfo.pair.isnot(None))
            .order_by(CryptoInfo.symbol)
        )
        result = await db.execute(stmt)
        # 튜플 리스트 [(symbol,), ...]를 리스트 [symbol, ...]로 변환
        symbols_list = [row[0] for row in result.all()]

        return SymbolsResponse(symbols=symbols_list)
    except Exception as e:
        logger.exception(f"심볼 조회 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"심볼 조회 중 오류 발생: {e}")
