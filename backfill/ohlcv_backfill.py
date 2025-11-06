from fastapi import APIRouter, HTTPException, Depends
from celery.result import AsyncResult
from typing import List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from loguru import logger
from .schemas import BackfillResponse, TaskInfo, TaskStatusResponse
from celery_task.rest_api_task import backfill_symbol_interval
from celery_task import celery_app
from db_module.connect_sqlalchemy_engine import get_async_db
from models import CryptoInfo, OHLCV_MODELS


router = APIRouter(prefix="/ohlcv", tags=["OHLCV"])


@router.post("/backfill", response_model=BackfillResponse)
async def start_backfill(db: AsyncSession = Depends(get_async_db)):
    """
    DB에 등록된 '모든' 심볼과 '모든' 인터벌에 대해 백필 작업을 시작합니다.
    """
    tasks_started = []

    try:
        # 1. DB에서 심볼과 페어 목록을 가져옵니다.
        stmt = select(CryptoInfo.symbol, CryptoInfo.pair).where(
            CryptoInfo.pair.isnot(None)
        )
        result = await db.execute(stmt)
        all_symbols = result.all()  # [(symbol, pair), ...]

        # 2. 코드에 정의된 인터벌 목록을 가져옵니다.
        all_intervals = list(OHLCV_MODELS.keys())

        if not all_symbols:
            raise HTTPException(
                status_code=404, detail="metadata.crypto_info에 등록된 심볼이 없습니다."
            )

        logger.info(
            f"총 {len(all_symbols)}개 심볼, {len(all_intervals)}개 인터벌에 대해 백필 시작."
        )

        for symbol_row in all_symbols:
            symbol = symbol_row.symbol
            pair = symbol_row.pair

            for interval in all_intervals:
                # Celery 작업을 비동기적으로 호출
                task = backfill_symbol_interval.delay(
                    symbol=symbol, pair=pair, interval=interval
                )
                tasks_started.append(
                    TaskInfo(task_id=task.id, symbol=symbol, interval=interval)
                )

        return BackfillResponse(
            message=f"총 {len(tasks_started)}개의 백필 작업을 시작했습니다.",
            tasks=tasks_started,
        )

    except Exception as e:
        logger.exception(f"백필 작업 시작 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"백필 작업 시작 실패: {e}")


@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    Celery 작업 ID를 사용하여 현재 상태와 진행률을 조회합니다.
    """
    task_result = AsyncResult(task_id, app=celery_app)

    if not task_result:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다.")

    response_meta = None
    if task_result.state == "PROGRESS":
        response_meta = task_result.info
    elif task_result.state == "FAILURE":
        response_meta = {"error": str(task_result.info)}
    elif task_result.state == "SUCCESS":
        response_meta = task_result.get()

    return TaskStatusResponse(
        task_id=task_id, state=task_result.state, meta=response_meta
    )


@router.post("/stop/{task_id}", response_model=dict)
async def stop_backfill_task(task_id: str):
    """
    실행 중인 작업을 중지시킵니다.
    """
    try:
        # terminate=True: 작업을 강제로 종료
        celery_app.control.revoke(task_id, terminate=True, signal="SIGKILL")
        return {"message": f"작업 {task_id}에 중지 신호를 보냈습니다."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
