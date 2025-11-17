# db_pipeline/pipeline_control.py

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, List

from loguru import logger
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from models.pipeline_state import (
    PipelineState,
    EngineStatus,
    PIPELINE_STATE_PK,
    ENGINE_ID_MAP,
)


async def get_pipeline_state(db: AsyncSession) -> bool:
    """
    현재 전체 파이프라인(id=1)의 is_active 값을 반환.
    row가 없으면 False 로 간주.
    """
    row = await db.get(PipelineState, PIPELINE_STATE_PK)
    if row is None:
        logger.info("pipeline_state row 가 없어 기본값 False 반환.")
        return False
    return bool(row.is_active)


async def set_pipeline_state(db: AsyncSession, active: bool) -> PipelineState:
    """
    전체 파이프라인(id=1)의 ON/OFF 상태를 설정하고, 해당 row 를 반환.
    """
    now = datetime.now(timezone.utc)

    stmt = (
        update(PipelineState)
        .where(PipelineState.id == PIPELINE_STATE_PK)
        .values(is_active=active, updated_at=now)
    )
    result = await db.execute(stmt)

    if result.rowcount == 0:
        logger.info("pipeline_state row 가 없어 INSERT 수행.")
        row = PipelineState(
            id=PIPELINE_STATE_PK,
            is_active=active,
            updated_at=now,
        )
        db.add(row)
        await db.flush()
    await db.commit()

    row = await db.get(PipelineState, PIPELINE_STATE_PK)
    logger.info(f"Pipeline state set to {active}.")
    return row


async def ensure_engine_status_rows(db: AsyncSession) -> List[EngineStatus]:
    """
    engine_status 테이블에 websocket/backfill/rest_maintenance/indicator
    4개 row가 없으면 생성.
    """
    now = datetime.now(timezone.utc)

    result = await db.execute(select(EngineStatus))
    existing = {row.name: row for row in result.scalars()}

    created = []
    for name, engine_id in ENGINE_ID_MAP.items():
        if name not in existing:
            row = EngineStatus(
                id=engine_id,
                name=name,
                status="WAIT",
                last_error=None,
                updated_at=now,
            )
            db.add(row)
            created.append(row)

    if created:
        await db.commit()

    result = await db.execute(select(EngineStatus))
    return list(result.scalars())


async def reset_engine_statuses_for_new_run(db: AsyncSession) -> None:
    """
    파이프라인을 ON 으로 전환할 때,
    모든 엔진 상태를 WAIT + last_error=None 으로 리셋.
    (프론트에서 '이번 실행 이후 에러만 보기'를 위해)
    """
    now = datetime.now(timezone.utc)
    await db.execute(
        update(EngineStatus).values(
            status="WAIT",
            last_error=None,
            updated_at=now,
        )
    )
    await db.commit()


async def get_all_engine_statuses(db: AsyncSession) -> List[EngineStatus]:
    """
    모든 엔진 상태를 조회해서 반환.
    """
    result = await db.execute(select(EngineStatus))
    return list(result.scalars())
