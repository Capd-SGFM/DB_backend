# routers/pipeline.py
from fastapi import APIRouter
from pydantic import BaseModel
from loguru import logger

from celery_task.pipeline_task import start_pipeline, stop_pipeline
from models.pipeline_state import (
    is_pipeline_active,
    set_pipeline_active,
    get_all_pipeline_states,
)
from db_module.connect_sqlalchemy_engine import SyncSessionLocal
from models.backfill_progress import BackfillProgress, reset_backfill_progress
from sqlalchemy import select, desc

router = APIRouter(prefix="/pipeline", tags=["Pipeline"])


class PipelineToggleResponse(BaseModel):
    message: str
    is_active: bool


class EngineStatusModel(BaseModel):
    id: int
    is_active: bool
    status: str  # "WAIT" | "PROGRESS" | "FAIL" | "UNKNOWN"
    last_error: str | None = None
    updated_at: str | None = None


class PipelineStatusResponse(BaseModel):
    is_active: bool
    websocket: EngineStatusModel
    backfill: EngineStatusModel
    rest_maintenance: EngineStatusModel
    indicator: EngineStatusModel


def _map_engine_status(
    is_pipeline_on: bool,
    comp: dict,
) -> EngineStatusModel:
    """
    get_all_pipeline_states() 에서 내려준 dict(comp)을
    프론트에서 쓰기 좋은 EngineStatusModel로 변환.

    comp 예시:
      {
        "id": 2,
        "is_active": True,
        "updated_at": "2025-11-17T00:00:00+00:00",
        "last_error": "some error" or None
      }
    """
    comp_id = int(comp.get("id"))
    is_active = bool(comp.get("is_active", False))
    last_error = comp.get("last_error")
    updated_at = comp.get("updated_at")

    # 기본값은 WAIT
    status = "WAIT"

    if not is_pipeline_on:
        # 전체 파이프라인 OFF면 전부 WAIT로 처리
        status = "WAIT"
    else:
        if last_error:
            status = "FAIL"
        elif is_active:
            status = "PROGRESS"
        else:
            # 파이프라인은 ON인데, 이 컴포넌트는 비활성
            status = "WAIT"

    return EngineStatusModel(
        id=comp_id,
        is_active=is_active,
        status=status,
        last_error=last_error,
        updated_at=updated_at,
    )


@router.post("/on", response_model=PipelineToggleResponse)
async def activate_pipeline():
    """
    전체 파이프라인 ON.
    - trading_data.pipeline_state(id=1).is_active = TRUE
    - 기존 backfill_progress 기록은 모두 초기화
    - Celery 에 pipeline.start_pipeline 태스크 enqueue
    """
    if is_pipeline_active():
        return PipelineToggleResponse(
            message="이미 파이프라인이 ON 상태입니다.",
            is_active=True,
        )

    # ✅ 이전 Backfill 진행률 기록을 전부 삭제
    try:
        reset_backfill_progress()
        logger.info("[Pipeline] reset backfill_progress table (fresh run).")
    except Exception:
        # 진행률 초기화 실패 때문에 파이프라인 자체가 안 켜지는 건 너무 과하니,
        # 일단 로그만 남기고 이어서 진행
        logger.exception("[Pipeline] failed to reset backfill_progress")

    # 파이프라인 플래그 ON
    set_pipeline_active(True)

    # Celery 에 실제 파이프라인 시작 태스크 enqueue
    start_pipeline.delay()
    logger.info("[Pipeline] activated (start_pipeline task enqueued)")

    return PipelineToggleResponse(
        message="데이터 수집 파이프라인을 활성화했습니다.",
        is_active=True,
    )


@router.post("/off", response_model=PipelineToggleResponse)
async def deactivate_pipeline():
    """
    전체 파이프라인 OFF.
    - trading_data.pipeline_state(id=1).is_active = FALSE
    - 하위 컴포넌트(id=2~5) 도 False 로 초기화 (set_pipeline_active 내부에서 처리)
    - Celery 에 pipeline.stop_pipeline 태스크 enqueue
    """
    if not is_pipeline_active():
        return PipelineToggleResponse(
            message="이미 파이프라인이 OFF 상태입니다.",
            is_active=False,
        )

    set_pipeline_active(False)
    stop_pipeline.delay()
    logger.info("[Pipeline] deactivated (stop_pipeline task enqueued)")

    return PipelineToggleResponse(
        message="데이터 수집 파이프라인을 비활성화했습니다.",
        is_active=False,
    )


@router.get("/status", response_model=PipelineStatusResponse)
async def get_pipeline_status():
    """
    프론트엔드 Progress UI 용 상태 조회.
    - get_all_pipeline_states() 에서
      pipeline / websocket / backfill / rest_maintenance / indicator 상태를 받아서
      각 컴포넌트별 status + last_error + updated_at까지 내려줌.
    """
    states = get_all_pipeline_states()

    is_on = bool(states.get("pipeline", {}).get("is_active", False))

    ws = states.get(
        "websocket",
        {
            "id": 2,
            "is_active": False,
            "updated_at": None,
            "last_error": None,
        },
    )
    bf = states.get(
        "backfill",
        {
            "id": 3,
            "is_active": False,
            "updated_at": None,
            "last_error": None,
        },
    )
    rm = states.get(
        "rest_maintenance",
        {
            "id": 4,
            "is_active": False,
            "updated_at": None,
            "last_error": None,
        },
    )
    ind = states.get(
        "indicator",
        {
            "id": 5,
            "is_active": False,
            "updated_at": None,
            "last_error": None,
        },
    )

    websocket_status = _map_engine_status(is_on, ws)
    backfill_status = _map_engine_status(is_on, bf)
    rest_status = _map_engine_status(is_on, rm)
    indicator_status = _map_engine_status(is_on, ind)

    return PipelineStatusResponse(
        is_active=is_on,
        websocket=websocket_status,
        backfill=backfill_status,
        rest_maintenance=rest_status,
        indicator=indicator_status,
    )


# ==============================
#   Backfill 진행률 조회용 API
# ==============================


class BackfillIntervalProgress(BaseModel):
    interval: str
    state: str  # PENDING / PROGRESS / COMPLETE / FAIL / SUCCESS / FAILURE 등
    pct_time: float
    last_updated_iso: str | None


class BackfillSymbolProgress(BaseModel):
    symbol: str
    state: str  # 전체 심볼 상태 (PENDING / PROGRESS / SUCCESS / FAILURE / UNKNOWN)
    intervals: dict[str, BackfillIntervalProgress]


class BackfillProgressResponse(BaseModel):
    run_id: str | None
    symbols: dict[str, BackfillSymbolProgress]


@router.get("/backfill/progress", response_model=BackfillProgressResponse)
async def get_backfill_progress():
    """
    현재 진행 중이거나 가장 최근 실행된 Backfill 엔진의 전체 진행률 조회 API.
    - 가장 최근 run_id 기준
    - 각 symbol × interval 별로 BackfillProgress 테이블의 row를 집계해서 반환
    - 프론트는 이 API만 폴링하면, localStorage 없이도 전체 진행 상황을 볼 수 있음
    """
    with SyncSessionLocal() as session:
        # 1) 가장 최근 run_id 찾기 (updated_at 기준)
        q = (
            select(BackfillProgress.run_id)
            .order_by(desc(BackfillProgress.updated_at))
            .limit(1)
        )
        latest_run = session.execute(q).scalar_one_or_none()

        if not latest_run:
            # 아직 Backfill 기록이 전혀 없음
            return BackfillProgressResponse(run_id=None, symbols={})

        run_id = latest_run

        # 2) 해당 run_id의 전체 row 조회
        q2 = select(BackfillProgress).where(BackfillProgress.run_id == run_id)
        rows = session.execute(q2).scalars().all()

    # 3) symbol → interval → progress 구조로 정리
    result: dict[str, BackfillSymbolProgress] = {}

    for row in rows:
        if row.symbol not in result:
            result[row.symbol] = BackfillSymbolProgress(
                symbol=row.symbol,
                state="UNKNOWN",  # 일단 UNKNOWN, 나중에 아래에서 재계산
                intervals={},
            )

        result[row.symbol].intervals[row.interval] = BackfillIntervalProgress(
            interval=row.interval,
            state=row.state,
            pct_time=float(row.pct_time or 0.0),
            # updated_at 기반 ISO 문자열
            last_updated_iso=row.updated_at.isoformat() if row.updated_at else None,
        )

    # 4) 심볼 단위 state 재계산
    for sym_model in result.values():
        states = [iv.state for iv in sym_model.intervals.values()]
        if not states:
            sym_model.state = "UNKNOWN"
        elif any(s in ("FAIL", "FAILURE") for s in states):
            sym_model.state = "FAILURE"
        elif all(s in ("COMPLETE", "SUCCESS") for s in states):
            sym_model.state = "SUCCESS"
        elif any(s in ("PROGRESS", "STARTED") for s in states):
            sym_model.state = "PROGRESS"
        elif all(s == "PENDING" for s in states):
            sym_model.state = "PENDING"
        else:
            sym_model.state = "UNKNOWN"

    return BackfillProgressResponse(
        run_id=run_id,
        symbols=result,
    )
