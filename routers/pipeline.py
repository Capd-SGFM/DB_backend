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

# 🔹 REST / Indicator 진행현황용 모델
from models.rest_progress import RestProgress
from models.indicator_progress import IndicatorProgress
from models.websocket_progress import WebSocketProgress

router = APIRouter(prefix="/pipeline", tags=["Pipeline"])


# ===============================
#   공통 Response 모델
# ===============================


class PipelineToggleResponse(BaseModel):
    message: str
    is_active: bool


class EngineStatusModel(BaseModel):
    id: int
    is_active: bool
    status: str  # WAIT | PROGRESS | SUCCESS | FAIL | UNKNOWN
    last_error: str | None = None
    updated_at: str | None = None


class PipelineStatusResponse(BaseModel):
    is_active: bool
    websocket: EngineStatusModel
    backfill: EngineStatusModel
    rest_maintenance: EngineStatusModel
    indicator: EngineStatusModel


# ==================================================
#   기존 엔진 상태 매핑 (WebSocket / REST / Indicator)
# ==================================================


def _map_engine_status(is_pipeline_on: bool, comp: dict) -> EngineStatusModel:
    comp_id = int(comp.get("id"))
    is_active = bool(comp.get("is_active", False))
    last_error = comp.get("last_error")
    updated_at = comp.get("updated_at")

    status = "WAIT"

    if not is_pipeline_on:
        status = "WAIT"
    else:
        if last_error:
            status = "FAIL"
        elif is_active:
            status = "PROGRESS"
        else:
            status = "WAIT"

    return EngineStatusModel(
        id=comp_id,
        is_active=is_active,
        status=status,
        last_error=last_error,
        updated_at=updated_at,
    )


# ==================================================
#   ☆ Backfill 엔진 전용 상태 매핑
# ==================================================


def _map_backfill_status(is_pipeline_on: bool, comp: dict) -> EngineStatusModel:
    comp_id = int(comp.get("id"))
    is_active = bool(comp.get("is_active", False))
    last_error = comp.get("last_error")
    updated_at = comp.get("updated_at")

    # pipeline OFF → 무조건 WAIT
    if not is_pipeline_on:
        return EngineStatusModel(
            id=comp_id,
            is_active=False,
            status="WAIT",
            last_error=last_error,
            updated_at=updated_at,
        )

    # error 있으면 FAIL
    if last_error:
        return EngineStatusModel(
            id=comp_id,
            is_active=False,
            status="FAIL",
            last_error=last_error,
            updated_at=updated_at,
        )

    # 현재 실행중이면 PROGRESS
    if is_active:
        return EngineStatusModel(
            id=comp_id,
            is_active=True,
            status="PROGRESS",
            last_error=last_error,
            updated_at=updated_at,
        )

    # 여기부터가 핵심
    # ================================
    #   BackfillProgress 전체 상태 확인
    # ================================
    with SyncSessionLocal() as session:
        rows = session.execute(select(BackfillProgress.state)).scalars().all()

    if not rows:
        final_status = "WAIT"
    elif any(s in ("FAIL", "FAILURE") for s in rows):
        final_status = "FAIL"
    elif all(s in ("COMPLETE", "SUCCESS") for s in rows):
        final_status = "SUCCESS"
    elif any(s in ("PROGRESS", "STARTED") for s in rows):
        final_status = "PROGRESS"
    else:
        final_status = "WAIT"

    return EngineStatusModel(
        id=comp_id,
        is_active=False,
        status=final_status,
        last_error=last_error,
        updated_at=updated_at,
    )


# ==================================================
#   파이프라인 ON
# ==================================================


@router.post("/on", response_model=PipelineToggleResponse)
async def activate_pipeline():
    if is_pipeline_active():
        return PipelineToggleResponse(
            message="이미 파이프라인이 ON 상태입니다.",
            is_active=True,
        )

    # 기존 BackfillProgress 모두 초기화
    try:
        reset_backfill_progress()
        logger.info("[Pipeline] reset backfill_progress table")
    except Exception:
        logger.exception("[Pipeline] failed to reset backfill_progress")

    set_pipeline_active(True)

    start_pipeline.delay()
    logger.info("[Pipeline] activated")

    return PipelineToggleResponse(
        message="데이터 수집 파이프라인을 활성화했습니다.",
        is_active=True,
    )


# ==================================================
#   파이프라인 OFF
# ==================================================


@router.post("/off", response_model=PipelineToggleResponse)
async def deactivate_pipeline():
    if not is_pipeline_active():
        return PipelineToggleResponse(
            message="이미 파이프라인이 OFF 상태입니다.",
            is_active=False,
        )

    set_pipeline_active(False)
    stop_pipeline.delay()

    logger.info("[Pipeline] deactivated")

    return PipelineToggleResponse(
        message="데이터 수집 파이프라인을 비활성화했습니다.",
        is_active=False,
    )


# ==================================================
#   파이프라인 상태 조회 API
# ==================================================


@router.get("/status", response_model=PipelineStatusResponse)
async def get_pipeline_status():
    states = get_all_pipeline_states()
    is_on = bool(states.get("pipeline", {}).get("is_active", False))

    ws = states.get("websocket", {"id": 2, "is_active": False})
    bf = states.get("backfill", {"id": 3, "is_active": False})
    rm = states.get("rest_maintenance", {"id": 4, "is_active": False})
    ind = states.get("indicator", {"id": 5, "is_active": False})

    websocket_status = _map_engine_status(is_on, ws)
    backfill_status = _map_backfill_status(is_on, bf)  # ★ Backfill 전용 로직
    rest_status = _map_engine_status(is_on, rm)
    indicator_status = _map_engine_status(is_on, ind)

    return PipelineStatusResponse(
        is_active=is_on,
        websocket=websocket_status,
        backfill=backfill_status,
        rest_maintenance=rest_status,
        indicator=indicator_status,
    )


# ==================================================
#   Backfill 진행률 조회 API
# ==================================================


class BackfillIntervalProgress(BaseModel):
    interval: str
    state: str
    pct_time: float
    last_updated_iso: str | None


class BackfillSymbolProgress(BaseModel):
    symbol: str
    state: str
    intervals: dict[str, BackfillIntervalProgress]


class BackfillProgressResponse(BaseModel):
    run_id: str | None
    symbols: dict[str, BackfillSymbolProgress]


@router.get("/backfill/progress", response_model=BackfillProgressResponse)
async def get_backfill_progress():
    with SyncSessionLocal() as session:
        q = (
            select(BackfillProgress.run_id)
            .order_by(desc(BackfillProgress.updated_at))
            .limit(1)
        )
        latest_run = session.execute(q).scalar_one_or_none()

        if not latest_run:
            return BackfillProgressResponse(run_id=None, symbols={})

        run_id = latest_run

        q2 = select(BackfillProgress).where(BackfillProgress.run_id == run_id)
        rows = session.execute(q2).scalars().all()

    result: dict[str, BackfillSymbolProgress] = {}

    for row in rows:
        if row.symbol not in result:
            result[row.symbol] = BackfillSymbolProgress(
                symbol=row.symbol,
                state="UNKNOWN",
                intervals={},
            )

        result[row.symbol].intervals[row.interval] = BackfillIntervalProgress(
            interval=row.interval,
            state=row.state,
            pct_time=float(row.pct_time or 0.0),
            last_updated_iso=row.updated_at.isoformat() if row.updated_at else None,
        )

    # 심볼 단위 상태 재계산
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


# ==================================================
#   REST 유지보수 진행현황 조회 API
# ==================================================


class RestIntervalModel(BaseModel):
    interval: str
    state: str
    updated_at: str | None


class RestSymbolModel(BaseModel):
    symbol: str
    intervals: dict[str, RestIntervalModel]


class RestProgressResponse(BaseModel):
    run_id: str | None
    symbols: dict[str, RestSymbolModel]


@router.get("/rest/progress", response_model=RestProgressResponse)
async def get_rest_progress():
    with SyncSessionLocal() as session:
        # 최신 run_id 찾기
        run_id = session.execute(
            select(RestProgress.run_id)
            .order_by(RestProgress.updated_at.desc())
            .limit(1)
        ).scalar_one_or_none()

        if not run_id:
            return RestProgressResponse(run_id=None, symbols={})

        rows = (
            session.execute(select(RestProgress).where(RestProgress.run_id == run_id))
            .scalars()
            .all()
        )

    symbols: dict[str, RestSymbolModel] = {}

    for row in rows:
        if row.symbol not in symbols:
            symbols[row.symbol] = RestSymbolModel(symbol=row.symbol, intervals={})

        symbols[row.symbol].intervals[row.interval] = RestIntervalModel(
            interval=row.interval,
            state=row.state,
            updated_at=row.updated_at.isoformat() if row.updated_at else None,
        )

    return RestProgressResponse(run_id=run_id, symbols=symbols)


# ==================================================
#   Indicator 유지보수 진행현황 조회 API
#   (/pipeline/indicator/progress)
# ==================================================


class IndicatorIntervalModel(BaseModel):
    interval: str
    state: str
    pct_time: float
    updated_at: str | None


class IndicatorSymbolModel(BaseModel):
    symbol: str
    intervals: dict[str, IndicatorIntervalModel]


class IndicatorProgressResponse(BaseModel):
    run_id: str | None
    symbols: dict[str, IndicatorSymbolModel]


@router.get("/indicator/progress", response_model=IndicatorProgressResponse)
async def get_indicator_progress():
    with SyncSessionLocal() as session:
        # 최신 run_id 찾기
        run_id = session.execute(
            select(IndicatorProgress.run_id)
            .order_by(IndicatorProgress.updated_at.desc())
            .limit(1)
        ).scalar_one_or_none()

        if not run_id:
            return IndicatorProgressResponse(run_id=None, symbols={})

        rows = (
            session.execute(
                select(IndicatorProgress).where(IndicatorProgress.run_id == run_id)
            )
            .scalars()
            .all()
        )

    symbols: dict[str, IndicatorSymbolModel] = {}

    for row in rows:
        if row.symbol not in symbols:
            symbols[row.symbol] = IndicatorSymbolModel(symbol=row.symbol, intervals={})

        symbols[row.symbol].intervals[row.interval] = IndicatorIntervalModel(
            interval=row.interval,
            state=row.state,
            pct_time=float(row.pct_time or 0.0),
            updated_at=row.updated_at.isoformat() if row.updated_at else None,
        )


    return IndicatorProgressResponse(run_id=run_id, symbols=symbols)


# ==================================================
#   WebSocket 실시간 연결 상태 조회 API
#   (/pipeline/websocket/progress)
# ==================================================


class WebSocketIntervalModel(BaseModel):
    interval: str
    state: str
    message_count: int
    last_message_ts: str | None
    last_error: str | None


class WebSocketSymbolModel(BaseModel):
    symbol: str
    intervals: dict[str, WebSocketIntervalModel]


class WebSocketProgressResponse(BaseModel):
    run_id: str | None
    symbols: dict[str, WebSocketSymbolModel]


@router.get("/websocket/progress", response_model=WebSocketProgressResponse)
async def get_websocket_progress():
    with SyncSessionLocal() as session:
        # 최신 run_id 찾기
        run_id = session.execute(
            select(WebSocketProgress.run_id)
            .order_by(WebSocketProgress.updated_at.desc())
            .limit(1)
        ).scalar_one_or_none()

        if not run_id:
            return WebSocketProgressResponse(run_id=None, symbols={})

        rows = (
            session.execute(
                select(WebSocketProgress).where(WebSocketProgress.run_id == run_id)
            )
            .scalars()
            .all()
        )

    symbols: dict[str, WebSocketSymbolModel] = {}

    for row in rows:
        if row.symbol not in symbols:
            symbols[row.symbol] = WebSocketSymbolModel(symbol=row.symbol, intervals={})

        symbols[row.symbol].intervals[row.interval] = WebSocketIntervalModel(
            interval=row.interval,
            state=row.state,
            message_count=row.message_count,
            last_message_ts=row.last_message_ts.isoformat() if row.last_message_ts else None,
            last_error=row.last_error,
        )

    return WebSocketProgressResponse(run_id=run_id, symbols=symbols)

