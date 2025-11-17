# celery_task/rest_maintenance_task.py
from celery import shared_task, group
from loguru import logger
import httpx

from sqlalchemy import select
from db_module.connect_sqlalchemy_engine import SyncSessionLocal
from models import CryptoInfo, OHLCV_MODELS
from models.pipeline_state import (
    is_pipeline_active,
    set_component_error,
    PipelineComponent,
)
from .rest_api_task import backfill_symbol_interval


# ===========================================================
# DB 기반 WS frontier 계산
# ===========================================================
def get_ws_frontier_from_db():
    """
    WebSocket으로 들어온 is_ended=False 중 가장 오래된 timestamp를 WS frontier로 사용.
    없다면 해당 interval의 is_ended=False 가 생길 때까지 대기하도록 None 반환.
    """
    frontier_candidates = []

    with SyncSessionLocal() as session:
        for interval, Model in OHLCV_MODELS.items():
            row = session.execute(
                select(Model.timestamp)
                .where(Model.is_ended == False)
                .order_by(Model.timestamp.asc())
                .limit(1)
            ).scalar_one_or_none()
            if row:
                frontier_candidates.append(row)

    if not frontier_candidates:
        return None  # 아직 websocket 데이터가 없음 → 대기해야 함

    # 가장 오래된 is_ended=False timestamp
    oldest = min(frontier_candidates)
    return int(oldest.timestamp() * 1000)


# ===========================================================
# REST 유지보수 엔진 (최종 패치 완료)
# ===========================================================
@shared_task(name="pipeline.run_maintenance_cycle")
def run_maintenance_cycle():
    """
    REST 유지보수 엔진:
    1) 유지보수할 데이터가(is_ended=False) 없으면 즉시 SUCCESS 반환
    2) 유지보수 대상이 있으면 해당 대상만 REST 백필 실행
    """
    if not is_pipeline_active():
        logger.info("[REST] Pipeline OFF -> Skip")
        return {"status": "SKIP"}

    try:
        # ---------------------------------------------------
        # 1) 유지보수할 데이터 유무 확인
        # ---------------------------------------------------
        logger.info("[REST] 유지보수 대상 스캔 시작")

        maintenance_targets = []

        with SyncSessionLocal() as session:

            for interval, Model in OHLCV_MODELS.items():
                row = session.execute(
                    select(Model)
                    .where(Model.is_ended == False)
                    .order_by(Model.timestamp.asc())
                    .limit(1)
                ).scalar_one_or_none()
                if row:
                    maintenance_targets.append((row.symbol, interval))

        # ---------------------------------------------------
        # 1-1) 유지보수 대상 없음 → 즉시 SUCCESS
        # ---------------------------------------------------
        if not maintenance_targets:
            logger.info("[REST] 유지보수 대상 없음 → 즉시 SUCCESS")
            return {"status": "SUCCESS", "updated": 0}

        logger.info(f"[REST] 유지보수 대상 발견: {len(maintenance_targets)}개")

        # ---------------------------------------------------
        # 2) DB 기반 WS frontier 계산
        # ---------------------------------------------------
        ws_frontier_ms = get_ws_frontier_from_db()
        if ws_frontier_ms is None:
            logger.info("[REST] is_ended=False 캔들이 아직 없음 → WS 데이터 생성 대기")
            return {"status": "WAITING_WS"}

        logger.info(f"[REST] WS frontier timestamp(ms) = {ws_frontier_ms}")

        # ---------------------------------------------------
        # 3) 실제 유지보수 수행 (필요한 구간만 REST backfill)
        # ---------------------------------------------------
        jobs = []

        with SyncSessionLocal() as session:
            symbols = (
                session.query(CryptoInfo.symbol, CryptoInfo.pair)
                .filter(CryptoInfo.pair.isnot(None))
                .all()
            )

        symbol_pair_map = {sym: pair for sym, pair in symbols}

        for sym, interval in maintenance_targets:
            pair = symbol_pair_map.get(sym)
            if not pair:
                continue

            jobs.append(
                backfill_symbol_interval.s(
                    symbol=sym,
                    pair=pair,
                    interval=interval,
                    ws_frontier_ms=ws_frontier_ms,
                    run_id=None,
                )
            )

        if not jobs:
            logger.info("[REST] 실행할 작업 없음 → SUCCESS")
            return {"status": "SUCCESS", "updated": 0}

        logger.info(f"[REST] 유지보수 실행: jobs={len(jobs)}")

        group(jobs).apply_async()

        return {"status": "SUCCESS", "updated": len(jobs)}

    except Exception as e:
        logger.exception(e)
        set_component_error(PipelineComponent.REST_MAINTENANCE, f"REST error: {e}")
        return {"status": "FAIL", "error": str(e)}
