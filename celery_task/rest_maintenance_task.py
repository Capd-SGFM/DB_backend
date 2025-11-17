# celery_task/rest_maintenance_task.py
from celery import shared_task, group
from loguru import logger
import httpx

from db_module.connect_sqlalchemy_engine import SyncSessionLocal
from models import CryptoInfo, OHLCV_MODELS
from models.pipeline_state import (
    is_pipeline_active,
    set_component_error,
    PipelineComponent,
)
from .rest_api_task import backfill_symbol_interval


@shared_task(name="pipeline.run_maintenance_cycle")
def run_maintenance_cycle():
    """
    일정 주기로 호출 (Celery beat).
    - pipeline이 active가 아니면 즉시 종료
    - active면: Binance serverTime 기준으로 WebSocket 이전 구간을 REST로 틈 메우기
    - 에러 발생 시 pipeline_state(id=4: REST 유지보수)의 last_error에 기록
    """
    # 전체 파이프라인 OFF면 그냥 종료
    if not is_pipeline_active():
        logger.info("[Maintenance] pipeline inactive -> skip")
        return

    try:
        # 1) Binance serverTime 가져옴
        with httpx.Client(timeout=10.0) as client:
            server_time_res = client.get("https://fapi.binance.com/fapi/v1/time")
            server_time_res.raise_for_status()
            server_time_ms = server_time_res.json()["serverTime"]

        # WebSocket이 약간 앞에서 수집 중이라고 가정하고,
        # 딱 server_time_ms 이전까지만 REST 유지보수
        ws_cutoff_ms = server_time_ms

        # 2) 모든 심볼/interval 조회
        with SyncSessionLocal() as session:
            symbols = (
                session.query(CryptoInfo.symbol, CryptoInfo.pair)
                .filter(CryptoInfo.pair.isnot(None))
                .all()
            )

        intervals = list(OHLCV_MODELS.keys())

        # 3) 각 (symbol, interval)에 대해 짧은 backfill 실행
        jobs = []
        for sym, pair in symbols:
            for interval in intervals:
                jobs.append(
                    backfill_symbol_interval.s(
                        symbol=sym,
                        pair=pair,
                        interval=interval,
                        ws_frontier_ms=ws_cutoff_ms,  # ← 시그니처에 맞게 수정
                    )
                )

        if not jobs:
            logger.info("[Maintenance] no symbols/intervals to maintain.")
            return

        logger.info(
            f"[Maintenance] start maintenance cycle: jobs={len(jobs)}, ws_cutoff_ms={ws_cutoff_ms}"
        )
        group(jobs).apply_async()

    except Exception as e:
        logger.error(f"[Maintenance] run_maintenance_cycle failed: {e}")
        # REST 유지보수 엔진 에러 로그 기록 (id=4)
        try:
            set_component_error(
                PipelineComponent.REST_MAINTENANCE,
                f"Maintenance cycle error: {type(e).__name__}: {e}",
            )
        except Exception:
            logger.exception("[Maintenance] failed to save last_error")
        # Celery 쪽에는 에러로 남겨 두기
        raise
