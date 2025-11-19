# celery_task/pipeline_task.py
from __future__ import annotations

import time
import uuid

import httpx
from celery import group, chain, chord
from loguru import logger
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from . import celery_app
from . import websocket_task
from .rest_api_task import backfill_symbol_interval

from db_module.connect_sqlalchemy_engine import SyncSessionLocal
from models import CryptoInfo
from models.pipeline_state import (
    is_pipeline_active,
    set_component_active,
    PipelineComponent,
)
from models.backfill_progress import BackfillProgress
from celery_task.rest_maintenance_task import run_rest_maintenance
from celery_task.indicator_task import run_indicator_maintenance

__all__ = ["start_pipeline", "stop_pipeline", "run_maintenance_cycle"]


# ================================================================
# 1) 전체 파이프라인 시작 (WebSocket → Backfill → Maintenance)
# ================================================================
@celery_app.task(name="pipeline.start_pipeline")
def start_pipeline():
    logger.info("[pipeline] 파이프라인 시작")

    if not is_pipeline_active():
        logger.info("[pipeline] pipeline_state.id=1 이 OFF → 종료")
        return

    # -----------------------------
    # 1) WebSocket 엔진 시작
    # -----------------------------
    set_component_active(PipelineComponent.WEBSOCKET, True)
    websocket_task.websocket_collector.delay()
    logger.info("[pipeline] WebSocket collector started")

    # -----------------------------
    # 2) Backfill 준비 및 실행 (Chain)
    # -----------------------------
    # wait_for_ws_data -> prepare_backfill -> execute_backfill_chord
    chain(
        wait_for_ws_data.s(),
        prepare_backfill_and_execute.s()
    ).apply_async()


@celery_app.task(name="pipeline.wait_for_ws_data", bind=True, max_retries=10)
def wait_for_ws_data(self):
    """
    WebSocket 데이터가 수집되기 시작할 때까지 대기 (최대 30초)
    """
    if not is_pipeline_active():
        return False

    # 간단히 3초 대기 후 리트라이 하는 방식으로 구현 (기존 sleep(30) 대체)
    # 실제로는 DB에 데이터가 들어왔는지 체크하는 것이 더 정확하지만,
    # 기존 로직의 의도(안정화 대기)를 살려 단순 딜레이로 처리하거나,
    # 혹은 단순히 다음 단계로 넘김. 여기서는 30초를 쪼개서 대기하지 않고
    # 별도 체크 없이 30초 후에 실행되도록 countdown을 줄 수도 있음.
    # 하지만 chain 상에서는 앞 task가 끝나야 뒤가 실행되므로,
    # 여기서는 단순히 sleep 하고 리턴하거나,
    # 좀 더 우아하게는 retry를 사용.
    
    # 여기서는 기존 로직(time.sleep(30))을 비동기적으로 처리하기 위해
    # 이 태스크 자체가 30초 딜레이를 갖는 것보다,
    # start_pipeline에서 이 태스크를 호출할 때 countdown=30을 주는 것이 나을 수 있으나,
    # chain 내부라 복잡함.
    # 단순히 여기서 sleep(30)을 하는 것은 worker를 블로킹하므로 좋지 않음.
    # 하지만 초기화 1회성이므로 큰 문제는 아님.
    # 개선: 그냥 바로 리턴하고, 다음 단계에서 필요한 데이터를 체크하도록 함.
    # 혹은 retry를 이용해 데이터가 들어왔는지 확인.
    
    # 여기서는 "안정화 대기"라는 명목하에 30초를 기다렸던 것이므로,
    # 실제 데이터가 들어왔는지 확인하는 로직으로 변경하는 것이 바람직함.
    # 하지만 간단히 구현하기 위해 sleep 대신, retry backoff를 사용하지 않고
    # 그냥 통과시키되, backfill task 내부에서 데이터 없으면 retry 하도록 위임.
    
    # 일단 기존 로직 존중하여 30초 대기 (블로킹이지만 1회성)
    # *더 나은 방법*: 이 태스크를 soft_time_limit 등으로 제어하거나
    # 그냥 넘어가고 backfill에서 frontier 체크를 강화.
    
    logger.info("[pipeline] WebSocket 안정화 대기 (30s)...")
    time.sleep(30) 
    return True


@celery_app.task(name="pipeline.prepare_backfill_and_execute")
def prepare_backfill_and_execute(prev_result):
    if not is_pipeline_active():
        return

    # -----------------------------
    # Binance 서버 시간 조회
    # -----------------------------
    try:
        with httpx.Client(timeout=10.0) as client:
            res = client.get("https://fapi.binance.com/fapi/v1/time")
            res.raise_for_status()
            server_time_ms = int(res.json()["serverTime"])
    except Exception as e:
        logger.error(f"[pipeline] serverTime 조회 실패: {e}")
        # 실패 시 재시도 로직이 필요하다면 여기서 raise하여 retry
        return

    ws_frontier_ms = server_time_ms
    logger.info(f"[pipeline] ws_frontier_ms={ws_frontier_ms}")

    # -----------------------------
    # 3) Backfill 시작
    # -----------------------------
    set_component_active(PipelineComponent.BACKFILL, True)
    run_id = f"run-{uuid.uuid4().hex}"
    logger.info(f"[pipeline] Backfill run_id={run_id}")

    # -----------------------------
    # 심볼 가져오기
    # -----------------------------
    with SyncSessionLocal() as session:
        symbols = (
            session.query(CryptoInfo.symbol, CryptoInfo.pair)
            .filter(CryptoInfo.pair.isnot(None))
            .all()
        )

    intervals = ["1h", "4h", "1d", "1w", "1M"]

    # -----------------------------
    # BackfillProgress Dummy row 생성
    # -----------------------------
    with SyncSessionLocal() as session, session.begin():
        for sym, _pair in symbols:
            for interval in intervals:
                stmt = (
                    insert(BackfillProgress)
                    .values(
                        run_id=run_id,
                        symbol=sym,
                        interval=interval,
                        state="PENDING",
                        pct_time=0.0,
                        last_candle_ts=None,
                        last_error=None,
                    )
                    .on_conflict_do_nothing()
                )
                session.execute(stmt)

    logger.info("[pipeline] BackfillProgress dummy rows inserted")

    # -----------------------------
    # Backfill 병렬 잡 생성 (Chord)
    # -----------------------------
    header = []
    for sym, pair in symbols:
        for interval in intervals:
            header.append(
                backfill_symbol_interval.s(
                    symbol=sym,
                    pair=pair,
                    interval=interval,
                    ws_frontier_ms=ws_frontier_ms,
                    run_id=run_id,
                )
            )

    # Chord: 모든 backfill 작업이 끝나면 on_backfill_complete 실행
    callback = on_backfill_complete.s(run_id=run_id)
    chord(header)(callback)
    logger.info("[pipeline] Backfill chord started")


@celery_app.task(name="pipeline.on_backfill_complete")
def on_backfill_complete(results, run_id):
    """
    Backfill 완료 후 호출되는 콜백
    """
    logger.info(f"[pipeline] Backfill 완료 (run_id={run_id})")
    set_component_active(PipelineComponent.BACKFILL, False)

    if not is_pipeline_active():
        return

    # 실패 확인
    # results는 각 태스크의 리턴값 리스트
    # 예: [{'status': 'COMPLETE', ...}, {'status': 'SKIP'}, ...]
    
    # 간단히 DB에서 최종 상태 확인
    if not is_backfill_success(run_id):
        logger.error("[pipeline] Backfill 실패 포함됨 → Maintenance 진입 중단")
        return

    logger.info("[pipeline] Backfill SUCCESS → Maintenance 사이클 시작")
    run_maintenance_cycle.delay()


def is_backfill_success(run_id: str) -> bool:
    with SyncSessionLocal() as session:
        rows = (
            session.execute(
                select(BackfillProgress.state).where(BackfillProgress.run_id == run_id)
            )
            .scalars()
            .all()
        )
    if not rows:
        return False
    if any(state in ("FAILURE", "FAIL") for state in rows):
        return False
    return True


# ================================================================
# 2) 파이프라인 OFF
# ================================================================
@celery_app.task(name="pipeline.stop_pipeline")
def stop_pipeline():
    logger.info("[pipeline] 전체 pipeline OFF")
    # 각 컴포넌트(Websocket, Maintenance)는 is_pipeline_active()를 체크하여 스스로 종료됨
    return


# ================================================================
# 3) Maintenance Cycle (Loop 대신 재귀적 호출 or Periodic)
# ================================================================
@celery_app.task(name="pipeline.run_maintenance_cycle")
def run_maintenance_cycle():
    """
    REST 유지보수 -> Indicator 유지보수 -> (잠시 대기) -> 다시 REST 유지보수 ...
    무한 루프 대신 Chain + 재귀 호출로 구현하여 Worker Blocking 방지
    """
    if not is_pipeline_active():
        logger.info("[pipeline] Maintenance cycle stopped (Pipeline inactive)")
        return

    logger.info("[pipeline] Maintenance cycle step started")
    
    # Chain: REST -> Indicator -> Next Cycle
    chain(
        run_rest_maintenance_wrapper.s(),
        run_indicator_maintenance_wrapper.s(),
        schedule_next_maintenance.s()
    ).apply_async()


@celery_app.task(name="pipeline.run_rest_maintenance_wrapper")
def run_rest_maintenance_wrapper():
    if not is_pipeline_active():
        return {"status": "INACTIVE"}
    
    set_component_active(PipelineComponent.REST_MAINTENANCE, True)
    logger.info("[pipeline] REST 유지보수 시작")
    
    # 동기 함수인 run_rest_maintenance를 호출하거나, 
    # 만약 run_rest_maintenance가 async task라면 .delay()가 아니라 직접 호출해야 함.
    # 현재 rest_maintenance_task.py의 run_rest_maintenance는 @shared_task임.
    # 따라서 여기서는 동기적으로 실행하거나, 
    # 아니면 chain을 더 쪼개야 함.
    # run_rest_maintenance가 내부적으로 blocking이면 여기서도 blocking됨.
    # 하지만 run_rest_maintenance는 내부에서 loop를 돌며 처리하므로 
    # 하나의 거대한 task임. 
    # 이를 쪼개는 건 더 큰 공사므로, 일단은 wrapper에서 호출하고 완료를 기다림.
    
    # 주의: run_rest_maintenance가 celery task로 정의되어 있으므로,
    # 함수로 직접 호출하면 일반 함수처럼 실행됨 (Celery app context 내에서).
    # 단, @shared_task 데코레이터가 있으면 .delay() 없이 호출 시 원본 함수 실행됨.
    
    try:
        result = run_rest_maintenance() # 직접 호출 (동기 실행)
    except Exception as e:
        logger.error(f"[pipeline] REST maintenance error: {e}")
        result = {"status": "FAILURE"}
        
    set_component_active(PipelineComponent.REST_MAINTENANCE, False)
    logger.info("[pipeline] REST 유지보수 종료")
    return result


@celery_app.task(name="pipeline.run_indicator_maintenance_wrapper")
def run_indicator_maintenance_wrapper(prev_result):
    if not is_pipeline_active():
        return {"status": "INACTIVE"}

    set_component_active(PipelineComponent.INDICATOR, True)
    logger.info("[pipeline] Indicator 계산 시작")
    
    try:
        result = run_indicator_maintenance() # 직접 호출
    except Exception as e:
        logger.error(f"[pipeline] Indicator maintenance error: {e}")
        result = {"status": "FAILURE"}

    set_component_active(PipelineComponent.INDICATOR, False)
    logger.info("[pipeline] Indicator 계산 완료")
    return result


@celery_app.task(name="pipeline.schedule_next_maintenance")
def schedule_next_maintenance(prev_result):
    if not is_pipeline_active():
        return
    
    # 1초 대기 후 다음 사이클 스케줄링
    # 재귀 호출 (countdown 사용)
    run_maintenance_cycle.apply_async(countdown=1)

