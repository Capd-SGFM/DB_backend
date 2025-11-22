# celery_task/rest_maintenance_task.py

import uuid
from datetime import datetime, timezone
from typing import Optional, List

import httpx
from celery import shared_task
from loguru import logger

from sqlalchemy import select, delete, asc
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.sql import text

from db_module.connect_sqlalchemy_engine import SyncSessionLocal
from models import CryptoInfo, OHLCV_MODELS
from models.pipeline_state import (
    is_pipeline_active,
    set_component_error,
    PipelineComponent,
)
from models.rest_progress import RestProgress


# =========================================================
#   RestProgress Upsert 함수
# =========================================================
def upsert_rest_progress(
    run_id: str,
    symbol: str,
    interval: str,
    state: str,
    last_ts: Optional[datetime] = None,
    error: Optional[str] = None,
):
    """rest_progress 테이블에 상태 업데이트 (UPSERT)."""
    with SyncSessionLocal() as session, session.begin():
        stmt = (
            insert(RestProgress)
            .values(
                run_id=run_id,
                symbol=symbol,
                interval=interval,
                state=state,
                last_candle_ts=last_ts,
                last_error=error,
            )
            .on_conflict_do_update(
                index_elements=["run_id", "symbol", "interval"],
                set_={
                    "state": state,
                    "last_candle_ts": last_ts,
                    "last_error": error,
                    "updated_at": text("NOW()"),
                },
            )
        )
        session.execute(stmt)


# =========================================================
#  is_ended = FALSE 구간 탐색 헬퍼
# =========================================================
def find_false_range(
    model, symbol: str
) -> tuple[int, Optional[datetime], Optional[datetime]]:
    """
    해당 심볼에 대해 is_ended = FALSE 인 캔들들을 기준으로
    REST 유지보수 대상 구간을 찾는다.

    return:
        (count_false, first_false_ts, ws_current_ts)

        - count_false == 0 → 유지보수 대상 없음
        - count_false >= 1 → first_false_ts = 가장 오래된 FALSE,
                             ws_current_ts = 가장 최근 FALSE (WS 현재 캔들)
    """
    with SyncSessionLocal() as session:
        q = (
            select(model.timestamp)
            .where(model.symbol == symbol, model.is_ended.is_(False))
            .order_by(asc(model.timestamp))
        )
        rows: List[datetime] = session.execute(q).scalars().all()

    if not rows:
        return 0, None, None

    first_false_ts = rows[0]
    ws_current_ts = rows[-1]
    return len(rows), first_false_ts, ws_current_ts


# =========================================================
#   Binance kline REST 호출 헬퍼
# =========================================================
def fetch_klines_range(
    client: httpx.Client,
    pair: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    max_per_req: int = 1000,
):
    """
    Binance Futures kline을 start_ms~end_ms 구간에 대해 여러 번 나눠 가져오는 헬퍼.
    open_time 기준으로 end_ms를 넘으면 중단.
    """
    base_url = "https://fapi.binance.com/fapi/v1/klines"
    result = []

    if end_ms < start_ms:
        return result

    cur_start = start_ms
    while True:
        params = {
            "symbol": pair,
            "interval": interval,
            "startTime": cur_start,
            "endTime": end_ms,
            "limit": max_per_req,
        }
        r = client.get(base_url, params=params)
        r.raise_for_status()
        data = r.json()
        if not data:
            break

        # data: [[open_time, open, high, low, close, volume, ...], ...]
        for item in data:
            open_time_ms = int(item[0])
            if open_time_ms > end_ms:
                return result
            result.append(item)

        # 더 가져올게 없으면 종료
        if len(data) < max_per_req:
            break

        # 다음 시작 시간 = 마지막 캔들 + 1ms
        last_open_ms = int(data[-1][0])
        if last_open_ms >= end_ms:
            break
        cur_start = last_open_ms + 1

    return result


# =========================================================
#   REST 유지보수 메인 태스크 — A-1 (개정 버전)
# =========================================================
@shared_task(name="pipeline.run_rest_maintenance")
def run_rest_maintenance():
    """REST 유지보수:

    각 심볼 × 인터벌에 대해:

    1) is_ended = FALSE 캔들을 timestamp 오름차순으로 정렬했을 때
       - 개수가 0개 → 유지보수 할 것 없음 → 바로 SUCCESS
       - 개수가 1개 이상 → 가장 오래된 is_ended = FALSE인 캔들부터
         "현재 WebSocket이 수집중인 캔들의 바로 이전 timestamp"까지
         전체 구간을 REST로 한 번에 가져와서
         해당 구간을 모두 덮어쓰고 is_ended = TRUE 로 저장

       ※ 현재 WS 캔들은 is_ended = FALSE 인 캔들 중 "가장 최근 timestamp"라고 가정.
       ※ 단, 구간이 비어버리는 경우(start > end)는 실제 REST 호출 없이 SUCCESS 처리.
    """
    logger.info("[REST] run_rest_maintenance 시작")

    if not is_pipeline_active():
        logger.info("[REST] pipeline inactive → 종료")
        return {"status": "INACTIVE"}

    # ----------------------------------
    # 1) run_id 생성
    # ----------------------------------
    run_id = f"rest-{uuid.uuid4().hex}"
    logger.info(f"[REST] rest_run_id={run_id}")

    # ----------------------------------
    # 2) 심볼 목록 가져오기
    # ----------------------------------
    with SyncSessionLocal() as session:
        symbols = (
            session.query(CryptoInfo.symbol, CryptoInfo.pair)
            .filter(CryptoInfo.pair.isnot(None))
            .all()
        )

    intervals = list(OHLCV_MODELS.keys())  # 예: ["1h", "4h", "1d", "1w", "1M"]

    total_jobs = 0
    success_cnt = 0

    # Binance HTTP 클라이언트 재사용
    try:
        client = httpx.Client(timeout=10.0)
    except Exception as e:
        logger.error(f"[REST] httpx.Client 생성 실패: {e}")
        set_component_error(PipelineComponent.REST_MAINTENANCE, str(e))
        return {"status": "FAILURE", "error": str(e)}

    try:
        # serverTime은 필요 시 upper bound 검증용으로 참고
        try:
            res = client.get("https://fapi.binance.com/fapi/v1/time")
            res.raise_for_status()
            server_time_ms = int(res.json()["serverTime"])
            logger.info(f"[REST] Binance serverTime={server_time_ms}")
        except Exception as e:
            logger.error(f"[REST] serverTime 조회 실패 (무시 가능): {e}")
            server_time_ms = None

        # ----------------------------------
        # 3) 심볼 × 인터벌 루프
        # ----------------------------------
        for sym, pair in symbols:
            if not is_pipeline_active():
                logger.info("[REST] pipeline OFF 감지 → 조기 종료")
                break

            for interval in intervals:
                total_jobs += 1

                model = OHLCV_MODELS[interval]

                # (1) is_ended = FALSE 캔들 구간 탐색
                cnt_false, first_false_ts, ws_current_ts = find_false_range(model, sym)

                # 0개 → 유지보수 없음 → SUCCESS
                if cnt_false == 0 or not first_false_ts or not ws_current_ts:
                    upsert_rest_progress(run_id, sym, interval, "SUCCESS", None, None)
                    logger.info(
                        f"[REST] {sym} {interval}: is_ended=FALSE 없음 → SUCCESS(스킵)"
                    )
                    continue

                # start = 가장 오래된 FALSE
                start_ts = first_false_ts
                start_ms = int(start_ts.replace(tzinfo=timezone.utc).timestamp() * 1000)

                # end = "현재 WS 캔들 바로 이전" → ws_current_ms - 1ms
                # 사용자 요구사항: "is_ended = FALSE인 데이터 중 가장 최신인 데이터 캔들 - 1 인터벌까지"
                # ws_current_ts가 가장 최신 FALSE 캔들이므로,
                # ws_current_ms - 1ms를 end_ms로 잡으면,
                # fetch_klines_range(open_time <= end_ms) 로직에 의해
                # 정확히 "최신 FALSE 캔들 바로 직전 캔들"까지만 포함됨.
                # (즉, 최신 FALSE 캔들 자체는 포함되지 않음)
                ws_current_ms = int(
                    ws_current_ts.replace(tzinfo=timezone.utc).timestamp() * 1000
                )
                end_ms = ws_current_ms - 1

                # 구간이 비어버리면(예: FALSE 1개 뿐인 경우) 실제 작업 없이 SUCCESS
                if end_ms < start_ms:
                    upsert_rest_progress(run_id, sym, interval, "SUCCESS", None, None)
                    logger.info(
                        f"[REST] {sym} {interval}: FALSE는 있으나 "
                        f"start_ms > end_ms (실제 유지보수 구간 없음) → SUCCESS"
                    )
                    continue

                # delete조건 / 로그에서 사용할 end_ts 는 end_ms 기준으로 생성
                end_ts_for_db = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc)

                logger.info(
                    f"[REST] {sym} {interval}: 유지보수 대상 구간 "
                    f"{start_ts.isoformat()} ~ {end_ts_for_db.isoformat()} "
                    f"(FALSE 개수={cnt_false}, WS 현재 캔들={ws_current_ts.isoformat()})"
                )

                # 작업 시작 마킹
                upsert_rest_progress(run_id, sym, interval, "PROGRESS", None, None)

                # (2) REST로 kline 조회
                try:
                    klines = fetch_klines_range(
                        client=client,
                        pair=pair,
                        interval=interval,
                        start_ms=start_ms,
                        end_ms=end_ms,
                    )
                except Exception as e:
                    logger.exception(f"[REST] {sym} {interval}: kline 조회 실패: {e}")
                    err_msg = f"kline fetch error: {e}"
                    upsert_rest_progress(
                        run_id, sym, interval, "FAILURE", None, err_msg
                    )
                    set_component_error(PipelineComponent.REST_MAINTENANCE, err_msg)
                    continue

                if not klines:
                    # 이 구간에서 REST도 아무것도 안 주면 → 실패로 기록(정책에 따라 조정 가능)
                    logger.warning(
                        f"[REST] {sym} {interval}: REST 데이터 없음 (start~end) → FAILURE"
                    )
                    upsert_rest_progress(
                        run_id,
                        sym,
                        interval,
                        "FAILURE",
                        None,
                        "No REST data returned in range",
                    )
                    continue

                # (3) DB 덮어쓰기 (해당 구간 전체 삭제 후, REST 데이터 insert/upsert)
                from decimal import Decimal

                with SyncSessionLocal() as session, session.begin():
                    # 해당 구간 기존 데이터 삭제
                    session.execute(
                        delete(model).where(
                            model.symbol == sym,
                            model.timestamp >= start_ts,
                            model.timestamp <= end_ts_for_db,
                        )
                    )

                    last_filled_ts: Optional[datetime] = None

                    for item in klines:
                        open_time_ms = int(item[0])
                        open_time_dt = datetime.fromtimestamp(
                            open_time_ms / 1000, tz=timezone.utc
                        )
                        # 이론상 fetch_klines_range 에서 이미 end_ms를 넘는건 거르지만 한 번 더 방어
                        if open_time_ms > end_ms:
                            continue

                        o = Decimal(item[1])
                        h = Decimal(item[2])
                        l = Decimal(item[3])
                        c = Decimal(item[4])
                        v = Decimal(item[5])

                        stmt = (
                            insert(model)
                            .values(
                                symbol=sym,
                                timestamp=open_time_dt,
                                open=o,
                                high=h,
                                low=l,
                                close=c,
                                volume=v,
                                is_ended=True,
                            )
                            .on_conflict_do_update(
                                index_elements=["symbol", "timestamp"],
                                set_={
                                    "open": o,
                                    "high": h,
                                    "low": l,
                                    "close": c,
                                    "volume": v,
                                    "is_ended": True,
                                },
                            )
                        )
                        session.execute(stmt)
                        last_filled_ts = open_time_dt

                # (4) 성공 기록
                upsert_rest_progress(
                    run_id, sym, interval, "SUCCESS", last_filled_ts, None
                )
                success_cnt += 1
                logger.info(
                    f"[REST] {sym} {interval}: 유지보수 완료 "
                    f"{start_ts.isoformat()} ~ {end_ts_for_db.isoformat()}"
                )

    finally:
        try:
            client.close()
        except Exception:
            pass

    logger.info(f"[REST] total_jobs={total_jobs}, success_cnt={success_cnt}")

    overall_status = "SUCCESS" if total_jobs == success_cnt else "PARTIAL"
    return {
        "status": overall_status,
        "run_id": run_id,
        "total_jobs": total_jobs,
        "success_cnt": success_cnt,
    }
