import time
from datetime import datetime, timezone
from typing import Tuple, Optional

import httpx
import pandas as pd
from celery import Task
from loguru import logger
from sqlalchemy import select, func, text
from sqlalchemy.dialects.postgresql import insert

from db_module.connect_sqlalchemy_engine import SyncSessionLocal
from models import OHLCV_MODELS  # 🔹 보조지표 모델은 백필에서 사용 안 함
from models.backfill_progress import BackfillProgress
from models.pipeline_state import (
    is_pipeline_active,
    set_component_error,
    PipelineComponent,
)
from . import celery_app

# ──────────────────────────────────────────────────────────────
#  Binance API 기본 설정
# ──────────────────────────────────────────────────────────────
BINANCE_FAPI_URL = "https://fapi.binance.com/fapi/v1/klines"
KLINE_LIMIT = 1000

# Rate Limit 제어용 상수 (바이낸스 공식 기준)
RATE_LIMIT_HEADER = "x-mbx-used-weight-1m"
MAX_WEIGHT_PER_MINUTE = 2400
SAFETY_MARGIN_PERCENT = 0.8
TARGET_WEIGHT = MAX_WEIGHT_PER_MINUTE * SAFETY_MARGIN_PERCENT

# 인터벌별 밀리초
INTERVAL_TO_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
    "1w": 7 * 86_400_000,
    "1M": 30 * 86_400_000,  # Binance 1M 정의는 실제 달이지만, interval 길이 추정용
}


# ──────────────────────────────────────────────────────────────
#  헬퍼: backfill_progress upsert
# ──────────────────────────────────────────────────────────────
def upsert_backfill_progress(
    run_id: str,
    symbol: str,
    interval: str,
    state: str,
    pct_time: float,
    last_candle_ts: Optional[datetime],
    last_error: Optional[str],
):
    """
    trading_data.backfill_progress 에 현재 상태를 UPSERT.
    """
    if not run_id:
        return

    with SyncSessionLocal() as session, session.begin():
        stmt = insert(BackfillProgress).values(
            run_id=run_id,
            symbol=symbol,
            interval=interval,
            state=state,
            pct_time=pct_time,
            last_candle_ts=last_candle_ts,
            last_error=last_error,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["run_id", "symbol", "interval"],
            set_={
                "state": stmt.excluded.state,
                "pct_time": stmt.excluded.pct_time,
                "last_candle_ts": stmt.excluded.last_candle_ts,
                "last_error": stmt.excluded.last_error,
                "updated_at": text("now()"),
            },
        )
        session.execute(stmt)


# ──────────────────────────────────────────────────────────────
#  DB에서 REST 백필 시작 시각 계산
# ──────────────────────────────────────────────────────────────
def get_start_time_ms(
    symbol: str,
    interval: str,
    OhlcvModel,
    ws_frontier_ms: int,
) -> Tuple[Optional[int], bool]:
    """
    ws_frontier_ms(밀리초) 이전 구간에서,
    is_ended = TRUE 인 마지막 캔들의 다음 캔들부터 백필하기 위해
    startTime(ms)를 계산한다.

    return:
        (start_time_ms, has_any_row)
        - start_time_ms: None 이면 'is_ended=TRUE 기준 시작점을 찾지 못했다'는 의미
        - has_any_row: 이 심볼/인터벌에 row가 하나라도 있는지 여부
    """
    ws_frontier_dt = datetime.fromtimestamp(ws_frontier_ms / 1000, tz=timezone.utc)

    with SyncSessionLocal() as session:
        # 이 심볼/인터벌에 row가 하나라도 있는지 (데이터는 있는데 ended가 없을 수도 있음)
        total_count = session.execute(
            select(func.count()).where(OhlcvModel.symbol == symbol)
        ).scalar_one()
        has_any_row = total_count > 0

        # ws_frontier 이전에서 is_ended = TRUE인 마지막 캔들
        stmt = select(func.max(OhlcvModel.timestamp)).where(
            OhlcvModel.symbol == symbol,
            OhlcvModel.is_ended == True,  # noqa: E712
            OhlcvModel.timestamp < ws_frontier_dt,
        )
        latest_timestamp: Optional[datetime] = session.execute(
            stmt
        ).scalar_one_or_none()

        if latest_timestamp:
            interval_ms = INTERVAL_TO_MS.get(interval, 60_000)
            start_ms = int(latest_timestamp.timestamp() * 1000) + interval_ms
            return start_ms, has_any_row
        else:
            # ended 캔들을 기준으로 한 시작점은 없음
            return None, has_any_row


# ──────────────────────────────────────────────────────────────
#  OHLCV 저장 (REST: 항상 is_ended = TRUE 기준)
#   ⚠️ 백필에서는 보조지표를 계산/저장하지 않는다!
# ──────────────────────────────────────────────────────────────
def save_data(OhlcvModel, symbol: str, all_klines: list) -> int:
    """
    all_klines:
      [
        {
          "symbol": str,
          "open_time_ms": int,
          "open": float,
          "high": float,
          "low": float,
          "close": float,
          "volume": float,
          "is_ended": bool,  # REST에서는 항상 True
        }, ...
      ]

    백필 단계에서는 OHLCV 테이블만 UPSERT하고,
    보조지표(indicators_*)는 별도의 보조지표 엔진에서 계산/저장한다.
    """
    if not all_klines:
        return 0

    # DataFrame 변환
    df = pd.DataFrame(all_klines)
    df["timestamp"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)

    # OHLCV 저장용
    ohlcv_data_to_save = df[
        ["symbol", "timestamp", "open", "high", "low", "close", "volume", "is_ended"]
    ].to_dict("records")

    # ---- DB 저장 (OHLCV UPSERT만 수행) ----
    with SyncSessionLocal() as session:
        with session.begin():
            if ohlcv_data_to_save:
                ohlcv_stmt = insert(OhlcvModel).values(ohlcv_data_to_save)
                ohlcv_keys = ohlcv_data_to_save[0].keys()
                update_ohlcv_cols = {
                    key: getattr(ohlcv_stmt.excluded, key)
                    for key in ohlcv_keys
                    if key not in ["symbol", "timestamp"]
                }
                ohlcv_stmt = ohlcv_stmt.on_conflict_do_update(
                    index_elements=["symbol", "timestamp"],
                    set_=update_ohlcv_cols,
                )
                session.execute(ohlcv_stmt)

    return len(ohlcv_data_to_save)


# ──────────────────────────────────────────────────────────────
#  REST 백필 Celery Task
#   - ws_frontier_ms 이전 구간만 수집
#   - REST로 저장하는 캔들은 항상 is_ended = TRUE
#   - 파이프라인(id=1)이 OFF 되면 중간에도 종료
#   - 예외 발생 시 pipeline_state(BACKFILL)의 last_error에 기록
#   - ⚠️ 백필에서는 indicators_* 테이블에 전혀 쓰지 않음
# ──────────────────────────────────────────────────────────────
@celery_app.task(bind=True, name="ohlcv.backfill_symbol_interval")
def backfill_symbol_interval(
    self: Task,
    symbol: str,
    pair: str,
    interval: str,
    ws_frontier_ms: Optional[int] = None,  # ✅ Optional 로 변경
    run_id: str | None = None,
):
    """
    REST 백필 태스크 (동기):
      - ws_frontier_ms 이전 구간만 백필
      - DB 상태에 따라:
        1) 해당 심볼/인터벌에 is_ended = TRUE 캔들이 있으면 → 그 이후부터 ws_frontier 직전까지 증분 백필
        2) row는 있는데 is_ended = TRUE가 하나도 없거나, 아예 데이터가 없으면 →
           Binance에서 가져올 수 있는 가장 오래된 캔들부터 ws_frontier 직전까지 전체 백필
      - REST가 저장하는 모든 캔들은 is_ended = TRUE 로 저장
      - 보조지표(indicators_*) 저장은 하지 않음 (별도 엔진 담당)
    """
    OhlcvModel = OHLCV_MODELS.get(interval)
    if not OhlcvModel:
        raise ValueError(f"지원하지 않는 인터벌입니다: {interval}")

    # 한번에 메모리에 보관했다 저장할 최대 개수 (대량 수집 대비)
    BATCH_SAVE_SIZE = 50_000
    all_klines_data = []
    total_saved_count = 0
    last_known_pct = 0.0

    try:
        # ───── 파이프라인 OFF 면 바로 종료 ─────
        if not is_pipeline_active():
            logger.info(f"[{symbol}-{interval}] pipeline inactive → skip backfill.")
            if run_id:
                upsert_backfill_progress(
                    run_id,
                    symbol,
                    interval,
                    state="PENDING",
                    pct_time=0.0,
                    last_candle_ts=None,
                    last_error=None,
                )
            return {
                "status": "SKIP",
                "symbol": symbol,
                "interval": interval,
                "saved_count": 0,
            }

        # ───── 0. ws_frontier_ms 없으면 Binance serverTime 으로 자동 설정 ─────
        if ws_frontier_ms is None:
            with httpx.Client(timeout=30.0) as client:
                time_res = client.get("https://fapi.binance.com/fapi/v1/time")
                time_res.raise_for_status()
                ws_frontier_ms = int(time_res.json()["serverTime"])

        # ───── 1. DB 상태 + ws_frontier 기반 시작 시각 결정 ─────
        db_start_time_ms, has_any_row = get_start_time_ms(
            symbol, interval, OhlcvModel, ws_frontier_ms
        )

        current_start_time_ms: Optional[int] = None
        progress_start_ms: Optional[int] = None
        progress_end_ms: int = ws_frontier_ms  # REST 백필 목표 구간 끝점

        if has_any_row and db_start_time_ms is not None:
            # (1) 데이터 있고 is_ended = TRUE도 있음 → 마지막 ended 캔들 이후부터 시작
            logger.info(
                f"[{symbol}-{interval}] 증분 백필 시작 (db_start_time_ms={db_start_time_ms}, ws_frontier_ms={ws_frontier_ms})"
            )
            current_start_time_ms = db_start_time_ms
            progress_start_ms = db_start_time_ms

        else:
            # (2) row는 있는데 is_ended=TRUE가 하나도 없거나, 아예 데이터가 없음 → 전체 백필
            logger.info(
                f"[{symbol}-{interval}] 전체 백필 시작 (DB 비어있거나 is_ended=TRUE 없음). "
                "Binance에서 실제 첫 캔들 시간 조회..."
            )
            with httpx.Client(timeout=30.0) as client:
                params = {
                    "symbol": pair,
                    "interval": interval,
                    "startTime": 1,
                    "limit": 1,
                }
                res = client.get(BINANCE_FAPI_URL, params=params)
                res.raise_for_status()
                first_candle_data = res.json()

                if not first_candle_data:
                    logger.warning(
                        f"[{symbol}-{interval}] API에 데이터가 전혀 없습니다. 작업 종료."
                    )
                    if run_id:
                        upsert_backfill_progress(
                            run_id,
                            symbol,
                            interval,
                            state="SUCCESS",
                            pct_time=100.0,
                            last_candle_ts=None,
                            last_error=None,
                        )
                    return {
                        "status": "COMPLETE",
                        "symbol": symbol,
                        "interval": interval,
                        "saved_count": 0,
                    }

                actual_first_candle_ms = int(first_candle_data[0][0])
                current_start_time_ms = actual_first_candle_ms
                progress_start_ms = actual_first_candle_ms

                first_candle_dt = datetime.fromtimestamp(
                    actual_first_candle_ms / 1000, tz=timezone.utc
                )
                logger.info(
                    f"[{symbol}-{interval}] 실제 시작 시간 확인: {first_candle_dt.isoformat()}, "
                    f"ws_frontier_ms={ws_frontier_ms}"
                )

        if current_start_time_ms is None or progress_start_ms is None:
            raise Exception("Start time could not be determined.")

        # 이미 ws_frontier보다 뒤면 할 일이 없음
        if current_start_time_ms >= ws_frontier_ms:
            logger.info(
                f"[{symbol}-{interval}] current_start_time_ms({current_start_time_ms}) >= ws_frontier_ms({ws_frontier_ms}) "
                "→ 백필할 구간이 없습니다."
            )
            if run_id:
                upsert_backfill_progress(
                    run_id,
                    symbol,
                    interval,
                    state="SUCCESS",
                    pct_time=100.0,
                    last_candle_ts=None,
                    last_error=None,
                )
            return {
                "status": "COMPLETE",
                "symbol": symbol,
                "interval": interval,
                "saved_count": 0,
            }

        interval_ms = INTERVAL_TO_MS.get(interval, 60_000)

        # ───── 2. Binance REST 루프 (ws_frontier 이전까지만) ─────
        with httpx.Client(timeout=30.0) as client:
            while True:
                # 파이프라인이 중간에 꺼지면 종료
                if not is_pipeline_active():
                    logger.info(
                        f"[{symbol}-{interval}] pipeline OFF 감지 → backfill 중단."
                    )
                    if run_id:
                        upsert_backfill_progress(
                            run_id,
                            symbol,
                            interval,
                            state="PENDING",
                            pct_time=last_known_pct,
                            last_candle_ts=None,
                            last_error=None,
                        )
                    break

                # 더 이상 수집할 구간이 없으면 종료
                if current_start_time_ms >= ws_frontier_ms:
                    logger.info(
                        f"[{symbol}-{interval}] ws_frontier({ws_frontier_ms})까지 모두 수집하여 종료."
                    )
                    break

                params = {
                    "symbol": pair,
                    "interval": interval,
                    "limit": KLINE_LIMIT,
                    "startTime": current_start_time_ms,
                    "endTime": ws_frontier_ms - 1,  # WebSocket 담당 구간 직전까지만
                }

                try:
                    res = client.get(BINANCE_FAPI_URL, params=params)

                    # 레이트 리밋 보호
                    if res.status_code in (429, 418):
                        retry_after = res.headers.get("Retry-After")
                        sleep_time = 60
                        if retry_after and retry_after.isdigit():
                            sleep_time = int(retry_after)

                        logger.warning(
                            f"[{symbol}-{interval}] Rate limit hit (Status {res.status_code}). "
                            f"Sleeping for {sleep_time} seconds..."
                        )
                        self.update_state(
                            state="PROGRESS",
                            meta={
                                "symbol": symbol,
                                "interval": interval,
                                "pct": last_known_pct,
                                "last_candle_time": datetime.fromtimestamp(
                                    current_start_time_ms / 1000, tz=timezone.utc
                                ).isoformat(),
                                "status": f"Rate limit. Paused for {sleep_time}s.",
                            },
                        )
                        if run_id:
                            upsert_backfill_progress(
                                run_id,
                                symbol,
                                interval,
                                state="PROGRESS",
                                pct_time=last_known_pct,
                                last_candle_ts=datetime.fromtimestamp(
                                    current_start_time_ms / 1000, tz=timezone.utc
                                ),
                                last_error=None,
                            )
                        time.sleep(sleep_time)
                        continue

                    res.raise_for_status()
                    klines = res.json()

                except httpx.HTTPStatusError as e:
                    logger.error(f"[{symbol}-{interval}] HTTP Error: {e}")
                    raise Exception(f"HTTP Error: {e.response.status_code}")
                except httpx.RequestError as e:
                    logger.error(f"[{symbol}-{interval}] Connection Error: {e}")
                    raise Exception(f"Connection Error: {e}")

                if not klines:
                    logger.info(
                        f"[{symbol}-{interval}] API가 빈 목록을 반환. 루프 종료."
                    )
                    break

                new_klines_count = 0
                last_saved_open_ms: Optional[int] = None

                for k in klines:
                    open_time_ms = int(k[0])

                    # 방어적: ws_frontier 이후의 캔들은 REST 백필 대상이 아님
                    if open_time_ms >= ws_frontier_ms:
                        continue

                    # 증분 모드라면, 이전 ended 이후 시점만
                    if db_start_time_ms and open_time_ms < db_start_time_ms:
                        continue

                    all_klines_data.append(
                        {
                            "symbol": symbol,
                            "open_time_ms": open_time_ms,
                            "open": float(k[1]),
                            "high": float(k[2]),
                            "low": float(k[3]),
                            "close": float(k[4]),
                            "volume": float(k[5]),
                            # REST로 들어오는 캔들은 모두 '닫힌 캔들'로 보고 is_ended = TRUE
                            "is_ended": True,
                        }
                    )
                    new_klines_count += 1
                    last_saved_open_ms = open_time_ms

                # 진행률 계산 (ws_frontier 기준)
                if (
                    last_saved_open_ms is not None
                    and progress_start_ms is not None
                    and progress_end_ms > progress_start_ms
                ):
                    pct = (
                        (last_saved_open_ms - progress_start_ms)
                        / (progress_end_ms - progress_start_ms)
                    ) * 100
                else:
                    pct = 0

                last_known_pct = min(round(pct, 2), 100.0)

                last_ts = datetime.fromtimestamp(
                    (last_saved_open_ms or current_start_time_ms) / 1000,
                    tz=timezone.utc,
                )

                self.update_state(
                    state="PROGRESS",
                    meta={
                        "symbol": symbol,
                        "interval": interval,
                        "pct": last_known_pct,
                        "last_candle_time": last_ts.isoformat(),
                        "status": "Running...",
                    },
                )

                if run_id:
                    upsert_backfill_progress(
                        run_id,
                        symbol,
                        interval,
                        state="PROGRESS",
                        pct_time=last_known_pct,
                        last_candle_ts=last_ts,
                        last_error=None,
                    )

                # 메모리 배치 저장
                if len(all_klines_data) >= BATCH_SAVE_SIZE:
                    logger.info(
                        f"[{symbol}-{interval}] 메모리 배치 {len(all_klines_data)}개 저장 시도..."
                    )
                    saved_in_batch = save_data(OhlcvModel, symbol, all_klines_data)
                    total_saved_count += saved_in_batch
                    all_klines_data.clear()

                # 새로 저장된 캔들이 하나도 없다면 종료
                if new_klines_count == 0:
                    logger.info(
                        f"[{symbol}-{interval}] 새로 저장된 캔들이 없으므로 루프 종료."
                    )
                    break

                # 다음 배치 시작 시각 결정
                if last_saved_open_ms is None:
                    logger.info(
                        f"[{symbol}-{interval}] last_saved_open_ms가 없어 루프 종료."
                    )
                    break

                current_start_time_ms = last_saved_open_ms + interval_ms

                # ws_frontier에 도달했으면 종료
                if current_start_time_ms >= ws_frontier_ms:
                    logger.info(
                        f"[{symbol}-{interval}] ws_frontier({ws_frontier_ms})까지 수집 완료."
                    )
                    break

                # Rate Limit 완화용 weight 체크
                try:
                    used_weight = int(res.headers.get(RATE_LIMIT_HEADER, "0"))
                    if used_weight > TARGET_WEIGHT:
                        sleep_duration = 10
                        logger.warning(
                            f"[{symbol}-{interval}] High weight ({used_weight}). "
                            f"Pausing for {sleep_duration}s."
                        )
                        self.update_state(
                            state="PROGRESS",
                            meta={
                                "symbol": symbol,
                                "interval": interval,
                                "pct": last_known_pct,
                                "last_candle_time": datetime.fromtimestamp(
                                    last_saved_open_ms / 1000, tz=timezone.utc
                                ).isoformat(),
                                "status": f"Pacing weight. Paused for {sleep_duration}s.",
                            },
                        )
                        if run_id:
                            upsert_backfill_progress(
                                run_id,
                                symbol,
                                interval,
                                state="PROGRESS",
                                pct_time=last_known_pct,
                                last_candle_ts=datetime.fromtimestamp(
                                    last_saved_open_ms / 1000, tz=timezone.utc
                                ),
                                last_error=None,
                            )
                        time.sleep(sleep_duration)
                except Exception:
                    time.sleep(0.5)

        # ───── 3. 남은 메모리 배치 저장 ─────
        if all_klines_data:
            logger.info(
                f"[{symbol}-{interval}] 마지막 남은 배치 {len(all_klines_data)}개 캔들 저장 시도..."
            )
            saved_in_batch = save_data(OhlcvModel, symbol, all_klines_data)
            total_saved_count += saved_in_batch
            all_klines_data.clear()

        # 성공적으로 끝난 경우 상태 SUCCESS 로 기록
        if run_id:
            upsert_backfill_progress(
                run_id,
                symbol,
                interval,
                state="SUCCESS",
                pct_time=100.0,
                last_candle_ts=None,
                last_error=None,
            )

        return {
            "status": "COMPLETE",
            "symbol": symbol,
            "interval": interval,
            "saved_count": total_saved_count,
        }

    except Exception as e:
        logger.error(
            f"Task {getattr(self.request, 'id', 'unknown')} "
            f"(Symbol: {symbol}, Interval: {interval}) failed: {e}"
        )
        # BACKFILL 컴포넌트 에러 기록
        try:
            set_component_error(
                PipelineComponent.BACKFILL,
                f"{type(e).__name__}: {e}",
            )
        except Exception:
            logger.exception("[BACKFILL] failed to save last_error")

        # backfill_progress 에 FAILURE 기록
        if run_id:
            upsert_backfill_progress(
                run_id,
                symbol,
                interval,
                state="FAILURE",
                pct_time=last_known_pct,
                last_candle_ts=None,
                last_error=str(e),
            )

        raise Exception(f"Task failed for {symbol} {interval}: {str(e)}")
