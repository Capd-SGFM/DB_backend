import uuid
from datetime import datetime
from typing import Optional

import pandas as pd
import pandas_ta as ta
from celery import Task
from loguru import logger
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert

from db_module.connect_sqlalchemy_engine import SyncSessionLocal
from models import OHLCV_MODELS, INDICATOR_MODELS, CryptoInfo
from models.indicator_progress import IndicatorProgress
from models.pipeline_state import (
    is_pipeline_active,
    set_component_error,
    PipelineComponent,
)
from . import celery_app


# Indicator 유지보수 대상 인터벌(Backfill/REST와 맞춤)
INTERVALS = ["1h", "4h", "1d", "1w", "1M"]


# =========================================================
#   공통: OHLCV 로딩 + 보조지표 계산 + UPSERT
# =========================================================
def _load_ohlcv_ended_df(
    symbol: str, interval: str, limit: Optional[int] = None
) -> Optional[pd.DataFrame]:
    """
    trading_data.ohlcv_{interval} 에서
    symbol, is_ended = true 인 캔들을 timestamp 오름차순으로 DataFrame으로 로드.
    limit가 지정되면 '가장 최신 limit개'만 사용.
    """
    OhlcvModel = OHLCV_MODELS.get(interval)
    if OhlcvModel is None:
        logger.error(f"[indicator] 지원하지 않는 인터벌: {interval}")
        return None

    with SyncSessionLocal() as session:
        stmt = (
            select(
                OhlcvModel.timestamp,
                OhlcvModel.open,
                OhlcvModel.high,
                OhlcvModel.low,
                OhlcvModel.close,
                OhlcvModel.volume,
            )
            .where(OhlcvModel.symbol == symbol, OhlcvModel.is_ended.is_(True))
            .order_by(OhlcvModel.timestamp.desc())
        )

        if limit is not None:
            stmt = stmt.limit(limit)

        rows = session.execute(stmt).all()

    if not rows:
        return None

    # 최신 → 과거 순으로 가져왔으니 다시 정렬
    df = pd.DataFrame(
        [
            {
                "timestamp": r[0],
                "open": float(r[1]),
                "high": float(r[2]),
                "low": float(r[3]),
                "close": float(r[4]),
                "volume": float(r[5]),
            }
            for r in rows
        ]
    )

    df = df.sort_values("timestamp").set_index("timestamp")
    return df


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    OHLCV DataFrame(index=timestamp)에 보조지표 컬럼들을 계산해서 리턴.
    (rsi_14, ema_7, ema_21, ema_99, macd, macd_signal, macd_hist, bb_*, volume_20)
    """
    if df.empty:
        return df

    # pandas_ta가 요구하는 컬럼명이 대강 맞을 거라서 그대로 사용
    df.ta.rsi(length=14, append=True, col_names=("rsi_14",))
    df.ta.ema(length=7, append=True, col_names=("ema_7",))
    df.ta.ema(length=21, append=True, col_names=("ema_21",))
    df.ta.ema(length=99, append=True, col_names=("ema_99",))

    # macd: (fast=12, slow=26, signal=9 기본값)
    df.ta.macd(append=True, col_names=("macd", "macd_hist", "macd_signal"))

    # Bollinger Bands (20, 2)
    df.ta.bbands(
        length=20,
        std=2.0,
        append=True,
        col_names=("bb_lower", "bb_middle", "bb_upper", "bb_bandwidth", "bb_percent"),
    )

    # volume 20 이동평균
    df["volume_20"] = df["volume"].rolling(20).mean()

    # 실제 DB에 넣을 컬럼만 추출 + NaN 제거
    wanted_cols = [
        "rsi_14",
        "ema_7",
        "ema_21",
        "ema_99",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "volume_20",
    ]
    df_ind = df[wanted_cols].dropna()

    return df_ind


def _upsert_indicators(
    symbol: str, interval: str, df_ind: pd.DataFrame, only_last: bool = False
) -> int:
    """
    계산된 보조지표 df_ind(index=timestamp)를
    trading_data.indicators_{interval}에 UPSERT.

    only_last = True면 마지막 1개만 upsert (실시간 업데이트용).
    리턴: upsert된 row 개수
    """
    IndicatorModel = INDICATOR_MODELS.get(interval)
    if IndicatorModel is None:
        logger.error(
            f"[indicator] 지원하지 않는 인터벌(IndicatorModel 없음): {interval}"
        )
        return 0

    if df_ind.empty:
        return 0

    if only_last:
        df_ind = df_ind.tail(1)

    df_reset = df_ind.reset_index()  # timestamp 컬럼으로 복구
    records = []
    for _, row in df_reset.iterrows():
        records.append(
            {
                "symbol": symbol,
                "timestamp": row["timestamp"],
                "rsi_14": float(row["rsi_14"]),
                "ema_7": float(row["ema_7"]),
                "ema_21": float(row["ema_21"]),
                "ema_99": float(row["ema_99"]) if not pd.isna(row["ema_99"]) else None,
                "macd": float(row["macd"]),
                "macd_signal": float(row["macd_signal"]),
                "macd_hist": float(row["macd_hist"]),
                "bb_upper": float(row["bb_upper"]),
                "bb_middle": float(row["bb_middle"]),
                "bb_lower": float(row["bb_lower"]),
                "volume_20": float(row["volume_20"]),
            }
        )

    if not records:
        return 0

    with SyncSessionLocal() as session, session.begin():
        stmt = insert(IndicatorModel).values(records)
        keys = records[0].keys()

        update_cols = {
            k: getattr(stmt.excluded, k)
            for k in keys
            if k not in ("symbol", "timestamp")
        }

        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol", "timestamp"],
            set_=update_cols,
        )
        session.execute(stmt)

    return len(records)


# =========================================================
#   indicator_progress UPSERT (유지보수 엔진 UI용)
# =========================================================
def upsert_indicator_progress(
    run_id: str,
    symbol: str,
    interval: str,
    state: str,
    pct_time: float = 0.0,
    last_ts: Optional[datetime] = None,
    error: Optional[str] = None,
):
    """trading_data.indicator_progress UPSERT."""
    if not run_id:
        return

    with SyncSessionLocal() as session, session.begin():
        stmt = (
            insert(IndicatorProgress)
            .values(
                run_id=run_id,
                symbol=symbol,
                interval=interval,
                state=state,
                pct_time=pct_time,
                last_candle_ts=last_ts,
                last_error=error,
            )
            .on_conflict_do_update(
                index_elements=["run_id", "symbol", "interval"],
                set_={
                    "state": state,
                    "pct_time": pct_time,
                    "last_candle_ts": last_ts,
                    "last_error": error,
                    "updated_at": text("now()"),
                },
            )
        )
        session.execute(stmt)


# =====================
# ① 최초 대량 계산(심볼/인터벌 단위) — 필요시 사용
# =====================
@celery_app.task(bind=True, name="indicator.bulk_init_indicators_symbol_interval")
def bulk_init_indicators_symbol_interval(
    self: Task, symbol: str, interval: str
) -> dict:
    """
    최초 대량 보조지표 계산 태스크.
    - 해당 symbol, interval에 대해 is_ended = true 인 모든 캔들을 기준으로
      보조지표를 전부 다시 계산해서 indicators_* 테이블에 upsert.
    - 파이프라인이 OFF 상태면 바로 SKIP 리턴.
    """
    # 파이프라인 OFF면 작업 스킵
    if not is_pipeline_active():
        logger.info(
            f"[indicator.bulk_init] pipeline inactive -> skip ({symbol} {interval})"
        )
        return {
            "status": "SKIP_PIPELINE_OFF",
            "symbol": symbol,
            "interval": interval,
        }

    logger.info(f"[indicator.bulk_init] start: {symbol} {interval}")

    try:
        df_ohlcv = _load_ohlcv_ended_df(symbol, interval, limit=None)
        if df_ohlcv is None or df_ohlcv.empty:
            logger.warning(
                f"[indicator.bulk_init] no OHLCV data for {symbol} {interval}"
            )
            return {
                "status": "NO_DATA",
                "symbol": symbol,
                "interval": interval,
            }

        df_ind = _compute_indicators(df_ohlcv)
        if df_ind.empty:
            logger.warning(
                f"[indicator.bulk_init] indicators empty for {symbol} {interval}"
            )
            return {
                "status": "EMPTY_INDICATORS",
                "symbol": symbol,
                "interval": interval,
            }

        count = _upsert_indicators(symbol, interval, df_ind, only_last=False)

        logger.info(f"[indicator.bulk_init] done {symbol} {interval} (rows={count})")
        return {
            "status": "COMPLETE",
            "symbol": symbol,
            "interval": interval,
            "rows": count,
        }

    except Exception as e:
        msg = f"bulk_init failed for {symbol} {interval}: {type(e).__name__}: {e}"
        logger.error(f"[indicator.bulk_init] {msg}")
        # Indicator 엔진 에러 기록 (id=5)
        try:
            set_component_error(PipelineComponent.INDICATOR, msg)
        except Exception:
            logger.exception("[indicator.bulk_init] failed to save last_error")
        # Celery 쪽에도 실패로 남기기
        raise


# =====================
# ② 실시간용 per-symbol 태스크 (웹소켓에서 사용)
# =====================
@celery_app.task(bind=True, name="indicator.update_last_indicator_for_symbol_interval")
def update_last_indicator_for_symbol_interval(
    self: Task, symbol: str, interval: str
) -> dict:
    """
    실시간 유지보수용 태스크.
    - 최근 N개 OHLCV만 불러와서 지표 계산
    - 마지막 1개만 indicators_*에 upsert
    - websocket_task에서 '캔들이 닫혔다(is_ended=True)' 이벤트 발생 시 호출하는 용도
    - 파이프라인이 OFF면 바로 SKIP.
    """
    if not is_pipeline_active():
        logger.info(
            f"[indicator.update_last] pipeline inactive -> skip ({symbol} {interval})"
        )
        return {
            "status": "SKIP_PIPELINE_OFF",
            "symbol": symbol,
            "interval": interval,
        }

    logger.info(f"[indicator.update_last] start: {symbol} {interval}")

    try:
        df_ohlcv = _load_ohlcv_ended_df(symbol, interval, limit=200)
        if df_ohlcv is None or df_ohlcv.empty:
            logger.warning(
                f"[indicator.update_last] no OHLCV data for {symbol} {interval}"
            )
            return {
                "status": "NO_DATA",
                "symbol": symbol,
                "interval": interval,
            }

        df_ind = _compute_indicators(df_ohlcv)
        if df_ind.empty:
            logger.warning(
                f"[indicator.update_last] indicators empty for {symbol} {interval}"
            )
            return {
                "status": "EMPTY_INDICATORS",
                "symbol": symbol,
                "interval": interval,
            }

        count = _upsert_indicators(symbol, interval, df_ind, only_last=True)
        last_ts: datetime = df_ind.index[-1]

        logger.info(
            f"[indicator.update_last] done {symbol} {interval} "
            f"(rows={count}, last_ts={last_ts.isoformat()})"
        )

        return {
            "status": "COMPLETE",
            "symbol": symbol,
            "interval": interval,
            "rows": count,
            "last_timestamp": last_ts.isoformat(),
        }

    except Exception as e:
        msg = f"update_last failed for {symbol} {interval}: {type(e).__name__}: {e}"
        logger.error(f"[indicator.update_last] {msg}")
        try:
            set_component_error(PipelineComponent.INDICATOR, msg)
        except Exception:
            logger.exception("[indicator.update_last] failed to save last_error")
        raise


# =====================
# ③ 파이프라인용 Indicator 유지보수 엔진
#     (모든 심볼×인터벌을 한 번에 돌리고 진행현황 저장)
# =====================
@celery_app.task(name="indicator.run_indicator_maintenance")
def run_indicator_maintenance() -> dict:
    """
    파이프라인 Maintenance 사이클에서 호출되는 보조지표 엔진.
    - 모든 심볼 × INTERVALS 에 대해:
        * OHLCV(is_ended=True) 로부터 보조지표 전부 재계산
        * indicators_{interval} 에 upsert
        * indicator_progress 에 PENDING/PROGRESS/SUCCESS/FAILURE 기록
    - 진행률 pct_time 은 간단히
        * 작업 시작 시 0
        * 계산/저장 완료 시 100 으로만 사용 (세밀한 %는 생략)
    """
    logger.info("[Indicator] 유지보수 엔진 시작")

    if not is_pipeline_active():
        logger.info("[Indicator] pipeline inactive → 종료")
        return {"status": "INACTIVE"}

    run_id = f"ind-{uuid.uuid4().hex}"
    logger.info(f"[Indicator] run_id={run_id}")

    # 1) 심볼 목록 조회
    with SyncSessionLocal() as session:
        symbols = (
            session.query(CryptoInfo.symbol, CryptoInfo.pair)
            .filter(CryptoInfo.pair.isnot(None))
            .all()
        )

    # 2) 모든 심볼×인터벌에 PENDING dummy row 생성
    with SyncSessionLocal() as session, session.begin():
        for sym, _ in symbols:
            for interval in INTERVALS:
                stmt = (
                    insert(IndicatorProgress)
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

    # 3) 실제 계산 루프
    total = 0
    success = 0

    for sym, _pair in symbols:
        if not is_pipeline_active():
            logger.info("[Indicator] pipeline OFF → 중단")
            return {"status": "STOPPED", "run_id": run_id}

        for interval in INTERVALS:
            total += 1
            try:
                df_ohlcv = _load_ohlcv_ended_df(sym, interval, limit=None)

                # OHLCV 자체가 없으면 "유지보수할 것 없음" → SUCCESS(100)
                if df_ohlcv is None or df_ohlcv.empty:
                    upsert_indicator_progress(
                        run_id, sym, interval, "SUCCESS", 100.0, None, None
                    )
                    continue

                # 작업 시작
                upsert_indicator_progress(
                    run_id, sym, interval, "PROGRESS", 0.0, None, None
                )

                df_ind = _compute_indicators(df_ohlcv)
                if df_ind.empty:
                    # 지표 계산 결과가 없으면 그냥 성공 취급
                    upsert_indicator_progress(
                        run_id, sym, interval, "SUCCESS", 100.0, None, None
                    )
                    success += 1
                    continue

                _upsert_indicators(sym, interval, df_ind, only_last=False)
                last_ts = df_ind.index[-1]
                if isinstance(last_ts, pd.Timestamp):
                    last_ts = last_ts.to_pydatetime()

                upsert_indicator_progress(
                    run_id,
                    sym,
                    interval,
                    "SUCCESS",
                    100.0,
                    last_ts,
                    None,
                )
                success += 1

            except Exception as e:
                logger.exception(f"[Indicator] 유지보수 오류: {sym} {interval}: {e!r}")
                try:
                    set_component_error(PipelineComponent.INDICATOR, str(e))
                except Exception:
                    logger.exception(
                        "[Indicator] failed to save last_error to pipeline_state"
                    )

                upsert_indicator_progress(
                    run_id, sym, interval, "FAILURE", 0.0, None, str(e)
                )

    logger.info(
        f"[Indicator] 유지보수 완료: total={total}, success={success}, run_id={run_id}"
    )
    return {
        "status": "SUCCESS" if total == success else "PARTIAL",
        "run_id": run_id,
        "total": total,
        "success": success,
    }
