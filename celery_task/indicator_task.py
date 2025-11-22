import uuid
from datetime import datetime
from typing import Optional

import pandas as pd
import pandas_ta as ta
from celery import Task
from loguru import logger
from sqlalchemy import select, text, func
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
# 짧은 인터벌(1m~30m)은 배치 처리로 메모리 절약
INTERVALS = ["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M"]


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


def _load_ohlcv_incremental(
    symbol: str, interval: str, last_indicator_ts: Optional[datetime] = None
) -> Optional[pd.DataFrame]:
    """
    증분 계산용 OHLCV 로드.
    
    - last_indicator_ts가 None이면: 전체 데이터 로드 (최초 계산)
    - last_indicator_ts가 있으면: 그 이후 데이터만 로드
      + 단, 보조지표 계산을 위해 lookback 기간(100개) 포함
    
    Args:
        symbol: 심볼
        interval: 인터벌
        last_indicator_ts: 마지막으로 계산된 지표의 timestamp
        
    Returns:
        OHLCV DataFrame (index=timestamp) 또는 None
    """
    OhlcvModel = OHLCV_MODELS.get(interval)
    if OhlcvModel is None:
        logger.error(f"[indicator] 지원하지 않는 인터벌: {interval}")
        return None

    with SyncSessionLocal() as session:
        # 기본 쿼리: is_ended=True인 캔들만
        stmt = select(
            OhlcvModel.timestamp,
            OhlcvModel.open,
            OhlcvModel.high,
            OhlcvModel.low,
            OhlcvModel.close,
            OhlcvModel.volume,
        ).where(OhlcvModel.symbol == symbol, OhlcvModel.is_ended.is_(True))

        if last_indicator_ts is not None:
            # 증분 계산: last_indicator_ts 이후 데이터만
            # 단, lookback 기간을 위해 100개 이전부터 로드
            
            # 1) last_indicator_ts 이후의 모든 데이터
            stmt_new = stmt.where(OhlcvModel.timestamp > last_indicator_ts)
            
            # 2) last_indicator_ts 이전 100개 (warm-up용)
            stmt_lookback = (
                select(
                    OhlcvModel.timestamp,
                    OhlcvModel.open,
                    OhlcvModel.high,
                    OhlcvModel.low,
                    OhlcvModel.close,
                    OhlcvModel.volume,
                )
                .where(
                    OhlcvModel.symbol == symbol,
                    OhlcvModel.is_ended.is_(True),
                    OhlcvModel.timestamp <= last_indicator_ts,
                )
                .order_by(OhlcvModel.timestamp.desc())
                .limit(100)
            )
            
            # 3) 두 쿼리 결과 합치기
            rows_new = session.execute(stmt_new.order_by(OhlcvModel.timestamp.asc())).all()
            rows_lookback = session.execute(stmt_lookback).all()
            
            # lookback은 desc로 가져왔으니 reverse
            rows = list(reversed(rows_lookback)) + list(rows_new)
            
        else:
            # 최초 계산: 전체 데이터
            stmt = stmt.order_by(OhlcvModel.timestamp.asc())
            rows = session.execute(stmt).all()

    if not rows:
        return None

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


def _get_last_indicator_timestamp(symbol: str, interval: str) -> Optional[datetime]:
    """
    indicators_{interval} 테이블에서 해당 symbol의 마지막 timestamp 조회.
    
    Returns:
        마지막 timestamp 또는 None (데이터 없으면)
    """
    IndicatorModel = INDICATOR_MODELS.get(interval)
    if IndicatorModel is None:
        return None
    
    with SyncSessionLocal() as session:
        result = (
            session.query(IndicatorModel.timestamp)
            .filter(IndicatorModel.symbol == symbol)
            .order_by(IndicatorModel.timestamp.desc())
            .limit(1)
            .first()
        )
    
    return result[0] if result else None


def _process_indicator_in_batches(
    symbol: str,
    interval: str,
    batch_months: int = 3,
    lookback_count: int = 100
) -> int:
    """
    배치 단위로 보조지표 계산 (메모리 절약)
    
    대량의 캔들 데이터를 한 번에 로드하면 OOM이 발생하므로,
    시간 범위를 여러 배치로 나눠서 처리합니다.
    
    Args:
        symbol: 심볼
        interval: 인터벌
        batch_months: 배치 크기 (개월 단위)
        lookback_count: 각 배치에 포함할 이전 캔들 개수 (연속성 유지용)
    
    Returns:
        저장된 레코드 수
    """
    from datetime import timedelta
    from dateutil.relativedelta import relativedelta
    
    OhlcvModel = OHLCV_MODELS.get(interval)
    IndicatorModel = INDICATOR_MODELS.get(interval)
    
    if not OhlcvModel or not IndicatorModel:
        logger.error(f"[indicator_batch] 지원하지 않는 인터벌: {interval}")
        return 0
    
    # 1. OHLCV 데이터의 시간 범위 조회
    with SyncSessionLocal() as session:
        result = session.execute(
            select(
                func.min(OhlcvModel.timestamp),
                func.max(OhlcvModel.timestamp)
            ).where(
                OhlcvModel.symbol == symbol,
                OhlcvModel.is_ended.is_(True)
            )
        ).first()
        
        if not result or not result[0]:
            logger.info(f"[indicator_batch] {symbol} {interval}: OHLCV 데이터 없음")
            return 0
        
        min_ts, max_ts = result
    
    # 2. 이미 계산된 indicator의 마지막 timestamp 조회
    last_indicator_ts = _get_last_indicator_timestamp(symbol, interval)
    
    if last_indicator_ts:
        # 증분 계산: 마지막 이후부터 처리
        start_ts = last_indicator_ts
        logger.info(
            f"[indicator_batch] {symbol} {interval}: "
            f"증분 계산 시작 (마지막={start_ts.isoformat()})"
        )
    else:
        # 최초 계산: 전체 처리
        start_ts = min_ts
        logger.info(
            f"[indicator_batch] {symbol} {interval}: "
            f"최초 계산 시작 (범위={min_ts.isoformat()} ~ {max_ts.isoformat()})"
        )
    
    # 3. 배치 단위로 처리
    total_saved = 0
    current_start = start_ts
    batch_num = 0
    
    while current_start < max_ts:
        batch_num += 1
        # 배치 종료 시각 계산 (N개월 후)
        batch_end = current_start + relativedelta(months=batch_months)
        if batch_end > max_ts:
            batch_end = max_ts
        
        logger.info(
            f"[indicator_batch] {symbol} {interval} 배치 #{batch_num}: "
            f"{current_start.isoformat()} ~ {batch_end.isoformat()}"
        )
        
        # 배치 데이터 + lookback 로드
        with SyncSessionLocal() as session:
            # lookback용: current_start 이전 N개
            stmt_lookback = (
                select(
                    OhlcvModel.timestamp,
                    OhlcvModel.open,
                    OhlcvModel.high,
                    OhlcvModel.low,
                    OhlcvModel.close,
                    OhlcvModel.volume,
                )
                .where(
                    OhlcvModel.symbol == symbol,
                    OhlcvModel.is_ended.is_(True),
                    OhlcvModel.timestamp < current_start,
                )
                .order_by(OhlcvModel.timestamp.desc())
                .limit(lookback_count)
            )
            
            # 실제 배치: current_start ~ batch_end
            stmt_batch = (
                select(
                    OhlcvModel.timestamp,
                    OhlcvModel.open,
                    OhlcvModel.high,
                    OhlcvModel.low,
                    OhlcvModel.close,
                    OhlcvModel.volume,
                )
                .where(
                    OhlcvModel.symbol == symbol,
                    OhlcvModel.is_ended.is_(True),
                    OhlcvModel.timestamp >= current_start,
                    OhlcvModel.timestamp < batch_end,
                )
                .order_by(OhlcvModel.timestamp.asc())
            )
            
            rows_lookback = session.execute(stmt_lookback).all()
            rows_batch = session.execute(stmt_batch).all()
        
        if not rows_batch:
            logger.info(f"[indicator_batch] 배치 #{batch_num}: 데이터 없음, 종료")
            break
        
        # lookback 역순 정렬 + batch 합치기
        rows_combined = list(reversed(rows_lookback)) + list(rows_batch)
        
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
                for r in rows_combined
            ]
        )
        df = df.set_index("timestamp")
        
        # 보조지표 계산
        df_ind = _compute_indicators(df)
        
        if df_ind.empty:
            logger.warning(f"[indicator_batch] 배치 #{batch_num}: 지표 계산 결과 없음")
            current_start = batch_end
            continue
        
        # lookback 제외하고 current_start 이후만 저장
        df_to_save = df_ind[df_ind.index >= current_start]
        
        if df_to_save.empty:
            logger.info(f"[indicator_batch] 배치 #{batch_num}: 저장할 데이터 없음")
        else:
            saved = _upsert_indicators(symbol, interval, df_to_save, only_last=False)
            total_saved += saved
            logger.info(
                f"[indicator_batch] 배치 #{batch_num}: {saved}개 저장 "
                f"(누적={total_saved})"
            )
        
        # 다음 배치로 이동
        current_start = batch_end
    
    logger.info(
        f"[indicator_batch] {symbol} {interval}: "
        f"배치 처리 완료 (총 {batch_num}개 배치, {total_saved}개 저장)"
    )
    return total_saved



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

    # 실제 DB에 넣을 컬럼만 추출
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
    for col in wanted_cols:
        if col not in df.columns:
            logger.warning(
                f"[indicator] missing column '{col}' in computed df, filling with NaN"
            )
            df[col] = pd.NA

    # ema_99는 99개의 캔들이 필요하므로 1M 같은 경우 데이터가 부족할 수 있음
    # ema_99를 제외한 컬럼들만 dropna() 적용
    required_cols = [
        "rsi_14",
        "ema_7",
        "ema_21",
        # "ema_99",  # 제외: nullable
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "volume_20",
    ]
    
    # required_cols에 대해서만 dropna 수행
    df_ind = df[wanted_cols].dropna(subset=required_cols)
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
                # 짧은 인터벌은 배치 처리 사용 (메모리 절약)
                if interval in ("1m", "3m", "5m", "15m", "30m"):
                    # 배치 크기 결정
                    if interval == "1m":
                        batch_months = 1  # 1개월 (~43,000 캔들)
                    elif interval == "3m":
                        batch_months = 2  # 2개월 (~28,000 캔들)
                    else:
                        batch_months = 3  # 3개월 (~12,000~18,000 캔들)
                    
                    # 작업 시작
                    upsert_indicator_progress(
                        run_id, sym, interval, "PROGRESS", 0.0, None, None
                    )
                    
                    # 배치 처리 실행
                    saved_count = _process_indicator_in_batches(
                        sym, interval, batch_months=batch_months
                    )
                    
                    if saved_count > 0:
                        # 성공: 최신 timestamp 조회
                        last_ts = _get_last_indicator_timestamp(sym, interval)
                        upsert_indicator_progress(
                            run_id, sym, interval, "SUCCESS", 100.0, last_ts, None
                        )
                        success += 1
                        logger.info(
                            f"[Indicator] {sym} {interval}: 배치 처리 완료 ({saved_count}개)"
                        )
                    else:
                        # 데이터 없음
                        upsert_indicator_progress(
                            run_id, sym, interval, "SUCCESS", 100.0, None, None
                        )
                        success += 1
                    continue
                
                # 긴 인터벌은 기존 증분 계산 사용
                # 증분 계산: 마지막으로 계산된 timestamp 조회
                last_indicator_ts = _get_last_indicator_timestamp(sym, interval)
                
                if last_indicator_ts:
                    logger.info(
                        f"[Indicator] {sym} {interval}: 증분 계산 (마지막={last_indicator_ts.isoformat()})"
                    )
                else:
                    logger.info(f"[Indicator] {sym} {interval}: 최초 계산")
                
                # 증분 로드: 마지막 이후 데이터만 + lookback 100개
                df_ohlcv = _load_ohlcv_incremental(sym, interval, last_indicator_ts)

                # OHLCV 자체가 없으면 "유지보수할 것 없음" → SUCCESS(100)
                if df_ohlcv is None or df_ohlcv.empty:
                    upsert_indicator_progress(
                        run_id, sym, interval, "SUCCESS", 100.0, None, None
                    )
                    success += 1
                    continue

                # 새로운 데이터가 있는지 확인
                if last_indicator_ts is not None:
                    # last_indicator_ts 이후의 데이터만 필터링
                    df_new = df_ohlcv[df_ohlcv.index > last_indicator_ts]
                    
                    if df_new.empty:
                        # 새로운 데이터 없음 → 이미 최신 상태
                        logger.info(f"[Indicator] {sym} {interval}: 새 데이터 없음 (이미 최신)")
                        upsert_indicator_progress(
                            run_id, sym, interval, "SUCCESS", 100.0, last_indicator_ts, None
                        )
                        success += 1
                        continue
                    
                    logger.info(
                        f"[Indicator] {sym} {interval}: {len(df_new)}개 새 캔들 발견"
                    )

                # 작업 시작
                upsert_indicator_progress(
                    run_id, sym, interval, "PROGRESS", 0.0, None, None
                )

                # 전체 데이터로 지표 계산 (lookback 포함)
                df_ind = _compute_indicators(df_ohlcv)
                if df_ind.empty:
                    # 지표 계산 결과가 없으면 그냥 성공 취급
                    upsert_indicator_progress(
                        run_id, sym, interval, "SUCCESS", 100.0, None, None
                    )
                    success += 1
                    continue

                # 증분 저장: 마지막 이후 데이터만 저장
                if last_indicator_ts is not None:
                    # last_indicator_ts 이후의 지표만 저장
                    df_ind_to_save = df_ind[df_ind.index > last_indicator_ts]
                    
                    if df_ind_to_save.empty:
                        logger.warning(
                            f"[Indicator] {sym} {interval}: 계산했지만 저장할 지표 없음"
                        )
                        upsert_indicator_progress(
                            run_id, sym, interval, "SUCCESS", 100.0, last_indicator_ts, None
                        )
                        success += 1
                        continue
                    
                    saved_count = _upsert_indicators(sym, interval, df_ind_to_save, only_last=False)
                    logger.info(
                        f"[Indicator] {sym} {interval}: {saved_count}개 지표 저장 (증분)"
                    )
                else:
                    # 최초 계산: 전체 저장
                    saved_count = _upsert_indicators(sym, interval, df_ind, only_last=False)
                    logger.info(
                        f"[Indicator] {sym} {interval}: {saved_count}개 지표 저장 (최초)"
                    )
                
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
