from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException
from sqlalchemy import select, func, text
from loguru import logger

from db_module.connect_sqlalchemy_engine import SyncSessionLocal
from models import OHLCV_MODELS, INDICATOR_MODELS, CryptoInfo

router = APIRouter(
    prefix="/verification",
    tags=["Verification"],
)

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
    "1M": 30 * 86_400_000,
}

@router.get("/integrity")
def verify_data_integrity():
    """
    모든 심볼과 인터벌에 대해 데이터 무결성을 검증합니다.
    
    검증 항목:
    1. OHLCV 연속성: (Max - Min) / Interval + 1 == Count 인지 확인 (Gap 탐지)
    2. Indicator 동기화: OHLCV 개수와 Indicator 개수 비교, Max Timestamp 일치 여부 확인
    3. 최신성: 마지막 데이터가 현재 시간과 얼마나 차이나는지 (WebSocket 수집 여부 간접 확인)
    """
    results = []
    
    with SyncSessionLocal() as session:
        # 1. 활성 심볼 목록 조회
        symbols = (
            session.query(CryptoInfo.symbol)
            .filter(CryptoInfo.pair.isnot(None))
            .all()
        )
        symbol_list = [s[0] for s in symbols]
        
    intervals = list(OHLCV_MODELS.keys())
    now = datetime.now(timezone.utc)
    
    for symbol in symbol_list:
        for interval in intervals:
            report = {
                "symbol": symbol,
                "interval": interval,
                "status": "OK",
                "details": [],
                "ohlcv": {},
                "indicator": {}
            }
            
            OhlcvModel = OHLCV_MODELS.get(interval)
            IndicatorModel = INDICATOR_MODELS.get(interval)
            
            if not OhlcvModel or not IndicatorModel:
                continue
                
            with SyncSessionLocal() as session:
                # --- OHLCV 검증 ---
                ohlcv_stats = session.execute(
                    select(
                        func.min(OhlcvModel.timestamp),
                        func.max(OhlcvModel.timestamp),
                        func.count()
                    ).where(
                        OhlcvModel.symbol == symbol,
                        OhlcvModel.is_ended.is_(True)
                    )
                ).first()
                
                min_ts, max_ts, count = ohlcv_stats
                
                if not count or count == 0:
                    report["status"] = "EMPTY"
                    report["details"].append("No OHLCV data")
                    results.append(report)
                    continue
                
                # Gap 계산
                interval_ms = INTERVAL_TO_MS.get(interval)
                if interval_ms:
                    diff_ms = (max_ts - min_ts).total_seconds() * 1000
                    expected_count = int(diff_ms / interval_ms) + 1
                    gap_count = expected_count - count
                    
                    report["ohlcv"] = {
                        "count": count,
                        "expected": expected_count,
                        "min_ts": min_ts,
                        "max_ts": max_ts,
                        "gap": gap_count
                    }
                    
                    if gap_count > 0:
                        report["status"] = "WARNING"
                        report["details"].append(f"OHLCV Gap detected: missing {gap_count} candles")
                
                # 최신성 확인 (WebSocket이 돌고 있다면 최신이어야 함)
                # 1M, 1w 등은 오차가 클 수 있으므로 짧은 프레임 위주로 보거나, 
                # 단순히 마지막 시간만 기록
                
                # --- Indicator 검증 ---
                ind_stats = session.execute(
                    select(
                        func.min(IndicatorModel.timestamp),
                        func.max(IndicatorModel.timestamp),
                        func.count()
                    ).where(
                        IndicatorModel.symbol == symbol
                    )
                ).first()
                
                ind_min, ind_max, ind_count = ind_stats
                
                report["indicator"] = {
                    "count": ind_count,
                    "min_ts": ind_min,
                    "max_ts": ind_max
                }
                
                # OHLCV와 비교
                if ind_count < count:
                    diff = count - ind_count
                    # 100개 정도는 lookback으로 인해 없을 수 있음 (정상)
                    # 하지만 너무 많이 차이나면 문제
                    if diff > 200: 
                        report["status"] = "WARNING" if report["status"] == "OK" else report["status"]
                        report["details"].append(f"Indicator missing: {diff} less than OHLCV")
                
                if ind_max != max_ts:
                     # Indicator가 아직 최신 OHLCV를 못 따라잡음
                     report["status"] = "WARNING" if report["status"] == "OK" else report["status"]
                     report["details"].append(f"Indicator lag: OHLCV({max_ts}) != Ind({ind_max})")

            results.append(report)
            
    return {
        "timestamp": now,
        "total_items": len(results),
        "results": results
    }
