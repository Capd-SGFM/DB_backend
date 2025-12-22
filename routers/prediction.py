# routers/prediction.py
"""
LSTM 모델 예측 API
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from datetime import datetime, timezone, timedelta
from typing import Optional
import numpy as np
from loguru import logger
import os
import json
import uuid
import asyncio

from db_module.connect_sqlalchemy_engine import SyncSessionLocal
from sqlalchemy import text
from services.lstm_predictor import _run_fast_backtest_numpy, FEE_MAKER, FEE_TAKER

router = APIRouter(prefix="/predict", tags=["Prediction"])


# ========================
#  Request/Response Models
# ========================

class PredictRequest(BaseModel):
    symbol: str  # 예: "ETH"
    interval: str  # 예: "15m"
    timestamp: datetime  # 예측 시점
    model_name: str  # 모델 파일명


class FeatureData(BaseModel):
    timestamp: str
    log_return: Optional[float]
    ema_ratio: Optional[float]
    macd_hist: Optional[float]
    bandwidth: Optional[float]
    pct_b: Optional[float]
    rsi: Optional[float]
    mfi: Optional[float]


class PredictResponse(BaseModel):
    model_name: str
    seq_length: int
    symbol: str
    interval: str
    target_timestamp: str
    predictions: dict  # {sl_probability, tp_probability, neither_probability}
    input_features: list[FeatureData]





class FindModelRequest(BaseModel):
    interval: str  # 예: "15m", "4h"
    sl_pct: float  # SL 배수 (예: 1.0)
    rr_ratio: float  # RR 비율 (예: 1.0, 1.5, 2.0)
    m_value: int  # M 파라미터 (예: 20, 40, 60)
    rank: int  # 랭크 (예: 1, 2, 3)


class FindModelResponse(BaseModel):
    found: bool
    filename: Optional[str] = None
    full_path: Optional[str] = None  # Full path for model loading
    seq_length: Optional[int] = None
    interval: Optional[str] = None
    sl_pct: Optional[float] = None
    rr_ratio: Optional[float] = None
    m_value: Optional[int] = None
    rank: Optional[int] = None


@router.post("/find-model", response_model=FindModelResponse)
def find_model(request: FindModelRequest):
    """
    파라미터로 모델 파일 검색 (seq는 파일명에서 자동 추출)
    """
    from services.lstm_predictor import find_model_by_params
    
    result = find_model_by_params(
        interval=request.interval,
        sl_pct=request.sl_pct,
        rr_ratio=request.rr_ratio,
        m_value=request.m_value,
        rank=request.rank,
    )
    
    if result is None:
        return FindModelResponse(found=False)
    
    return FindModelResponse(
        found=True,
        filename=result["filename"],
        full_path=result["full_path"],
        seq_length=result["seq_length"],
        interval=result["interval"],
        sl_pct=result["sl_pct"],
        rr_ratio=result["rr_ratio"],
        m_value=result["m_value"],
        rank=result["rank"],
    )


class GridSearchRequest(BaseModel):
    interval: str  # 예: "15m", "4h"
    sl_pct: float  # SL 배수 (예: 1.0)
    rr_ratio: float  # RR 비율 (예: 1.0, 1.5, 2.0)
    m_value: int  # M 파라미터 (예: 20, 40, 60)


class GridSearchDataPoint(BaseModel):
    seq_len: int
    score: float


class GridSearchResponse(BaseModel):
    found: bool
    data: list[GridSearchDataPoint] = []
    available_ranks: list[dict] = []  # [{rank: 1, seq_len: 40, score: 0.7}, ...]
    message: Optional[str] = None


@router.post("/grid-search", response_model=GridSearchResponse)
def get_grid_search_results(request: GridSearchRequest):
    """
    Grid Search 결과 조회 - Rank 선택 전에 seq_len vs score 그래프용 데이터 제공
    """
    import csv
    import re
    from pathlib import Path
    
    # Build directory path: {interval}/BTC_{interval}_{sl}_{rr}_{m}
    subdir = f"{request.interval}/BTC_{request.interval}_{request.sl_pct:.1f}_{request.rr_ratio:.1f}_{request.m_value}"
    search_dir = Path("/app/ml_models") / subdir
    csv_path = search_dir / "grid_search_results.csv"
    
    if not csv_path.exists():
        logger.warning(f"[Prediction] Grid search CSV not found: {csv_path}")
        return GridSearchResponse(found=False, message=f"CSV not found: {subdir}/grid_search_results.csv")
    
    try:
        data_points = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                seq_len = int(row.get('seq_len', 0))
                # Find score column (might be score_risk_X.XX)
                score = 0.0
                for key in row:
                    if key.startswith('score'):
                        score = float(row[key])
                        break
                data_points.append(GridSearchDataPoint(seq_len=seq_len, score=score))
        
        # Sort by seq_len
        data_points.sort(key=lambda x: x.seq_len)
        
        # Find available models (ranks) in this directory
        available_ranks = []
        if search_dir.exists():
            for f in search_dir.glob("lstm_*.keras"):
                # Extract rank and seq from filename
                rank_match = re.search(r'rank(\d+)', f.stem)
                seq_match = re.search(r'seq(\d+)', f.stem)
                if rank_match and seq_match:
                    rank = int(rank_match.group(1))
                    seq = int(seq_match.group(1))
                    # Find score for this seq_len
                    score = next((dp.score for dp in data_points if dp.seq_len == seq), 0.0)
                    available_ranks.append({
                        "rank": rank,
                        "seq_len": seq,
                        "score": score,
                        "filename": f.name
                    })
        
        # Sort by rank
        available_ranks.sort(key=lambda x: x["rank"])
        
        logger.info(f"[Prediction] Grid search data loaded: {len(data_points)} points, {len(available_ranks)} ranks")
        
        return GridSearchResponse(found=True, data=data_points, available_ranks=available_ranks)
    
    except Exception as e:
        logger.error(f"[Prediction] Failed to read grid search CSV: {e}")
        return GridSearchResponse(found=False, message=str(e))


class ModelHistoryRequest(BaseModel):
    interval: str  # 예: "15m", "4h"
    sl_pct: float
    rr_ratio: float
    m_value: int
    rank: int
    seq_len: int


class HistoryDataPoint(BaseModel):
    epoch: int
    loss: float
    accuracy: float
    val_loss: float
    val_accuracy: float


class ModelHistoryResponse(BaseModel):
    found: bool
    data: list[HistoryDataPoint] = []
    filename: Optional[str] = None
    message: Optional[str] = None


@router.post("/model-history", response_model=ModelHistoryResponse)
def get_model_history(request: ModelHistoryRequest):
    """
    모델 학습 히스토리 조회 - loss, accuracy, val_loss, val_accuracy 시각화용
    """
    import csv
    from pathlib import Path
    
    # Build directory path: {interval}/BTC_{interval}_{sl}_{rr}_{m}
    subdir = f"{request.interval}/BTC_{request.interval}_{request.sl_pct:.1f}_{request.rr_ratio:.1f}_{request.m_value}"
    search_dir = Path("/app/ml_models") / subdir
    
    # History filename pattern: history_{interval}_{sl}_{rr}_{m}_rank{rank}_seq{seq}.csv
    history_filename = f"history_{request.interval}_{request.sl_pct:.1f}_{request.rr_ratio:.1f}_{request.m_value}_rank{request.rank}_seq{request.seq_len}.csv"
    csv_path = search_dir / history_filename
    
    if not csv_path.exists():
        logger.warning(f"[Prediction] Model history CSV not found: {csv_path}")
        return ModelHistoryResponse(found=False, message=f"History not found: {history_filename}")
    
    try:
        data_points = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for epoch, row in enumerate(reader, start=1):
                data_points.append(HistoryDataPoint(
                    epoch=epoch,
                    loss=float(row.get('loss', 0)),
                    accuracy=float(row.get('accuracy', 0)),
                    val_loss=float(row.get('val_loss', 0)),
                    val_accuracy=float(row.get('val_accuracy', 0)),
                ))
        
        logger.info(f"[Prediction] Model history loaded: {len(data_points)} epochs from {history_filename}")
        
        return ModelHistoryResponse(found=True, data=data_points, filename=history_filename)
    
    except Exception as e:
        logger.error(f"[Prediction] Failed to read model history CSV: {e}")
        return ModelHistoryResponse(found=False, message=str(e))


@router.post("/lstm", response_model=PredictResponse)
def predict_lstm(request: PredictRequest):
    """
    LSTM 모델을 사용한 예측
    
    - symbol: 심볼 (예: ETH)
    - interval: 인터벌 (예: 15m)
    - timestamp: 예측 시점 (이 시점 이전 n개 캔들 사용)
    - model_name: 모델 파일명
    """
    from services.lstm_predictor import get_predictor
    
    try:
        # 1. Load Model
        predictor = get_predictor(request.model_name)
        seq_length = predictor.seq_length
        
        # 2. Fetch Features from DB
        table_name = f"backtesting_features_{request.interval}"
        
        # Ensure timestamp has timezone
        target_ts = request.timestamp
        if target_ts.tzinfo is None:
            target_ts = target_ts.replace(tzinfo=timezone.utc)
        
        query = text(f"""
            SELECT timestamp, log_return, ema_ratio, macd_hist, bandwidth, pct_b, rsi, mfi
            FROM trading_data.{table_name}
            WHERE symbol = :symbol
              AND timestamp <= :target_ts
            ORDER BY timestamp DESC
            LIMIT :limit
        """)
        
        with SyncSessionLocal() as session:
            result = session.execute(query, {
                "symbol": request.symbol,
                "target_ts": target_ts,
                "limit": seq_length
            }).fetchall()
        
        if len(result) < seq_length:
            raise HTTPException(
                status_code=400, 
                detail=f"Not enough data. Need {seq_length} candles, got {len(result)}"
            )
        
        # 3. Prepare Features Array
        # Reverse to chronological order (oldest first)
        rows = list(reversed(result))
        
        feature_names = ["log_return", "ema_ratio", "macd_hist", "bandwidth", "pct_b", "rsi", "mfi"]
        features = np.zeros((1, seq_length, 7), dtype=np.float32)
        
        input_features = []
        for i, row in enumerate(rows):
            for j, fname in enumerate(feature_names):
                val = getattr(row, fname) if hasattr(row, fname) else row[j + 1]
                features[0, i, j] = float(val) if val is not None else 0.0
            
            input_features.append(FeatureData(
                timestamp=row[0].isoformat() if hasattr(row[0], 'isoformat') else str(row[0]),
                log_return=float(row[1]) if row[1] is not None else None,
                ema_ratio=float(row[2]) if row[2] is not None else None,
                macd_hist=float(row[3]) if row[3] is not None else None,
                bandwidth=float(row[4]) if row[4] is not None else None,
                pct_b=float(row[5]) if row[5] is not None else None,
                rsi=float(row[6]) if row[6] is not None else None,
                mfi=float(row[7]) if row[7] is not None else None,
            ))
        
        # 4. Run Prediction
        predictions = predictor.predict(features)
        
        return PredictResponse(
            model_name=request.model_name,
            seq_length=seq_length,
            symbol=request.symbol,
            interval=request.interval,
            target_timestamp=target_ts.isoformat(),
            predictions=predictions,
            input_features=input_features,
        )
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Model not found: {e}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"[Prediction] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))





# ========================
#  Batch Scan Models
# ========================

class BatchScanRequest(BaseModel):
    symbol: str
    interval: str
    start_time: datetime
    end_time: datetime
    model_name: str
    tp_threshold: float = 0.6  # TP 확률 임계값 (기본 60%)


class ScanResult(BaseModel):
    timestamp: str
    sl_probability: float
    tp_probability: float
    neither_probability: float


class BatchScanResponse(BaseModel):
    model_name: str
    symbol: str
    interval: str
    start_time: str
    end_time: str
    tp_threshold: float
    total_scanned: int
    matches_found: int
    results: list[ScanResult]


@router.post("/lstm/scan", response_model=BatchScanResponse)
def batch_scan_lstm(request: BatchScanRequest):
    """
    기간 내 모든 timestamp를 스캔하여 TP 확률이 임계값을 초과하는 시점을 찾습니다.
    배치 처리로 몇 년치 데이터도 빠르게 스캔 가능합니다.
    
    - symbol: 심볼 (예: ETH)
    - interval: 인터벌 (예: 15m)
    - start_time: 스캔 시작 시점
    - end_time: 스캔 종료 시점
    - model_name: 모델 파일명
    - tp_threshold: TP 확률 임계값 (기본 0.6 = 60%)
    """
    from services.lstm_predictor import get_predictor
    
    try:
        # 1. Load Model
        predictor = get_predictor(request.model_name)
        seq_length = predictor.seq_length
        
        # 2. Ensure timestamps have timezone
        start_ts = request.start_time
        end_ts = request.end_time
        if start_ts.tzinfo is None:
            start_ts = start_ts.replace(tzinfo=timezone.utc)
        if end_ts.tzinfo is None:
            end_ts = end_ts.replace(tzinfo=timezone.utc)
        
        # 3. Load ALL feature data in range (including seq_length before start for first window)
        table_name = f"backtesting_features_{request.interval}"
        
        # We need seq_length rows before start_ts to form the first prediction window
        data_query = text(f"""
            WITH all_data AS (
                SELECT timestamp, log_return, ema_ratio, macd_hist, bandwidth, pct_b, rsi, mfi,
                       ROW_NUMBER() OVER (ORDER BY timestamp) as rn
                FROM trading_data.{table_name}
                WHERE symbol = :symbol
                  AND timestamp <= :end_ts
                ORDER BY timestamp
            ),
            start_rn AS (
                SELECT MIN(rn) as min_rn FROM all_data WHERE timestamp >= :start_ts
            )
            SELECT timestamp, log_return, ema_ratio, macd_hist, bandwidth, pct_b, rsi, mfi
            FROM all_data
            WHERE rn >= (SELECT GREATEST(1, min_rn - :seq_length + 1) FROM start_rn)
            ORDER BY timestamp
        """)
        
        with SyncSessionLocal() as session:
            all_data = session.execute(data_query, {
                "symbol": request.symbol,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "seq_length": seq_length
            }).fetchall()
        
        if len(all_data) < seq_length:
            return BatchScanResponse(
                model_name=request.model_name,
                symbol=request.symbol,
                interval=request.interval,
                start_time=start_ts.isoformat(),
                end_time=end_ts.isoformat(),
                tp_threshold=request.tp_threshold,
                total_scanned=0,
                matches_found=0,
                results=[]
            )
        
        # 4. Convert to numpy arrays
        timestamps = [row[0] for row in all_data]
        features_raw = np.array([[
            float(row[1]) if row[1] is not None else 0.0,
            float(row[2]) if row[2] is not None else 0.0,
            float(row[3]) if row[3] is not None else 0.0,
            float(row[4]) if row[4] is not None else 0.0,
            float(row[5]) if row[5] is not None else 0.0,
            float(row[6]) if row[6] is not None else 0.0,
            float(row[7]) if row[7] is not None else 0.0,
        ] for row in all_data], dtype=np.float32)
        
        # 5. Find indices where timestamp >= start_ts
        scan_start_idx = 0
        for i, ts in enumerate(timestamps):
            if ts >= start_ts:
                scan_start_idx = i
                break
        
        # We need at least seq_length rows before scan_start_idx
        if scan_start_idx < seq_length - 1:
            # Adjust - not enough history for some early timestamps
            scan_start_idx = seq_length - 1
        
        # 6. Build sliding windows for batch prediction
        num_windows = len(timestamps) - seq_length + 1 - (scan_start_idx - (seq_length - 1))
        if num_windows <= 0:
            return BatchScanResponse(
                model_name=request.model_name,
                symbol=request.symbol,
                interval=request.interval,
                start_time=start_ts.isoformat(),
                end_time=end_ts.isoformat(),
                tp_threshold=request.tp_threshold,
                total_scanned=0,
                matches_found=0,
                results=[]
            )
        
        # Create windows efficiently using stride tricks
        valid_indices = list(range(scan_start_idx, len(timestamps)))
        num_samples = len(valid_indices)
        
        logger.info(f"[BatchScan] Building {num_samples} windows for prediction")
        
        # Pre-allocate feature array
        features_batch = np.zeros((num_samples, seq_length, 7), dtype=np.float32)
        
        for i, end_idx in enumerate(valid_indices):
            start_idx = end_idx - seq_length + 1
            features_batch[i] = features_raw[start_idx:end_idx + 1]
        
        # 7. Run batch prediction
        logger.info(f"[BatchScan] Running batch prediction on {num_samples} samples")
        predictions = predictor.batch_predict(features_batch)
        
        # 8. Filter by TP threshold
        results: list[ScanResult] = []
        for i, end_idx in enumerate(valid_indices):
            tp_prob = float(predictions[i][1])
            if tp_prob >= request.tp_threshold:
                results.append(ScanResult(
                    timestamp=timestamps[end_idx].isoformat(),
                    sl_probability=float(predictions[i][0]),
                    tp_probability=tp_prob,
                    neither_probability=float(predictions[i][2])
                ))
        
        # Sort by TP probability (highest first)
        results.sort(key=lambda x: x.tp_probability, reverse=True)
        
        logger.info(f"[BatchScan] Found {len(results)} matches out of {num_samples} scanned")
        
        return BatchScanResponse(
            model_name=request.model_name,
            symbol=request.symbol,
            interval=request.interval,
            start_time=start_ts.isoformat(),
            end_time=end_ts.isoformat(),
            tp_threshold=request.tp_threshold,
            total_scanned=num_samples,
            matches_found=len(results),
            results=results
        )
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Model not found: {e}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"[Prediction] Batch scan error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception(f"[Prediction] Batch scan error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================
#  Backtesting Simulation
# ========================

class BacktestRequest(BaseModel):
    symbol: str
    interval: str
    start_time: datetime
    end_time: datetime
    model_name: str
    tp_threshold: float = 0.6
    sl_mult: float  # From model
    rr_ratio: float # From model
    m_value: int    # From model
    risk_pct: float = 0.02 # Fixed Risk per Trade (e.g. 2%)
    allow_overlap: bool = False
    max_positions: int = 10  # Maximum concurrent positions (only for overlap mode)

class BacktestTrade(BaseModel):
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    side: str = "Long" # Only Long for now? (Or assume model predicts Long). LSTM usually predicts direction. But implicit here is Long if TP/SL logic is E - ATR.
    reason: str # "TP", "SL", "Time"
    pnl_pct: float # Realized PnL % (with leverage)
    leverage: float
    raw_pnl: float # Unleveraged PnL
    holding_bars: int

class BacktestSummary(BaseModel):
    total_trades: int
    win_rate: float
    total_pnl_pct: float
    avg_pnl_pct: float
    max_drawdown_pct: float
    avg_leverage: float

class BacktestResponse(BaseModel):
    summary: BacktestSummary
    trades: list[BacktestTrade]

@router.post("/lstm/backtest")
def backtest_lstm(request: BacktestRequest, background_tasks: BackgroundTasks):
    """
    Async Backtest Dispatch
    """
    job_id = str(uuid.uuid4())
    background_jobs[job_id] = {
        "status": "queued",
        "progress": "Pending...",
        "data": None,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    save_jobs()
    
    background_tasks.add_task(run_backtest_task, job_id, request)
    
    return {"job_id": job_id, "message": "Backtest simulation started in background."}


@router.get("/lstm/backtest/{job_id}")
def get_backtest_status(job_id: str):
    """
    Poll status of backtest job.
    """
    if job_id not in background_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Return full job info including data if completed
    return background_jobs[job_id]


# ========================
#  Cross Margin Backtest (V2)
# ========================

class CrossMarginBacktestRequest(BaseModel):
    symbol: str
    interval: str
    start_time: datetime
    end_time: datetime
    model_name: str
    tp_threshold: float = 0.6
    sl_mult: float
    rr_ratio: float
    m_value: int
    risk_pct: float = 0.02
    initial_balance: float = 10000.0  # NEW: Starting balance
    max_leverage: float = 50.0  # NEW: Maximum effective leverage


class CrossMarginBacktestTrade(BaseModel):
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    side: str = "Long"
    reason: str
    pnl_pct: float
    pnl_usdt: float
    position_size: float
    leverage: float
    holding_bars: int
    balance_after: float


class CrossMarginBacktestSummary(BaseModel):
    initial_balance: float
    final_balance: float
    total_pnl_usdt: float
    total_pnl_pct: float
    total_trades: int
    win_rate: float
    max_drawdown_pct: float


class CrossMarginBacktestResponse(BaseModel):
    summary: CrossMarginBacktestSummary
    trades: list[CrossMarginBacktestTrade]


@router.post("/lstm/backtest-v2")
def backtest_lstm_v2(request: CrossMarginBacktestRequest, background_tasks: BackgroundTasks):
    """
    Cross Margin Backtest - 잔고 추적 + 동적 레버리지
    """
    job_id = str(uuid.uuid4())
    background_jobs[job_id] = {
        "status": "queued",
        "progress": "Pending...",
        "data": None,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    save_jobs()
    
    background_tasks.add_task(run_cross_margin_backtest_task, job_id, request)
    
    return {"job_id": job_id, "message": "Cross Margin Backtest started."}





class OptimizeRequest(BaseModel):
    symbol: str
    interval: str
    allow_overlap: bool = False

class OptimizationResultItem(BaseModel):
    rank: int
    model_name: str
    threshold: float
    max_pnl: float
    avg_pnl: float
    win_rate: float
    total_trades: int

# ========================
#  Async Jobs (Optimization & Backtesting)
# ========================

import uuid
from fastapi import BackgroundTasks
import json
import os
from datetime import datetime, timezone
from sqlalchemy import text
from db_module.connect_sqlalchemy_engine import SyncSessionLocal


# Simple persistent job store
JOBS_FILE = "jobs_store.json"

def load_jobs():
    if os.path.exists(JOBS_FILE):
        try:
            with open(JOBS_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load jobs: {e}")
            return {}
    return {}

background_jobs = load_jobs()

def save_jobs():
    try:
        with open(JOBS_FILE, "w") as f:
            json.dump(background_jobs, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Failed to save jobs: {e}")

# ... (optimization task logic remains) ...


def run_backtest_task(job_id: str, request: BacktestRequest):
    """
    Background task for LSTM Backtesting simulation.
    """
    from services.lstm_predictor import get_predictor
    import numpy as np
    import pandas as pd
    
    # Fees
    FEE_MAKER = 0.0002
    FEE_TAKER = 0.0005
    FEE_IN = FEE_TAKER
    FEE_SL = FEE_TAKER
    FEE_TP = FEE_MAKER

    try:
        background_jobs[job_id]["status"] = "running"
        background_jobs[job_id]["progress"] = "Initializing (Wait < 1m)..."
        save_jobs()

        # 1. Load Model
        predictor = get_predictor(request.model_name)
        seq_length = predictor.seq_length
        background_jobs[job_id]["total_steps"] = 5
        background_jobs[job_id]["current_step"] = 1
        save_jobs()

        # 2. Timezone
        start_ts = request.start_time
        end_ts = request.end_time
        if start_ts.tzinfo is None:
            start_ts = start_ts.replace(tzinfo=timezone.utc)
        if end_ts.tzinfo is None:
            end_ts = end_ts.replace(tzinfo=timezone.utc)

        # 3. Fetch Data with Warmup
        interval_min = get_interval_minutes(request.interval)
        # 2x seq_length safety margin
        warmup_minutes = 120 * interval_min 
        fetch_start_ts = start_ts - timedelta(minutes=warmup_minutes)

        background_jobs[job_id]["progress"] = "Fetching Data..."
        save_jobs()
        
        feat_table = f"backtesting_features_{request.interval}"
        ohlcv_table = f"ohlcv_{request.interval}"
        atr_table = f"atr_{request.interval}"

        # Fetch extra data for warmup
        query = text(f"""
            SELECT 
                f.timestamp,
                f.log_return, f.ema_ratio, f.macd_hist, f.bandwidth, f.pct_b, f.rsi, f.mfi,
                o.open, o.high, o.low, o.close,
                a.atr
            FROM trading_data.{feat_table} f
            JOIN trading_data.{ohlcv_table} o ON f.timestamp = o.timestamp AND f.symbol = o.symbol
            JOIN trading_data.{atr_table} a ON f.timestamp = a.timestamp AND f.symbol = a.symbol
            WHERE f.symbol = :symbol
              AND f.timestamp >= :fetch_start_ts
              AND f.timestamp <= :end_ts
            ORDER BY f.timestamp ASC
        """)

        with SyncSessionLocal() as session:
            rows = session.execute(query, {
                "symbol": request.symbol,
                "fetch_start_ts": fetch_start_ts,
                "end_ts": end_ts
            }).fetchall()

        if not rows:
            raise ValueError("No data found for backtesting")
            
        # Convert to DataFrame
        df = pd.DataFrame(rows, columns=[
            'timestamp', 
            'log_return', 'ema_ratio', 'macd_hist', 'bandwidth', 'pct_b', 'rsi', 'mfi',
            'open', 'high', 'low', 'close', 'atr'
        ])
        
        feature_cols = ['log_return', 'ema_ratio', 'macd_hist', 'bandwidth', 'pct_b', 'rsi', 'mfi']
        df[feature_cols] = df[feature_cols].fillna(method='ffill')
        df[feature_cols] = df[feature_cols].fillna(0.0)
        df['atr'] = df['atr'].astype(float)
        df['close'] = df['close'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        
        # 4. Prepare Feature Windows
        if len(df) < seq_length:
            raise ValueError(f"Not enough data: need at least {seq_length} rows, got {len(df)}")
            
        timestamps = df['timestamp'].tolist()
        num_samples = len(df) - seq_length + 1
        features_np = df[feature_cols].values
        batch_input = np.zeros((num_samples, seq_length, 7), dtype=np.float32)
        
        for i in range(num_samples):
            batch_input[i] = features_np[i:i + seq_length]
            
        # 5. Run Prediction on ALL data
        background_jobs[job_id]["progress"] = f"Running Inference ({num_samples} samples)..."
        save_jobs()
        
        preds = predictor.batch_predict(batch_input)
        
        # 6. Run Unified Backtest Simulation
        background_jobs[job_id]["progress"] = "Simulating Trades (Unified Logic)..."
        save_jobs()
        
        # Slice Data such that it matches PREDICTION output
        # preds[i] corresponds to window ending at df[seq_length-1+i].
        # So preds[i] is prediction for timestamp df[seq_length-1+i].
        
        full_slice_df = df.iloc[seq_length - 1:].reset_index(drop=True)
        # Note: timestamps list matches df index
        full_slice_timestamps = timestamps[seq_length - 1:]
        
        if len(preds) != len(full_slice_df):
             min_len = min(len(preds), len(full_slice_df))
             preds = preds[:min_len]
             full_slice_df = full_slice_df.iloc[:min_len]
             full_slice_timestamps = full_slice_timestamps[:min_len]

        # FILTER: Keep only rows where timestamp >= request.start_time
        # This ensures we trade strictly within the requested window (Evaluation Phase)
        # but we used prior data (Warmup) to generate the signal for the very first timestamp.
        
        # Must handle timezone aware comparison
        valid_mask = [ts >= start_ts for ts in full_slice_timestamps]
        
        # If no valid data
        if not any(valid_mask):
             raise ValueError("No valid predictions within the requested time range after filtering.")
             
        # Apply filter
        # numpy bool array
        import numpy as np
        valid_mask_np = np.array(valid_mask)
        
        df_slice = full_slice_df[valid_mask_np].reset_index(drop=True)
        sliced_timestamps = [ts for ts, m in zip(full_slice_timestamps, valid_mask) if m]
        preds = preds[valid_mask_np]


        # Call the unified NumPy-based backtest function (Same as Optimization)
        # NOTE: Frontend already converts tp_threshold from percentage (40) to decimal (0.40)
        # So request.tp_threshold is already in decimal form - do NOT divide again!
        
        pnl, trade_count, w_count, d_count, trade_list = _run_fast_backtest_numpy(
            df_slice, 
            preds, 
            threshold=request.tp_threshold,  # Already 0.40, not 40
            rr_ratio=request.rr_ratio, 
            allow_overlap=request.allow_overlap, 
            m_value=request.m_value,
            return_trades=True,
            risk_pct=request.risk_pct,
            sl_mult=request.sl_mult,
            max_positions=request.max_positions
        )
        
        # Convert raw trade list to BacktestTrade objects
        trades = []
        if trade_list:
            for t in trade_list:
                t_entry_idx = t['entry_idx']
                t_exit_idx = t['exit_idx']
                
                # Safety check
                if t_entry_idx < len(sliced_timestamps) and t_exit_idx < len(sliced_timestamps):
                    entry_ts = sliced_timestamps[t_entry_idx]
                    exit_ts = sliced_timestamps[t_exit_idx]
                    
                    trades.append(BacktestTrade(
                        entry_time=entry_ts.isoformat(),
                        exit_time=exit_ts.isoformat(),
                        entry_price=t['entry_price'],
                        exit_price=t['exit_price'],
                        side="Long",
                        reason=t['reason'],
                        pnl_pct=t['pnl_pct'],
                        leverage=t['leverage'],
                        raw_pnl=float(t['pnl_pct'] / t['leverage']) if t['leverage'] > 0 else 0.0,
                        holding_bars=t['holding_bars']
                    ))
            
        # Summary Calculation
        if not trades:
            summary = BacktestSummary(
                 total_trades=0, win_rate=0, total_pnl_pct=0, avg_pnl_pct=0, max_drawdown_pct=0, avg_leverage=0
            )
        else:
            # Cumprod 방식으로 Total PnL 계산
            df_res = pd.DataFrame([t.dict() for t in trades])
            
            # Safety: clamp pnl_pct to prevent extreme values
            df_res['pnl_pct'] = df_res['pnl_pct'].clip(-99, 1000)
            df_res['pnl_pct'] = df_res['pnl_pct'].replace([np.inf, -np.inf], 0).fillna(0)
            
            df_res['equity_curve'] = (1 + df_res['pnl_pct'] / 100).cumprod()
            total_pnl = (df_res['equity_curve'].iloc[-1] - 1) * 100
            
            # Safety clamp for total_pnl
            total_pnl = max(min(float(total_pnl), 100000), -100) if np.isfinite(total_pnl) else 0
            
            wins = df_res[df_res['pnl_pct'] > 0]
            win_rate = len(wins) / len(df_res) * 100
            avg_pnl = df_res['pnl_pct'].mean()
            avg_lev = df_res['leverage'].clip(0, 100).mean()
            
            # MDD calculation
            running_max = df_res['equity_curve'].cummax()
            drawdown = (df_res['equity_curve'] - running_max) / running_max.replace(0, 1) * 100
            max_dd = drawdown.min()
            
            # Safety for all values
            win_rate = float(win_rate) if np.isfinite(win_rate) else 0
            avg_pnl = float(avg_pnl) if np.isfinite(avg_pnl) else 0
            avg_lev = float(avg_lev) if np.isfinite(avg_lev) else 1
            max_dd = float(max_dd) if np.isfinite(max_dd) else 0
            
            summary = BacktestSummary(
                total_trades=len(trades),
                win_rate=win_rate,
                total_pnl_pct=total_pnl,
                avg_pnl_pct=avg_pnl,
                max_drawdown_pct=max_dd,
                avg_leverage=avg_lev
            )
        
        # Calculate Advanced Metrics
        from services.lstm_predictor import calculate_advanced_metrics
        advanced_metrics = calculate_advanced_metrics(
            trade_list=trade_list if trade_list else [],
            timestamps=sliced_timestamps if 'sliced_timestamps' in dir() else None,
            interval=request.interval
        )
        
        # Save Result
        # Convert models to dicts for JSON serialization
        result_data = BacktestResponse(summary=summary, trades=trades).dict()
        result_data['advanced_metrics'] = advanced_metrics
        
        # Add prediction data for CSV export
        result_data['all_predictions'] = [
            {
                "timestamp": ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
                "p_hold": float(p[0]),
                "p_long": float(p[1]),
                "p_short": float(p[2])
            }
            for ts, p in zip(sliced_timestamps, preds)
        ]
        result_data['filtered_predictions'] = [
            {
                "timestamp": ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
                "p_hold": float(p[0]),
                "p_long": float(p[1]),
                "p_short": float(p[2])
            }
            for ts, p in zip(sliced_timestamps, preds) if float(p[1]) >= request.tp_threshold
        ]
        
        # Add equity curve data for chart
        # Build cumulative returns from trade list
        equity_curve_data = []
        cumulative_return = 1.0
        if trade_list:
            for t in trade_list:
                t_exit_idx = t['exit_idx']
                if t_exit_idx < len(sliced_timestamps):
                    exit_ts = sliced_timestamps[t_exit_idx]
                    cumulative_return *= (1 + t['pnl_pct'] / 100)
                    equity_curve_data.append({
                        "timestamp": exit_ts.isoformat() if hasattr(exit_ts, 'isoformat') else str(exit_ts),
                        "cumulative_return": float((cumulative_return - 1) * 100)
                    })
        result_data['equity_curve'] = equity_curve_data
        
        # Calculate Buy and Hold returns
        buy_hold_data = []
        if len(sliced_timestamps) > 0 and len(df_slice) > 0:
            initial_price = df_slice['close'].iloc[0]
            # Sample every N candles to avoid too many data points
            sample_interval = max(1, len(df_slice) // 100)
            for idx in range(0, len(df_slice), sample_interval):
                if idx < len(sliced_timestamps):
                    ts = sliced_timestamps[idx]
                    current_price = df_slice['close'].iloc[idx]
                    bh_return = ((current_price / initial_price) - 1) * 100
                    buy_hold_data.append({
                        "timestamp": ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
                        "buy_hold_return": float(bh_return)
                    })
            # Always include last point
            if len(df_slice) > 1:
                final_ts = sliced_timestamps[-1]
                final_price = df_slice['close'].iloc[-1]
                final_return = ((final_price / initial_price) - 1) * 100
                buy_hold_data.append({
                    "timestamp": final_ts.isoformat() if hasattr(final_ts, 'isoformat') else str(final_ts),
                    "buy_hold_return": float(final_return)
                })
        result_data['buy_hold'] = buy_hold_data
        
        background_jobs[job_id]["data"] = result_data
        background_jobs[job_id]["status"] = "completed"
        background_jobs[job_id]["progress"] = "Done"
        save_jobs()


    except Exception as e:
        logger.exception(f"[Backtest Task] Error: {e}")
        background_jobs[job_id]["status"] = "failed"
        background_jobs[job_id]["message"] = str(e)
        save_jobs()


def get_interval_minutes(interval: str) -> int:
    unit = interval[-1]
    val = int(interval[:-1])
    if unit == 'm': return val
    if unit == 'h': return val * 60
    if unit == 'd': return val * 1440
    return 15 # Default


def run_cross_margin_backtest_task(job_id: str, request: CrossMarginBacktestRequest):
    """
    Background task for Cross Margin Backtesting.
    Uses balance tracking, unrealized PnL, and dynamic leverage.
    """
    from services.lstm_predictor import get_predictor, _run_cross_margin_backtest, FEE_MAKER, FEE_TAKER
    import numpy as np
    import pandas as pd
    
    try:
        background_jobs[job_id]["status"] = "running"
        background_jobs[job_id]["progress"] = "Loading Model..."
        save_jobs()

        # 1. Load Model
        predictor = get_predictor(request.model_name)
        seq_length = predictor.seq_length

        # 2. Timezone
        start_ts = request.start_time
        end_ts = request.end_time
        if start_ts.tzinfo is None:
            start_ts = start_ts.replace(tzinfo=timezone.utc)
        if end_ts.tzinfo is None:
            end_ts = end_ts.replace(tzinfo=timezone.utc)

        # 3. Fetch Data with Warmup
        interval_min = get_interval_minutes(request.interval)
        warmup_minutes = 120 * interval_min 
        fetch_start_ts = start_ts - timedelta(minutes=warmup_minutes)

        background_jobs[job_id]["progress"] = "Fetching Data..."
        save_jobs()
        
        feat_table = f"backtesting_features_{request.interval}"
        ohlcv_table = f"ohlcv_{request.interval}"
        atr_table = f"atr_{request.interval}"

        query = text(f"""
            SELECT 
                f.timestamp,
                f.log_return, f.ema_ratio, f.macd_hist, f.bandwidth, f.pct_b, f.rsi, f.mfi,
                o.open, o.high, o.low, o.close,
                a.atr
            FROM trading_data.{feat_table} f
            JOIN trading_data.{ohlcv_table} o ON f.timestamp = o.timestamp AND f.symbol = o.symbol
            JOIN trading_data.{atr_table} a ON f.timestamp = a.timestamp AND f.symbol = a.symbol
            WHERE f.symbol = :symbol
              AND f.timestamp >= :fetch_start_ts
              AND f.timestamp <= :end_ts
            ORDER BY f.timestamp ASC
        """)

        with SyncSessionLocal() as session:
            rows = session.execute(query, {
                "symbol": request.symbol,
                "fetch_start_ts": fetch_start_ts,
                "end_ts": end_ts
            }).fetchall()

        if not rows:
            raise ValueError("No data found for backtesting")
            
        df = pd.DataFrame(rows, columns=[
            'timestamp', 
            'log_return', 'ema_ratio', 'macd_hist', 'bandwidth', 'pct_b', 'rsi', 'mfi',
            'open', 'high', 'low', 'close', 'atr'
        ])
        
        feature_cols = ['log_return', 'ema_ratio', 'macd_hist', 'bandwidth', 'pct_b', 'rsi', 'mfi']
        df[feature_cols] = df[feature_cols].fillna(method='ffill')
        df[feature_cols] = df[feature_cols].fillna(0.0)
        df['atr'] = df['atr'].astype(float)
        df['close'] = df['close'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        
        # 4. Prepare Feature Windows
        if len(df) < seq_length:
            raise ValueError(f"Not enough data: need at least {seq_length} rows, got {len(df)}")
            
        timestamps = df['timestamp'].tolist()
        num_samples = len(df) - seq_length + 1
        features_np = df[feature_cols].values
        batch_input = np.zeros((num_samples, seq_length, 7), dtype=np.float32)
        
        for i in range(num_samples):
            batch_input[i] = features_np[i:i + seq_length]
            
        # 5. Run Prediction
        background_jobs[job_id]["progress"] = f"Running Inference ({num_samples} samples)..."
        save_jobs()
        
        preds = predictor.batch_predict(batch_input)
        
        # 6. Slice and Filter
        full_slice_df = df.iloc[seq_length - 1:].reset_index(drop=True)
        full_slice_timestamps = timestamps[seq_length - 1:]
        
        if len(preds) != len(full_slice_df):
            min_len = min(len(preds), len(full_slice_df))
            preds = preds[:min_len]
            full_slice_df = full_slice_df.iloc[:min_len]
            full_slice_timestamps = full_slice_timestamps[:min_len]

        valid_mask = [ts >= start_ts for ts in full_slice_timestamps]
        
        if not any(valid_mask):
            raise ValueError("No valid predictions within the requested time range.")
            
        valid_mask_np = np.array(valid_mask)
        df_slice = full_slice_df[valid_mask_np].reset_index(drop=True)
        sliced_timestamps = [ts for ts, m in zip(full_slice_timestamps, valid_mask) if m]
        preds = preds[valid_mask_np]

        # 7. Run Cross Margin Backtest
        background_jobs[job_id]["progress"] = "Running Cross Margin Simulation..."
        save_jobs()
        
        final_balance, trade_count, w_count, d_count, trade_list = _run_cross_margin_backtest(
            df_slice, 
            preds, 
            threshold=request.tp_threshold,
            rr_ratio=request.rr_ratio, 
            initial_balance=request.initial_balance,
            risk_pct=request.risk_pct,
            sl_mult=request.sl_mult,
            m_value=request.m_value,
            max_leverage=request.max_leverage,
            return_trades=True
        )
        
        # 8. Build Response
        trades = []
        if trade_list:
            for t in trade_list:
                entry_idx = t['entry_idx']
                exit_idx = t['exit_idx']
                
                if entry_idx < len(sliced_timestamps) and exit_idx < len(sliced_timestamps):
                    entry_ts = sliced_timestamps[entry_idx]
                    exit_ts = sliced_timestamps[exit_idx]
                    
                    trades.append(CrossMarginBacktestTrade(
                        entry_time=entry_ts.isoformat(),
                        exit_time=exit_ts.isoformat(),
                        entry_price=t['entry_price'],
                        exit_price=t['exit_price'],
                        side="Long",
                        reason=t['reason'],
                        pnl_pct=t['pnl_pct'],
                        pnl_usdt=t['pnl_usdt'],
                        position_size=t['position_size'],
                        leverage=t['leverage'],
                        holding_bars=t['holding_bars'],
                        balance_after=t['balance_after']
                    ))
        
        # Calculate MDD from balance_after
        if trades:
            balances = [request.initial_balance] + [t.balance_after for t in trades]
            balance_arr = np.array(balances)
            peak = np.maximum.accumulate(balance_arr)
            drawdown = (balance_arr - peak) / peak * 100
            max_dd = float(np.min(drawdown))
            win_rate = (w_count / trade_count * 100) if trade_count > 0 else 0
        else:
            max_dd = 0.0
            win_rate = 0.0
        
        total_pnl_usdt = final_balance - request.initial_balance
        total_pnl_pct = (total_pnl_usdt / request.initial_balance) * 100
        
        summary = CrossMarginBacktestSummary(
            initial_balance=request.initial_balance,
            final_balance=final_balance,
            total_pnl_usdt=total_pnl_usdt,
            total_pnl_pct=total_pnl_pct,
            total_trades=trade_count,
            win_rate=win_rate,
            max_drawdown_pct=max_dd
        )
        
        result_data = CrossMarginBacktestResponse(summary=summary, trades=trades).dict()
        
        # Add equity curve data for chart (balance over time)
        equity_curve_data = []
        if trades:
            for t in trades:
                equity_curve_data.append({
                    "timestamp": t.exit_time,
                    "cumulative_return": float((t.balance_after / request.initial_balance - 1) * 100)
                })
        result_data['equity_curve'] = equity_curve_data
        
        # Calculate Buy and Hold returns
        buy_hold_data = []
        if len(sliced_timestamps) > 0 and len(df_slice) > 0:
            initial_price = df_slice['close'].iloc[0]
            sample_interval = max(1, len(df_slice) // 100)
            for idx in range(0, len(df_slice), sample_interval):
                if idx < len(sliced_timestamps):
                    ts = sliced_timestamps[idx]
                    current_price = df_slice['close'].iloc[idx]
                    bh_return = ((current_price / initial_price) - 1) * 100
                    buy_hold_data.append({
                        "timestamp": ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
                        "buy_hold_return": float(bh_return)
                    })
            if len(df_slice) > 1:
                final_ts = sliced_timestamps[-1]
                final_price = df_slice['close'].iloc[-1]
                final_return = ((final_price / initial_price) - 1) * 100
                buy_hold_data.append({
                    "timestamp": final_ts.isoformat() if hasattr(final_ts, 'isoformat') else str(final_ts),
                    "buy_hold_return": float(final_return)
                })
        result_data['buy_hold'] = buy_hold_data
        
        background_jobs[job_id]["data"] = result_data
        background_jobs[job_id]["status"] = "completed"
        background_jobs[job_id]["progress"] = "Done"
        save_jobs()

    except Exception as e:
        logger.exception(f"[Cross Margin Backtest] Error: {e}")
        background_jobs[job_id]["status"] = "failed"
        background_jobs[job_id]["message"] = str(e)
        save_jobs()

def run_optimization_task(job_id: str, request: OptimizeRequest):
    from services.lstm_predictor import list_available_models, optimize_model_parameter_portfolio
    import pandas as pd
    from datetime import timedelta
    
    try:
        background_jobs[job_id]["status"] = "running"
        background_jobs[job_id]["progress"] = "Initializing..."
        save_jobs()
        
        if request.symbol:
             PORTFOLIO_SYMBOLS = [request.symbol]
             logger.info(f"[Optimize] Optimization restricted to Single Symbol: {request.symbol}")
        else:
             # Default Portfolio if no symbol specified (though UI forces one)
             PORTFOLIO_SYMBOLS = ["BTC", "ETH", "XRP", "SOL", "ZEC"]
             logger.info(f"[Optimize] Optimization for Default Portfolio: {PORTFOLIO_SYMBOLS}")
        
        # 1. Fetch Data for Portfolio
        portfolio_data = {}
        
        feat_table = f"trading_data.backtesting_features_{request.interval}"
        ohlcv_table = f"trading_data.ohlcv_{request.interval}"
        atr_table = f"trading_data.atr_{request.interval}"
        
        # First, capture the current max timestamp
        with SyncSessionLocal() as session:
            max_ts_query = text(f"""
                SELECT MAX(timestamp) as max_ts 
                FROM {ohlcv_table} 
                WHERE symbol = :sym AND timestamp >= '2024-01-01 00:00:00+00'
            """)
            max_ts_result = session.execute(max_ts_query, {"sym": PORTFOLIO_SYMBOLS[0]}).fetchone()
            captured_end_time = max_ts_result[0] if max_ts_result and max_ts_result[0] else None
        
        if not captured_end_time:
            raise Exception("Could not determine max timestamp for optimization")
        
        # Calculate Warmup Start Time (Start Time - (SeqLen * Interval))
        # We need a safe upper bound for SeqLen (e.g. 100) or check models first?
        # Standard models max seq is 60. Let's use 100 to be safe.
        interval_min = get_interval_minutes(request.interval)
        warmup_minutes = 120 * interval_min # 120 candles back to be very safe
        start_time_dt = datetime.fromisoformat("2024-01-01T00:00:00+00:00")
        fetch_start_time = start_time_dt - timedelta(minutes=warmup_minutes)
        
        logger.info(f"[Optimize] Captured end_time: {captured_end_time}")
        logger.info(f"[Optimize] Fetching portfolio data (Start: {fetch_start_time} for Warmup) ~ {captured_end_time}...")
        background_jobs[job_id]["progress"] = f"Fetching Data..."
        save_jobs()
        
        with SyncSessionLocal() as session:
            for sym in PORTFOLIO_SYMBOLS:
                query = text(f"""
                    SELECT t1.timestamp, t1.open, t1.high, t1.low, t1.close, t1.volume,
                           t2.log_return, t2.ema_ratio, t2.macd_hist, t2.bandwidth, t2.pct_b, t2.rsi, t2.mfi,
                           t3.atr
                    FROM {ohlcv_table} t1
                    JOIN {feat_table} t2 ON t1.timestamp = t2.timestamp AND t1.symbol = t2.symbol
                    JOIN {atr_table} t3 ON t1.timestamp = t3.timestamp AND t1.symbol = t3.symbol
                    WHERE t1.symbol = :sym
                      AND t1.timestamp >= :fetch_start_ts
                      AND t1.timestamp <= :end_ts
                    ORDER BY t1.timestamp ASC
                """)
                df_res = pd.read_sql(query, session.bind, params={"sym": sym, "end_ts": captured_end_time, "fetch_start_ts": fetch_start_time})
                
                if not df_res.empty:
                    # Fill NaN
                    df_res.fillna(method='ffill', inplace=True)
                    df_res.fillna(0.0, inplace=True) 
                    portfolio_data[sym] = df_res
                    logger.info(f"Loaded {len(df_res)} rows for {sym}")
                else:
                    logger.warning(f"No data found for {sym}")

        if not portfolio_data:
            raise Exception("No portfolio data found for optimization.")

        # 2. List Models
        model_interval = request.interval
        if model_interval in ("1m", "5m"):
            model_interval = "15m"
        
        all_models = list_available_models("/app/ml_models")
        target_models = [m for m in all_models if m['interval'] == model_interval]
        
        results_map = {sym: {} for sym in PORTFOLIO_SYMBOLS}
        
        # 3. Optimize Each Model
        logger.info(f"Optimizing {len(target_models)} models...")
        
        for idx, model in enumerate(target_models):
            if background_jobs.get(job_id, {}).get("status") == "cancelled":
                return
            
            progress_msg = f"Processing model {idx + 1}/{len(target_models)}: {model['filename']}..."
            background_jobs[job_id]["progress"] = progress_msg
            save_jobs()
            
            res = optimize_model_parameter_portfolio(
                model_path=model['full_path'],
                portfolio_data=portfolio_data,
                rr_ratio=float(model.get('tp_pct', 2.0)), 
                allow_overlap=request.allow_overlap,
                evaluation_start_time="2024-01-01T00:00:00+00:00"
            )
            
            if res:
                for sym, thresholds_list in res.items():
                    if sym in results_map and thresholds_list:
                        results_map[sym][model['filename']] = thresholds_list

        # ... (rest of logic) ...
        final_results = {}
        for sym, models_dict in results_map.items():
            models_with_best = []
            for model_name, thresholds in models_dict.items():
                if thresholds:
                    # Filter results? No, result is already computed.
                    best_pnl = thresholds[0]['final_pnl'] 
                    models_with_best.append((model_name, best_pnl, thresholds))
            
            models_with_best.sort(key=lambda x: x[1], reverse=True)
            
            final_results[sym] = []
            for i, (name, pnl, thresholds) in enumerate(models_with_best[:5]):
                best_th = thresholds[0]
                final_results[sym].append({
                    "rank": i + 1,
                    "model_name": name,
                    "best_pnl": pnl,
                    "max_pnl": pnl,
                    "threshold": best_th['threshold'],
                    "win_rate": best_th['win_rate'],
                    "total_trades": best_th['total_trades'],
                    "avg_pnl": best_th['avg_pnl'],
                    "thresholds": thresholds
                })
        
        # Save to JSON
        import os
        results_dir = "/app/ml_models/optimization_results"
        os.makedirs(results_dir, exist_ok=True)
        results_path = f"{results_dir}/{request.interval}.json"
        
        # Determine strict start time for metadata (User wanted 2024-01-01)
        # Even though we fetched earlier, the "Result" period is effectively 2024-01-01 if we filter right.
        meta_start = "2024-01-01T00:00:00+00:00"
        
        final_results["_metadata"] = {
            "start_time": meta_start,
            "end_time": captured_end_time.isoformat() if hasattr(captured_end_time, 'isoformat') else str(captured_end_time) if captured_end_time else None,
            "interval": request.interval,
            "allow_overlap": request.allow_overlap,
            "optimized_at": datetime.now(timezone.utc).isoformat()
        }
        
        try:
            with open(results_path, "w") as f:
                json.dump(final_results, f, indent=2)
            logger.info(f"[Optimize] Saved results to {results_path}")
        except Exception as e:
            logger.error(f"[Optimize] Failed to save results JSON: {e}")
            
        background_jobs[job_id]["data"] = final_results
        background_jobs[job_id]["status"] = "completed"
        background_jobs[job_id]["progress"] = "Done"
        save_jobs()


        
    except Exception as e:
        logger.exception(f"[Optimize] Error: {e}")
        background_jobs[job_id]["status"] = "failed"
        background_jobs[job_id]["message"] = str(e)
        save_jobs()



@router.post("/lstm/optimize")
def optimize_models(request: OptimizeRequest, background_tasks: BackgroundTasks):
    """
    Start asynchronous optimization task. Returns a Job ID.
    Exhaustive optimization for all models using a Portfolio of Representative Symbols.
    """
    job_id = str(uuid.uuid4())
    background_jobs[job_id] = {
        "status": "pending",
        "progress": "Starting...",
        "data": None,
        "message": None
    }
    
    background_tasks.add_task(run_optimization_task, job_id, request)
    
    return {"job_id": job_id, "message": "Optimization started in background"}


# ========================
#  Batch Optimization (5 VIP × 6 Intervals)
# ========================

# Store batch job state
batch_jobs = {}  # { batch_id: { "status": ..., "tasks": [...], "completed": 0, "total": 20 } }

VIP_SYMBOLS = ["BTC", "ETH", "XRP", "SOL", "ZEC"]
ALL_INTERVALS = ["15m", "1h", "4h", "1d"]  # 1m, 5m use 15m models


@router.post("/lstm/optimize/batch")
def start_batch_optimization(allow_overlap: bool = True):
    """
    Start parallel optimization for ALL VIP symbols × ALL intervals.
    Uses Celery GPU workers for parallel execution.
    Returns batch_id for tracking.
    """
    from celery_task.optimization_task import run_single_optimization
    from celery import group
    
    batch_id = str(uuid.uuid4())
    total_tasks = len(VIP_SYMBOLS) * len(ALL_INTERVALS)
    
    # Create task group
    tasks = []
    for symbol in VIP_SYMBOLS:
        for interval in ALL_INTERVALS:
            tasks.append(run_single_optimization.s(symbol, interval, allow_overlap))
    
    # Execute all tasks in parallel
    job = group(tasks)
    async_result = job.apply_async()
    
    batch_jobs[batch_id] = {
        "status": "running",
        "async_result_id": async_result.id,
        "total": total_tasks,
        "completed": 0,
        "tasks": [{"symbol": s, "interval": i, "status": "pending"} 
                  for s in VIP_SYMBOLS for i in ALL_INTERVALS]
    }
    
    logger.info(f"[Optimize] Batch started: {batch_id} with {total_tasks} tasks")
    
    return {
        "batch_id": batch_id,
        "total_tasks": total_tasks,
        "message": f"Batch optimization started for {len(VIP_SYMBOLS)} symbols × {len(ALL_INTERVALS)} intervals"
    }


@router.get("/lstm/optimize/batch/{batch_id}")
def get_batch_status(batch_id: str):
    """
    Get status of batch optimization job.
    """
    if batch_id not in batch_jobs:
        raise HTTPException(status_code=404, detail="Batch job not found")
    
    job = batch_jobs[batch_id]
    
    # Check Celery async result for progress
    from celery.result import GroupResult
    try:
        if job["async_result_id"]:
            group_result = GroupResult.restore(job["async_result_id"])
            if group_result:
                completed = sum(1 for r in group_result.results if r.ready())
                job["completed"] = completed
                
                # Update individual task statuses
                for i, r in enumerate(group_result.results):
                    if r.ready():
                        result = r.get(timeout=1)
                        if isinstance(result, dict):
                            job["tasks"][i]["status"] = result.get("status", "completed")
                
                if group_result.ready():
                    job["status"] = "completed"
    except Exception as e:
        logger.warning(f"Could not check group result: {e}")
    
    return {
        "batch_id": batch_id,
        "status": job["status"],
        "completed": job["completed"],
        "total": job["total"],
        "progress": f"{job['completed']}/{job['total']}",
        "tasks": job["tasks"]
    }


@router.get("/lstm/optimize/batch/results")
def get_all_batch_results():
    """
    Get all saved optimization results from DB.
    """
    results = {}
    
    try:
        with SyncSessionLocal() as session:
            rows = session.execute(text("""
                SELECT symbol, interval, results, created_at
                FROM trading_data.optimization_results
                ORDER BY symbol, interval
            """)).fetchall()
            
            for row in rows:
                key = f"{row.symbol}_{row.interval}"
                results[key] = {
                    "symbol": row.symbol,
                    "interval": row.interval,
                    "results": row.results,
                    "created_at": row.created_at.isoformat() if row.created_at else None
                }
    except Exception as e:
        logger.error(f"Failed to fetch batch results: {e}")
    
    return {"results": results, "count": len(results)}


@router.get("/lstm/optimize/jobs")
def get_all_jobs():
    """
    Get list of all optimization jobs (IDs and Status).
    Useful for reconnecting to a running job after refresh.
    """
    # Return list of { job_id: ..., status: ..., progress: ... }
    # Sort by creation? Dict preserves insertion order in Py3.7+
    job_list = []
    for jid, info in background_jobs.items():
        job_list.append({
            "job_id": jid,
            "status": info["status"],
            "progress": info["progress"],
            "has_data": info["data"] is not None
        })
    return job_list

@router.get("/lstm/optimize/{job_id}")
def get_optimization_status(job_id: str):
    """
    Check status of optimization job.
    """
    if job_id not in background_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
        
    return background_jobs[job_id]


@router.delete("/lstm/optimize/{job_id}")
def cancel_optimization_job(job_id: str):
    """
    Cancel a running optimization job.
    """
    if job_id not in background_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = background_jobs[job_id]
    if job["status"] not in ["running", "pending"]:
        raise HTTPException(status_code=400, detail=f"Cannot cancel job with status: {job['status']}")
    
    # Mark as cancelled
    background_jobs[job_id]["status"] = "cancelled"
    background_jobs[job_id]["progress"] = "사용자에 의해 취소됨"
    background_jobs[job_id]["message"] = "Job cancelled by user"
    save_jobs()
    
    logger.info(f"[Optimize] Job {job_id} cancelled by user")
    return {"message": "Job cancelled successfully", "job_id": job_id}


# ========================
#  Optimization Recommendations API (for LSTMTestPage)
# ========================

class OptimizationRecommendationsRequest(BaseModel):
    symbol: str
    interval: str

class ThresholdResult(BaseModel):
    threshold: int
    final_pnl: float
    avg_pnl: float
    win_rate: float
    total_trades: int

class ModelRecommendation(BaseModel):
    model_name: str
    best_pnl: float
    thresholds: list[ThresholdResult]

class OptimizationRecommendationsResponse(BaseModel):
    symbol: str
    interval: str
    models: list[ModelRecommendation]

@router.get("/lstm/recommendations")
def get_optimization_recommendations(symbol: str, interval: str):
    """
    Get Top 5 recommended models with Top 5 thresholds each for a symbol/interval.
    Reads from pre-computed optimization results JSON.
    """
    import os
    
    results_path = f"/app/ml_models/optimization_results/{interval}.json"
    
    if not os.path.exists(results_path):
        raise HTTPException(
            status_code=404, 
            detail=f"No optimization results found for interval {interval}. Run optimization first."
        )
    
    try:
        with open(results_path, "r") as f:
            all_results = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read optimization results: {e}")
        raise HTTPException(status_code=500, detail="Failed to read optimization results")
    
    if symbol not in all_results:
        raise HTTPException(
            status_code=404,
            detail=f"No results found for symbol {symbol}. Available: {list(all_results.keys())}"
        )
    
    # Extract metadata if present
    metadata = all_results.get("_metadata", {})
    
    raw_models = all_results[symbol]
    models_list = []
    
    # Transform Dict { "model_name": [thresholds] } -> List [ {model_name, best_pnl, thresholds} ]
    if isinstance(raw_models, dict):
        for m_name, thresholds in raw_models.items():
            best_pnl = 0.0
            if isinstance(thresholds, list) and len(thresholds) > 0:
                best_pnl = thresholds[0].get("final_pnl", 0.0)
            
            models_list.append({
                "model_name": m_name,
                "best_pnl": best_pnl,
                "thresholds": thresholds
            })
        
        # Sort by best_pnl descending
        models_list.sort(key=lambda x: x["best_pnl"], reverse=True)
    elif isinstance(raw_models, list):
        models_list = raw_models
    
    return {
        "symbol": symbol,
        "interval": interval,
        "models": models_list,
        "metadata": metadata  # Include start_time, end_time for backtest sync
    }


# ========================
#  Legacy Recommend Endpoint (keep for backwards compatibility)
# ========================

class RecommendRequest(BaseModel):
    interval: str

class RecommendedModel(BaseModel):
    model_name: str
    interval: str
    sl_pct: float
    rr_ratio: float
    m_value: int
    rank: int
    seq_length: int
    score: float
    avg_precision: float = 0.0
    avg_recall: float = 0.0
    precisions: list[float] = []
    recalls: list[float] = []
    # Optimization Fields
    optimal_threshold: Optional[float] = None
    max_pnl: Optional[float] = None
    optimal_win_rate: Optional[float] = None
    total_trades: Optional[int] = None
    mdd: Optional[float] = None

class RecommendResponse(BaseModel):
    recommendations: dict[str, list[RecommendedModel]]

@router.post("/recommend", response_model=RecommendResponse)
def recommend_models(request: RecommendRequest):
    """
    해당 인터벌에서 가장 성능(VAL_LOG_SCORE)이 좋은 상위 5개 모델 추천 (심볼별)
    Returns: { "BTC": [...], "ETH": [...] }
    """
    from services.lstm_predictor import recommend_models_by_interval
    
    try:
        results = recommend_models_by_interval(request.interval, top_k=5)
        return RecommendResponse(recommendations=results)
    except Exception as e:
        logger.error(f"[Prediction] Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

