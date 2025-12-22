# routers/pipeline.py
from fastapi import APIRouter
from pydantic import BaseModel
from loguru import logger

from celery_task.pipeline_task import start_pipeline, stop_pipeline
from models.pipeline_state import (
    is_pipeline_active,
    set_pipeline_active,
    set_component_active,
    get_all_pipeline_states,
    PipelineComponent,
)
from db_module.connect_sqlalchemy_engine import SyncSessionLocal
from models.backfill_progress import BackfillProgress, reset_backfill_progress
from sqlalchemy import select, desc

# 🔹 REST / Indicator 진행현황용 모델
from models.rest_progress import RestProgress
from models.indicator_progress import IndicatorProgress
from models.websocket_progress import WebSocketProgress, reset_websocket_progress
from models.error_log import ErrorLogCurrent
from datetime import datetime
from sqlalchemy import delete

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


class ErrorLogModel(BaseModel):
    id: int
    component: str
    symbol: str | None
    interval: str | None
    error_message: str
    occurred_at: datetime

    class Config:
        from_attributes = True



class PipelineStatusResponse(BaseModel):
    is_active: bool
    websocket: EngineStatusModel
    backfill: EngineStatusModel
    rest_maintenance: EngineStatusModel
    indicator: EngineStatusModel
    backtesting: EngineStatusModel
    vip_backtesting: EngineStatusModel


@router.get("/status", response_model=PipelineStatusResponse)
def get_pipeline_status():
    states = get_all_pipeline_states()
    # states dict: {"pipeline": {...}, "websocket": {...}, ...}

    # 1. Main Pipeline
    main_p = states.get("pipeline", {})
    is_main_active = bool(main_p.get("is_active", False))

    # 2. Map Sub-components
    ws_st = _map_websocket_status(is_main_active, states.get("websocket", {"id": 2}))
    bf_st = _map_backfill_status(is_main_active, states.get("backfill", {"id": 3}))
    rm_st = _map_engine_status(is_main_active, states.get("rest_maintenance", {"id": 4}))
    ind_st = _map_engine_status(is_main_active, states.get("indicator", {"id": 5}))
    
    # Backtesting (General) - can run independently
    bt_st = _map_engine_status(True, states.get("backtesting", {"id": 6}))
    
    # VIP Backtesting - can run independently
    vip_st = _map_engine_status(True, states.get("vip_backtesting", {"id": 7}))

    return PipelineStatusResponse(
        is_active=is_main_active,
        websocket=ws_st,
        backfill=bf_st,
        rest_maintenance=rm_st,
        indicator=ind_st,
        backtesting=bt_st,
        vip_backtesting=vip_st,
    )


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
#   파이프라인 상태 조회 API
# ==================================================


def _map_websocket_status(is_pipeline_on: bool, comp: dict) -> EngineStatusModel:
    comp_id = int(comp.get("id"))
    is_active = bool(comp.get("is_active", False))
    last_error = comp.get("last_error")
    updated_at = comp.get("updated_at")

    if not is_pipeline_on:
        return EngineStatusModel(
            id=comp_id,
            is_active=False,
            status="WAIT",
            last_error=last_error,
            updated_at=updated_at,
        )

    # WebSocketProgress 전체 상태 확인
    # WebSocketProgress 전체 상태 확인
    with SyncSessionLocal() as session:
        # 1. 최신 run_id 조회
        latest_run_id = session.execute(
            select(WebSocketProgress.run_id)
            .order_by(desc(WebSocketProgress.updated_at))
            .limit(1)
        ).scalar_one_or_none()

        if not latest_run_id:
            final_status = "WAIT"
        else:
            # 2. 최신 run_id에 해당하는 row들의 state 조회
            rows = (
                session.execute(
                    select(WebSocketProgress.state).where(
                        WebSocketProgress.run_id == latest_run_id
                    )
                )
                .scalars()
                .all()
            )

            if not rows:
                final_status = "WAIT"
            elif any(s in ("ERROR", "DISCONNECTED", "FAIL", "FAILURE") for s in rows):
                final_status = "FAIL"
            elif all(s == "CONNECTED" for s in rows):
                final_status = "PROGRESS"
            else:
                final_status = "PROGRESS"

    # 상태가 PROGRESS(정상)이면, 과거의 last_error가 남아있더라도 UI에는 표시하지 않음
    if final_status == "PROGRESS":
        last_error = None

    return EngineStatusModel(
        id=comp_id,
        is_active=is_active,
        status=final_status,
        last_error=last_error,
        updated_at=updated_at,
    )









# ==================================================
#   Backfill 진행률 조회 API
# ==================================================

class BackfillIntervalProgress(BaseModel):
    interval: str
    state: str
    pct_time: float
    last_updated_iso: str | None
    last_error: str | None

class BackfillSymbolProgress(BaseModel):
    symbol: str
    state: str
    intervals: dict[str, BackfillIntervalProgress]

class BackfillProgressResponse(BaseModel):
    run_id: str | None
    symbols: dict[str, BackfillSymbolProgress]


@router.post("/backtesting/on", response_model=PipelineToggleResponse)
async def start_backtesting():
    from celery_task.pipeline_task import start_backtesting_pipeline
    from models.pipeline_state import is_pipeline_active, set_component_active, PipelineComponent, _get_state

    # Check state directly
    st = _get_state(PipelineComponent.BACKTESTING)
    if st and st.is_active:
        return PipelineToggleResponse(
            message="Backtesting pipeline is already active.",
            is_active=True,
        )

    # Set active
    set_component_active(PipelineComponent.BACKTESTING, True)
    
    # Trigger task
    start_backtesting_pipeline.delay()
    logger.info("[Pipeline] Backtesting pipeline activated")

    return PipelineToggleResponse(
        message="Backtesting pipeline started.",
        is_active=True,
    )


@router.post("/backtesting/off", response_model=PipelineToggleResponse)
async def stop_backtesting():
    from models.pipeline_state import set_component_active, PipelineComponent, _get_state

    st = _get_state(PipelineComponent.BACKTESTING)
    if not st or not st.is_active:
         return PipelineToggleResponse(
            message="Backtesting pipeline is already inactive.",
            is_active=False,
        )

    set_component_active(PipelineComponent.BACKTESTING, False)
    logger.info("[Pipeline] Backtesting pipeline deactivated")

    return PipelineToggleResponse(
        message="Backtesting pipeline stopped.",
        is_active=False,
    )


@router.get("/backtesting/progress", response_model=BackfillProgressResponse)
async def get_backtesting_progress():
    with SyncSessionLocal() as session:
        # Find latest run_id starting with 'backtest-'
        q = (
            select(BackfillProgress.run_id)
            .where(BackfillProgress.run_id.like("backtest-%"))
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
            last_error=row.last_error,
        )

    # Symbol State Calculation
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
#   VIP PIPELINE: 5 Symbols / Reserved Workers
# ==================================================
@router.post("/backtesting/vip/start")
def start_vip_backtesting_endpoint():
    # 1. Pipeline Activate
    set_component_active(PipelineComponent.VIP_BACKTESTING, True)
    
    # 2. Trigger Task
    from celery_task.pipeline_task import start_vip_backtesting_pipeline
    start_vip_backtesting_pipeline.delay()
    
    return PipelineToggleResponse(message="VIP Backtesting Started", is_active=True)


@router.post("/backtesting/vip/stop")
def stop_vip_backtesting_endpoint():
    # 1. Pipeline Deactivate
    set_component_active(PipelineComponent.VIP_BACKTESTING, False)
    
    return PipelineToggleResponse(message="VIP Backtesting Stopped", is_active=False)


@router.get("/backtesting/vip/progress", response_model=BackfillProgressResponse)
async def get_vip_backtesting_progress():
    """VIP Backtesting 진행률 조회 (run_id가 'vip-'로 시작하는 것만)"""
    with SyncSessionLocal() as session:
        # Find latest run_id starting with 'vip-'
        q = (
            select(BackfillProgress.run_id)
            .where(BackfillProgress.run_id.like("vip-%"))
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
            last_error=row.last_error,
        )

    # Symbol State Calculation
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
#   LSTM Test Results API
# ==================================================

class ModelReport(BaseModel):
    risk_level: float = 0.0
    val_log_score: float = 0.0
    precs: list[float] = []
    recalls: list[float] = []

class TestResultsResponse(BaseModel):
    found: bool
    confusion_matrix: list[list[int]] | None = None
    class_labels: list[str] | None = None
    prob_dist: dict | None = None  # {"bins": [...], "counts": [...]}
    equity_curve: list[dict] | None = None  # [{"time": ..., "cumulative_return": ...}]
    summary: dict | None = None  # {"accuracy": ..., "total_samples": ...}
    report: ModelReport | None = None

def parse_report_file(filepath):
    report = {
        "risk_level": 0.0,
        "val_log_score": 0.0,
        "precs": [],
        "recalls": []
    }
    try:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                if line.startswith("RISK_LEVEL="):
                    report["risk_level"] = float(line.split("=")[1])
                elif line.startswith("VAL_LOG_SCORE="):
                    report["val_log_score"] = float(line.split("=")[1])
                elif line.startswith("FINAL_TEST_PRECs="):
                    val_str = line.split("=")[1].strip("[] ")
                    report["precs"] = [float(x) for x in val_str.split() if x]
                elif line.startswith("FINAL_TEST_RECALLs="):
                    val_str = line.split("=")[1].strip("[] ")
                    report["recalls"] = [float(x) for x in val_str.split() if x]
    except Exception as e:
        logger.error(f"Failed to parse report file {filepath}: {e}")
        return None
    return report


@router.get("/lstm/test-results", response_model=TestResultsResponse)
async def get_lstm_test_results(
    symbol: str = "BTC",
    interval: str = "4h",
    sl_pct: float = 1.0,
    rr_ratio: float = 1.5,
    m: int = 60,
    rank: int = 1,
    seq_len: int | None = None,
):
    """LSTM 모델 Test 결과 조회 (Confusion Matrix, 확률 분포, 누적 수익 곡선)"""
    import pandas as pd
    import numpy as np
    from pathlib import Path
    
    # Find test_pred CSV file
    base_dir = Path("/app/ml_models") / interval / f"{symbol}_{interval}_{sl_pct}_{rr_ratio}_{m}"
    
    # Find matching file with rank
    # Find matching file with rank
    csv_pattern = f"test_pred_{interval}_{sl_pct}_{rr_ratio}_{m}_rank{rank}_seq*.csv"
    if seq_len is not None:
        csv_pattern = f"test_pred_{interval}_{sl_pct}_{rr_ratio}_{m}_rank{rank}_seq{seq_len}.csv"

    test_files = list(base_dir.glob(csv_pattern))
    
    if not test_files:
        return TestResultsResponse(found=False)
    
    test_file = test_files[0]
    
    try:
        df = pd.read_csv(test_file)
    except Exception as e:
        logger.error(f"Error reading test file: {e}")
        return TestResultsResponse(found=False)
    
    # 1. Confusion Matrix (3x3)
    # y_true_internal: 0=HOLD, 1=LONG, 2=SHORT (internal mapping)
    # y_pred_internal: same
    y_true = df["y_true_internal"].values.astype(int)
    y_pred = df["y_pred_internal"].values.astype(int)
    
    # Create 3x3 confusion matrix
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < 3 and 0 <= p < 3:
            cm[t, p] += 1
    
    class_labels = ["HOLD", "LONG", "SHORT"]
    
    # 2. Probability Distribution (for predicted class)
    # Get max probability for each prediction
    max_probs = df[["p_class0", "p_class1", "p_class2"]].max(axis=1).values
    
    # Create histogram
    hist, bin_edges = np.histogram(max_probs, bins=20, range=(0, 1))
    prob_dist = {
        "bins": [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(hist))],
        "counts": hist.tolist()
    }
    
    # 3. Equity Curve (Cumulative Return)
    # Simple strategy: +1 for correct, -1 for incorrect
    # More realistic: based on actual price movement
    correct = (y_true == y_pred).astype(int)
    cumulative_correct = np.cumsum(correct)
    
    # Alternative: simulate returns based on prediction
    # HOLD=0, LONG=1, SHORT=2
    # For simplicity: correct LONG/SHORT = +1%, incorrect = -1%
    returns = []
    for i, (pred, true) in enumerate(zip(y_pred, y_true)):
        if pred == 0:  # HOLD prediction
            returns.append(0)
        elif pred == true:  # Correct LONG or SHORT
            returns.append(1.0)  # Gain
        else:  # Wrong LONG or SHORT
            returns.append(-1.0)  # Loss
    
    cumulative_return = np.cumsum(returns)
    
    # Sample every N points for performance (max 200 points)
    total_points = len(df)
    step = max(1, total_points // 200)
    
    equity_curve = []
    for i in range(0, len(df), step):
        equity_curve.append({
            "time": df.iloc[i]["open_time"],
            "cumulative_return": float(cumulative_return[i]),
            "cumulative_accuracy": float(cumulative_correct[i]) / (i + 1) * 100
        })
    
    # 4. Summary stats
    accuracy = (y_true == y_pred).mean() * 100
    summary = {
        "accuracy": round(accuracy, 2),
        "total_samples": len(df),
        "file": test_file.name
    }
    
    # 5. Parse Report File
    report_data = None
    report_pattern = f"report_{interval}_{sl_pct}_{rr_ratio}_{m}_Rank{rank}_seq*.txt"
    if seq_len is not None:
        report_pattern = f"report_{interval}_{sl_pct}_{rr_ratio}_{m}_Rank{rank}_seq{seq_len}.txt"
    
    report_files = list(base_dir.glob(report_pattern))
    if report_files:
        report_data = parse_report_file(report_files[0])

    return TestResultsResponse(
        found=True,
        confusion_matrix=cm.tolist(),
        class_labels=class_labels,
        prob_dist=prob_dist,
        equity_curve=equity_curve,
        summary=summary,
        report=report_data
    )

