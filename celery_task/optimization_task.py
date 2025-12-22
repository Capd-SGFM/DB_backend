"""
Celery task for running LSTM model optimization on GPU workers.
Executes optimization for a single symbol/interval combination.
"""
from celery import shared_task
from loguru import logger
import numpy as np
import pandas as pd
from sqlalchemy import text

from celery_task import celery_app
from db_module.connect_sqlalchemy_engine import SyncSessionLocal


VIP_SYMBOLS = ["BTC", "ETH", "XRP", "SOL", "ZEC"]
ALL_INTERVALS = ["15m", "1h", "4h", "1d"]  # 1m, 5m use 15m models


@celery_app.task(name="optimization.run_single", queue="indicators", bind=True)
def run_single_optimization(self, symbol: str, interval: str, allow_overlap: bool = True):
    """
    Run optimization for a single symbol/interval combination.
    Uses GPU worker queue 'indicators'.
    """
    from services.lstm_predictor import list_available_models, optimize_model_parameter_portfolio
    
    logger.info(f"[Optimize] Starting optimization for {symbol}/{interval}")
    
    try:
        # 1. Fetch Data
        ohlcv_table = f"trading_data.ohlcv_{interval}"
        feat_table = f"trading_data.backtesting_features_{interval}"
        atr_table = f"trading_data.atr_{interval}"
        
        portfolio_data = {}
        
        with SyncSessionLocal() as session:
            query = text(f"""
                SELECT t1.timestamp, t1.open, t1.high, t1.low, t1.close, t1.volume,
                       t2.log_return, t2.ema_ratio, t2.macd_hist, t2.bandwidth, t2.pct_b, t2.rsi, t2.mfi,
                       t3.atr
                FROM {ohlcv_table} t1
                JOIN {feat_table} t2 ON t1.timestamp = t2.timestamp AND t1.symbol = t2.symbol
                JOIN {atr_table} t3 ON t1.timestamp = t3.timestamp AND t1.symbol = t3.symbol
                WHERE t1.symbol = :sym
                  AND t1.timestamp >= '2024-01-01 00:00:00+00'
                ORDER BY t1.timestamp ASC
            """)
            df_res = pd.read_sql(query, session.bind, params={"sym": symbol})
            
            if not df_res.empty:
                df_res.fillna(method='ffill', inplace=True)
                portfolio_data[symbol] = df_res
                logger.info(f"Loaded {len(df_res)} rows for {symbol}/{interval}")
            else:
                logger.warning(f"No data found for {symbol}/{interval}")
                return {"symbol": symbol, "interval": interval, "status": "no_data", "results": {}}
        
        if not portfolio_data:
            return {"symbol": symbol, "interval": interval, "status": "no_data", "results": {}}
        
        # 2. List Models for Interval
        model_interval = interval
        if model_interval in ("1m", "5m"):
            model_interval = "15m"
        
        all_models = list_available_models()
        models = [m for m in all_models if m.get('interval') == model_interval]
        
        if not models:
            logger.warning(f"No models found for interval {model_interval}")
            return {"symbol": symbol, "interval": interval, "status": "no_models", "results": {}}
        
        logger.info(f"[Optimize] Found {len(models)} models for {model_interval}")
        
        # 3. Optimize Each Model
        results_map = {symbol: {}}
        
        for model in models:
            # Extract RR ratio from filename
            rr_ratio = 2.0
            try:
                parts = model['filename'].replace(".keras", "").split("_")
                if len(parts) >= 4:
                    rr_ratio = float(parts[3])
            except:
                pass
            
            res = optimize_model_parameter_portfolio(
                model_path=model['full_path'],
                portfolio_data=portfolio_data,
                rr_ratio=rr_ratio,
                allow_overlap=allow_overlap
            )
            
            if res and symbol in res:
                results_map[symbol][model['filename']] = res[symbol]
        
        # 4. Select Top 5 Models
        final_results = {}
        models_with_best = []
        for model_name, thresholds in results_map[symbol].items():
            if thresholds:
                best_pnl = thresholds[0]['final_pnl']
                models_with_best.append((model_name, best_pnl, thresholds))
        
        models_with_best.sort(key=lambda x: x[1], reverse=True)
        top_models = models_with_best[:5]
        
        final_results[symbol] = [
            {
                "model_name": m[0],
                "best_pnl": m[1],
                "thresholds": m[2]
            }
            for m in top_models
        ]
        
        # 5. Save to DB
        _save_optimization_results(symbol, interval, final_results[symbol])
        
        logger.info(f"[Optimize] Completed {symbol}/{interval}")
        return {"symbol": symbol, "interval": interval, "status": "completed", "results": final_results}
        
    except Exception as e:
        logger.error(f"[Optimize] Error for {symbol}/{interval}: {e}")
        return {"symbol": symbol, "interval": interval, "status": "error", "error": str(e)}


def _save_optimization_results(symbol: str, interval: str, results: list):
    """Save optimization results to DB table."""
    from datetime import datetime, timezone
    
    try:
        with SyncSessionLocal() as session:
            # Create table if not exists
            session.execute(text("""
                CREATE TABLE IF NOT EXISTS trading_data.optimization_results (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(30) NOT NULL,
                    interval VARCHAR(10) NOT NULL,
                    results JSONB NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(symbol, interval)
                )
            """))
            
            # Upsert results
            import json
            session.execute(text("""
                INSERT INTO trading_data.optimization_results (symbol, interval, results, created_at)
                VALUES (:sym, :interval, :results, :created_at)
                ON CONFLICT (symbol, interval) 
                DO UPDATE SET results = :results, created_at = :created_at
            """), {
                "sym": symbol,
                "interval": interval,
                "results": json.dumps(results),
                "created_at": datetime.now(timezone.utc)
            })
            session.commit()
            logger.info(f"Saved optimization results for {symbol}/{interval}")
    except Exception as e:
        logger.error(f"Failed to save optimization results: {e}")


@celery_app.task(name="optimization.run_batch", queue="indicators")
def run_batch_optimization(allow_overlap: bool = True):
    """
    Dispatch optimization tasks for all VIP symbols × all intervals.
    Returns batch_id for tracking.
    """
    from celery import group
    import uuid
    
    batch_id = str(uuid.uuid4())
    logger.info(f"[Optimize] Starting batch optimization (batch_id={batch_id})")
    
    tasks = []
    for symbol in VIP_SYMBOLS:
        for interval in ALL_INTERVALS:
            tasks.append(run_single_optimization.s(symbol, interval, allow_overlap))
    
    # Execute all tasks in parallel
    job = group(tasks)
    result = job.apply_async()
    
    logger.info(f"[Optimize] Dispatched {len(tasks)} optimization tasks")
    
    return {
        "batch_id": batch_id,
        "total_tasks": len(tasks),
        "async_result_id": result.id
    }
