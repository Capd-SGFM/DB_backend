# LSTM Predictor Service
import os
import re
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from loguru import logger
import json
import pandas as pd # Ensure pandas is imported

# TensorFlow will be imported lazily to avoid startup issues if not installed
_tf = None
_keras = None

def _load_tensorflow():
    global _tf, _keras
    if _tf is None:
        import tensorflow as tf
        _tf = tf
        _keras = tf.keras
    return _tf, _keras


class LSTMPredictor:
    """
    LSTM 모델 로딩 및 예측 서비스
    
    모델 파일명 패턴: lstm_{interval}_{sl}_{tp}_{param}_{rank}_seq{length}.keras
    예: lstm_15m_1.0_2.0_40_rank1_seq60.keras
    """
    
    def __init__(self, model_path: str):
        """
        Args:
            model_path: .keras 모델 파일 경로
        """
        tf, keras = _load_tensorflow()
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(f"[LSTMPredictor] Loading model from {model_path}")
        self.model = keras.models.load_model(model_path)
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        
        # Extract sequence length from filename
        self.seq_length = self._extract_seq_length(self.model_name)
        logger.info(f"[LSTMPredictor] Model loaded. seq_length={self.seq_length}")
    
    def _extract_seq_length(self, filename: str) -> int:
        """
        파일명에서 시퀀스 길이 추출
        예: lstm_15m_1.0_2.0_40_rank1_seq60.keras -> 60
        """
        match = re.search(r'seq(\d+)', filename)
        if match:
            return int(match.group(1))
        
        # 기본값
        logger.warning(f"[LSTMPredictor] Could not extract seq_length from {filename}, using default 60")
        return 60
    
    def predict(self, features: np.ndarray) -> dict:
        """
        예측 실행
        
        Args:
            features: shape (1, seq_length, 7) - 7개 보조지표
                      순서: log_return, ema_ratio, macd_hist, bandwidth, pct_b, rsi, mfi
        
        Returns:
            dict with probabilities for each state:
            - sl_probability: SL 도달 확률
            - tp_probability: TP 도달 확률
            - neither_probability: 둘다 X 확률
        """
        if features.shape[1] != self.seq_length:
            raise ValueError(f"Expected seq_length={self.seq_length}, got {features.shape[1]}")
        
        if features.shape[2] != 7:
            raise ValueError(f"Expected 7 features, got {features.shape[2]}")
        
        # Predict
        probs = self.model.predict(features, verbose=0)
        
        # Assuming output shape is (1, 3) with [sl_prob, tp_prob, neither_prob]
        return {
            "sl_probability": float(probs[0][0]),
            "tp_probability": float(probs[0][1]),
            "neither_probability": float(probs[0][2]),
        }
    
    def batch_predict(self, features: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """
        배치 예측 실행 (대량 데이터용)
        
        Args:
            features: shape (N, seq_length, 7) - N개 샘플
            batch_size: 한 번에 예측할 샘플 수
        
        Returns:
            np.ndarray: shape (N, 3) - 각 샘플의 [sl_prob, tp_prob, neither_prob]
        """
        if features.shape[1] != self.seq_length:
            raise ValueError(f"Expected seq_length={self.seq_length}, got {features.shape[1]}")
        
        if features.shape[2] != 7:
            raise ValueError(f"Expected 7 features, got {features.shape[2]}")
        
        # Use model.predict with batch_size for efficiency
        probs = self.model.predict(features, batch_size=batch_size, verbose=0)
        return probs
    
    def get_model_info(self) -> dict:
        """모델 정보 반환"""
        return {
            "model_name": self.model_name,
            "seq_length": self.seq_length,
            "input_shape": str(self.model.input_shape),
            "output_shape": str(self.model.output_shape),
        }


# Model Cache (singleton pattern for efficiency)
_model_cache: dict[str, LSTMPredictor] = {}


def get_predictor(model_name: str, models_dir: str = "/app/ml_models") -> LSTMPredictor:
    """
    모델 캐시에서 예측기 가져오기 (없으면 로딩)
    
    Args:
        model_name: 모델 파일명 또는 전체 경로 (확장자 포함)
        models_dir: 모델 디렉토리 경로 (model_name이 절대경로면 무시됨)
    
    Returns:
        LSTMPredictor instance
    """
    # Determine full path
    if model_name.startswith('/'):
        # model_name is already a full path
        model_path = model_name
        cache_key = model_name
    else:
        # model_name is just filename, use models_dir
        model_path = os.path.join(models_dir, model_name)
        cache_key = model_path
    
    if cache_key not in _model_cache:
        _model_cache[cache_key] = LSTMPredictor(model_path)
    
    return _model_cache[cache_key]


def find_model_by_params(
    interval: str,
    sl_pct: float,
    rr_ratio: float,
    m_value: int,
    rank: int,
    models_dir: str = "/app/ml_models"
) -> dict | None:
    """
    파라미터로 모델 파일 검색 (seq는 자동 추출)
    
    디렉토리 구조:
        {models_dir}/{interval}/BTC_{interval}_{sl}_{rr}_{m}/lstm_..._rank{rank}_seq{seq}.keras
    
    예: /app/ml_models/1d/BTC_1d_2.0_2.0_60/lstm_1d_2.0_2.0_60_rank3_seq108.keras
    
    Note: 모든 심볼이 BTC 모델을 공유합니다.
    
    Args:
        interval: 인터벌 (예: 15m, 1h, 4h, 1d)
        sl_pct: SL 배수 (예: 1.0, 1.5, 2.0)
        rr_ratio: RR 비율 (예: 1.0, 1.5, 2.0)
        m_value: M 파라미터 (예: 20, 40, 60)
        rank: 랭크 (예: 1, 2, 3)
        models_dir: 모델 디렉토리 경로
    
    Returns:
        dict with filename, full_path, and seq_length, or None if not found
    """
    # Build subdirectory path: {interval}/BTC_{interval}_{sl}_{rr}_{m}
    subdir = f"{interval}/BTC_{interval}_{sl_pct:.1f}_{rr_ratio:.1f}_{m_value}"
    search_dir = Path(models_dir) / subdir
    
    if not search_dir.exists():
        logger.warning(f"[LSTMPredictor] Directory not found: {search_dir}")
        return None
    
    # Build pattern: lstm_{interval}_{sl}_{rr}_{m}_rank{rank}_seq*.keras
    pattern = f"lstm_{interval}_{sl_pct:.1f}_{rr_ratio:.1f}_{m_value}_rank{rank}_seq*.keras"
    
    # Search for matching files
    matches = list(search_dir.glob(pattern))
    
    if not matches:
        logger.warning(f"[LSTMPredictor] No model found matching pattern: {search_dir}/{pattern}")
        return None
    
    # Take the first match (there should only be one)
    matched_file = matches[0]
    
    # Extract seq_length from filename
    seq_match = re.search(r'seq(\d+)', matched_file.stem)
    seq_length = int(seq_match.group(1)) if seq_match else 60
    
    logger.info(f"[LSTMPredictor] Found model: {matched_file} (seq_length={seq_length})")
    
    return {
        "filename": matched_file.name,
        "full_path": str(matched_file),
        "seq_length": seq_length,
        "interval": interval,
        "sl_pct": sl_pct,
        "rr_ratio": rr_ratio,
        "m_value": m_value,
        "rank": rank,
    }


def list_available_models(models_dir: str = "/app/ml_models") -> list[dict]:
    """
    사용 가능한 모델 목록 반환 (중첩 디렉토리 구조 지원)
    
    디렉토리 구조:
        {models_dir}/{interval}/{sl}_{tp}_{m}/lstm_..._rank{rank}_seq{seq}.keras
    
    Returns:
        List of model info dicts
    """
    models = []
    models_path = Path(models_dir)
    
    if not models_path.exists():
        return models
    
    # Recursively search for .keras files in nested directories
    for f in models_path.glob("**/*.keras"):
        # Extract info from filename
        match = re.search(r'lstm_(\w+)_([\d.]+)_([\d.]+)_(\d+)_rank(\d+)_seq(\d+)', f.stem)
        if match:
            models.append({
                "filename": f.name,
                "full_path": str(f),
                "interval": match.group(1),
                "sl_pct": float(match.group(2)),
                "tp_pct": float(match.group(3)),
                "param": int(match.group(4)),
                "rank": int(match.group(5)),
                "seq_length": int(match.group(6)),
            })
        else:
            # Generic entry for non-matching filenames
            seq_match = re.search(r'seq(\d+)', f.stem)
            models.append({
                "filename": f.name,
                "full_path": str(f),
                "interval": "unknown",
                "seq_length": int(seq_match.group(1)) if seq_match else 60,
            })
    
    return models


def recommend_models_by_interval(interval: str, top_k: int = 5, models_dir: str = "/app/ml_models") -> dict[str, list[dict]]:
    """
    Scans models for the given interval and returns top k recommendations PER SYMBOL.
    
    Returns:
        {
            "BTC": [ {model_info, max_pnl: ...}, ... ],
            "ETH": [ ... ]
        }
    """
    from collections import defaultdict
    candidates_map = defaultdict(list)
    
    models_path = Path(models_dir) / interval
    
    if not models_path.exists():
        logger.warning(f"[LSTMPredictor] Interval directory not found: {models_path}")
        return {}

    # Iterate over parameter directories: e.g. BTC_15m_1.0_1.0_20
    for param_dir in models_path.iterdir():
        if not param_dir.is_dir():
            continue
            
        # Check for model files
        for keras_file in param_dir.glob("lstm_*.keras"):
            # Parse filename
            match = re.search(r'lstm_(\w+)_([\d.]+)_([\d.]+)_(\d+)_rank(\d+)_seq(\d+)', keras_file.stem)
            if not match:
                continue
            
            p_interval, sl, tp, param, rank, seq = match.groups()
            
            # Infer training symbol from directory name (usually starts with BTC_)
            # param_dir.name example: "BTC_15m_..."
            training_symbol = param_dir.name.split("_")[0]
            
            # Construct report filename
            report_name = f"report_{p_interval}_{sl}_{tp}_{param}_Rank{rank}_seq{seq}.txt"
            report_path = param_dir / report_name
            
            score = -999.0
            precs = []
            recalls = []
            
            if report_path.exists():
                try:
                    with open(report_path, "r") as f:
                        for line in f:
                            if line.startswith("VAL_LOG_SCORE="):
                                score = float(line.strip().split("=")[1])
                            elif line.startswith("FINAL_TEST_PRECs="):
                                val_str = line.strip().split("=")[1].strip("[]")
                                precs = [float(x) for x in val_str.split()]
                            elif line.startswith("FINAL_TEST_RECALLs="):
                                val_str = line.strip().split("=")[1].strip("[]")
                                recalls = [float(x) for x in val_str.split()]
                except Exception as e:
                    logger.warning(f"Failed to read report {report_path}: {e}")
            
            avg_precision = sum(precs) / len(precs) if precs else 0.0
            avg_recall = sum(recalls) / len(recalls) if recalls else 0.0

            # Check for optimization results (per symbol)
            # NEW: _optimization.json (replacing _optimization_2025.json)
            # Try both for backward compatibility or prefer new one
            opt_path = keras_file.with_name(f"{keras_file.stem}_optimization.json")
            if not opt_path.exists():
                 opt_path = keras_file.with_name(f"{keras_file.stem}_optimization_2025.json")

            opt_data = {}
            if opt_path.exists():
                try:
                    with open(opt_path, "r") as f:
                        opt_data = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to read optimization {opt_path}: {e}")

            # Base Model Info
            base_info = {
                "model_name": keras_file.name,
                "interval": p_interval,
                "sl_pct": float(sl),
                "rr_ratio": float(tp),
                "m_value": int(param),
                "rank": int(rank),
                "seq_length": int(seq),
                "score": score,
                "avg_precision": avg_precision,
                "avg_recall": avg_recall,
                "precisions": precs,
                "recalls": recalls,
                "full_path": str(keras_file),
            }

            # If optimization data exists, create an entry for each symbol found
            if opt_data:
                for sym, stats in opt_data.items():
                    # Skip metadata key
                    if sym == "_metadata":
                        continue
                    
                    # Handle both old format (dict) and new format (list)
                    if isinstance(stats, list) and len(stats) > 0:
                        # New format: stats is a list of threshold results, take the best one
                        best_stat = stats[0]  # Already sorted by final_pnl descending
                        candidate = base_info.copy()
                        candidate.update({
                            "optimal_threshold": best_stat.get("threshold"),
                            "max_pnl": best_stat.get("final_pnl"),
                            "optimal_win_rate": best_stat.get("win_rate"),
                            "total_trades": best_stat.get("total_trades"),
                            "mdd": best_stat.get("max_drawdown"),
                        })
                        candidates_map[sym].append(candidate)
                    elif isinstance(stats, dict):
                        # Old format: stats is a dict with optimization metrics
                        candidate = base_info.copy()
                        candidate.update({
                            "optimal_threshold": stats.get("optimal_threshold"),
                            "max_pnl": stats.get("max_pnl"),
                            "optimal_win_rate": stats.get("win_rate"),
                            "total_trades": stats.get("total_trades"),
                            "mdd": stats.get("max_drawdown"),
                        })
                        candidates_map[sym].append(candidate)
            else:
                # Fallback: Add to training symbol list (unoptimized)
                candidates_map[training_symbol].append(base_info)

    # Sort and slice for each symbol
    final_results = {}
    for sym, items in candidates_map.items():
        # Sort by max_pnl desc if available, else by score
        items.sort(key=lambda x: x.get("max_pnl") if x.get("max_pnl") is not None else -9999.0, reverse=True)
        final_results[sym] = items[:top_k]
    
    return final_results

def optimize_model_parameter_portfolio(model_path: str, portfolio_data: dict[str, pd.DataFrame], rr_ratio: float = 2.0, allow_overlap: bool = False, evaluation_start_time: str = None):
    """
    Exhaustive optimization for a single model across a PORTFOLIO of symbols.
    
    Args:
        model_path: Path to .keras model
        portfolio_data: Dict { "BTCUSDT": df, "ETHUSDT": df, ... } (Must have features + ohlcv + atr)
        rr_ratio: Risk Reward Ratio (determines TP search range)
        allow_overlap: If True, allows concurrent trades (high frequency).
        evaluation_start_time: ISO string. If set, optimization only considers trades AFTER this time.
        
    Returns:
        dict: { "BTC": { "optimal_threshold": int, "max_pnl": float ... }, ... }
    """
    tf, keras = _load_tensorflow()
    try:
        model = keras.models.load_model(model_path, compile=False)
    except Exception as e:
        logger.error(f"Failed to load model {model_path}: {e}")
        return None

    # Extract M value, SL mult, and seq_length from model filename
    # Pattern: lstm_{interval}_{sl}_{rr}_{M}_rank{N}_seq{L}.keras
    import os
    import re
    filename = os.path.basename(model_path)
    m_value = 60  # Default fallback
    sl_mult = 1.0  # Default fallback
    seq_length = 60  # Default fallback
    try:
        parts = filename.replace(".keras", "").split("_")
        # Expected: ['lstm', '1d', '1.0', '2.0', '60', 'rank3', 'seq44']
        if len(parts) >= 7:
            sl_mult = float(parts[2])  # SL is at index 2
            m_value = int(parts[4])  # M is at index 4
            # Extract seq from 'seq44' format
            seq_match = re.search(r'seq(\d+)', parts[6])
            if seq_match:
                seq_length = int(seq_match.group(1))
            logger.debug(f"Extracted sl_mult={sl_mult}, m_value={m_value}, seq_length={seq_length} from {filename}")
    except (ValueError, IndexError) as e:
        logger.warning(f"Could not parse params from {filename}, using defaults")

    # Determine TP Threshold Range based on RR
    if rr_ratio < 1.3:
        thresholds = range(50, 75, 1) # 50% ~ 74%
    elif rr_ratio < 1.8:
        thresholds = range(45, 70, 1) # 45% ~ 69%
    else:
        thresholds = range(40, 65, 1) # 40% ~ 64%

    # 1. Batch Predict for ALL symbols in portfolio using SLIDING WINDOWS
    # Store predictions: { "BTCUSDT": (probs, df_slice), ... }
    predictions_map = {}
    
    feature_cols = [
        'log_return', 'ema_ratio', 'macd_hist', 'bandwidth', 'pct_b', 'rsi', 'mfi'
    ]
    
    for sym, df in portfolio_data.items():
        if df.empty or len(df) < seq_length:
            logger.warning(f"[{sym}] Not enough data (need {seq_length}, got {len(df)}), skipping.")
            continue
        
        # Check if columns exist
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            logger.warning(f"[{sym}] Missing features {missing}, skipping.")
            continue
        
        # Build sliding windows (same as backtest simulation)
        features_raw = df[feature_cols].values.astype('float32')  # (N, 7)
        n_samples = len(df) - seq_length + 1
        
        # Pre-allocate feature array for batch prediction
        features_batch = np.zeros((n_samples, seq_length, 7), dtype=np.float32)
        
        for i in range(n_samples):
            features_batch[i] = features_raw[i:i + seq_length]
        
        # Batch Predict
        probs = model.predict(features_batch, batch_size=512, verbose=0)  # (n_samples, 3)
        
        # Slice df to align with predictions (predictions[i] corresponds to df.iloc[seq_length-1+i])
        # i.e., first prediction is for row at index seq_length-1
        df_slice = df.iloc[seq_length - 1:].reset_index(drop=True)
        
        # Filter by evaluation_start_time if provided
        if evaluation_start_time:
            eval_start_dt = pd.to_datetime(evaluation_start_time, utc=True)
            # Normalize df_slice timestamp timezone if needed
            if df_slice['timestamp'].dt.tz is None:
                df_slice['timestamp'] = df_slice['timestamp'].dt.tz_localize('UTC')
                
            mask = df_slice['timestamp'] >= eval_start_dt
            if mask.sum() == 0:
                logger.warning(f"[{sym}] No data left after filtering start_time {evaluation_start_time}")
                continue
            
            df_slice = df_slice[mask].reset_index(drop=True)
            probs = probs[mask]
        
        predictions_map[sym] = (probs, df_slice)

    if not predictions_map:
        return None

    # 2. Optimize per symbol - Return TOP 5 thresholds (not just best)
    # Result structure: { "BTC": [ {threshold, final_pnl, ...}, ... ], ... }
    results_per_symbol = {}

    for sym, (probs, df) in predictions_map.items():
        all_threshold_results = []
        
        # Test ALL thresholds for THIS symbol
        for t in thresholds:
            threshold_val = t / 100.0
            
            # Pass m_value extracted from model filename for time-limited backtest
            # Use fixed risk_pct=0.02 (2%) and sl_mult from model filename
            pnl, trades, wins, draws, _ = _run_fast_backtest_numpy(
                df, probs, threshold_val, rr_ratio, allow_overlap, m_value=m_value,
                risk_pct=0.02, sl_mult=sl_mult, max_positions=30
            )
            
            if trades > 0:
                win_rate = (wins / trades) * 100
                draw_rate = (draws / trades) * 100
                # Convert log return to percentage: exp(log_pnl) - 1
                final_pnl_pct = np.exp(pnl) - 1.0
                # Geometric mean (nth root): exp(log_pnl / trades) - 1
                avg_pnl_geometric = np.exp(pnl / trades) - 1.0
            else:
                win_rate = 0.0
                draw_rate = 0.0
                final_pnl_pct = 0.0
                avg_pnl_geometric = 0.0
            
            all_threshold_results.append({
                "threshold": int(t),
                "final_pnl": float(round(final_pnl_pct, 4)),  # Actual % return
                "avg_pnl": float(round(avg_pnl_geometric, 6)),  # Geometric mean
                "win_rate": float(round(win_rate / 100.0, 4)),
                "draw_rate": float(round(draw_rate / 100.0, 4)),  # Time-exit rate
                "total_trades": int(trades),
            })
        
        # Sort by final_pnl descending and keep Top 5
        all_threshold_results.sort(key=lambda x: x['final_pnl'], reverse=True)
        results_per_symbol[sym] = all_threshold_results[:1]

    # Save Optimization Result to JSON file (per-model)
    # Filename: model_name_optimization.json
    try:
        opt_path = model_path.replace(".keras", "_optimization.json")
        with open(opt_path, "w") as f:
            json.dump(results_per_symbol, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save optimization JSON: {e}")

    return results_per_symbol


# Binance Futures Fee Constants
FEE_MAKER = 0.0002  # 0.02% (지정가)
FEE_TAKER = 0.0005  # 0.05% (시장가)


def calculate_advanced_metrics(trade_list: list, timestamps: list = None, interval: str = "1d") -> dict:
    """
    Calculate advanced backtesting metrics from trade list.
    
    Args:
        trade_list: List of trade dicts with 'pnl_pct', 'entry_idx', 'exit_idx', 'reason'
        timestamps: Optional list of timestamps corresponding to data indices
        interval: Trading interval for annualization factor
    
    Returns:
        dict: {
            'total_return': float,
            'cagr': float,
            'win_rate': float,
            'profit_factor': float,
            'sharpe_ratio': float,
            'sortino_ratio': float,
            'mdd': float,
            'var_95': float,
            'cvar_95': float,
            'avg_win': float,
            'avg_loss': float,
            'equity_curve': list,
            'drawdown_curve': list,
            'trade_returns': list,
            'position_heatmap': dict
        }
    """
    if not trade_list:
        return {
            'total_return': 0.0, 'cagr': 0.0, 'win_rate': 0.0, 'profit_factor': 0.0,
            'sharpe_ratio': 0.0, 'sortino_ratio': 0.0, 'mdd': 0.0,
            'var_95': 0.0, 'cvar_95': 0.0, 'avg_win': 0.0, 'avg_loss': 0.0,
            'equity_curve': [], 'drawdown_curve': [], 'trade_returns': [],
            'position_heatmap': {}
        }
    
    # Extract returns from trade list
    returns = np.array([t.get('pnl_pct', 0) / 100 for t in trade_list])  # Convert % to decimal
    
    # --- Profitability Metrics ---
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    
    win_rate = len(wins) / len(returns) if len(returns) > 0 else 0.0
    avg_win = np.mean(wins) if len(wins) > 0 else 0.0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0.0
    
    total_wins = np.sum(wins) if len(wins) > 0 else 0.0
    total_losses = np.abs(np.sum(losses)) if len(losses) > 0 else 0.0
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
    
    # --- Equity Curve & Drawdown ---
    equity = [1.0]  # Start with 1.0 (100%)
    for r in returns:
        equity.append(equity[-1] * (1 + r))
    equity = np.array(equity)
    
    # Drawdown calculation
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    mdd = float(np.min(drawdown))  # Most negative value
    
    # --- Total Return & CAGR ---
    total_return = float(equity[-1] - 1)  # Final equity - 1
    
    # Annualization factor based on interval
    candles_per_year = {
        '1m': 525600, '5m': 105120, '15m': 35040,
        '1h': 8760, '4h': 2190, '1d': 365
    }
    periods_per_year = candles_per_year.get(interval, 365)
    
    # Estimate trading period in years
    n_trades = len(returns)
    avg_trade_duration = 30  # Approximate (M value average)
    total_candles = n_trades * avg_trade_duration
    years = total_candles / periods_per_year if periods_per_year > 0 else 1.0
    years = max(years, 0.01)  # Avoid division by zero
    
    cagr = float((equity[-1]) ** (1 / years) - 1) if equity[-1] > 0 else 0.0
    
    # --- Risk Metrics ---
    if len(returns) > 1:
        # Sharpe Ratio (assuming 0 risk-free rate)
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        sharpe = float(mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0.0
        
        # Sortino Ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns, ddof=1) if len(downside_returns) > 1 else std_return
        sortino = float(mean_return / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0
        
        # VaR & CVaR (95%)
        var_95 = float(np.percentile(returns, 5))  # 5th percentile = 95% VaR
        cvar_95 = float(np.mean(returns[returns <= var_95])) if np.any(returns <= var_95) else var_95
    else:
        sharpe = sortino = var_95 = cvar_95 = 0.0
    
    # --- Position Heatmap (by hour of day if timestamps available) ---
    position_heatmap = {}
    if timestamps and len(timestamps) > 0:
        hour_counts = {}
        hour_wins = {}
        for t in trade_list:
            entry_idx = t.get('entry_idx', 0)
            if entry_idx < len(timestamps):
                try:
                    ts = pd.to_datetime(timestamps[entry_idx])
                    hour = ts.hour
                    hour_counts[hour] = hour_counts.get(hour, 0) + 1
                    if t.get('pnl_pct', 0) > 0:
                        hour_wins[hour] = hour_wins.get(hour, 0) + 1
                except:
                    pass
        
        for h in range(24):
            total = hour_counts.get(h, 0)
            wins = hour_wins.get(h, 0)
            position_heatmap[h] = {
                'count': total,
                'win_rate': wins / total if total > 0 else 0
            }
    
    # Format output
    equity_curve = [{'index': i, 'equity': float(e)} for i, e in enumerate(equity)]
    drawdown_curve = [{'index': i, 'drawdown': float(d) * 100} for i, d in enumerate(drawdown)]
    trade_returns = [float(r * 100) for r in returns]
    
    return {
        'total_return': round(total_return * 100, 2),  # As percentage
        'cagr': round(cagr * 100, 2),
        'win_rate': round(win_rate * 100, 2),
        'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 999.99,
        'sharpe_ratio': round(sharpe, 2),
        'sortino_ratio': round(sortino, 2),
        'mdd': round(mdd * 100, 2),  # As percentage (negative)
        'var_95': round(var_95 * 100, 2),
        'cvar_95': round(cvar_95 * 100, 2),
        'avg_win': round(avg_win * 100, 2),
        'avg_loss': round(avg_loss * 100, 2),
        'equity_curve': equity_curve,
        'drawdown_curve': drawdown_curve,
        'trade_returns': trade_returns,
        'position_heatmap': position_heatmap
    }


def _run_fast_backtest_numpy(df, probs, threshold, rr_ratio, allow_overlap=False, m_value=60, return_trades=False, risk_pct=0.02, sl_mult=1.0, max_positions=10):
    """
    Optimized Backtest using NumPy with realistic fee structure.
    
    TP Condition: High >= Entry + (ATR * sl_mult * rr_ratio) + Fees
    SL Condition: Low <= Entry - (ATR * sl_mult) - Fees
    Time Limit: After M candles, exit at close price as DRAW.
    max_positions: Maximum concurrent positions allowed (only for overlap mode)
    
    Returns: (pnl, trades, wins, draws, trade_list)
    """
    if df.empty or len(probs) != len(df):
        return 0.0, 0, 0, 0, []

    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    atrs = df['atr'].values
    
    if probs.shape[1] == 3:
        prob_up = probs[:, 1]
    else:
        return 0.0, 0, 0, 0, []

    # 1. Generate Signals
    buy_signals = (prob_up >= threshold)
    n = len(closes)
    
    trade_list = [] if return_trades else None

    # --- Overlap Logic (For High Frequency Simulation) ---
    if allow_overlap:
        # ============================================
        # STEP 1: Find all entry indices (vectorized)
        # ============================================
        entry_indices = np.where(buy_signals)[0]
        
        if len(entry_indices) == 0:
            return 0.0, 0, 0, 0, []
        
        # ============================================
        # STEP 2: Pre-calculate TP/SL prices for each entry (vectorized)
        # ============================================
        entry_prices = closes[entry_indices]
        entry_atrs = atrs[entry_indices]
        
        sl_dists = entry_atrs * sl_mult
        total_fees = FEE_MAKER + FEE_TAKER
        fee_compensations = entry_prices * total_fees * rr_ratio
        tp_dists = sl_dists * rr_ratio + fee_compensations
        
        tp_prices = entry_prices + tp_dists
        sl_prices = entry_prices - sl_dists
        
        # Calculate leverages
        sl_pcts = sl_dists / entry_prices
        loss_pcts = sl_pcts + FEE_MAKER + FEE_TAKER
        leverages = np.minimum(risk_pct / np.maximum(loss_pcts, 1e-8), 50.0)
        
        # ============================================
        # STEP 3: Find exit index for each entry (vectorized inner loop optimization)
        # ============================================
        num_entries = len(entry_indices)
        exit_indices = np.full(num_entries, n - 1, dtype=np.int32)
        exit_reasons = np.full(num_entries, 3, dtype=np.int8)  # 0=SL, 1=TP, 2=Time, 3=End
        exit_prices_arr = np.full(num_entries, closes[-1], dtype=np.float64)
        
        for idx in range(num_entries):
            entry_idx = entry_indices[idx]
            tp_price = tp_prices[idx]
            sl_price = sl_prices[idx]
            max_exit = min(entry_idx + m_value + 1, n)
            
            # Check each bar from entry+1 to max_exit
            for j in range(entry_idx + 1, max_exit):
                if lows[j] <= sl_price:
                    exit_indices[idx] = j
                    exit_reasons[idx] = 0  # SL
                    exit_prices_arr[idx] = sl_price
                    break
                elif highs[j] >= tp_price:
                    exit_indices[idx] = j
                    exit_reasons[idx] = 1  # TP
                    exit_prices_arr[idx] = tp_price
                    break
            else:
                # No TP/SL hit within m_value
                if entry_idx + m_value < n:
                    exit_indices[idx] = entry_idx + m_value
                    exit_reasons[idx] = 2  # Time
                    exit_prices_arr[idx] = closes[entry_idx + m_value]
                else:
                    exit_indices[idx] = n - 1
                    exit_reasons[idx] = 3  # End
                    exit_prices_arr[idx] = closes[-1]
        
        # ============================================
        # STEP 4: Create events and process sequentially
        # ============================================
        # Events: (time, type, trade_idx) where type: 0=entry, 1=exit
        events = []
        for idx in range(num_entries):
            events.append((entry_indices[idx], 0, idx))  # Entry event
            events.append((exit_indices[idx], 1, idx))   # Exit event
        
        # Sort by time, then exit before entry at same time
        events.sort(key=lambda x: (x[0], -x[1]))
        
        # Process events
        balance = 1.0
        trades = 0
        wins = 0
        draws = 0
        
        # Track: risk_amount for each trade, whether position is open
        risk_amounts = np.zeros(num_entries, dtype=np.float64)
        is_open = np.zeros(num_entries, dtype=bool)
        open_count = 0
        
        for event_time, event_type, trade_idx in events:
            if event_type == 0:  # Entry
                # Check max_positions limit
                if open_count >= max_positions:
                    # Skip this entry
                    exit_reasons[trade_idx] = -1  # Mark as skipped
                    continue
                
                # Calculate expected equity (assuming all open positions hit TP)
                expected_tp_profit = np.sum(risk_amounts[is_open] * rr_ratio)
                expected_equity = max(balance + expected_tp_profit, 0.001)  # Prevent negative
                
                # Calculate risk_amount based on expected equity
                risk_amounts[trade_idx] = expected_equity * risk_pct
                is_open[trade_idx] = True
                open_count += 1
                
            else:  # Exit
                if exit_reasons[trade_idx] == -1:  # Was skipped
                    continue
                if not is_open[trade_idx]:  # Already closed (shouldn't happen)
                    continue
                
                pos_risk_amount = risk_amounts[trade_idx]
                reason = exit_reasons[trade_idx]
                
                if reason == 0:  # SL
                    balance -= pos_risk_amount
                    balance = max(balance, 0.001)  # Prevent going negative
                elif reason == 1:  # TP
                    balance += pos_risk_amount * rr_ratio
                    wins += 1
                else:  # Time or End
                    entry_price = entry_prices[trade_idx]
                    exit_price = exit_prices_arr[trade_idx]
                    leveraged_return = 0.0
                    if entry_price > 0:
                        raw_return = (exit_price / entry_price) - 1.0
                        net_return = raw_return - (2 * FEE_MAKER)
                        leverage = leverages[trade_idx]
                        leveraged_return = net_return * leverage
                        # Clamp to prevent extreme values
                        leveraged_return = max(min(leveraged_return, 10.0), -0.99)
                    pnl_amount = pos_risk_amount * (leveraged_return / risk_pct) if risk_pct > 0 else 0
                    balance += pnl_amount
                    balance = max(balance, 0.001)  # Prevent negative
                    draws += 1
                
                trades += 1
                is_open[trade_idx] = False
                open_count -= 1
        
        # ============================================
        # STEP 5: Build trade list if requested
        # ============================================
        if return_trades:
            for idx in range(num_entries):
                if exit_reasons[idx] == -1:  # Skipped
                    continue
                
                reason = exit_reasons[idx]
                reason_str = ["SL", "TP", "Time", "End"][reason]
                pos_risk_amount = risk_amounts[idx]
                
                # Safety check for risk_amount
                if not np.isfinite(pos_risk_amount) or pos_risk_amount <= 0:
                    pos_risk_amount = 0.02  # Default to 2%
                
                if reason == 0:  # SL
                    pnl_pct = -pos_risk_amount * 100
                elif reason == 1:  # TP
                    pnl_pct = pos_risk_amount * rr_ratio * 100
                else:  # Time or End
                    entry_price = entry_prices[idx]
                    exit_price = exit_prices_arr[idx]
                    if entry_price > 0:
                        raw_return = (exit_price / entry_price) - 1.0
                        net_return = raw_return - (2 * FEE_MAKER)
                        leverage = leverages[idx]
                        leveraged_return = net_return * leverage
                        # Clamp leveraged_return
                        leveraged_return = max(min(leveraged_return, 10.0), -0.99)
                        pnl_pct = (pos_risk_amount * (leveraged_return / risk_pct) * 100) if risk_pct > 0 else 0
                    else:
                        pnl_pct = 0.0
                
                # Final safety clamp
                pnl_pct = max(min(float(pnl_pct), 10000.0), -10000.0)
                if not np.isfinite(pnl_pct):
                    pnl_pct = 0.0
                
                trade_list.append({
                    "entry_idx": int(entry_indices[idx]),
                    "exit_idx": int(exit_indices[idx]),
                    "entry_price": float(entry_prices[idx]),
                    "exit_price": float(exit_prices_arr[idx]),
                    "reason": reason_str,
                    "pnl_pct": float(pnl_pct),
                    "leverage": float(min(leverages[idx], 100.0)),
                    "holding_bars": int(exit_indices[idx] - entry_indices[idx])
                })
        
        # Convert balance to log return for compatibility
        pnl = np.log(balance) if balance > 0.001 else np.log(0.001)
        return pnl, trades, wins, draws, trade_list

    # --- Standard Logic (No Overlap) ---
    in_position = False
    entry_price = 0.0
    entry_idx = 0
    
    # State for current position
    curr_tp_price = 0.0
    curr_sl_price = 0.0
    curr_leverage = 1.0
    
    pnl = 0.0
    trades = 0
    wins = 0
    draws = 0
    
    for i in range(n):
        if in_position:
            current_low = lows[i]
            current_high = highs[i]
            
            exit_idx = -1
            exit_price = 0.0
            reason = ""
            trade_pnl_pct = 0.0
            completed = False

            # Check SL First (Conservative)
            if current_low <= curr_sl_price:
                actual_loss_pct = (entry_price - curr_sl_price) / entry_price
                total_loss_pct = actual_loss_pct + FEE_MAKER + FEE_TAKER
                
                leveraged_loss = total_loss_pct * curr_leverage
                remaining = 1.0 - leveraged_loss
                pnl += np.log(remaining if remaining > 0.001 else 0.001)
                
                trades += 1
                in_position = False
                
                exit_idx = i
                exit_price = curr_sl_price
                reason = "SL"
                trade_pnl_pct = -leveraged_loss * 100
                completed = True

            # Check TP
            elif current_high >= curr_tp_price:
                # TP Hit - fee compensation already included in tp_dist
                actual_profit_pct = (curr_tp_price - entry_price) / entry_price
                # No fee subtraction needed (already compensated in tp_dist)
                leveraged_profit = actual_profit_pct * curr_leverage
                pnl += np.log(1.0 + leveraged_profit)
                
                wins += 1
                trades += 1
                in_position = False
                
                exit_idx = i
                exit_price = curr_tp_price
                reason = "TP"
                trade_pnl_pct = leveraged_profit * 100
                completed = True

            # Check Time Limit
            elif i - entry_idx >= m_value:
                exit_price = closes[i]
                raw_return = (exit_price / entry_price) - 1.0
                net_return = raw_return - (2 * FEE_MAKER)
                
                leveraged_return = net_return * curr_leverage
                pnl += np.log(1.0 + leveraged_return) if (1.0 + leveraged_return) > 0.001 else np.log(0.001)
                
                draws += 1
                trades += 1
                in_position = False
                
                exit_idx = i
                reason = "Time"
                trade_pnl_pct = leveraged_return * 100
                completed = True
            
            if completed and return_trades:
                trade_list.append({
                    "entry_idx": entry_idx,
                    "exit_idx": exit_idx,
                    "entry_price": float(entry_price),
                    "exit_price": float(exit_price),
                    "reason": reason,
                    "pnl_pct": float(trade_pnl_pct),
                    "leverage": float(curr_leverage),
                    "holding_bars": int(exit_idx - entry_idx)
                })

        # Check for Entry
        if not in_position:
            if buy_signals[i]:
                in_position = True
                entry_price = closes[i]
                entry_idx = i
                current_atr = atrs[i]
                
                # Setup Position Params
                sl_dist = current_atr * sl_mult
                # Fee compensation: tp_pct = rr_ratio × (sl_pct + fees)
                total_fees = FEE_MAKER + FEE_TAKER
                fee_compensation = entry_price * total_fees * rr_ratio
                tp_dist = sl_dist * rr_ratio + fee_compensation
                
                curr_sl_price = entry_price - sl_dist
                curr_tp_price = entry_price + tp_dist
                
                # Setup Leverage
                sl_pct_val = sl_dist / entry_price
                loss_pct_with_fees = sl_pct_val + FEE_MAKER + FEE_TAKER
                
                if loss_pct_with_fees > 0:
                    base_leverage = risk_pct / loss_pct_with_fees
                else:
                    base_leverage = 1.0
                
                curr_leverage = min(base_leverage, 50.0)
    
    return pnl, trades, wins, draws, trade_list


def _run_cross_margin_backtest(
    df, probs, threshold, rr_ratio, 
    initial_balance=10000.0, 
    risk_pct=0.02, 
    sl_mult=1.0, 
    m_value=60,
    max_leverage=50.0,
    return_trades=False
):
    """
    Cross Margin 방식의 백테스트. 
    잔고 추적 + 동적 포지션 사이징 + 미실현 손익 반영.
    
    Args:
        df: DataFrame with close, high, low, atr columns
        probs: Model predictions (N, 3)
        threshold: TP probability threshold
        rr_ratio: Risk/Reward ratio
        initial_balance: Starting balance in USDT
        risk_pct: Fixed risk % per trade (e.g., 0.02 = 2%)
        sl_mult: SL distance multiplier (ATR * sl_mult)
        m_value: Max holding candles
        max_leverage: Maximum allowed effective leverage
        return_trades: Whether to return detailed trade list
    
    Returns:
        (final_balance, total_trades, wins, draws, trade_list)
    """
    if df.empty or len(probs) != len(df):
        return initial_balance, 0, 0, 0, []

    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    atrs = df['atr'].values
    
    if probs.shape[1] == 3:
        prob_up = probs[:, 1]
    else:
        return initial_balance, 0, 0, 0, []

    buy_signals = (prob_up >= threshold)
    n = len(closes)
    
    # State tracking
    balance = initial_balance  # Realized balance (실현 잔고)
    open_positions = []  # List of open position dicts
    trade_list = [] if return_trades else None
    
    total_trades = 0
    wins = 0
    draws = 0
    
    for i in range(n):
        current_price = closes[i]
        current_high = highs[i]
        current_low = lows[i]
        
        # ============================================
        # 1. Check existing positions for TP/SL/Time
        # ============================================
        positions_to_close = []
        
        for pos_idx, pos in enumerate(open_positions):
            # Check SL
            if current_low <= pos['sl_price']:
                # SL Hit
                loss = pos['position_size'] * (pos['entry_price'] - pos['sl_price'])
                fee = pos['position_size'] * pos['entry_price'] * (FEE_MAKER + FEE_TAKER)
                net_loss = loss + fee
                
                balance -= net_loss
                total_trades += 1
                
                if return_trades:
                    pnl_pct = -net_loss / initial_balance * 100
                    trade_list.append({
                        "entry_idx": pos['entry_idx'],
                        "exit_idx": i,
                        "entry_price": float(pos['entry_price']),
                        "exit_price": float(pos['sl_price']),
                        "reason": "SL",
                        "pnl_pct": float(pnl_pct),
                        "pnl_usdt": float(-net_loss),
                        "position_size": float(pos['position_size']),
                        "leverage": float(pos['leverage']),
                        "holding_bars": int(i - pos['entry_idx']),
                        "balance_after": float(balance)
                    })
                
                positions_to_close.append(pos_idx)
                continue
            
            # Check TP
            elif current_high >= pos['tp_price']:
                # TP Hit - fee compensation already included in tp_dist
                profit = pos['position_size'] * (pos['tp_price'] - pos['entry_price'])
                # No fee subtraction needed (already compensated in tp_dist)
                
                balance += profit
                total_trades += 1
                wins += 1
                
                if return_trades:
                    pnl_pct = profit / initial_balance * 100
                    trade_list.append({
                        "entry_idx": pos['entry_idx'],
                        "exit_idx": i,
                        "entry_price": float(pos['entry_price']),
                        "exit_price": float(pos['tp_price']),
                        "reason": "TP",
                        "pnl_pct": float(pnl_pct),
                        "pnl_usdt": float(profit),
                        "position_size": float(pos['position_size']),
                        "leverage": float(pos['leverage']),
                        "holding_bars": int(i - pos['entry_idx']),
                        "balance_after": float(balance)
                    })
                
                positions_to_close.append(pos_idx)
                continue
            
            # Check Time Limit
            elif i - pos['entry_idx'] >= m_value:
                # Time Exit
                exit_price = current_price
                raw_pnl = pos['position_size'] * (exit_price - pos['entry_price'])
                fee = pos['position_size'] * pos['entry_price'] * (2 * FEE_MAKER)
                net_pnl = raw_pnl - fee
                
                balance += net_pnl
                total_trades += 1
                draws += 1
                
                if return_trades:
                    pnl_pct = net_pnl / initial_balance * 100
                    trade_list.append({
                        "entry_idx": pos['entry_idx'],
                        "exit_idx": i,
                        "entry_price": float(pos['entry_price']),
                        "exit_price": float(exit_price),
                        "reason": "Time",
                        "pnl_pct": float(pnl_pct),
                        "pnl_usdt": float(net_pnl),
                        "position_size": float(pos['position_size']),
                        "leverage": float(pos['leverage']),
                        "holding_bars": int(i - pos['entry_idx']),
                        "balance_after": float(balance)
                    })
                
                positions_to_close.append(pos_idx)
                continue
        
        # Remove closed positions (reverse order to avoid index issues)
        for pos_idx in sorted(positions_to_close, reverse=True):
            open_positions.pop(pos_idx)
        
        # ============================================
        # 2. Calculate Current Equity (Balance + Unrealized PnL)
        # ============================================
        unrealized_pnl = sum(
            pos['position_size'] * (current_price - pos['entry_price'])
            for pos in open_positions
        )
        equity = balance + unrealized_pnl
        
        # ============================================
        # 3. Calculate Current Effective Leverage
        # ============================================
        total_exposure = sum(
            pos['position_size'] * pos['entry_price']
            for pos in open_positions
        )
        current_effective_leverage = total_exposure / equity if equity > 0 else 0
        
        # ============================================
        # 4. Check for New Entry Signal
        # ============================================
        if buy_signals[i] and equity > 0:
            # Check if we can add more leverage
            if current_effective_leverage < max_leverage:
                entry_price = current_price
                current_atr = atrs[i]
                
                # SL/TP calculation with fee compensation
                sl_dist = current_atr * sl_mult
                # Fee compensation: tp_pct = rr_ratio × (sl_pct + fees)
                total_fees = FEE_MAKER + FEE_TAKER
                fee_compensation = entry_price * total_fees * rr_ratio
                tp_dist = sl_dist * rr_ratio + fee_compensation
                sl_price = entry_price - sl_dist
                tp_price = entry_price + tp_dist
                
                # Position sizing: Risk-based
                risk_amount = equity * risk_pct  # e.g., 10500 * 0.02 = 210 USDT
                
                # Position size (in base currency, e.g., BTC)
                # If SL is hit, loss = position_size * sl_dist
                # We want: position_size * sl_dist = risk_amount
                position_size = risk_amount / sl_dist if sl_dist > 0 else 0
                
                # Calculate position notional value
                position_notional = position_size * entry_price
                
                # Calculate new leverage after this position
                new_total_exposure = total_exposure + position_notional
                new_effective_leverage = new_total_exposure / equity if equity > 0 else 0
                
                # Cap to max leverage
                if new_effective_leverage > max_leverage:
                    # Reduce position size to fit max leverage
                    max_additional_exposure = (max_leverage * equity) - total_exposure
                    if max_additional_exposure > 0:
                        position_notional = max_additional_exposure
                        position_size = position_notional / entry_price
                        new_effective_leverage = max_leverage
                    else:
                        position_size = 0  # Can't add any more
                
                if position_size > 0:
                    open_positions.append({
                        'entry_idx': i,
                        'entry_price': entry_price,
                        'sl_price': sl_price,
                        'tp_price': tp_price,
                        'position_size': position_size,
                        'leverage': new_effective_leverage,
                        'risk_amount': risk_amount
                    })
    
    # Close any remaining positions at the end
    final_price = closes[-1]
    for pos in open_positions:
        raw_pnl = pos['position_size'] * (final_price - pos['entry_price'])
        fee = pos['position_size'] * pos['entry_price'] * (2 * FEE_MAKER)
        net_pnl = raw_pnl - fee
        balance += net_pnl
        total_trades += 1
        draws += 1
        
        if return_trades:
            pnl_pct = net_pnl / initial_balance * 100
            trade_list.append({
                "entry_idx": pos['entry_idx'],
                "exit_idx": n - 1,
                "entry_price": float(pos['entry_price']),
                "exit_price": float(final_price),
                "reason": "End",
                "pnl_pct": float(pnl_pct),
                "pnl_usdt": float(net_pnl),
                "position_size": float(pos['position_size']),
                "leverage": float(pos['leverage']),
                "holding_bars": int(n - 1 - pos['entry_idx']),
                "balance_after": float(balance)
            })
    
    return balance, total_trades, wins, draws, trade_list

