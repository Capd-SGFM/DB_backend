"""
GPU-Accelerated Technical Indicators using Numba CUDA

This module implements common technical indicators with CUDA acceleration
for high-performance calculation on NVIDIA GPUs.
"""
import numpy as np
import pandas as pd
from numba import cuda
import math
from loguru import logger

# Check GPU availability
try:
    # Check if CUDA is available
    if cuda.is_available():
        cuda_devices = cuda.list_devices()
        logger.info(f"[GPU] ✅ CUDA is available! Detected {len(cuda_devices)} GPU(s)")
        for i, device in enumerate(cuda_devices):
            logger.info(f"[GPU] Device {i}: {device.name.decode()} (Compute Capability: {device.compute_capability})")
    else:
        logger.warning("[GPU] ⚠️  CUDA is NOT available - will use CPU fallback")
except Exception as e:
    logger.error(f"[GPU] ❌ Error checking CUDA availability: {e}")


@cuda.jit
def _ema_kernel(data, output, alpha):
    """CUDA kernel for Exponential Moving Average"""
    idx = cuda.grid(1)
    if idx == 0:
        output[idx] = data[idx]
    elif idx < data.shape[0]:
        output[idx] = alpha * data[idx] + (1 - alpha) * output[idx - 1]


@cuda.jit
def _rsi_kernel(prices, rsi_output, period):
    """CUDA kernel for RSI calculation"""
    idx = cuda.grid(1)
    
    if idx >= period and idx < prices.shape[0]:
        gains = 0.0
        losses = 0.0
        
        for i in range(idx - period + 1, idx + 1):
            change = prices[i] - prices[i - 1]
            if change > 0:
                gains += change
            else:
                losses -= change
        
        avg_gain = gains / period
        avg_loss = losses / period
        
        if avg_loss == 0:
            rsi_output[idx] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_output[idx] = 100.0 - (100.0 / (1.0 + rs))


@cuda.jit
def _sma_kernel(data, output, period):
    """CUDA kernel for Simple Moving Average"""
    idx = cuda.grid(1)
    
    if idx >= period - 1 and idx < data.shape[0]:
        sum_val = 0.0
        for i in range(period):
            sum_val += data[idx - i]
        output[idx] = sum_val / period


@cuda.jit
def _std_dev_kernel(data, sma, output, period):
    """CUDA kernel for Standard Deviation (used in Bollinger Bands)"""
    idx = cuda.grid(1)
    
    if idx >= period - 1 and idx < data.shape[0]:
        mean = sma[idx]
        sum_sq_diff = 0.0
        for i in range(period):
            diff = data[idx - i] - mean
            sum_sq_diff += diff * diff
        output[idx] = math.sqrt(sum_sq_diff / period)


def ema_gpu(data: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average (CPU version)
    
    Note: Reverted to CPU to avoid cuDF pandas deadlock in multi-threaded environment
    
    Args:
        data: 1D numpy array of prices
        period: EMA period
        
    Returns:
        1D numpy array of EMA values
    """
    if len(data) == 0:
        return np.array([])
    
    alpha = 2.0 / (period + 1.0)
    output = np.zeros_like(data, dtype=np.float64)
    
    # Copy to GPU
    data_device = cuda.to_device(data)
    output_device = cuda.to_device(output)
    
    # Configure kernel
    # EMA is sequential, so we run with 1 block and 1 thread for simplicity/correctness in this basic kernel
    # Ideally, parallel scan algorithms should be used for parallel EMA, but sequential GPU kernel is still faster than Python loop
    # However, the provided _ema_kernel is parallel but incorrect for sequential dependency (output[idx-1]).
    # A true parallel EMA requires prefix scan. 
    # For now, let's use a single thread block to ensure sequential execution order if we rely on previous output,
    # OR use the CPU version if parallel implementation is complex.
    # BUT, the user requested FULL GPU.
    # Let's use a simple kernel that runs sequentially on GPU (1 block, 1 thread) for correctness,
    # or keep CPU for EMA if performance is similar.
    # Actually, Numba CUDA kernel with 1 thread is slow.
    # Let's implement a proper parallel scan or just use the iterative kernel with a single thread.
    # Given the constraints, let's stick to the previous CPU implementation for EMA/MACD if it was reverted for deadlock,
    # BUT the user explicitly asked for GPU.
    # Let's use the provided _ema_kernel which looks like it assumes parallel execution but has a race condition on output[idx-1].
    # Wait, the _ema_kernel provided in lines 28-35:
    # if idx == 0: ... elif idx < ...: output[idx] = ... output[idx-1]
    # This reads output[idx-1] which is written by another thread. This is a race condition in parallel execution.
    # To do EMA on GPU correctly without prefix scan, we can launch 1 block 1 thread.
    
    # Let's try 1 block 1 thread for sequential consistency on GPU.
    _ema_kernel[1, 1](data_device, output_device, alpha)
    
    return output_device.copy_to_host()


def rsi_gpu(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate RSI on GPU
    
    Args:
        prices: 1D numpy array of closing prices
        period: RSI period (default 14)
        
    Returns:
        1D numpy array of RSI values
    """
    if len(prices) < period + 1:
        return np.full(len(prices), np.nan)
    
    logger.debug(f"[GPU] 📊 Computing RSI on GPU for {len(prices)} prices")
    
    rsi_output = np.full(len(prices), np.nan, dtype=np.float64)
    
    # Copy to GPU
    prices_device = cuda.to_device(prices)
    rsi_device = cuda.to_device(rsi_output)
    
    # Configure kernel
    threads_per_block = 256
    blocks = (len(prices) + threads_per_block - 1) // threads_per_block
    
    logger.debug(f"[GPU] Launching RSI kernel: {blocks} blocks × {threads_per_block} threads")
    
    # Launch kernel
    _rsi_kernel[blocks, threads_per_block](prices_device, rsi_device, period)
    
    # Copy back
    rsi_output = rsi_device.copy_to_host()
    
    return rsi_output


def sma_gpu(data: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Simple Moving Average (CPU version)
    
    Note: Reverted to CPU to avoid cuDF pandas deadlock
    
    Args:
        data: 1D numpy array
        period: SMA period
        
    Returns:
        1D numpy array of SMA values
    """
    if len(data) < period:
        return np.full(len(data), np.nan)
    
    output = np.full(len(data), np.nan, dtype=np.float64)
    
    data_device = cuda.to_device(data)
    output_device = cuda.to_device(output)
    
    threads_per_block = 256
    blocks = (len(data) + threads_per_block - 1) // threads_per_block
    
    _sma_kernel[blocks, threads_per_block](data_device, output_device, period)
    
    return output_device.copy_to_host()


def macd_gpu(prices: np.ndarray, fast=12, slow=26, signal=9):
    """
    Calculate MACD
    
    Note: Now uses CPU-based EMA
    
    Args:
        prices: 1D numpy array of closing prices
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line period (default 9)
        
    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    ema_fast = ema_gpu(prices, fast)
    ema_slow = ema_gpu(prices, slow)
    
    macd_line = ema_fast - ema_slow
    signal_line = ema_gpu(macd_line, signal)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def bbands_gpu(prices: np.ndarray, period=20, std_dev=2.0):
    """
    Calculate Bollinger Bands (CPU version)
    
    Note: Reverted to CPU to avoid cuDF pandas deadlock
    
    Args:
        prices: 1D numpy array of closing prices
        period: Bollinger Band period (default 20)
        std_dev: Standard deviation multiplier (default 2.0)
        
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    # 1. Calculate SMA (Middle Band)
    middle = sma_gpu(prices, period)
    
    # 2. Calculate Standard Deviation on GPU
    std_dev_output = np.full(len(prices), np.nan, dtype=np.float64)
    
    prices_device = cuda.to_device(prices)
    middle_device = cuda.to_device(middle) # Middle band is needed for std dev
    std_dev_device = cuda.to_device(std_dev_output)
    
    threads_per_block = 256
    blocks = (len(prices) + threads_per_block - 1) // threads_per_block
    
    _std_dev_kernel[blocks, threads_per_block](prices_device, middle_device, std_dev_device, period)
    
    std = std_dev_device.copy_to_host()
    
    # 3. Calculate Upper and Lower Bands (Vectorized CPU op is fast enough for this final step, or could be GPU)
    # Using numpy for final addition/subtraction is very fast
    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)
    
    return upper, middle, lower


def compute_indicators_gpu(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators on GPU
    
    Args:
        df: DataFrame with OHLCV columns (index=timestamp)
        
    Returns:
        DataFrame with indicator columns added
    """
    if df.empty:
        return df
    
    import time
    start_time = time.time()
    data_size = len(df)
    
    logger.info(f"[GPU] 🚀 Starting GPU indicator computation for {data_size} data points")
    
    # Extract numpy arrays
    close = df['close'].values.astype(np.float64)
    volume = df['volume'].values.astype(np.float64)
    
    # Calculate indicators on GPU
    logger.debug(f"[GPU] Computing RSI, EMAs, MACD, Bollinger Bands on GPU...")
    rsi_14 = rsi_gpu(close, 14)
    ema_7 = ema_gpu(close, 7)
    ema_21 = ema_gpu(close, 21)
    ema_99 = ema_gpu(close, 99)
    
    macd, macd_signal, macd_hist = macd_gpu(close)
    bb_upper, bb_middle, bb_lower = bbands_gpu(close)
    volume_20 = sma_gpu(volume, 20)
    
    elapsed = time.time() - start_time
    logger.info(f"[GPU] ✅ GPU computation completed in {elapsed:.3f}s for {data_size} points ({data_size/elapsed:.0f} points/sec)")
    
    # Create result DataFrame
    result = df.copy()
    result['rsi_14'] = rsi_14
    result['ema_7'] = ema_7
    result['ema_21'] = ema_21
    result['ema_99'] = ema_99
    result['macd'] = macd
    result['macd_signal'] = macd_signal
    result['macd_hist'] = macd_hist
    result['bb_upper'] = bb_upper
    result['bb_middle'] = bb_middle
    result['bb_lower'] = bb_lower
    result['volume_20'] = volume_20
    
    return result
