from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Body, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert
from loguru import logger
from pathlib import Path
import pandas as pd
import httpx

from db_module.connect_sqlalchemy_engine import get_async_db
from models import CryptoInfo, SymbolTradingRules

router = APIRouter(prefix="/get_symbol_info", tags=["get_symbol_info"])

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "initial_settings" / "symbol_data" / "live_trading.csv"
BACKTESTING_CSV_PATH = BASE_DIR / "initial_settings" / "symbol_data" / "only_backtesting.csv"

BINANCE_FAPI_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"


def parse_filters(filters: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    API의 'filters' 리스트를 DB 스키마에 맞는 딕셔너리로 변환
    """
    parsed = {}
    for f in filters:
        ft = f.get("filterType")
        if ft == "PRICE_FILTER":
            parsed["tick_size"] = f.get("tickSize")
        elif ft == "LOT_SIZE":
            parsed["min_qty"] = f.get("minQty")
            parsed["max_qty"] = f.get("maxQty")
            parsed["step_size"] = f.get("stepSize")
        elif ft == "MARKET_LOT_SIZE":
            parsed["market_min_qty"] = f.get("minQty")
            parsed["market_max_qty"] = f.get("maxQty")
            parsed["market_step_size"] = f.get("stepSize")
        elif ft == "MIN_NOTIONAL":
            parsed["min_notional"] = f.get("notional")
        elif ft == "MAX_NUM_ORDERS":
            parsed["max_num_orders"] = f.get("limit")
    schema_keys = [
        "tick_size",
        "min_qty",
        "max_qty",
        "step_size",
        "market_min_qty",
        "market_max_qty",
        "market_step_size",
        "min_notional",
        "max_num_orders",
    ]

    for key in schema_keys:
        if key not in parsed:
            parsed[key] = None

    return parsed


async def _process_symbols(db: AsyncSession, csv_path: Path, is_backtesting_only: bool):
    """
    공통 심볼 등록 로직 처리
    """
    try:
        # 1. CSV에서 기준 심볼 로드
        logger.info(f"CSV 경로 확인: {csv_path}")
        if not csv_path.exists():
            msg = f"CSV 파일이 없습니다: {csv_path}"
            logger.error(msg)
            raise HTTPException(status_code=404, detail=msg)

        df = pd.read_csv(csv_path)
        if "symbol" not in df.columns:
            msg = "CSV에 'symbol' 컬럼이 없습니다."
            logger.error(msg)
            raise HTTPException(status_code=400, detail=msg)

        s = (
            df["symbol"]
            .astype(str)
            .str.strip()
            .str.upper()
            .replace({"": None})
            .dropna()
        )
        s = s[s.str.len() <= 30].drop_duplicates()

        if s.empty:
            msg = "등록할 심볼이 없습니다(전처리 후 빈 목록)."
            logger.warning(msg)
            return {"message": msg, "upserted_count": 0}

        symbols_from_csv_set = set(s.tolist())
        logger.info(f"CSV에서 {len(symbols_from_csv_set)}개 기준 심볼 로드 완료.")

        # 2. Binance fapi/v1/exchangeInfo 호출
        logger.info(f"Binance API 호출: {BINANCE_FAPI_URL}")
        async with httpx.AsyncClient() as client:
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                    "Accept": "application/json"
                }
                response = await client.get(BINANCE_FAPI_URL, headers=headers, timeout=10.0)
                response.raise_for_status()
            except httpx.RequestError as e:
                msg = f"Binance API 요청 실패: {e}"
                logger.error(msg)
                raise HTTPException(status_code=502, detail=msg)

        api_data = response.json()
        logger.success("Binance API 데이터 로드 완료.")

        # 3. 데이터 필터링
        crypto_info_upsert = []
        trading_rules_upsert = []
        
        api_symbols = api_data.get("symbols", [])

        if not api_symbols:
            logger.warning("API에서 'symbols' 데이터를 찾을 수 없습니다.")
            raise HTTPException(
                status_code=500, detail="API response missing 'symbols'"
            )

        for item in api_symbols:
            base_asset = item.get("baseAsset")

            if base_asset not in symbols_from_csv_set:
                continue

            if not (
                item.get("status") == "TRADING"
                and item.get("contractType") == "PERPETUAL"
                and item.get("quoteAsset") == "USDT"
            ):
                continue

            filters_data = parse_filters(item.get("filters", []))

            # 1) metadata.crypto_info 데이터 준비
            row_info = {
                "symbol": base_asset,
                "pair": item.get("symbol"),
                "is_backtesting_only": is_backtesting_only,
            }
            crypto_info_upsert.append(row_info)

            # 2) futures.symbol_trading_rules 데이터 준비
            row_rules = {
                "symbol": base_asset,
                "price_precision": item.get("pricePrecision"),
                "quantity_precision": item.get("quantityPrecision"),
                "required_margin_percent": item.get("requiredMarginPercent"),
                "maint_margin_percent": item.get("maintMarginPercent"),
                "liquidation_fee": item.get("liquidationFee"),
                **filters_data,
            }
            trading_rules_upsert.append(row_rules)

        if not crypto_info_upsert:
            msg = "API에서 CSV와 일치하는 심볼 정보를 찾지 못했습니다."
            logger.warning(msg)
            return {"message": msg, "upserted_count": 0}

        logger.info(
            f"DB에 {len(crypto_info_upsert)}개 심볼 UPSERT (Insert or Update) 시도..."
        )

        # 4. UPSERT 실행 (1): metadata.crypto_info
        stmt_info = insert(CryptoInfo).values(crypto_info_upsert)
        update_cols_info = {
            key: getattr(stmt_info.excluded, key)
            for key in crypto_info_upsert[0].keys()
            if key != "symbol"
        }
        stmt_info = stmt_info.on_conflict_do_update(
            index_elements=["symbol"],
            set_=update_cols_info,
        )
        await db.execute(stmt_info)

        # 5. UPSERT 실행 (2): futures.symbol_trading_rules
        stmt_rules = insert(SymbolTradingRules).values(trading_rules_upsert)
        update_cols_rules = {
            key: getattr(stmt_rules.excluded, key)
            for key in trading_rules_upsert[0].keys()
            if key != "symbol"
        }
        stmt_rules = stmt_rules.on_conflict_do_update(
            index_elements=["symbol"],
            set_=update_cols_rules,
        )
        result = await db.execute(stmt_rules)

        # 6. 커밋
        await db.commit()

        # result.rowcount는 UPSERT로 인해 영향을 받은 총 행의 수를 반환
        upserted_count = result.rowcount

        msg = f"심볼 정보 {upserted_count}개 UPSERT 완료 (백테스팅 전용: {is_backtesting_only})."
        logger.success(msg)
        return {
            "message": msg,
            "upserted_count": upserted_count,
            "csv_symbols_found": len(crypto_info_upsert),
        }

    except HTTPException:
        await db.rollback()
        raise
    except Exception as e:
        await db.rollback()
        msg = f"심볼 등록/업데이트 중 오류 발생: {e}"
        logger.exception(msg)
        raise HTTPException(status_code=500, detail=msg)


@router.post("/register_symbols")
async def register_symbols(db: AsyncSession = Depends(get_async_db)):
    """
    일반 종목 등록 (Live Trading 용 - 'live_trading.csv')
    """
    return await _process_symbols(db, CSV_PATH, is_backtesting_only=False)


@router.post("/register_backtesting_symbols")
async def register_backtesting_symbols(db: AsyncSession = Depends(get_async_db)):
    """
    백테스팅 전용 종목 등록
    """
    return await _process_symbols(db, BACKTESTING_CSV_PATH, is_backtesting_only=True)
