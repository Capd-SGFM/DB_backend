# celery_task/websocket_task.py
import asyncio
import json
import uuid
import redis
from datetime import datetime, timezone

import websockets
from websockets.exceptions import ConnectionClosedError
from loguru import logger
from sqlalchemy.dialects.postgresql import insert

from celery_task import celery_app
from db_module.connect_sqlalchemy_engine import SyncSessionLocal
from models import OHLCV_MODELS, CryptoInfo
from models.websocket_progress import WebSocketProgress
from models.pipeline_state import (
    is_pipeline_active,
    set_component_error,
    log_pipeline_error,
    PipelineComponent,
)

STREAM_URL = "wss://fstream.binance.com/stream"

# WebSocket 옵션 / 재접속 딜레이 상수
PING_INTERVAL = 20  # 서버에 ping 보내는 주기(초)
PING_TIMEOUT = 60  # pong 응답 기다리는 최대 시간(초) – 넉넉하게
RECONNECT_DELAY = 5  # 에러 발생 시 재접속까지 대기(초)
PIPELINE_INACTIVE_SLEEP = 5  # 파이프라인 OFF일 때 재확인 주기(초)
NO_SYMBOL_SLEEP = 10  # 구독할 심볼 없을 때 대기(초)


# 전역 변수로 현재 WebSocket run_id 저장
CURRENT_WS_RUN_ID = None


def upsert_websocket_progress(
    run_id: str,
    symbol: str,
    interval: str,
    state: str,
    message_count: int = 0,
    last_error: str | None = None,
):
    """
    websocket_progress 테이블에 상태 업데이트 (UPSERT).
    """
    now = datetime.now(timezone.utc)
    with SyncSessionLocal() as session, session.begin():
        stmt = (
            insert(WebSocketProgress)
            .values(
                run_id=run_id,
                symbol=symbol,
                interval=interval,
                state=state,
                last_message_ts=now if state == "CONNECTED" else None,
                message_count=message_count,
                last_error=last_error,
                updated_at=now,
            )
            .on_conflict_do_update(
                index_elements=["run_id", "symbol", "interval"],
                set_={
                    "state": state,
                    "last_message_ts": now if state == "CONNECTED" else WebSocketProgress.last_message_ts,
                    "message_count": message_count,
                    "last_error": last_error,
                    "updated_at": now,
                },
            )
        )
        session.execute(stmt)


@celery_app.task(name="ohlcv.websocket_collector", queue="websocket")
def websocket_collector():
    """
    파이프라인이 is_active = TRUE 인 동안만
    Binance Futures WebSocket 캔들을 수집하는 무한 루프 태스크.

    - 태스크 시작 시: WebSocket 컴포넌트의 last_error 초기화
    - 치명적인 예외 발생 시: last_error 에 에러 메시지 기록
    """
    # 새로 시작할 때 이전 에러 로그 초기화
    try:
        set_component_error(PipelineComponent.WEBSOCKET, None)
    except Exception:
        # DB 에러 때문에 파이프라인이 죽으면 안 되므로, 여기서 죽이지 않고 로그만 찍음
        logger.exception("[WS] failed to reset last_error on start")

    try:
        asyncio.run(run_ws_loop())
    except Exception as e:
        # run_ws_loop 밖으로 예외가 튀어나온 경우 (치명적 오류)
        logger.exception("[WS] websocket_collector fatal error")
        try:
            set_component_error(
                PipelineComponent.WEBSOCKET,
                f"{type(e).__name__}: {e}",
            )
            log_pipeline_error(
                PipelineComponent.WEBSOCKET,
                f"Fatal error: {type(e).__name__}: {e}",
            )
        except Exception:
            logger.exception("[WS] failed to save fatal last_error")


def get_realtime_symbol_intervals():
    """
    실시간 수집 대상이 되는 (symbol, pair, interval) 목록을 반환.
    - metadata.crypto_info 에 등록된 모든 심볼 중 pair 가 있는 것들
    - OHLCV_MODELS 에 정의된 모든 interval
    """
    from models import OHLCV_MODELS  # 순환 참조 방지용 내부 import

    with SyncSessionLocal() as session:
        rows = (
            session.query(CryptoInfo.symbol, CryptoInfo.pair)
            .filter(CryptoInfo.pair.isnot(None))
            .all()
        )

    intervals = list(OHLCV_MODELS.keys())
    # (내부 symbol, 바이낸스 pair, interval)
    return [(row.symbol, row.pair, iv) for row in rows for iv in intervals]


async def run_ws_loop():
    """
    WebSocket 메인 루프.
    - pipeline_state.id=1 (PIPELINE)이 ON 인 동안에만 동작
    - 예외 발생 시 RECONNECT_DELAY 후 재접속
    - 재접속 에러 및 예상치 못한 에러는 모두 last_error 에 기록
    """
    global CURRENT_WS_RUN_ID
    
    # WebSocket 세션 run_id 생성
    if CURRENT_WS_RUN_ID is None:
        CURRENT_WS_RUN_ID = f"ws-{uuid.uuid4().hex}"
        logger.info(f"[WS] Generated new run_id: {CURRENT_WS_RUN_ID}")
    
    # 메시지 카운터 (심볼×인터벌별)
    message_counters: dict[tuple[str, str], int] = {}
    
    while True:
        try:
            # ───────── 파이프라인 OFF면 WebSocket 연결 시도 자체를 안 함 ─────────
            if not is_pipeline_active():
                logger.info(
                    f"[WS] pipeline inactive. sleep {PIPELINE_INACTIVE_SLEEP}s..."
                )
                # 모든 연결을 DISCONNECTED로 표시
                for (symbol, interval), _ in message_counters.items():
                    try:
                        upsert_websocket_progress(
                            run_id=CURRENT_WS_RUN_ID,
                            symbol=symbol,
                            interval=interval,
                            state="DISCONNECTED",
                            message_count=message_counters.get((symbol, interval), 0),
                        )
                    except Exception:
                        logger.exception(f"[WS] Failed to update DISCONNECTED status for {symbol}/{interval}")
                await asyncio.sleep(PIPELINE_INACTIVE_SLEEP)
                continue

            symbol_intervals = get_realtime_symbol_intervals()
            if not symbol_intervals:
                logger.warning(
                    f"[WS] 구독할 심볼/인터벌이 없습니다. {NO_SYMBOL_SLEEP}초 대기 후 재시도."
                )
                await asyncio.sleep(NO_SYMBOL_SLEEP)
                continue

            # pair -> 내부 symbol 매핑
            pair_to_symbol: dict[str, str] = {}
            for symbol, pair, _ in symbol_intervals:
                if pair:
                    pair_to_symbol[pair.upper()] = symbol

            # WebSocket 구독 스트림은 pair 기준으로
            streams = [
                f"{pair.lower()}@kline_{interval}"
                for symbol, pair, interval in symbol_intervals
            ]
            url = f"{STREAM_URL}?streams={'/'.join(streams)}"
            logger.info(f"[WS] connect: {url}")

            try:
                # ping_interval / ping_timeout 튜닝
                async with websockets.connect(
                    url,
                    ping_interval=PING_INTERVAL,
                    ping_timeout=PING_TIMEOUT,
                ) as ws:
                    logger.info("[WS] ✅ WebSocket connected")

                    # ✅ 연결 성공 시 메인 엔진 상태의 에러도 클리어
                    try:
                        set_component_error(PipelineComponent.WEBSOCKET, None)
                    except Exception:
                        logger.exception("[WS] Failed to clear component error")
                    
                    # 모든 심볼×인터벌 조합을 CONNECTED 상태로 초기화
                    for symbol, pair, interval in symbol_intervals:
                        key = (symbol, interval)
                        if key not in message_counters:
                            message_counters[key] = 0
                        try:
                            upsert_websocket_progress(
                                run_id=CURRENT_WS_RUN_ID,
                                symbol=symbol,
                                interval=interval,
                                state="CONNECTED",
                                message_count=message_counters[key],
                            )
                        except Exception:
                            logger.exception(f"[WS] Failed to update CONNECTED status for {symbol}/{interval}")

                    async for msg in ws:
                        # 파이프라인이 중간에 OFF 되면 연결을 정리하고 루프 상단으로 복귀
                        if not is_pipeline_active():
                            logger.info(
                                "[WS] pipeline deactivated. closing connection."
                            )
                            # async for 를 빠져나가면서 context manager 가 ws.close() 수행
                            break

                        try:
                            data = json.loads(msg)
                        except json.JSONDecodeError:
                            logger.warning("[WS] JSON decode error. message skipped.")
                            continue

                        # Binance multi-stream 포맷: {"stream": "...", "data": {...}}
                        if "data" not in data or "k" not in data["data"]:
                            logger.warning("[WS] unexpected message format. skipped.")
                            continue

                        k = data["data"]["k"]
                        pair = k["s"]  # ex) SOLUSDT
                        interval = k["i"]  # ex) 4h
                        open_time_ms = k["t"]
                        # is_closed = k["x"]  # 현재는 사용하지 않음

                        # pair -> 내부 symbol 로 변환
                        symbol = pair_to_symbol.get(pair.upper())
                        if symbol is None:
                            logger.warning(
                                f"[WS] pair={pair} 에 해당하는 내부 symbol 이 없습니다. 건너뜀."
                            )
                            continue

                        candle = {
                            "symbol": symbol,  # ✅ BTC / ETH / SOL 등 내부 심볼로 저장
                            "timestamp": datetime.fromtimestamp(
                                open_time_ms / 1000, tz=timezone.utc
                            ),
                            "open": float(k["o"]),
                            "high": float(k["h"]),
                            "low": float(k["l"]),
                            "close": float(k["c"]),
                            "volume": float(k["v"]),
                            # WebSocket 단계에서는 아직 '완성된 캔들' 여부를 확정하지 않으므로 항상 False
                            "is_ended": False,
                        }

                        # 메시지 카운터 업데이트 및 진행률 기록
                        key = (symbol, interval)
                        message_counters[key] = message_counters.get(key, 0) + 1
                        
                        # 100개마다 DB 업데이트 (성능 고려)
                        if message_counters[key] % 100 == 0:
                            try:
                                upsert_websocket_progress(
                                    run_id=CURRENT_WS_RUN_ID,
                                    symbol=symbol,
                                    interval=interval,
                                    state="CONNECTED",
                                    message_count=message_counters[key],
                                )
                            except Exception:
                                logger.exception(f"[WS] Failed to update message count for {symbol}/{interval}")
                        
                        # NOTE:
                        # SyncSessionLocal 을 사용하므로 DB가 느리면 이벤트 루프가 잠깐 막힐 수 있음.
                        # 이를 thread executor 로 넘겨서 처리.
                        loop = asyncio.get_running_loop()
                        await loop.run_in_executor(None, save_realtime_ohlcv, candle, interval)

            except ConnectionClosedError as e:
                # keepalive ping timeout 등으로 연결이 끊긴 경우 여기로 옴
                logger.warning(
                    f"[WS] connection closed (code={getattr(e, 'code', None)}, "
                    f"reason={getattr(e, 'reason', None)}). "
                    f"재접속 {RECONNECT_DELAY}s 후 시도."
                )
                error_msg = f"ConnectionClosedError: code={getattr(e, 'code', None)}, reason={getattr(e, 'reason', None)}"
                
                # 모든 연결을 DISCONNECTED로 표시
                for (symbol, interval), count in message_counters.items():
                    try:
                        upsert_websocket_progress(
                            run_id=CURRENT_WS_RUN_ID,
                            symbol=symbol,
                            interval=interval,
                            state="ERROR",
                            message_count=count,
                            last_error=error_msg,
                        )
                    except Exception:
                        logger.exception(f"[WS] Failed to update ERROR status for {symbol}/{interval}")
                
                try:
                    set_component_error(
                        PipelineComponent.WEBSOCKET,
                        error_msg,
                    )
                    log_pipeline_error(
                        PipelineComponent.WEBSOCKET,
                        error_msg,
                    )
                except Exception:
                    logger.exception(
                        "[WS] failed to save ConnectionClosedError last_error"
                    )
                await asyncio.sleep(RECONNECT_DELAY)

            except Exception as e:
                # WebSocket 연결이나 메시지 처리 중 발생하는 예외 (재접속 대상)
                logger.error(f"[WS] error: {e}, reconnect in {RECONNECT_DELAY}s")
                try:
                    set_component_error(
                        PipelineComponent.WEBSOCKET,
                        f"{type(e).__name__}: {e}",
                    )
                    log_pipeline_error(
                        PipelineComponent.WEBSOCKET,
                        f"{type(e).__name__}: {e}",
                    )
                except Exception:
                    logger.exception("[WS] failed to save transient last_error")
                await asyncio.sleep(RECONNECT_DELAY)
                # while True 의 다음 회차에서 다시 pipeline 상태를 보고 재연결 시도

        except Exception as outer_e:
            # 루프 전체를 감싸는 예기치 못한 예외 방지용
            logger.exception(f"[WS] unexpected outer error: {outer_e}")
            try:
                set_component_error(
                    PipelineComponent.WEBSOCKET,
                    f"{type(outer_e).__name__}: {outer_e}",
                )
                log_pipeline_error(
                    PipelineComponent.WEBSOCKET,
                    f"{type(outer_e).__name__}: {outer_e}",
                )
            except Exception:
                logger.exception("[WS] failed to save outer last_error")
            await asyncio.sleep(RECONNECT_DELAY)


def save_realtime_ohlcv(candle: dict, interval: str):
    """
    WebSocket 으로 들어온 실시간 캔들을 해당 interval 의 ohlcv_{interval} 테이블에 upsert.
    - symbol + timestamp 기준으로 ON CONFLICT DO UPDATE
    - is_ended 는 항상 False 로 유지 (REST 유지보수 엔진에서 TRUE 처리)
    """
    from sqlalchemy.dialects.postgresql import insert

    OhlcvModel = OHLCV_MODELS.get(interval)
    if not OhlcvModel:
        # 정의되지 않은 interval 은 무시
        return

    # Redis Publish (Fire and Forget)
    try:
        redis_client = redis.Redis.from_url("redis://redis:6379/0")
        message = json.dumps({
            "type": "ticker",
            "symbol": candle["symbol"],
            "price": candle["close"],
            "timestamp": candle["timestamp"].isoformat()
        })
        redis_client.publish("market_data", message)
    except Exception as e:
        logger.error(f"[WS] Failed to publish to Redis: {e}")

    with SyncSessionLocal() as session, session.begin():
        stmt = insert(OhlcvModel).values(**candle)

        # 동일 (symbol, timestamp) 가 이미 존재하면 가격/거래량만 업데이트
        update_cols = {
            k: getattr(stmt.excluded, k)
            for k in ["open", "high", "low", "close", "volume", "is_ended"]
        }

        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol", "timestamp"],
            set_=update_cols,
        )
        session.execute(stmt)
