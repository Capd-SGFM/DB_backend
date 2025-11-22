# models/pipeline_state.py
from __future__ import annotations

from datetime import datetime, timezone
from enum import IntEnum

from sqlalchemy import Integer, Boolean, String, Text, TIMESTAMP, text, select, insert, delete
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base
from db_module.connect_sqlalchemy_engine import SyncSessionLocal
from models.error_log import ErrorLogCurrent, ErrorLogHistory

class PipelineComponent(IntEnum):
    """
    1 : 전체 파이프라인 on/off
    2 : WebSocket 실시간 수집 on/off
    3 : Backfill 엔진 on/off
    4 : REST 유지보수 엔진 on/off
    5 : 보조지표 엔진 on/off
    """

    PIPELINE = 1
    WEBSOCKET = 2
    BACKFILL = 3
    REST_MAINTENANCE = 4
    INDICATOR = 5


class PipelineState(Base):
    """
    trading_data.pipeline_state 테이블과 매핑되는 SQLAlchemy 모델
    """

    __tablename__ = "pipeline_state"
    __table_args__ = {"schema": "trading_data"}

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("FALSE")
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("now()"),
        onupdate=text("now()"),
    )
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    current_backfill_run_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )


# 편의를 위한 상수 (기존 코드 호환용)
PIPELINE_STATE_PK = PipelineComponent.PIPELINE.value


def _get_state(component: PipelineComponent) -> PipelineState | None:
    with SyncSessionLocal() as session:
        return session.get(PipelineState, int(component))


def _set_state(component: PipelineComponent, active: bool) -> None:
    now = datetime.now(timezone.utc)
    with SyncSessionLocal() as session:
        obj = session.get(PipelineState, int(component))
        if obj is None:
            obj = PipelineState(
                id=int(component),
                is_active=active,
                updated_at=now,
                last_error=None,  # 새로 만들 땐 에러 없음
            )
            session.add(obj)
        else:
            obj.is_active = active
            obj.updated_at = now
        session.commit()


def set_component_error(
    component: PipelineComponent, error_message: str | None
) -> None:
    """
    해당 컴포넌트의 마지막 에러 메시지를 기록.
    - error_message=None 이면 에러 초기화
    - 에러 발생 시 is_active를 False 로 내려버리는 것도 선택사항
    """
    now = datetime.now(timezone.utc)
    with SyncSessionLocal() as session:
        obj = session.get(PipelineState, int(component))
        if obj is None:
            obj = PipelineState(
                id=int(component),
                is_active=False,
                updated_at=now,
                last_error=error_message,
            )
            session.add(obj)
        else:
            obj.last_error = error_message
            obj.updated_at = now
        session.commit()


def is_pipeline_active() -> bool:
    st = _get_state(PipelineComponent.PIPELINE)
    return bool(st and st.is_active)


def set_pipeline_active(active: bool) -> None:
    _set_state(PipelineComponent.PIPELINE, active)

    # 전체 파이프라인 OFF → 하위 컴포넌트도 모두 OFF
    if not active:
        for comp in (
            PipelineComponent.WEBSOCKET,
            PipelineComponent.BACKFILL,
            PipelineComponent.REST_MAINTENANCE,
            PipelineComponent.INDICATOR,
        ):
            _set_state(comp, False)
    else:
        # ★ 새로 ON 할 때는 이전 에러는 깔끔하게 초기화
        for comp in (
            PipelineComponent.WEBSOCKET,
            PipelineComponent.BACKFILL,
            PipelineComponent.REST_MAINTENANCE,
            PipelineComponent.INDICATOR,
        ):
            set_component_error(comp, None)  # last_error = NULL


def set_component_active(component: PipelineComponent, active: bool) -> None:
    """
    개별 엔진(WebSocket / Backfill / REST 유지보수 / 보조지표)의 on/off 플래그를 갱신.
    Celery 파이프라인 태스크에서 단계 전환 시 호출.
    """
    _set_state(component, active)


def get_all_pipeline_states() -> dict[str, dict]:
    with SyncSessionLocal() as session:
        rows = (
            session.query(PipelineState)
            .filter(PipelineState.id.in_([c.value for c in PipelineComponent]))
            .all()
        )
        mapping = {row.id: row for row in rows}

    result: dict[str, dict] = {}
    for comp in PipelineComponent:
        row = mapping.get(comp.value)
        key = {
            PipelineComponent.PIPELINE: "pipeline",
            PipelineComponent.WEBSOCKET: "websocket",
            PipelineComponent.BACKFILL: "backfill",
            PipelineComponent.REST_MAINTENANCE: "rest_maintenance",
            PipelineComponent.INDICATOR: "indicator",
        }[comp]
        result[key] = {
            "id": comp.value,
            "is_active": bool(row.is_active) if row else False,
            "updated_at": (
                row.updated_at.isoformat() if row and row.updated_at else None
            ),
            "last_error": row.last_error if row else None,
        }

    return result


def get_current_backfill_run_id() -> str | None:
    st = _get_state(PipelineComponent.PIPELINE)
    return st.current_backfill_run_id if st else None


def set_current_backfill_run_id(run_id: str | None) -> None:
    now = datetime.now(timezone.utc)
    with SyncSessionLocal() as session:
        obj = session.get(PipelineState, int(PipelineComponent.PIPELINE))
        if obj is None:
            obj = PipelineState(
                id=int(PipelineComponent.PIPELINE),
                is_active=False,
                updated_at=now,
                last_error=None,
                current_backfill_run_id=run_id,
            )
            session.add(obj)
        else:
            obj.current_backfill_run_id = run_id
            obj.updated_at = now
        session.commit()


def log_pipeline_error(
    component: PipelineComponent | str,
    error_message: str,
    symbol: str | None = None,
    interval: str | None = None,
) -> None:
    """
    현재 에러 로그 테이블(error_logs_current)에 에러 추가.
    """
    if not error_message:
        return

    comp_str = (
        component.name if isinstance(component, PipelineComponent) else str(component)
    )

    with SyncSessionLocal() as session, session.begin():
        stmt = insert(ErrorLogCurrent).values(
            component=comp_str,
            symbol=symbol,
            interval=interval,
            error_message=error_message,
        )
        session.execute(stmt)


def archive_current_errors() -> None:
    """
    파이프라인 OFF 시 호출.
    Current 테이블의 내용을 History 테이블로 이동하고 Current를 비움.
    """
    with SyncSessionLocal() as session, session.begin():
        # 1. Current 조회
        rows = session.execute(select(ErrorLogCurrent)).scalars().all()
        if not rows:
            return

        # 2. History에 삽입
        history_data = [
            {
                "component": row.component,
                "symbol": row.symbol,
                "interval": row.interval,
                "error_message": row.error_message,
                "occurred_at": row.occurred_at,
            }
            for row in rows
        ]
        session.execute(insert(ErrorLogHistory).values(history_data))

        # 3. Current 삭제
        session.execute(delete(ErrorLogCurrent))
