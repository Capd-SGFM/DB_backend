from pydantic import BaseModel
from typing import List, Optional, Any


class BackfillRequest(BaseModel):
    intervals: List[str]


class TaskInfo(BaseModel):
    task_id: str
    symbol: str
    interval: str


class BackfillResponse(BaseModel):
    message: str
    tasks: List[TaskInfo]


class TaskStatusResponse(BaseModel):
    task_id: str
    state: str  # PENDING, PROGRESS, SUCCESS, FAILURE
    meta: Optional[Any] = None  # {"pct": 10.5, ...}
