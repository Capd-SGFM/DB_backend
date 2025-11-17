# celery_task/__init__.py

from celery import Celery

# Redis 설정 (기존과 동일하게 유지)
CELERY_BROKER_URL = "redis://redis:6379/0"  # 작업 요청용
CELERY_RESULT_BACKEND = "redis://redis:6379/1"  # 진행률/결과 저장용

celery_app = Celery(
    "worker",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=[
        "celery_task.rest_api_task",
        "celery_task.websocket_task",
        "celery_task.rest_maintenance_task",
        "celery_task.indicator_task",
        "celery_task.pipeline_task",
    ],
)

celery_app.conf.update(
    task_track_started=True,  # 작업이 'STARTED' 상태를 보고하도록 설정
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
)
