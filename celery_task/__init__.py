from celery import Celery

CELERY_BROKER_URL = "redis://redis:6379/0"
CELERY_RESULT_BACKEND = "redis://redis:6379/1"

celery_app = Celery(
    "celery_task",
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
    task_track_started=True,
)
