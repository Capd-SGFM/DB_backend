from celery import Celery


CELERY_BROKER_URL = "redis://redis:6379/0"  # 작업 요청용
CELERY_RESULT_BACKEND = "redis://redis:6379/1"  # 진행률/결과 저장용

celery_app = Celery(
    "worker",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=[
        "celery_task.rest_api_task"
    ],  # 'rest_api_task.py' 파일을 작업 목록으로 포함
)
celery_app.conf.update(
    task_track_started=True,  # 작업이 'STARTED' 상태를 보고하도록 설정
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
)
