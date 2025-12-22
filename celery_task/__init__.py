from celery import Celery
from celery.signals import worker_process_init

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
        "celery_task.optimization_task",
    ],
)

celery_app.conf.update(
    task_track_started=True,
)

# Enable cuDF in each worker process
@worker_process_init.connect
def init_worker_process(**kwargs):
    try:
        import cudf.pandas
        cudf.pandas.install()
        print("[GPU] cuDF enabled in worker process")

        # 🚀 Fix for Deadlock: Pre-import pandas modules to avoid lazy import race conditions
        import pandas as pd
        import numpy as np
        # Force load internal modules that cause locking
        try:
            import pandas.compat.numpy.function
        except ImportError:
            pass # Might not exist in all versions, but good to try
            
        # Trigger dummy operations to force initialization of internal locks
        df = pd.DataFrame({'a': [1, 2, 3]})
        _ = df.mean()
        print("[GPU] Pandas modules pre-loaded to prevent deadlock")

    except Exception as e:
        print(f"[GPU] Failed to enable cuDF or pre-load pandas: {e}")
