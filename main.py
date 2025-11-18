# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from auth.auth import router as auth_router
from auth.auth_google import router as google_router
from routers import get_crypto_info
from backfill import ohlcv_backfill, symbols
from routers.pipeline import router as pipeline_router

# from routers.rest_progress import router as rest_progress_router

import models  # noqa: F401 (모델 로딩용)
from models.base import Base  # noqa: F401


app = FastAPI(title="DB backend")

# CORS 설정 (필요에 맞게 수정 가능)
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(auth_router)
app.include_router(google_router)
app.include_router(get_crypto_info.router)
app.include_router(ohlcv_backfill.router)
app.include_router(symbols.router)
app.include_router(pipeline_router)
# app.include_router(rest_progress_router)


@app.get("/")
def root():
    return {"msg": "DB backend"}
