from fastapi import FastAPI
from auth import auth_google
from routers import get_crypto_info
from backfill import ohlcv_backfill, symbols
from fastapi.middleware.cors import CORSMiddleware
from auth.auth import router as auth_router
from auth.auth_google import router as google_router

import models
from models.base import Base

app = FastAPI(title="DB backend")


origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(google_router)
app.include_router(get_crypto_info.router)
app.include_router(ohlcv_backfill.router)
app.include_router(symbols.router)


@app.get("/")
def root():
    return {"msg": "DB backend"}
