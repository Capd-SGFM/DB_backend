from fastapi import APIRouter, HTTPException, Depends
from celery.result import AsyncResult
from typing import List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from loguru import logger
from .schemas import BackfillResponse, TaskInfo, TaskStatusResponse
from celery_task.rest_api_task import backfill_symbol_interval
from celery_task import celery_app
from db_module.connect_sqlalchemy_engine import get_async_db
from models import CryptoInfo, OHLCV_MODELS


router = APIRouter(prefix="/ohlcv", tags=["OHLCV"])



