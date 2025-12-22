from datetime import datetime, timedelta, timezone
import os

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, constr
from jose import jwt, JWTError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from db_module.connect_sqlalchemy_engine import get_async_db

router = APIRouter(prefix="/auth", tags=["auth"])

# ===== JWT =====
JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "10080"))
security = HTTPBearer(auto_error=True)


class TokenData(BaseModel):
    # sub=email, id=google_id
    sub: str | None = None
    id: str | None = None
    name: str | None = None
    exp: int | None = None


def create_access_token(data: dict, minutes: int | None = None) -> str:
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=minutes or ACCESS_TOKEN_EXPIRE_MINUTES
    )
    payload = {**data, "exp": int(expire.timestamp())}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> TokenData:
    if not JWT_SECRET or JWT_SECRET.strip() == "":
        raise HTTPException(500, "Server misconfiguration: JWT_SECRET not set")

    token = credentials.credentials
    try:
        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
            options={
                "require_exp": True,
                "verify_signature": True,
            },
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired")
    except JWTError:
        raise HTTPException(401, "Invalid token or signature")

    td = TokenData(
        sub=payload.get("sub"),
        id=str(payload.get("id")) if payload.get("id") is not None else None,
        name=payload.get("name"),
        exp=payload.get("exp"),
    )

    if not td.sub or not td.id:
        raise HTTPException(401, "Invalid token payload")

    return td





# ===== 로그인 상태 확인 =====
@router.get("/me")
async def auth_me(token: TokenData = Depends(verify_token)):
    """JWT 토큰이 유효한지 확인하고, 사용자 정보 반환"""
    return {
        "email": token.sub,
        "google_id": token.id,
        "name": token.name,
        "exp": token.exp,
    }
