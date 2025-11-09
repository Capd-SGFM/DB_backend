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


# ===== 입력 모델 =====
class ProfileUpdateIn(BaseModel):
    username: constr(min_length=1, max_length=12)
    email_opt_in: bool = False


# ===== 로그인 상태 확인 =====
@router.get("/me")
async def auth_me(token: TokenData = Depends(verify_token)):
    return {
        "email": token.sub,
        "google_id": token.id,
        "name": token.name,
        "exp": token.exp,
    }


# ===== 회원가입 완료/프로필 업데이트  =====
@router.patch("/me/profile")
async def update_my_profile(
    body: ProfileUpdateIn,
    token: TokenData = Depends(verify_token),  # sub=email, id=google_id
    db: AsyncSession = Depends(get_async_db),
):
    username = body.username.strip()[:12]
    try:
        await db.execute(
            text(
                """
INSERT INTO users.accounts (google_id, username, email, email_opt_in)
VALUES (:google_id, :username, :email, :email_opt_in)
ON CONFLICT (google_id) DO UPDATE
SET username     = EXCLUDED.username,
    email        = EXCLUDED.email,       -- 동일 이메일 호출 전제
    email_opt_in = EXCLUDED.email_opt_in,
    updated_at   = NOW()
"""
            ),
            {
                "google_id": token.id,
                "username": username,
                "email": token.sub,
                "email_opt_in": bool(body.email_opt_in),
            },
        )
        await db.commit()
    except Exception as e:
        await db.rollback()
        raise HTTPException(409, f"profile upsert failed: {e}")
    return {"ok": True}


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
