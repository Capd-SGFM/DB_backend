from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import httpx, os, logging
from .auth import create_access_token

router = APIRouter(prefix="/auth/google", tags=["auth: google"])

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")

from db_module.connect_sqlalchemy_engine import get_async_db

log = logging.getLogger("auth.auth_google")


@router.get("/callback")
async def google_callback(code: str, db: AsyncSession = Depends(get_async_db)):
    if not (GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET and GOOGLE_REDIRECT_URI):
        raise HTTPException(500, "Google OAuth env missing")

    async with httpx.AsyncClient(timeout=15.0) as client:
        t = await client.post(
            GOOGLE_TOKEN_URL,
            data={
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": GOOGLE_REDIRECT_URI,
            },
        )
        tj = t.json()
        at = tj.get("access_token")
        if not at:
            raise HTTPException(400, f"google token missing: {tj}")

        u = await client.get(
            GOOGLE_USERINFO_URL, headers={"Authorization": f"Bearer {at}"}
        )
        ui = u.json()

    if not ui.get("verified_email"):
        raise HTTPException(400, "unverified email")

    email = ui.get("email")
    gid = ui.get("id") or ui.get("sub")
    name = ui.get("name")
    if not email or not gid:
        raise HTTPException(500, "google email/id missing")

    # 기존 유저 여부(email 기준)
    row = (
        await db.execute(
            text("SELECT google_id FROM users.accounts WHERE email=:e"), {"e": email}
        )
    ).first()

    jwt_token = create_access_token({"sub": email, "id": gid, "name": name})
    if row:
        await db.execute(
            text("UPDATE users.accounts SET last_login=NOW() WHERE email=:e"),
            {"e": email},
        )
        await db.commit()
        return RedirectResponse(
            f"{FRONTEND_URL}/main?jwt_token={jwt_token}", status_code=307
        )
    else:
        return RedirectResponse(
            f"{FRONTEND_URL}/signup?jwt_token={jwt_token}", status_code=307
        )
