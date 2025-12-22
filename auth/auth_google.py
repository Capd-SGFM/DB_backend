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



import traceback
import datetime

def log_debug(msg):
    with open("debug_auth.log", "a") as f:
        f.write(f"[{datetime.datetime.now()}] {msg}\n")

@router.get("/callback")
async def google_callback(code: str, db: AsyncSession = Depends(get_async_db)):
    try:
        log_debug(f"Start callback with code={code[:10]}...")
        if not (GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET and GOOGLE_REDIRECT_URI):
            log_debug("Env missing")
            raise HTTPException(500, "Google OAuth env missing")

        async with httpx.AsyncClient(timeout=15.0) as client:
            log_debug("Requesting token...")
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
            log_debug(f"Token response status: {t.status_code}")
            tj = t.json()
            at = tj.get("access_token")
            if not at:
                log_debug(f"Token missing: {tj}")
                raise HTTPException(400, f"google token missing: {tj}")

            log_debug("Requesting user info...")
            u = await client.get(
                GOOGLE_USERINFO_URL, headers={"Authorization": f"Bearer {at}"}
            )
            ui = u.json()
            log_debug(f"User info received: {ui.get('email')}")

        if not ui.get("verified_email"):
            log_debug("Email not verified")
            raise HTTPException(400, "unverified email")

        email = ui.get("email")
        gid = ui.get("id") or ui.get("sub")
        name = ui.get("name")
        if not email or not gid:
            log_debug("Email or GID missing")
            raise HTTPException(500, "google email/id missing")

        # 기존 유저 여부(email 기준)
        log_debug("Checking DB...")
        row = (
            await db.execute(
                text("SELECT google_id FROM users.accounts WHERE email=:e"), {"e": email}
            )
        ).first()

        jwt_token = create_access_token({"sub": email, "id": gid, "name": name})
        if row:
            log_debug("User exists, updating login time")
            await db.execute(
                text("UPDATE users.accounts SET last_login=NOW() WHERE email=:e"),
                {"e": email},
            )
            await db.commit()
            return RedirectResponse(
                f"{FRONTEND_URL}/main?jwt_token={jwt_token}", status_code=307
            )
        else:
            log_debug("New user, redirecting to signup")
            return RedirectResponse(
                f"{FRONTEND_URL}/signup?jwt_token={jwt_token}", status_code=307
            )
    except Exception as e:
        err_msg = traceback.format_exc()
        log_debug(f"EXCEPTION: {err_msg}")
        return {"error": "Internal Server Error", "details": str(e)}
