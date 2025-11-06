from typing import Optional
from sqlalchemy import String, Boolean
from sqlalchemy.orm import Mapped, mapped_column
from .base import Base


class User(Base):
    __tablename__ = "accounts"
    __table_args__ = {"schema": "users"}

    google_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    username: Mapped[str] = mapped_column(String(12), nullable=False)
    email: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    email_opt_in: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )
    is_admin: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )

    def __repr__(self) -> str:
        return f"User(google_id={self.google_id!r}, email={self.email!r})"
