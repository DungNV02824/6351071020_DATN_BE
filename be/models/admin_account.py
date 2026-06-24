from sqlalchemy import Column, Integer, String, DateTime, Boolean
from sqlalchemy.sql import func
from db.base import Base

# Bảnh admin
class AdminAccount(Base):
    """
    Bảng tài khoản để đăng nhập vào hệ thống quản trị.
    Độc lập, KHÔNG có RLS – không cần tenant_id khi đăng nhập.
    """
    __tablename__ = "admin_accounts"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)

    # Dùng cho chức năng quên mật khẩu
    reset_token = Column(String(255), nullable=True, index=True)
    reset_token_expires_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
