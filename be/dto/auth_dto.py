from pydantic import BaseModel, field_validator
from typing import Optional


class RegisterRequest(BaseModel):
    """Payload đăng ký tài khoản"""
    email: str
    password: str
    full_name: Optional[str] = None

    @field_validator("password")
    @classmethod
    def password_min_length(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Mật khẩu phải có ít nhất 8 ký tự")
        return v


class LoginRequest(BaseModel):
    """Payload đăng nhập"""
    email: str
    password: str


class LoginResponse(BaseModel):
    """Kết quả trả về sau khi đăng nhập thành công"""
    access_token: str
    token_type: str = "bearer"
    admin_id: int
    email: str
    full_name: Optional[str]


class ForgotPasswordRequest(BaseModel):
    """Yêu cầu đặt lại mật khẩu – nhập email"""
    email: str


class ResetPasswordRequest(BaseModel):
    """Đặt lại mật khẩu bằng token nhận được"""
    token: str
    new_password: str

    @field_validator("new_password")
    @classmethod
    def password_min_length(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Mật khẩu mới phải có ít nhất 8 ký tự")
        return v


class AdminAccountResponse(BaseModel):
    """Thông tin tài khoản (không chứa password)"""
    id: int
    email: str
    full_name: Optional[str]
    is_active: bool

    class Config:
        from_attributes = True


