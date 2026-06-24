import secrets
from datetime import datetime, timedelta

import bcrypt
from fastapi import APIRouter, HTTPException, status, Depends, BackgroundTasks # <-- Thêm BackgroundTasks
from jose import jwt
from sqlalchemy.orm import Session
# --- Thêm các thư viện gửi mail ---
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig, MessageType
from pydantic import EmailStr
# ----------------------------------

from core.config import JWT_SECRET_KEY, JWT_ALGORITHM, JWT_ACCESS_TOKEN_EXPIRE_MINUTES
from db.session import SessionLocal
from dto.auth_dto import (
    ForgotPasswordRequest,
    LoginRequest,
    LoginResponse,
    RegisterRequest,
    ResetPasswordRequest,
    AdminAccountResponse,
)
from models.admin_account import AdminAccount

router = APIRouter(prefix="/auth", tags=["Authentication"])

RESET_TOKEN_EXPIRE_MINUTES = 30

# ──────────────────────────────────────────────
# CẤU HÌNH GỬI EMAIL (Nên đưa vào file .env)
# ──────────────────────────────────────────────
conf = ConnectionConfig(
    MAIL_USERNAME="nguyenvandung30271@gmail.com",      
    MAIL_PASSWORD="gbtr naps wfxp wnks",     
    MAIL_FROM="nguyenvandung30271@gmail.com",          
    MAIL_PORT=587,
    MAIL_SERVER="smtp.gmail.com",
    MAIL_STARTTLS=True,
    MAIL_SSL_TLS=False,
    USE_CREDENTIALS=True,
    VALIDATE_CERTS=True
)

# # Hàm gửi mail chạy ngầm
# async def send_reset_password_email(email_to: str, token: str):
#     # Đường dẫn frontend để user bấm vào (thay thế bằng domain thực tế của bạn)
#     reset_link = f"http://localhost:5173/reset-password?token={token}"
    
#     html_content = f"""
#     <h3>Yêu cầu đặt lại mật khẩu OMEWA</h3>
#     <p>Bạn đã yêu cầu đặt lại mật khẩu. Vui lòng click vào link bên dưới để đặt mật khẩu mới:</p>
#     <a href="{reset_link}" style="padding: 10px 20px; background-color: #10b981; color: white; text-decoration: none; border-radius: 5px;">Đặt lại mật khẩu</a>
#     <p>Link này sẽ hết hạn sau {RESET_TOKEN_EXPIRE_MINUTES} phút.</p>
#     <p>Nếu bạn không yêu cầu, vui lòng bỏ qua email này.</p>
#     """

#     message = MessageSchema(
#         subject="[OMEWA] Đặt lại mật khẩu hệ thống",
#         recipients=[email_to], # List các email nhận
#         body=html_content,
#         subtype=MessageType.html
#     )

#     fm = FastMail(conf)
#     await fm.send_message(message)

async def send_reset_password_email(email_to: str, token: str):
    # 1. Đường dẫn Front-end: Nơi người dùng sẽ nhập mật khẩu mới
    # Link này sẽ dẫn tới trang giao diện, sau đó trang này mới gọi API POST /auth/reset-password trong ảnh
    # frontend_url = "http://localhost:5173" # Thay đổi theo domain thực tế của bạn
    frontend_url = "http://localhost:3000" 
    reset_link = f"{frontend_url}/auth/reset-password?token={token}" 
    
    # 2. Nội dung Email (Format lại cho đẹp và chuyên nghiệp hơn)
    html_content = f"""
    <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; border: 1px solid #e5e7eb; padding: 20px; border-radius: 8px;">
        <h2 style="color: #10b981; text-align: center;">Yêu cầu đặt lại mật khẩu Chatbot OMEWA</h2>
        <p>Chào bạn,</p>
        <p>Chúng tôi nhận được yêu cầu đặt lại mật khẩu cho tài khoản của bạn. Vui lòng click vào nút bên dưới để tiến hành thiết lập mật khẩu mới:</p>
        
        <div style="text-align: center; margin: 30px 0;">
            <a href="{reset_link}" 
               style="padding: 12px 25px; background-color: #10b981; color: white; text-decoration: none; border-radius: 5px; font-weight: bold; display: inline-block;">
               Đặt lại mật khẩu ngay
            </a>
        </div>
        
        <p style="color: #6b7280; font-size: 0.9em;">Link này sẽ hết hạn sau <strong>{RESET_TOKEN_EXPIRE_MINUTES}</strong> phút.</p>
        <hr style="border: 0; border-top: 1px solid #eee; margin: 20px 0;">
        <p style="font-size: 0.8em; color: #9ca3af;">Nếu bạn không yêu cầu thay đổi này, vui lòng bỏ qua email này. Tài khoản của bạn vẫn được an toàn.</p>
    </div>
    """

    message = MessageSchema(
        subject="[OMEWA] Xác nhận đặt lại mật khẩu",
        recipients=[email_to],
        body=html_content,
        subtype=MessageType.html
    )

    fm = FastMail(conf) # Đảm bảo 'conf' đã được định nghĩa đúng với tài khoản SMTP của bạn
    await fm.send_message(message)

# ──────────────────────────────────────────────
# Helpers nội bộ
# ──────────────────────────────────────────────

def _get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


def _create_access_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@router.post(
    "/register",
    response_model=AdminAccountResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Đăng ký tài khoản",
)
def register(payload: RegisterRequest, db: Session = Depends(_get_db)):
    """
    Tạo tài khoản mới. Email phải là duy nhất. Password tối thiểu 8 ký tự.
    """
    email = payload.email.lower().strip()

    existing = db.query(AdminAccount).filter(AdminAccount.email == email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email này đã được đăng ký",
        )

    account = AdminAccount(
        email=email,
        hashed_password=_hash_password(payload.password),
        full_name=payload.full_name,
        is_active=True,
    )
    db.add(account)
    db.commit()
    db.refresh(account)
    return account


@router.post(
    "/login",
    response_model=LoginResponse,
    summary="Đăng nhập",
)
def login(payload: LoginRequest, db: Session = Depends(_get_db)):
    """
    Xác thực email + password. Trả về JWT access token (mặc định hết hạn sau 24 giờ).
    """
    email = payload.email.lower().strip()

    account = db.query(AdminAccount).filter(AdminAccount.email == email).first()

    if not account or not _verify_password(payload.password, account.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email hoặc mật khẩu không đúng",
        )

    if not account.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tài khoản đã bị vô hiệu hóa",
        )

    token = _create_access_token({"sub": str(account.id), "email": account.email})

    return LoginResponse(
        access_token=token,
        token_type="bearer",
        admin_id=account.id,
        email=account.email,
        full_name=account.full_name,
    )

@router.post(
    "/forgot-password",
    summary="Quên mật khẩu – yêu cầu đặt lại",
)
def forgot_password(
    payload: ForgotPasswordRequest, 
    background_tasks: BackgroundTasks, # <-- Inject BackgroundTasks vào đây
    db: Session = Depends(_get_db)
):
    """
    Tạo reset-token hết hạn sau 30 phút và gửi qua email.
    """
    email = payload.email.lower().strip()

    account = db.query(AdminAccount).filter(AdminAccount.email == email).first()

    if not account or not account.is_active:
        return {"message": "Nếu email tồn tại, hướng dẫn đặt lại mật khẩu đã được gửi"}

    reset_token = secrets.token_urlsafe(32)
    account.reset_token = reset_token
    account.reset_token_expires_at = datetime.utcnow() + timedelta(minutes=RESET_TOKEN_EXPIRE_MINUTES)
    db.commit()

    # THÊM TASK GỬI MAIL VÀO BACKGROUND (Chạy ngầm)
    background_tasks.add_task(send_reset_password_email, account.email, reset_token)

    return {
        "message": "Nếu email tồn tại, hướng dẫn đặt lại mật khẩu đã được gửi vào hòm thư của bạn.",
        # Đã xóa trường "reset_token" ở đây vì nó đã được gửi vào mail bảo mật.
    }


# @router.post(
#     "/forgot-password",
#     summary="Quên mật khẩu – yêu cầu đặt lại",
# )
# def forgot_password(payload: ForgotPasswordRequest, db: Session = Depends(_get_db)):
#     """
#     Tạo reset-token hết hạn sau 30 phút.

#     > **Production**: gửi token qua email. Hiện tại trả về trong response để kiểm thử.
#     """
#     email = payload.email.lower().strip()

#     account = db.query(AdminAccount).filter(AdminAccount.email == email).first()

#     # Luôn trả về thành công để tránh lộ email tồn tại
#     if not account or not account.is_active:
#         return {"message": "Nếu email tồn tại, hướng dẫn đặt lại mật khẩu đã được gửi"}

#     reset_token = secrets.token_urlsafe(32)
#     account.reset_token = reset_token
#     account.reset_token_expires_at = datetime.utcnow() + timedelta(minutes=RESET_TOKEN_EXPIRE_MINUTES)
#     db.commit()

#     return {
#         "message": "Nếu email tồn tại, hướng dẫn đặt lại mật khẩu đã được gửi",
#         # TODO production: gửi qua email, xóa trường này
#         "reset_token": reset_token,
#         "expires_in_minutes": RESET_TOKEN_EXPIRE_MINUTES,
#     }


@router.post(
    "/reset-password",
    summary="Đặt lại mật khẩu bằng reset-token",
)
def reset_password(payload: ResetPasswordRequest, db: Session = Depends(_get_db)):
    """
    Đặt mật khẩu mới bằng token từ `/auth/forgot-password`.
    Token chỉ dùng được **một lần** và hết hạn sau 30 phút.
    """
    account = db.query(AdminAccount).filter(
        AdminAccount.reset_token == payload.token
    ).first()

    if not account:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Token không hợp lệ",
        )

    if account.reset_token_expires_at is None or datetime.utcnow() > account.reset_token_expires_at:
        account.reset_token = None
        account.reset_token_expires_at = None
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Token đã hết hạn, vui lòng yêu cầu đặt lại mật khẩu lần nữa",
        )

    account.hashed_password = _hash_password(payload.new_password)
    account.reset_token = None
    account.reset_token_expires_at = None
    db.commit()

    return {"message": "Mật khẩu đã được đặt lại thành công"}


