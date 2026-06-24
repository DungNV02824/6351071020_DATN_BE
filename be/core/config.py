import os
from urllib.parse import quote_plus
from dotenv import load_dotenv

load_dotenv()

# JWT Authentication
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-this-secret-in-production")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", 1440))  # 24 giờ



DATABASE_URL = os.getenv("DATABASE_URL")