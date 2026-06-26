# Dental AI API

Đây là backend FastAPI cho hệ thống phân tích nha khoa sử dụng các mô hình AI để nhận diện và phân tích dữ liệu nha khoa.

## Công nghệ sử dụng
- Python
- FastAPI
- Uvicorn
- SQLAlchemy
- JWT Authentication
- OpenCV / NumPy / SciPy

## Yêu cầu hệ thống
- Python 3.9+ (khuyến nghị 3.10 hoặc 3.11)
- pip
- Virtual environment (khuyến nghị)

## 1. Clone project và vào thư mục

```bash
git clone <repo-url>
cd be
```

## 2. Tạo môi trường ảo (Windows)

```bash
python -m venv .venv
.venv\Scripts\activate
```

Nếu bạn dùng PowerShell, có thể dùng:

```powershell
.\.venv\Scripts\Activate.ps1
```

## 3. Cài đặt dependency

```bash
pip install -r requirements.txt
```

## 4. Tạo file cấu hình môi trường

Tạo file `.env` ở thư mục gốc dự án với nội dung ví dụ sau:

```env
JWT_SECRET_KEY=your-secret-key
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=1440
DATABASE_URL=sqlite:///./app.db
LOG_LEVEL=INFO
```

> Nếu bạn có database PostgreSQL, hãy thay `DATABASE_URL` bằng connection string phù hợp.

## 5. Chạy ứng dụng

Chạy server bằng lệnh sau:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Sau khi chạy thành công, mở trình duyệt tại:
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

## 6. Kiểm tra health

Truy cập endpoint sau để kiểm tra server:

```bash
curl http://127.0.0.1:8000/
```

Kết quả mong đợi:

```json
{"message":"Dental AI API is running. Visit /docs for the Swagger UI."}
```

## 7. Cấu trúc thư mục chính

- `main.py`: entry point của FastAPI
- `routers/`: định nghĩa API routes
- `services/`: logic nghiệp vụ
- `models/`: các file mô hình AI và trọng số
- `db/`: cấu hình database
- `schemas/`: schema dữ liệu
- `tests/`: test cases

## 8. Ghi chú quan trọng

- Một số API có thể phụ thuộc vào các file trọng số mô hình trong thư mục `Model/`.
- Nếu bạn gặp lỗi liên quan đến thư viện `opencv`, `numpy` hoặc `scipy`, hãy đảm bảo đã cài đặt đúng dependency từ `requirements.txt`.
- Nếu API cần xác thực bằng JWT, hãy chắc chắn đã cung cấp `JWT_SECRET_KEY` hợp lệ.

## 9. Chạy test (nếu có)

```bash
pytest
```

Nếu bạn muốn, mình có thể viết thêm một phiên bản README chuyên nghiệp hơn với phần cấu hình Docker và cách deploy lên server. 
