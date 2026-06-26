import logging
import os

from dotenv import load_dotenv
load_dotenv()  # Load .env before any os.getenv() calls

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.lrc_router import router as predict_router
from routers.fdi_router import router as fdi_router

from routers.cv_router import router as dental_cv_router
from routers.dental_issues_router import router as dental_issues_router
from routers.diagnosis_router import router as diagnosis_router
from routers.auth import router as auth
from routers.soft_tissue_router import router as soft_tissue_router
# ---------------------------------------------------------------------------
# Logging – structured, INFO level by default; override via LOG_LEVEL env var
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Dental AI API",
    description=(
        "APIs for dental image analysis:\n"
        "- **FDI** tooth detection\n"
        "- **LRC** cephalometric landmark detection\n"
        "- **Cephalometric analysis** (angles + classification + AI interpretation)\n"
        "- **Dental CV** pathology detection (caries, pulpitis, bone loss, lesion)"
    ),
    version="1.0.0",
)

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # Hoặc dùng ["*"] để cho phép tất cả (không khuyến khích khi sản xuất)
    allow_credentials=True,
    allow_methods=["*"], # Cho phép tất cả các phương thức GET, POST, PUT...
    allow_headers=["*"], # Cho phép tất cả các headers
)

app.include_router(fdi_router)
app.include_router(predict_router)
app.include_router(dental_cv_router)
app.include_router(dental_issues_router)
app.include_router(diagnosis_router)
app.include_router(auth)
app.include_router(soft_tissue_router)

@app.get("/", tags=["Health"])
def root():
    return {"message": "Dental AI API is running. Visit /docs for the Swagger UI."}

