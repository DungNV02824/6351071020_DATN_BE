# services/dental_issues_service.py
"""
Pipeline kết hợp: FDI (nhận diện răng) + CV (phát hiện bệnh lý).

Các bước xử lý:
  1. Giải mã bytes ảnh tải lên.
  2. Chạy mô hình FDI → danh sách bounding box răng + số hiệu FDI.
  3. Chạy mô hình CV  → danh sách bounding box bệnh lý + loại bệnh.
  4. Ghép cặp bệnh lý vào đúng răng:
       - Ưu tiên: tâm của box bệnh lý nằm trong box răng.
       - Dự phòng : IoU giữa hai box > ngưỡng MIN_IOU.
  5. Trả về danh sách các răng có bệnh lý kèm chi tiết.
"""

import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Phân loại bệnh lý theo mức độ cần điều trị
#   NEEDS_TREATMENT : bệnh cần can thiệp tích cực (chưa được xử lý)
#   ALREADY_TREATED : tình trạng đã được điều trị (không cần can thiệp thêm)
# ---------------------------------------------------------------------------
NEEDS_TREATMENT_CLASSES: set = {
    "Caries",
    "Malaligned",
    "Missing teeth",
    "Periapical lesion",
    "Retained root",
    "Root Piece",
    "Impacted tooth",
    "Bone Loss",
    "Fracture teeth",
    "Supra Eruption",
    "Attrition",
    "Bone defect",
    "Cyst",
    "Root resorption",
}
ALREADY_TREATED_CLASSES: set = {
    "Crown",
    "Filling",
    "Implant",
    "Root Canal Treatment",
    "TAD",
    "Abutment",
    "Gingival former",
    "Metal band",
    "Orthodontic brackets",
    "Permanent retainer",
    "Post-core",
    "Plating",
    "Wire",
}

# ---------------------------------------------------------------------------
# Đường dẫn tới các file model
# ---------------------------------------------------------------------------
_BASE = Path(__file__).resolve().parent.parent / "Model"
FDI_MODEL_PATH = _BASE / "fdi.pt"
CV_MODEL_PATH  = _BASE / "CV2.pt"

# Ngưỡng IoU dự phòng khi tâm box bệnh lý không nằm trong bất kỳ box răng nào
MIN_IOU = 0.10

# ---------------------------------------------------------------------------
# Singleton cho hai model (load lười – chỉ load lần đầu khi được gọi)
# ---------------------------------------------------------------------------
_fdi_model = None
_cv_model  = None


def _get_fdi_model():
    """Load mô hình FDI một lần duy nhất và cache lại."""
    global _fdi_model
    if _fdi_model is None:
        if not FDI_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Không tìm thấy mô hình FDI tại '{FDI_MODEL_PATH}'."
            )
        from ultralytics import YOLO  # import chậm để khởi động nhanh
        logger.info("Đang tải mô hình FDI từ '%s' …", FDI_MODEL_PATH)
        _fdi_model = YOLO(str(FDI_MODEL_PATH))
    return _fdi_model


def _get_cv_model():
    """Load mô hình CV một lần duy nhất và cache lại."""
    global _cv_model
    if _cv_model is None:
        if not CV_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Không tìm thấy mô hình CV tại '{CV_MODEL_PATH}'."
            )
        from ultralytics import YOLO
        logger.info("Đang tải mô hình CV từ '%s' …", CV_MODEL_PATH)
        _cv_model = YOLO(str(CV_MODEL_PATH))
    return _cv_model


# ---------------------------------------------------------------------------
# Hàm tính toán hình học
# ---------------------------------------------------------------------------

def _center(box: List[float]) -> Tuple[float, float]:
    """Trả về tọa độ tâm (cx, cy) của bounding box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _point_in_box(px: float, py: float, box: List[float]) -> bool:
    """Kiểm tra xem điểm (px, py) có nằm trong box [x1,y1,x2,y2] không."""
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def _iou(box_a: List[float], box_b: List[float]) -> float:
    """
    Tính Intersection over Union (IoU) giữa hai bounding box.

    Args:
        box_a, box_b: [x1, y1, x2, y2]

    Returns:
        Giá trị IoU trong khoảng [0, 1].
    """
    # Tọa độ phần giao nhau
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h

    if inter_area == 0.0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


# ---------------------------------------------------------------------------
# Hàm ghép cặp bệnh lý → răng
# ---------------------------------------------------------------------------

def _map_issues_to_teeth(
    teeth: List[Dict[str, Any]],
    issues: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Ghép cặp từng box bệnh lý vào box răng phù hợp nhất.

    Chiến lược (theo thứ tự ưu tiên):
      1. Tâm của box bệnh lý nằm trong box răng → gán ngay.
      2. Nếu không khớp qua bước 1, tìm răng có IoU cao nhất và > MIN_IOU.
      3. Nếu vẫn không khớp → bệnh lý bị bỏ qua (không gán cho răng nào).

    Args:
        teeth : danh sách dict răng, mỗi phần tử có key 'tooth_bbox' [x1,y1,x2,y2].
        issues: danh sách dict bệnh lý, mỗi phần tử có key 'issue_bbox' [x1,y1,x2,y2].

    Returns:
        Danh sách các dict răng đã được gắn thêm key 'issues'.
    """
    # Khởi tạo danh sách issues rỗng cho mỗi răng
    for tooth in teeth:
        tooth["issues"] = []

    for issue in issues:
        i_bbox = issue["issue_bbox"]
        cx, cy = _center(i_bbox)

        matched_idx: Optional[int] = None

        # --- Bước 1: kiểm tra tâm ---
        for idx, tooth in enumerate(teeth):
            if _point_in_box(cx, cy, tooth["tooth_bbox"]):
                matched_idx = idx
                break

        # --- Bước 2: dự phòng IoU ---
        if matched_idx is None:
            best_iou = MIN_IOU
            for idx, tooth in enumerate(teeth):
                score = _iou(i_bbox, tooth["tooth_bbox"])
                if score > best_iou:
                    best_iou = score
                    matched_idx = idx

        # Gán bệnh lý cho răng tương ứng (nếu tìm được)
        if matched_idx is not None:
            teeth[matched_idx]["issues"].append(
                {
                    "issue_name":  issue["issue_name"],
                    "confidence":  issue["confidence"],
                    "issue_bbox":  issue["issue_bbox"],
                }
            )
        else:
            logger.debug(
                "Bệnh lý '%s' (bbox=%s) không khớp với bất kỳ răng nào – bỏ qua.",
                issue["issue_name"],
                i_bbox,
            )

    # Chỉ trả về các răng thực sự có bệnh lý
    return [t for t in teeth if t["issues"]]


# ---------------------------------------------------------------------------
# Hàm công khai chính
# ---------------------------------------------------------------------------

def predict_dental_issues(
    image_bytes: bytes,
    fdi_conf: float = 0.25,
    cv_conf: float = 0.30,
) -> Dict[str, Any]:
    """
    Phân tích ảnh X-quang răng: nhận diện răng + phát hiện bệnh lý + ghép cặp.

    Args:
        image_bytes: Dữ liệu bytes thô của ảnh tải lên (JPG / PNG / WebP).
        fdi_conf:    Ngưỡng confidence cho mô hình FDI (mặc định 0.25).
        cv_conf:     Ngưỡng confidence cho mô hình CV (mặc định 0.30).

    Returns:
        {
          "status": "success",
          "data": [
            {
              "tooth_number": "46",
              "tooth_bbox":   [x1, y1, x2, y2],
              "issues": [
                {
                  "issue_name":  "Cavity",
                  "confidence":  0.89,
                  "issue_bbox":  [x1, y1, x2, y2]
                }
              ]
            },
            ...
          ],
          "summary": {
            "total_teeth_detected": int,
            "teeth_with_issues":    int,
            "total_issues_found":   int,
          }
        }

    Raises:
        ValueError:       Nếu không giải mã được ảnh.
        FileNotFoundError: Nếu file model không tồn tại.
        RuntimeError:     Nếu có lỗi xảy ra trong quá trình inference.
    """
    # ------------------------------------------------------------------
    # 1. Giải mã ảnh
    # ------------------------------------------------------------------
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Không thể giải mã ảnh tải lên: {exc}") from exc

    logger.info(
        "Bắt đầu phân tích ảnh kích thước %dx%d px.",
        pil_image.width,
        pil_image.height,
    )

    # ------------------------------------------------------------------
    # 2. Chạy mô hình FDI → danh sách răng
    # ------------------------------------------------------------------
    fdi_model = _get_fdi_model()

    try:
        fdi_results = fdi_model.predict(
            source=pil_image,
            conf=fdi_conf,
            verbose=False,
        )
    except Exception as exc:
        raise RuntimeError(f"Lỗi inference mô hình FDI: {exc}") from exc

    teeth: List[Dict[str, Any]] = []
    for result in fdi_results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_idx   = int(box.cls.item())
            tooth_num = result.names.get(cls_idx, str(cls_idx))
            conf_val  = float(box.conf.item())

            teeth.append(
                {
                    "tooth_number": tooth_num,
                    "confidence":   round(conf_val, 4),
                    "tooth_bbox": [
                        round(x1, 2), round(y1, 2),
                        round(x2, 2), round(y2, 2),
                    ],
                }
            )

    logger.info("Mô hình FDI phát hiện %d răng.", len(teeth))

    # ------------------------------------------------------------------
    # 3. Chạy mô hình CV → danh sách bệnh lý
    # ------------------------------------------------------------------
    cv_model = _get_cv_model()

    try:
        cv_results = cv_model.predict(
            source=pil_image,
            conf=cv_conf,
            iou=0.45,
            verbose=False,
        )
    except Exception as exc:
        raise RuntimeError(f"Lỗi inference mô hình CV: {exc}") from exc

    issues: List[Dict[str, Any]] = []
    for result in cv_results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_idx    = int(box.cls.item())
            issue_name = result.names.get(cls_idx, str(cls_idx))
            conf_val   = float(box.conf.item())

            issues.append(
                {
                    "issue_name": issue_name,
                    "confidence": round(conf_val, 4),
                    "issue_bbox": [
                        round(x1, 2), round(y1, 2),
                        round(x2, 2), round(y2, 2),
                    ],
                }
            )

    logger.info("Mô hình CV phát hiện %d bệnh lý.", len(issues))

    # ------------------------------------------------------------------
    # 4. Ghép cặp bệnh lý → răng
    # ------------------------------------------------------------------
    matched_teeth = _map_issues_to_teeth(teeth, issues)

    # Sắp xếp kết quả theo số hiệu răng (FDI) để dễ đọc
    matched_teeth.sort(key=lambda t: t["tooth_number"])

    # ------------------------------------------------------------------
    # 5. Xây dựng response
    # ------------------------------------------------------------------
    # Loại bỏ trường confidence nội bộ của răng khỏi output cuối
    output_data = [
        {
            "tooth_number": t["tooth_number"],
            "tooth_bbox":   t["tooth_bbox"],
            "issues":       t["issues"],
        }
        for t in matched_teeth
    ]

    total_issues = sum(len(t["issues"]) for t in matched_teeth)

    return {
        "status": "success",
        "data": output_data,
        "summary": {
            "total_teeth_detected": len(teeth),
            "teeth_with_issues":    len(matched_teeth),
            "total_issues_found":   total_issues,
        },
    }


# ---------------------------------------------------------------------------
# Hàm công khai: thống kê tình trạng răng
# ---------------------------------------------------------------------------

def _run_both_models(
    pil_image,
    fdi_conf: float,
    cv_conf: float,
) -> tuple:
    """
    Hàm nội bộ dùng chung: chạy cả hai mô hình và trả về (teeth_all, issues).
    Tách riêng để `predict_dental_stats` không phải duplicate code.
    """
    # --- FDI ---
    fdi_model = _get_fdi_model()
    try:
        fdi_results = fdi_model.predict(source=pil_image, conf=fdi_conf, verbose=False)
    except Exception as exc:
        raise RuntimeError(f"Lỗi inference mô hình FDI: {exc}") from exc

    teeth_all: List[Dict[str, Any]] = []
    for result in fdi_results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_idx   = int(box.cls.item())
            tooth_num = result.names.get(cls_idx, str(cls_idx))
            conf_val  = float(box.conf.item())
            teeth_all.append(
                {
                    "tooth_number": tooth_num,
                    "confidence":   round(conf_val, 4),
                    "tooth_bbox":   [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                }
            )

    # --- CV ---
    cv_model = _get_cv_model()
    try:
        cv_results = cv_model.predict(source=pil_image, conf=cv_conf, iou=0.45, verbose=False)
    except Exception as exc:
        raise RuntimeError(f"Lỗi inference mô hình CV: {exc}") from exc

    issues: List[Dict[str, Any]] = []
    for result in cv_results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_idx    = int(box.cls.item())
            issue_name = result.names.get(cls_idx, str(cls_idx))
            conf_val   = float(box.conf.item())
            issues.append(
                {
                    "issue_name": issue_name,
                    "confidence": round(conf_val, 4),
                    "issue_bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                }
            )

    return teeth_all, issues


def predict_dental_stats(
    image_bytes: bytes,
    fdi_conf: float = 0.25,
    cv_conf: float = 0.30,
) -> Dict[str, Any]:
    """
    Thống kê tổng quan tình trạng sức khỏe răng miệng.

    Phân loại từng răng:
      - **Khỏe mạnh**   : FDI phát hiện được nhưng CV không tìm thấy bệnh lý nào.
      - **Có vấn đề**   : có ít nhất một bệnh lý (Cavity / Fillings / Impacted Tooth / Implant).
      - **Cần điều trị**: có ít nhất một bệnh lý thuộc nhóm cần can thiệp tích cực
                          (Cavity, Impacted Tooth). Fillings và Implant là đã được xử lý
                          nên KHÔNG tính vào nhóm này.

    Returns:
        {
          "status": "success",
          "statistics": {
            "total_teeth":          int,
            "healthy_teeth":        int,
            "teeth_with_issues":    int,
            "teeth_need_treatment": int,
          },
          "data": {
            "healthy": [
              { "tooth_number": "11", "tooth_bbox": [...] },
              ...
            ],
            "with_issues": [
              {
                "tooth_number":    "46",
                "tooth_bbox":      [...],
                "needs_treatment": true,
                "issues": [
                  { "issue_name": "Cavity", "confidence": 0.89, "issue_bbox": [...] }
                ]
              },
              ...
            ]
          }
        }
    """
    # ------------------------------------------------------------------
    # 1. Giải mã ảnh
    # ------------------------------------------------------------------
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Không thể giải mã ảnh tải lên: {exc}") from exc

    logger.info(
        "Bắt đầu thống kê tình trạng răng – ảnh %dx%d px.",
        pil_image.width, pil_image.height,
    )

    # ------------------------------------------------------------------
    # 2. Chạy cả hai mô hình
    # ------------------------------------------------------------------
    teeth_all, issues = _run_both_models(pil_image, fdi_conf, cv_conf)

    logger.info(
        "FDI: %d răng | CV: %d bệnh lý.", len(teeth_all), len(issues)
    )

    # ------------------------------------------------------------------
    # 3. Ghép cặp bệnh lý → răng (tất cả răng, kể cả răng khỏe mạnh)
    # ------------------------------------------------------------------
    # Khởi tạo danh sách issues rỗng cho mỗi răng
    for tooth in teeth_all:
        tooth["issues"] = []

    for issue in issues:
        i_bbox = issue["issue_bbox"]
        cx, cy = _center(i_bbox)
        matched_idx: Optional[int] = None

        # Bước 1: tâm box bệnh lý nằm trong box răng
        for idx, tooth in enumerate(teeth_all):
            if _point_in_box(cx, cy, tooth["tooth_bbox"]):
                matched_idx = idx
                break

        # Bước 2: dự phòng IoU
        if matched_idx is None:
            best_iou = MIN_IOU
            for idx, tooth in enumerate(teeth_all):
                score = _iou(i_bbox, tooth["tooth_bbox"])
                if score > best_iou:
                    best_iou = score
                    matched_idx = idx

        if matched_idx is not None:
            teeth_all[matched_idx]["issues"].append(
                {
                    "issue_name": issue["issue_name"],
                    "confidence": issue["confidence"],
                    "issue_bbox": issue["issue_bbox"],
                }
            )

    # ------------------------------------------------------------------
    # 4. Phân loại từng răng
    # ------------------------------------------------------------------
    healthy_teeth:        List[Dict[str, Any]] = []
    teeth_with_issues:    List[Dict[str, Any]] = []

    for tooth in teeth_all:
        if not tooth["issues"]:
            # Không có bệnh lý nào → khỏe mạnh
            healthy_teeth.append(
                {
                    "tooth_number": tooth["tooth_number"],
                    "tooth_bbox":   tooth["tooth_bbox"],
                }
            )
        else:
            # Kiểm tra xem có bệnh nào thuộc nhóm CẦN điều trị không
            needs_treatment = any(
                iss["issue_name"] in NEEDS_TREATMENT_CLASSES
                for iss in tooth["issues"]
            )
            teeth_with_issues.append(
                {
                    "tooth_number":    tooth["tooth_number"],
                    "tooth_bbox":      tooth["tooth_bbox"],
                    "needs_treatment": needs_treatment,
                    "issues":          tooth["issues"],
                }
            )

    # Sắp xếp theo số hiệu răng để dễ đọc
    healthy_teeth.sort(key=lambda t: t["tooth_number"])
    teeth_with_issues.sort(key=lambda t: t["tooth_number"])

    # Đếm số răng cần điều trị
    need_treatment_count = sum(
        1 for t in teeth_with_issues if t["needs_treatment"]
    )

    # ------------------------------------------------------------------
    # 5. Xây dựng response
    # ------------------------------------------------------------------
    return {
        "status": "success",
        "statistics": {
            "total_teeth":          len(teeth_all),
            "healthy_teeth":        len(healthy_teeth),
            "teeth_with_issues":    len(teeth_with_issues),
            "teeth_need_treatment": need_treatment_count,
        },
        "data": {
            "healthy":     healthy_teeth,
            "with_issues": teeth_with_issues,
        },
    }
