# services/diagnosis_service.py
"""
Orchestration service cho Full Dental Report.

Pipeline:
  1. Panoramic  → FDI + CV models  → ghép cặp răng / bệnh lý
  2. Cephalometric → HRNet model   → 29 landmarks → tính góc SNA/SNB/ANB
  3. OpenAI GPT-4o                 → tư vấn sức khỏe răng miệng (JSON)
  4. Tổng hợp → FullReportResponse

Kỹ thuật:
  - Model inference (CPU-bound) chạy qua asyncio.to_thread để không block event-loop.
  - OpenAI sử dụng AsyncOpenAI (openai >= 1.0).
  - Xử lý lỗi từng bước riêng biệt: nếu bước nào lỗi, phần còn lại vẫn trả về.
"""

import asyncio
import json
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

from schemas.diagnosis_schema import (
    CephAiAnalysis,
    CephAnalysis,
    CephMetrics,
    Consultation,
    ConsultationIssue,
    FullReportResponse,
    IssueItem,
    LandmarkPoint,
    ReportSummary,
    ToothAnalysis,
    ToothDetail,
)
from services.dental_issues_service import predict_dental_issues
from services.lrc_services import process_and_predict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cấu hình chỉ số landmark trong mảng 29 điểm của mô hình LRC
# (điều chỉnh nếu thứ tự train khác)
# ---------------------------------------------------------------------------
LM_NAMES: List[str] = [
    "Me", "Xi", "A",  "B",  "Or", "ANS", "PNS", "Po", "Pog", "Gn",
    "S",  "Go", "Ar", "Ba", "Pt", "CF",  "UIE", "LIE", "UIA",
    "LIA", "UL", "LL", "Stms", "Stmi", "Pg", "Dt", "N", "Pm", "Na",
]

LM_S   = 10  # Sella
LM_N   = 26  # Nasion
LM_Or  = 4   # Orbitale
LM_Po  = 7   # Porion
LM_A   = 2   # Point A (Subspinale)
LM_B   = 3   # Point B (Supramentale)
LM_Gn  = 9   # Gnathion
LM_Me  = 0   # Menton
LM_Go  = 11  # Gonion
LM_UIE = 16  # Upper Incisor Edge
LM_LIE = 17  # Lower Incisor Edge
LM_UIA = 18  # Upper Incisor Apex
LM_LIA = 19  # Lower Incisor Apex

# ---------------------------------------------------------------------------
# Metadata bệnh lý → thông tin điều trị
# ---------------------------------------------------------------------------
NEEDS_TREATMENT_CLASSES = {
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
ALREADY_TREATED_CLASSES = {
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

_DISEASE_META: Dict[str, Dict[str, Any]] = {
    "Caries": {
        "disease_name": "Sâu răng",
        "latin_name": "Caries dentis",
        "treatment_method": "Nạo bỏ mô sâu, trám composite hoặc GIC, bọc sứ nếu tổn thương lớn",
        "estimated_duration": "1–2 buổi (30–60 phút/buổi)",
        "severity_percent": 65,
    },
    "Impacted tooth": {
        "disease_name": "Răng mọc ngầm / lệch",
        "latin_name": "Dens impactus",
        "treatment_method": "Tiểu phẫu nhổ răng ngầm, phẫu thuật nếu cần thiết",
        "estimated_duration": "1 buổi (45–90 phút)",
        "severity_percent": 70,
    },
    "Malaligned": {
        "disease_name": "Răng lệch lạc",
        "latin_name": "Malocclusion",
        "treatment_method": "Chỉnh nha bằng mắc cài hoặc máng trong suốt",
        "estimated_duration": "12–24 tháng",
        "severity_percent": 55,
    },
    "Missing teeth": {
        "disease_name": "Mất răng",
        "latin_name": "Edentulism",
        "treatment_method": "Cấy ghép implant, cầu răng sứ hoặc hàm tháo lắp",
        "estimated_duration": "3–6 tháng (implant)",
        "severity_percent": 75,
    },
    "Periapical lesion": {
        "disease_name": "Tổn thương quanh chóp",
        "latin_name": "Lesio periapicalis",
        "treatment_method": "Điều trị tủy, phẫu thuật cắt chóp nếu cần",
        "estimated_duration": "2–4 buổi (45–60 phút/buổi)",
        "severity_percent": 80,
    },
    "Retained root": {
        "disease_name": "Chân răng còn sót",
        "latin_name": "Radix relicta",
        "treatment_method": "Nhổ bỏ chân răng sót, vệ sinh ổ xương",
        "estimated_duration": "1 buổi (30–45 phút)",
        "severity_percent": 60,
    },
    "Root Piece": {
        "disease_name": "Mảnh chân răng",
        "latin_name": "Fragmentum radicis",
        "treatment_method": "Lấy mảnh chân răng qua tiểu phẫu",
        "estimated_duration": "1 buổi (30–60 phút)",
        "severity_percent": 60,
    },
    "Bone Loss": {
        "disease_name": "Tiêu xương ổ răng",
        "latin_name": "Resorptio ossis alveolaris",
        "treatment_method": "Điều trị nha chu, ghép xương nếu mất xương nặng",
        "estimated_duration": "3–6 tháng",
        "severity_percent": 85,
    },
    "Fracture teeth": {
        "disease_name": "Răng gãy vỡ",
        "latin_name": "Fractura dentis",
        "treatment_method": "Trám phục hồi, bọc mão răng hoặc nhổ nếu không phục hồi được",
        "estimated_duration": "1–2 buổi (30–60 phút/buổi)",
        "severity_percent": 75,
    },
    "Supra Eruption": {
        "disease_name": "Răng trồi",
        "latin_name": "Supra-eruptio dentis",
        "treatment_method": "Mài điều chỉnh hoặc chỉnh nha, bổ sung răng đối diện",
        "estimated_duration": "Tùy mức độ (1 buổi – vài tháng)",
        "severity_percent": 50,
    },
    "Attrition": {
        "disease_name": "Mòn răng",
        "latin_name": "Attritio dentis",
        "treatment_method": "Trám phục hồi bề mặt, máng bảo vệ răng, điều trị nguyên nhân",
        "estimated_duration": "1–2 buổi",
        "severity_percent": 55,
    },
    "Bone defect": {
        "disease_name": "Khuyết hổng xương",
        "latin_name": "Defectus ossis",
        "treatment_method": "Ghép xương tái sinh, màng collagen, tái tạo xương có hướng dẫn",
        "estimated_duration": "4–6 tháng",
        "severity_percent": 80,
    },
    "Cyst": {
        "disease_name": "Nang xương hàm",
        "latin_name": "Cystis maxillaris",
        "treatment_method": "Phẫu thuật cắt bỏ nang, nạo sạch vùng bệnh",
        "estimated_duration": "1 buổi (60–120 phút)",
        "severity_percent": 85,
    },
    "Root resorption": {
        "disease_name": "Tiêu chân răng",
        "latin_name": "Resorptio radicis",
        "treatment_method": "Theo dõi, điều trị tủy hoặc nhổ răng nếu tiêu chân nặng",
        "estimated_duration": "Tùy mức độ",
        "severity_percent": 70,
    },
    # ------------------------------------------------------------------
    # ALREADY_TREATED – đã được can thiệp / đang điều trị
    # ------------------------------------------------------------------
    "Crown": {
        "disease_name": "Mão răng sứ",
        "latin_name": "Corona dentis artificialis",
        "treatment_method": "Theo dõi định kỳ; thay mão khi mòn, hở hoặc vỡ",
        "estimated_duration": "Đã được điều trị",
        "severity_percent": 20,
    },
    "Filling": {
        "disease_name": "Trám răng",
        "latin_name": "Restauratio dentis",
        "treatment_method": "Theo dõi định kỳ; thay trám khi bong, hở hoặc mòn",
        "estimated_duration": "Đã được điều trị",
        "severity_percent": 15,
    },
    "Implant": {
        "disease_name": "Cấy ghép Implant",
        "latin_name": "Implantum dentale",
        "treatment_method": "Vệ sinh kỹ vùng implant; kiểm tra định kỳ 6 tháng/lần",
        "estimated_duration": "Đã được điều trị",
        "severity_percent": 10,
    },
    "Root Canal Treatment": {
        "disease_name": "Điều trị tủy",
        "latin_name": "Therapia endodontica",
        "treatment_method": "Theo dõi định kỳ; kiểm tra tình trạng chóp răng hàng năm",
        "estimated_duration": "Đã được điều trị",
        "severity_percent": 20,
    },
    "TAD": {
        "disease_name": "Neo chỉnh nha (TAD)",
        "latin_name": "Anchoratio orthodontica temporaria",
        "treatment_method": "Theo dõi quá trình chỉnh nha; tháo bỏ sau khi hoàn tất",
        "estimated_duration": "Đang điều trị chỉnh nha",
        "severity_percent": 10,
    },
    "Abutment": {
        "disease_name": "Trụ implant (Abutment)",
        "latin_name": "Abutmentum implanti",
        "treatment_method": "Kiểm tra định kỳ; vệ sinh kỹ vùng trụ và đường viền nướu",
        "estimated_duration": "Đã được điều trị",
        "severity_percent": 10,
    },
    "Gingival former": {
        "disease_name": "Đầu lành thương nướu",
        "latin_name": "Formator gingivae",
        "treatment_method": "Theo dõi lành thương nướu; chuẩn bị gắn phục hình cuối cùng",
        "estimated_duration": "Đang trong quá trình điều trị",
        "severity_percent": 15,
    },
    "Metal band": {
        "disease_name": "Band kim loại chỉnh nha",
        "latin_name": "Banda metallica orthodontica",
        "treatment_method": "Đang trong quá trình chỉnh nha; theo dõi định kỳ, vệ sinh kỹ",
        "estimated_duration": "Đang điều trị chỉnh nha",
        "severity_percent": 10,
    },
    "Orthodontic brackets": {
        "disease_name": "Mắc cài chỉnh nha",
        "latin_name": "Brachia orthodontica",
        "treatment_method": "Đang trong quá trình chỉnh nha; vệ sinh kỹ vùng mắc cài sau mỗi bữa ăn",
        "estimated_duration": "Đang điều trị chỉnh nha",
        "severity_percent": 10,
    },
    "Permanent retainer": {
        "disease_name": "Hàm duy trì cố định",
        "latin_name": "Retentio permanens",
        "treatment_method": "Vệ sinh kỹ vùng dây duy trì bằng chỉ nha khoa; kiểm tra định kỳ",
        "estimated_duration": "Duy trì sau chỉnh nha",
        "severity_percent": 10,
    },
    "Post-core": {
        "disease_name": "Chốt lõi phục hình",
        "latin_name": "Post et core",
        "treatment_method": "Theo dõi định kỳ; kiểm tra độ bền của mão và chốt",
        "estimated_duration": "Đã được điều trị",
        "severity_percent": 20,
    },
    "Plating": {
        "disease_name": "Nẹp vít xương",
        "latin_name": "Osteosynthesis",
        "treatment_method": "Theo dõi lành thương xương; tháo nẹp vít khi cần thiết",
        "estimated_duration": "Đã được điều trị",
        "severity_percent": 15,
    },
    "Wire": {
        "disease_name": "Dây cung chỉnh nha",
        "latin_name": "Arcus orthodonticus",
        "treatment_method": "Đang trong quá trình chỉnh nha; điều chỉnh định kỳ theo lịch hẹn",
        "estimated_duration": "Đang điều trị chỉnh nha",
        "severity_percent": 10,
    },
    # ------------------------------------------------------------------
    # INFO – cấu trúc giải phẫu bình thường (không phải bệnh lý)
    # ------------------------------------------------------------------
    "Mandibular Canal": {
        "disease_name": "Ống thần kinh hàm dưới",
        "latin_name": "Canalis mandibularis",
        "treatment_method": "Không cần điều trị; cần lưu ý khi thực hiện thủ thuật vùng hàm dưới",
        "estimated_duration": "Không áp dụng",
        "severity_percent": 5,
    },
    "Maxillary sinus": {
        "disease_name": "Xoang hàm trên",
        "latin_name": "Sinus maxillaris",
        "treatment_method": "Không cần điều trị; cần lưu ý khi cấy ghép implant vùng hàm trên",
        "estimated_duration": "Không áp dụng",
        "severity_percent": 5,
    },
    "Permanent Teeth": {
        "disease_name": "Răng vĩnh viễn",
        "latin_name": "Dentes permanentes",
        "treatment_method": "Duy trì vệ sinh răng miệng tốt; khám định kỳ 6 tháng/lần",
        "estimated_duration": "Không áp dụng",
        "severity_percent": 5,
    },
    "Primary teeth": {
        "disease_name": "Răng sữa",
        "latin_name": "Dentes decidui",
        "treatment_method": "Theo dõi quá trình thay răng; bảo vệ đến khi răng vĩnh viễn mọc",
        "estimated_duration": "Không áp dụng",
        "severity_percent": 5,
    },
}

INFO_CLASSES = {
    "Mandibular Canal",
    "Maxillary sinus",
    "Permanent Teeth",
    "Primary teeth",
}

_ISSUE_STATUS_MAP: Dict[str, str] = {
    **{k: "NEEDS_TREATMENT" for k in NEEDS_TREATMENT_CLASSES},
    **{k: "ALREADY_TREATED" for k in ALREADY_TREATED_CLASSES},
    **{k: "INFO" for k in INFO_CLASSES},
}

# Lookup phân biệt hoa/thường: key.lower() → giá trị gốc
_DISEASE_META_LC: Dict[str, Dict[str, Any]] = {k.lower(): v for k, v in _DISEASE_META.items()}
_ISSUE_STATUS_MAP_LC: Dict[str, str] = {k.lower(): v for k, v in _ISSUE_STATUS_MAP.items()}


# ---------------------------------------------------------------------------
# Hàm hình học: tính góc tại đỉnh B tạo bởi 3 điểm A, B, C
# Sử dụng công thức vector cosin:
#   vec1 = A - B,  vec2 = C - B
#   cos(θ) = (vec1 · vec2) / (|vec1| × |vec2|)
# ---------------------------------------------------------------------------

def _angle_between_three_points(
    ax: float, ay: float,   # Điểm A
    bx: float, by: float,   # Điểm B (đỉnh góc)
    cx: float, cy: float,   # Điểm C
) -> float:
    """
    Trả về góc tại B (độ) tạo bởi 3 điểm A–B–C.

    Công thức vector cosin:
        vec1 = A - B
        vec2 = C - B
        θ = arccos((vec1 · vec2) / (|vec1| × |vec2|))
    """
    v1x, v1y = ax - bx, ay - by
    v2x, v2y = cx - bx, cy - by

    dot = v1x * v2x + v1y * v2y
    mag1 = math.sqrt(v1x ** 2 + v1y ** 2)
    mag2 = math.sqrt(v2x ** 2 + v2y ** 2)

    if mag1 < 1e-9 or mag2 < 1e-9:
        return 0.0

    # Kẹp vào [-1, 1] để tránh lỗi domain của arccos
    cos_theta = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return round(math.degrees(math.acos(cos_theta)), 2)


def _classify_anb(anb: float) -> Tuple[str, str]:
    """Returns (conclusion_title, conclusion_detail)."""
    if anb > 4.0:
        return (
            "Khớp cắn Loại II (Class II)",
            f"ANB = {anb}° — Có dấu hiệu hô (Skeletal Class II)",
        )
    if anb < 0.0:
        return (
            "Khớp cắn Loại III (Class III)",
            f"ANB = {anb}° — Có dấu hiệu móm (Skeletal Class III)",
        )
    return (
        "Khớp cắn Loại I (Class I)",
        f"ANB = {anb}° — Không có dấu hiệu hô hoặc móm đáng kể",
    )


def _angle_between_lines(
    ax: float, ay: float, bx: float, by: float,   # line 1: A → B
    cx: float, cy: float, dx: float, dy: float,   # line 2: C → D
) -> float:
    """Acute angle (°) between two lines AB and CD."""
    v1x, v1y = bx - ax, by - ay
    v2x, v2y = dx - cx, dy - cy
    mag1 = math.sqrt(v1x ** 2 + v1y ** 2)
    mag2 = math.sqrt(v2x ** 2 + v2y ** 2)
    if mag1 < 1e-9 or mag2 < 1e-9:
        return 0.0
    cos_theta = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (mag1 * mag2)))
    angle = math.degrees(math.acos(cos_theta))
    if angle > 90:
        angle = 180.0 - angle
    return round(angle, 2)


# ---------------------------------------------------------------------------
# Xử lý Panoramic
# ---------------------------------------------------------------------------

def _run_panoramic(image_bytes: bytes, fdi_conf: float, cv_conf: float) -> Dict[str, Any]:
    """Wrapper đồng bộ gọi predict_dental_issues, dùng với asyncio.to_thread."""
    return predict_dental_issues(image_bytes, fdi_conf=fdi_conf, cv_conf=cv_conf)


def _build_panoramic_result(
    raw: Dict[str, Any],
) -> Tuple[List[ToothAnalysis], ReportSummary]:
    """Chuyển đổi output của predict_dental_issues sang schema của full report."""
    data: List[Dict] = raw.get("data", [])
    summary_raw: Dict = raw.get("summary", {})

    total_teeth = summary_raw.get("total_teeth_detected", 0)
    teeth_with_issues = summary_raw.get("teeth_with_issues", 0)

    panoramic: List[ToothAnalysis] = []
    cavity_count = 0

    for tooth in data:
        tooth_num = str(tooth["tooth_number"])
        issues_out: List[IssueItem] = []

        for iss in tooth.get("issues", []):
            name = iss["issue_name"]
            status = _ISSUE_STATUS_MAP_LC.get(name.lower(), "NEEDS_TREATMENT")
            issues_out.append(
                IssueItem(
                    issue_name=name,
                    confidence=iss["confidence"],
                    issue_bbox=iss["issue_bbox"],
                    status=status,
                )
            )
            if name == "Caries":
                cavity_count += 1

        if issues_out:
            panoramic.append(
                ToothAnalysis(
                    tooth_number=tooth_num,
                    tooth_bbox=tooth["tooth_bbox"],
                    issues=issues_out,
                )
            )

    healthy = max(0, total_teeth - teeth_with_issues)
    summary = ReportSummary(
        total_teeth=total_teeth,
        healthy=healthy,
        cavities=cavity_count,
        needs_treatment=total_teeth - healthy,
    )

    return panoramic, summary


# ---------------------------------------------------------------------------
# Xử lý Cephalometric
# ---------------------------------------------------------------------------

def _run_ceph(image_bytes: bytes) -> List[Dict]:
    """Wrapper đồng bộ gọi process_and_predict, dùng với asyncio.to_thread."""
    return process_and_predict(image_bytes)


def _build_ceph_result(points: List[Dict]) -> CephAnalysis:
    """Tính toán góc SNA/SNB/ANB và xây dựng CephAnalysis từ danh sách điểm."""
    num_points = len(points)

    # Đảm bảo đủ điểm cần thiết
    required = max(LM_S, LM_N, LM_A, LM_B) + 1
    if num_points < required:
        raise ValueError(
            f"Mô hình LRC chỉ trả về {num_points} điểm, "
            f"cần ít nhất {required} điểm để tính góc."
        )

    # Xây dựng danh sách landmark có tên
    landmarks: List[LandmarkPoint] = []
    for i, pt in enumerate(points):
        name = LM_NAMES[i] if i < len(LM_NAMES) else f"P{i}"
        landmarks.append(LandmarkPoint(name=name, x=pt["x"], y=pt["y"]))

    # Lấy tọa độ 4 điểm chính
    s = points[LM_S]
    n = points[LM_N]
    a = points[LM_A]
    b = points[LM_B]

    # Tính SNA: góc tại N tạo bởi S–N–A
    sna = _angle_between_three_points(
        s["x"], s["y"],
        n["x"], n["y"],
        a["x"], a["y"],
    )

    # Tính SNB: góc tại N tạo bởi S–N–B
    snb = _angle_between_three_points(
        s["x"], s["y"],
        n["x"], n["y"],
        b["x"], b["y"],
    )

    anb = round(sna - snb, 2)

    logger.info("Ceph metrics – SNA: %.2f°, SNB: %.2f°, ANB: %.2f°", sna, snb, anb)

    conclusion, conclusion_detail = _classify_anb(anb)

    return CephAnalysis(
        landmarks=landmarks,
        metrics=CephMetrics(SNA=sna, SNB=snb, ANB=anb),
        conclusion=conclusion,
        conclusion_detail=conclusion_detail,
    )


# ---------------------------------------------------------------------------
# OpenAI Consultation
# ---------------------------------------------------------------------------

async def _get_openai_consultation(
    panoramic: List[ToothAnalysis],
) -> Consultation:
    """
    Gọi GPT-4o (openai >= 1.0 / AsyncOpenAI) để tạo lời tư vấn bằng tiếng Việt.
    Nếu thiếu API key hoặc gọi thất bại, trả về Consultation với nội dung lỗi.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        logger.warning("OPENAI_API_KEY không được cấu hình – bỏ qua bước tư vấn AI.")
        return Consultation(
            overall_assessment=["Chưa cấu hình OPENAI_API_KEY – không thể tạo tư vấn AI."],
            main_issues=[],
        )

    # --- Tóm tắt findings để đưa vào prompt ---
    # Gộp tất cả bệnh lý (cả đã điều trị lẫn cần điều trị) theo từng răng
    teeth_summary_parts: List[str] = []
    for tooth in panoramic:
        all_issues = [i.issue_name for i in tooth.issues]
        if all_issues:
            teeth_summary_parts.append(f"Răng {tooth.tooth_number}: {', '.join(all_issues)}")

    if not teeth_summary_parts:
        teeth_summary_parts = ["Không phát hiện bệnh lý nào trên ảnh Panoramic."]

    user_prompt = (
        "Dựa trên kết quả X-quang Panoramic, các bệnh lý răng miệng được phát hiện như sau:\n"
        + "\n".join(f"- {s}" for s in teeth_summary_parts)
        + "\n\n"
        "Hãy đóng vai nha sĩ chuyên nghiệp để viết tư vấn chi tiết về sức khỏe răng miệng "
        "bằng tiếng Việt. CHỈ tư vấn về tình trạng các răng, bệnh lý răng miệng, "
        "KHÔNG đề cập đến khớp cắn, hô, móm hay các chỉ số Cephalometric.\n\n"
        "Trả về JSON (không có markdown code-block) với cấu trúc chính xác:\n"
        "{\n"
        '  "overall_assessment": [\n'
        '    "<câu đánh giá tổng thể tình trạng sức khỏe răng miệng 1>",\n'
        '    "<câu đánh giá 2 về mức độ nghiêm trọng>",\n'
        '    "<câu đánh giá 3 về ưu tiên điều trị>"\n'
        "  ],\n"
        '  "main_issues": [\n'
        '    {\n'
        '      "issue": "<tên bệnh lý cụ thể – ví dụ: Sâu răng số 46>",\n'
        '      "detail": "<mô tả chi tiết về tình trạng, nguyên nhân, hậu quả nếu không điều trị>",\n'
        '      "recommendation": "<khuyến nghị điều trị cụ thể và thời gian>"\n'
        '    }\n'
        "  ]\n"
        "}\n\n"
        "Lưu ý: overall_assessment cần nhận xét tổng quan về tình trạng sức khỏe răng miệng "
        "dựa trên danh sách bệnh lý thực tế. main_issues liệt kê từng vấn đề với đầy đủ chi tiết."
    )

    system_prompt = (
        "Bạn là nha sĩ chuyên nghiệp với hơn 15 năm kinh nghiệm. "
        "Hãy phân tích kết quả X-quang răng và đưa ra tư vấn CHI TIẾT về từng bệnh lý răng miệng. "
        "Chỉ tập trung vào sức khỏe răng, KHÔNG đề cập khớp cắn hay Cephalometric. "
        "Trả về CHÍNH XÁC JSON theo schema được yêu cầu, không thêm bất kỳ văn bản nào ngoài JSON."
    )

    try:
        from openai import AsyncOpenAI  # openai >= 1.0

        client = AsyncOpenAI(api_key=api_key)
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            max_tokens=800,
            response_format={"type": "json_object"},
        )

        raw_json = response.choices[0].message.content or "{}"
        parsed = json.loads(raw_json)

        overall = parsed.get("overall_assessment", [])
        if isinstance(overall, str):
            overall = [overall]

        issues_raw = parsed.get("main_issues", [])
        main_issues = [
            ConsultationIssue(
                issue=item.get("issue", ""),
                detail=item.get("detail", ""),
                recommendation=item.get("recommendation", ""),
            )
            for item in issues_raw
            if isinstance(item, dict)
        ]

        return Consultation(overall_assessment=overall, main_issues=main_issues)

    except json.JSONDecodeError as exc:
        logger.error("Không parse được JSON từ OpenAI: %s", exc)
        return Consultation(
            overall_assessment=["Không thể phân tích phản hồi từ AI tư vấn."],
            main_issues=[],
        )
    except Exception as exc:
        logger.error("Lỗi khi gọi OpenAI API: %s", exc)
        return Consultation(
            overall_assessment=[f"Lỗi khi kết nối AI tư vấn: {exc}"],
            main_issues=[],
        )


# ---------------------------------------------------------------------------
# Phân tích Cephalometric chuyên sâu bằng AI
# ---------------------------------------------------------------------------

async def _get_ceph_ai_analysis(ceph: CephAnalysis) -> Optional[CephAiAnalysis]:
    """
    Gọi GPT-4o để phân tích chuyên sâu kết quả Cephalometric.
    Trả về CephAiAnalysis hoặc None nếu không có API key / gọi thất bại.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        logger.warning("OPENAI_API_KEY không được cấu hình – bỏ qua phân tích Cephalometric AI.")
        return None

    sna = ceph.metrics.SNA
    snb = ceph.metrics.SNB
    anb = ceph.metrics.ANB
    conclusion = ceph.conclusion
    conclusion_detail = ceph.conclusion_detail

    sna_status = "bình thường" if 80 <= sna <= 84 else ("cao" if sna > 84 else "thấp")
    snb_status = "bình thường" if 78 <= snb <= 82 else ("cao" if snb > 82 else "thấp")
    anb_status = "bình thường" if 0 <= anb <= 4 else ("dương tính tăng" if anb > 4 else "âm tính")

    system_prompt = (
        "Bạn là chuyên gia phân tích Cephalometric X-Ray với hơn 20 năm kinh nghiệm "
        "trong lĩnh vực Chỉnh nha (Orthodontics) và Phẫu thuật hàm mặt (Orthognathic Surgery). "
        "Hãy phân tích kết quả đo đạc Cephalometric bằng tiếng Việt theo chuẩn mực lâm sàng, "
        "trả về CHÍNH XÁC JSON theo schema được yêu cầu, không thêm văn bản nào ngoài JSON."
    )

    user_prompt = (
        "Phân tích kết quả Cephalometric X-Ray sau đây:\n\n"
        f"• SNA = {sna:.2f}° (Chuẩn: 82 ± 2°) → {sna_status}\n"
        f"• SNB = {snb:.2f}° (Chuẩn: 80 ± 2°) → {snb_status}\n"
        f"• ANB = {anb:.2f}° (Chuẩn: 2 ± 2°) → {anb_status}\n"
        f"• Kết luận phân loại: {conclusion}\n"
        f"• Chi tiết: {conclusion_detail}\n\n"
        "Hãy phân tích chuyên sâu về:\n"
        "1. Vị trí xương hàm trên (SNA) so với nền sọ — hàm trên nhô, lùi hay bình thường.\n"
        "2. Vị trí xương hàm dưới (SNB) so với nền sọ — hàm dưới nhô, lùi hay bình thường.\n"
        "3. Mối quan hệ sagittal hàm trên – hàm dưới (ANB) và hậu quả chức năng.\n"
        "4. Các hậu quả lâm sàng (thẩm mỹ, chức năng nhai, nguy cơ TMJ, v.v.).\n"
        "5. Kế hoạch điều trị đề xuất (chỉnh nha, phẫu thuật, theo dõi).\n\n"
        "Trả về JSON (không có markdown code-block) với cấu trúc chính xác:\n"
        "{\n"
        '  "skeletal_summary": "<tóm tắt 1-2 câu về tình trạng khớp cắn xương>",\n'
        '  "sna_interpretation": "<phân tích chi tiết góc SNA và ý nghĩa lâm sàng>",\n'
        '  "snb_interpretation": "<phân tích chi tiết góc SNB và ý nghĩa lâm sàng>",\n'
        '  "anb_interpretation": "<phân tích chi tiết góc ANB, mối quan hệ sagittal>",\n'
        '  "clinical_implications": ["<hậu quả lâm sàng 1>", "<hậu quả 2>", ...],\n'
        '  "treatment_plan": ["<bước điều trị 1>", "<bước 2>", ...],\n'
        '  "severity": "low|medium|high"\n'
        "}"
    )

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key)
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1000,
            response_format={"type": "json_object"},
        )

        raw_json = response.choices[0].message.content or "{}"
        parsed = json.loads(raw_json)

        clinical = parsed.get("clinical_implications", [])
        if isinstance(clinical, str):
            clinical = [clinical]

        treatment = parsed.get("treatment_plan", [])
        if isinstance(treatment, str):
            treatment = [treatment]

        severity_raw = str(parsed.get("severity", "medium")).lower()
        if severity_raw not in ("low", "medium", "high"):
            severity_raw = "medium"

        return CephAiAnalysis(
            skeletal_summary=parsed.get("skeletal_summary", ""),
            sna_interpretation=parsed.get("sna_interpretation", ""),
            snb_interpretation=parsed.get("snb_interpretation", ""),
            anb_interpretation=parsed.get("anb_interpretation", ""),
            clinical_implications=clinical,
            treatment_plan=treatment,
            severity=severity_raw,  # type: ignore[arg-type]
        )

    except json.JSONDecodeError as exc:
        logger.error("Không parse được JSON Cephalometric AI: %s", exc)
        return None
    except Exception as exc:
        logger.error("Lỗi khi gọi OpenAI cho Cephalometric AI: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Xây dựng teeth_details
# ---------------------------------------------------------------------------

def _build_teeth_details(panoramic: List[ToothAnalysis]) -> List[ToothDetail]:
    """Tạo danh sách chi tiết cho tất cả răng có vấn đề (cả đã điều trị lẫn cần điều trị)."""
    details: List[ToothDetail] = []
    seen: set = set()

    for tooth in panoramic:
        for issue in tooth.issues:
            key = (tooth.tooth_number, issue.issue_name)
            if key in seen:
                continue
            seen.add(key)

            meta = _DISEASE_META_LC.get(issue.issue_name.lower())
            status = _ISSUE_STATUS_MAP_LC.get(issue.issue_name.lower(), "NEEDS_TREATMENT")
            if meta is None:
                meta = {
                    "disease_name": issue.issue_name,
                    "latin_name": "",
                    "treatment_method": "Tham khảo nha sĩ",
                    "estimated_duration": "Tùy tình trạng",
                    "severity_percent": 50,
                }

            details.append(
                ToothDetail(
                    tooth_number=tooth.tooth_number,
                    disease_name=meta["disease_name"],
                    latin_name=meta["latin_name"],
                    treatment_method=meta["treatment_method"],
                    estimated_duration=meta["estimated_duration"],
                    severity_percent=meta["severity_percent"],
                    status=status,
                )
            )

    return details


# ---------------------------------------------------------------------------
# Hàm công khai chính
# ---------------------------------------------------------------------------

async def build_full_report(
    panoramic_bytes: bytes,
    ceph_bytes: Optional[bytes],
    fdi_conf: float = 0.25,
    cv_conf: float = 0.30,
) -> FullReportResponse:
    """
    Orchestrate toàn bộ pipeline:
      1. Phân tích Panoramic (FDI + CV)
      2. Phân tích Cephalometric (LRC + tính góc)
      3. Tư vấn OpenAI (GPT-4o)
      4. Tổng hợp FullReportResponse

    Args:
        panoramic_bytes : bytes của ảnh Panoramic.
        ceph_bytes      : bytes của ảnh Cephalometric (None = bỏ qua bước này).
        fdi_conf        : confidence threshold cho FDI model.
        cv_conf         : confidence threshold cho CV model.

    Returns:
        FullReportResponse được validate bởi Pydantic.

    Raises:
        ValueError   : ảnh không hợp lệ.
        RuntimeError : lỗi inference model.
    """

    # ------------------------------------------------------------------
    # Bước 1: Panoramic (chạy trong thread riêng để không block event-loop)
    # ------------------------------------------------------------------
    logger.info("Bắt đầu phân tích Panoramic …")
    try:
        raw_panoramic = await asyncio.to_thread(
            _run_panoramic, panoramic_bytes, fdi_conf, cv_conf
        )
        panoramic_list, summary = _build_panoramic_result(raw_panoramic)
    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Lỗi phân tích ảnh Panoramic: {exc}") from exc

    logger.info("Panoramic: %d răng bất thường.", len(panoramic_list))

    # ------------------------------------------------------------------
    # Bước 2: Cephalometric (tùy chọn)
    # ------------------------------------------------------------------
    ceph_result: Optional[CephAnalysis] = None
    if ceph_bytes:
        logger.info("Bắt đầu phân tích Cephalometric …")
        try:
            raw_points = await asyncio.to_thread(_run_ceph, ceph_bytes)
            ceph_result = _build_ceph_result(raw_points)
        except Exception as exc:
            logger.warning("Bỏ qua Cephalometric do lỗi: %s", exc)
            # Không raise – tiếp tục với ceph_result = None

    # ------------------------------------------------------------------
    # Bước 3: OpenAI tư vấn + phân tích Cephalometric AI (chạy song song)
    # ------------------------------------------------------------------
    logger.info("Gọi OpenAI GPT-4o để tạo tư vấn và phân tích Cephalometric …")
    tasks = [_get_openai_consultation(panoramic_list)]
    if ceph_result:
        tasks.append(_get_ceph_ai_analysis(ceph_result))  # type: ignore[arg-type]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    consultation = results[0] if not isinstance(results[0], Exception) else Consultation(
        overall_assessment=["Lỗi khi kết nối AI tư vấn."],
        main_issues=[],
    )
    ceph_ai: Optional[CephAiAnalysis] = None
    if ceph_result and len(results) > 1:
        ceph_ai = results[1] if not isinstance(results[1], Exception) else None

    # ------------------------------------------------------------------
    # Bước 4: Xây dựng teeth_details
    # ------------------------------------------------------------------
    teeth_details = _build_teeth_details(panoramic_list)

    return FullReportResponse(
        summary=summary,
        panoramic_analysis=panoramic_list,
        ceph_analysis=ceph_result,
        ceph_ai_analysis=ceph_ai,
        consultation=consultation,
        teeth_details=teeth_details,
    )
