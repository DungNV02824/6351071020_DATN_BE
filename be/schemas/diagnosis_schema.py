# schemas/diagnosis_schema.py
"""
Pydantic models for POST /api/v1/diagnosis/full-report.

Cấu trúc được thiết kế để Frontend (React) render trực tiếp không cần
biến đổi thêm.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 1. SUMMARY
# ---------------------------------------------------------------------------

class ReportSummary(BaseModel):
    total_teeth: int = Field(..., description="Tổng số răng detect được từ FDI")
    healthy: int = Field(..., description="Số răng không có vấn đề gì")
    cavities: int = Field(..., description="Số răng bị sâu (Cavity)")
    needs_treatment: int = Field(..., description="Số răng cần điều trị (Cavity + Impacted Tooth)")


# ---------------------------------------------------------------------------
# 2. PANORAMIC ANALYSIS
# ---------------------------------------------------------------------------

class IssueItem(BaseModel):
    issue_name: str = Field(..., examples=["Cavity"])
    confidence: float = Field(..., ge=0.0, le=1.0)
    issue_bbox: List[float] = Field(..., min_length=4, max_length=4)
    status: Literal["NEEDS_TREATMENT", "ALREADY_TREATED"] = Field(
        ..., description="Phân loại mức độ can thiệp"
    )


class ToothAnalysis(BaseModel):
    tooth_number: str = Field(..., examples=["36"])
    tooth_bbox: List[float] = Field(..., min_length=4, max_length=4)
    issues: List[IssueItem]


# ---------------------------------------------------------------------------
# 3. CEPHALOMETRIC ANALYSIS
# ---------------------------------------------------------------------------

class LandmarkPoint(BaseModel):
    name: str = Field(..., examples=["S"])
    x: float
    y: float


class CephMetrics(BaseModel):
    SNA: float = Field(..., description="Góc SNA (độ)")
    SNB: float = Field(..., description="Góc SNB (độ)")
    ANB: float = Field(..., description="Góc ANB = SNA - SNB (độ)")


class CephAnalysis(BaseModel):
    landmarks: List[LandmarkPoint] = Field(
        default_factory=list,
        description="29 điểm mốc cephalometric với tên và tọa độ",
    )
    metrics: CephMetrics
    conclusion: str = Field(
        ...,
        examples=["Khớp cắn Loại I (Class I)"],
    )
    conclusion_detail: str = Field(
        default="",
        examples=["ANB = 2.3° — Không có dấu hiệu hô hoặc móm đáng kể"],
    )


# ---------------------------------------------------------------------------
# 4. CEPHALOMETRIC AI ANALYSIS (OpenAI – chuyên sâu)
# ---------------------------------------------------------------------------

class CephAiAnalysis(BaseModel):
    skeletal_summary: str = Field(
        ...,
        description="Tóm tắt tình trạng khớp cắn xương (1-2 câu)",
    )
    sna_interpretation: str = Field(
        ...,
        description="Phân tích vị trí xương hàm trên so với nền sọ (góc SNA)",
    )
    snb_interpretation: str = Field(
        ...,
        description="Phân tích vị trí xương hàm dưới so với nền sọ (góc SNB)",
    )
    anb_interpretation: str = Field(
        ...,
        description="Phân tích mối quan hệ sagittal hàm trên – hàm dưới (góc ANB)",
    )
    clinical_implications: List[str] = Field(
        default_factory=list,
        description="Danh sách các hậu quả lâm sàng",
    )
    treatment_plan: List[str] = Field(
        default_factory=list,
        description="Danh sách các bước kế hoạch điều trị được đề xuất",
    )
    severity: Literal["low", "medium", "high"] = Field(
        ...,
        description="Mức độ nghiêm trọng tổng thể: low / medium / high",
    )


# ---------------------------------------------------------------------------
# 5. CONSULTATION (OpenAI)
# ---------------------------------------------------------------------------

class ConsultationIssue(BaseModel):
    issue: str = Field(..., description="Tên vấn đề răng miệng")
    detail: str = Field(..., description="Mô tả chi tiết")
    recommendation: str = Field(..., description="Khuyến nghị điều trị")


class Consultation(BaseModel):
    overall_assessment: List[str] = Field(
        default_factory=list,
        description="Danh sách các đánh giá tổng thể từ nha sĩ AI",
    )
    main_issues: List[ConsultationIssue] = Field(
        default_factory=list,
        description="Danh sách các vấn đề chính được trình bày chuyên nghiệp",
    )


# ---------------------------------------------------------------------------
# 7. TOOTH DETAIL (cho từng răng cần điều trị)
# ---------------------------------------------------------------------------

class ToothDetail(BaseModel):
    tooth_number: str = Field(..., examples=["46"])
    disease_name: str = Field(..., description="Tên bệnh tiếng Việt", examples=["Sâu răng"])
    latin_name: str = Field(..., description="Tên bệnh tiếng Latin", examples=["Caries dentis"])
    treatment_method: str = Field(..., description="Phương pháp điều trị được khuyến nghị")
    estimated_duration: str = Field(..., description="Thời gian điều trị dự kiến")
    severity_percent: int = Field(..., ge=0, le=100, description="Mức độ nghiêm trọng (%)")
    status: Literal["NEEDS_TREATMENT", "ALREADY_TREATED", "INFO"] = Field(
        default="NEEDS_TREATMENT",
        description="Phân loại: cần điều trị / đã điều trị / thông tin giải phẫu",
    )


# ---------------------------------------------------------------------------
# 8. FULL REPORT (root response)
# ---------------------------------------------------------------------------

class FullReportResponse(BaseModel):
    summary: ReportSummary
    panoramic_analysis: List[ToothAnalysis] = Field(
        default_factory=list,
        description="Danh sách các răng có bất thường từ ảnh Panoramic",
    )
    ceph_analysis: Optional[CephAnalysis] = Field(
        default=None,
        description="Kết quả phân tích ảnh Cephalometric (None nếu không cung cấp ảnh)",
    )
    ceph_ai_analysis: Optional[CephAiAnalysis] = Field(
        default=None,
        description="Phân tích AI chuyên sâu về Cephalometric (None nếu không có ảnh hoặc lỗi)",
    )
    consultation: Consultation
    teeth_details: List[ToothDetail] = Field(
        default_factory=list,
        description="Chi tiết điều trị cho từng răng cần can thiệp",
    )
