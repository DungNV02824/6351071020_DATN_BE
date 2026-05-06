# services/dental_cv_service.py
"""
Dental CV inference pipeline.

Responsibilities:
  1. Preprocess uploaded image bytes into a format accepted by YOLO.
  2. Run model inference via the singleton loaded in models/dental_cv_model.py.
  3. Post-process raw predictions (threshold filter, Vietnamese label mapping).
  4. Visualise results on the original image using OpenCV.
  5. Return structured detection data + base64-encoded annotated image.
"""

import base64
import io
import logging
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

from models.cv_model import get_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label mapping – English class name → Vietnamese display name
# The keys must match exactly the class names stored in CV.pt (model.names).
# If the model uses numeric indices only, the FALLBACK_LABELS dict is used.
# ---------------------------------------------------------------------------
LABEL_VI: Dict[str, str] = {
    "Caries":               "Sâu răng",
    "Crown":                "Mão răng sứ",
    "Filling":              "Trám răng",
    "Implant":              "Cấy ghép implant",
    "Malaligned":           "Răng lệch lạc",
    "Mandibular Canal":     "Ống thần kinh hàm dưới",
    "Missing teeth":        "Mất răng",
    "Periapical lesion":    "Tổn thương quanh chóp",
    "Retained root":        "Chân răng còn sót",
    "Root Canal Treatment": "Điều trị tủy",
    "Root Piece":           "Mảnh chân răng",
    "Impacted tooth":       "Răng mọc ngầm",
    "Maxillary sinus":      "Xoang hàm trên",
    "Bone Loss":            "Tiêu xương",
    "Fracture teeth":       "Răng gãy vỡ",
    "Permanent Teeth":      "Răng vĩnh viễn",
    "Supra Eruption":       "Răng trồi",
    "TAD":                  "Neo chỉnh nha (TAD)",
    "Abutment":             "Trụ implant",
    "Attrition":            "Mòn răng",
    "Bone defect":          "Khuyết hổng xương",
    "Gingival former":      "Đầu lành thương nướu",
    "Metal band":           "Band kim loại",
    "Orthodontic brackets": "Mắc cài chỉnh nha",
    "Permanent retainer":   "Hàm duy trì cố định",
    "Post-core":            "Chốt lõi răng",
    "Plating":              "Nẹp vít xương",
    "Wire":                 "Dây cung chỉnh nha",
    "Cyst":                 "Nang xương hàm",
    "Root resorption":      "Tiêu chân răng",
    "Primary teeth":        "Răng sữa",
}

# Fallback by index if the model's class names don't match the keys above
FALLBACK_LABELS: Dict[int, str] = {
    0:  "Sâu răng",
    1:  "Mão răng sứ",
    2:  "Trám răng",
    3:  "Cấy ghép implant",
    4:  "Răng lệch lạc",
    5:  "Ống thần kinh hàm dưới",
    6:  "Mất răng",
    7:  "Tổn thương quanh chóp",
    8:  "Chân răng còn sót",
    9:  "Điều trị tủy",
    10: "Mảnh chân răng",
    11: "Răng mọc ngầm",
    12: "Xoang hàm trên",
    13: "Tiêu xương",
    14: "Răng gãy vỡ",
    15: "Răng vĩnh viễn",
    16: "Răng trồi",
    17: "Neo chỉnh nha (TAD)",
    18: "Trụ implant",
    19: "Mòn răng",
    20: "Khuyết hổng xương",
    21: "Đầu lành thương nướu",
    22: "Band kim loại",
    23: "Mắc cài chỉnh nha",
    24: "Hàm duy trì cố định",
    25: "Chốt lõi răng",
    26: "Nẹp vít xương",
    27: "Dây cung chỉnh nha",
    28: "Nang xương hàm",
    29: "Tiêu chân răng",
    30: "Răng sữa",
}

# ---------------------------------------------------------------------------
# Colour palette (BGR) – one distinct colour per class index
# ---------------------------------------------------------------------------
_COLOURS: List[Tuple[int, int, int]] = [
    (56,  56,  255),   # 0  – Caries
    (31, 112,  255),   # 1  – Crown
    (29, 178,  255),   # 2  – Filling
    (10, 249,  72),    # 3  – Implant
    (134, 219, 61),    # 4  – Malaligned
    (255, 157, 151),   # 5  – Mandibular Canal
    (255, 178, 29),    # 6  – Missing teeth
    (147, 210, 204),   # 7  – Periapical lesion
    (255,  69,   0),   # 8  – Retained root
    (0,   191, 255),   # 9  – Root Canal Treatment
    (255, 215,   0),   # 10 – Root Piece
    (148,   0, 211),   # 11 – Impacted tooth
    (0,   255, 127),   # 12 – Maxillary sinus
    (220,  20,  60),   # 13 – Bone Loss
    (255, 140,   0),   # 14 – Fracture teeth
    (100, 149, 237),   # 15 – Permanent Teeth
    (50,  205,  50),   # 16 – Supra Eruption
    (255,  20, 147),   # 17 – TAD
    (64,  224, 208),   # 18 – Abutment
    (255, 165,   0),   # 19 – Attrition
    (139,  69,  19),   # 20 – Bone defect
    (0,   128, 128),   # 21 – Gingival former
    (192, 192, 192),   # 22 – Metal band
    (70,  130, 180),   # 23 – Orthodontic brackets
    (0,   100,   0),   # 24 – Permanent retainer
    (128,   0,   0),   # 25 – Post-core
    (85,  107,  47),   # 26 – Plating
    (135, 206, 235),   # 27 – Wire
    (75,    0, 130),   # 28 – Cyst
    (210, 105,  30),   # 29 – Root resorption
    (127, 255, 212),   # 30 – Primary teeth
]

_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.55
_FONT_THICK = 1
_BOX_THICK  = 2


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_label(class_name: str, class_idx: int) -> str:
    """
    Return the Vietnamese label for a detection.
    Try the English name first; fall back to numeric index.
    """
    vi = LABEL_VI.get(class_name)
    if vi:
        return vi
    vi = FALLBACK_LABELS.get(class_idx)
    return vi if vi else class_name   # last resort: raw name from model


def _colour_for(class_idx: int) -> Tuple[int, int, int]:
    return _COLOURS[class_idx % len(_COLOURS)]


def _draw_detections(
    bgr_image: np.ndarray,
    detections: List[Dict[str, Any]],
) -> np.ndarray:
    """
    Draw bounding boxes and label+confidence text onto a copy of bgr_image.

    Args:
        bgr_image:  OpenCV image in BGR format (H×W×3, uint8).
        detections: list of detection dicts (must include label_vi, confidence,
                    bbox as [x1,y1,x2,y2], class_idx).

    Returns:
        Annotated BGR image (same size, new array).
    """
    canvas = bgr_image.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        colour = _colour_for(det["class_idx"])
        label_text = f"{det['label_vi']}  {det['confidence']:.0%}"

        # Bounding box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, _BOX_THICK)

        # Label background pill
        (tw, th), baseline = cv2.getTextSize(
            label_text, _FONT, _FONT_SCALE, _FONT_THICK
        )
        bg_y1 = max(y1 - th - baseline - 6, 0)
        bg_y2 = max(y1, th + baseline + 6)
        cv2.rectangle(canvas, (x1, bg_y1), (x1 + tw + 6, bg_y2), colour, cv2.FILLED)

        # Label text (white, readable on any colour background)
        cv2.putText(
            canvas,
            label_text,
            (x1 + 3, bg_y2 - baseline - 2),
            _FONT,
            _FONT_SCALE,
            (255, 255, 255),
            _FONT_THICK,
            cv2.LINE_AA,
        )

    return canvas


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_dental(
    image_bytes: bytes,
    conf_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Full inference pipeline for dental pathology detection.

    Args:
        image_bytes:     Raw bytes of the uploaded image (JPG / PNG).
        conf_threshold:  Minimum confidence score to keep a detection (0–1).

    Returns:
        {
          "detections":    [ { label, confidence, bbox, label_vi } ],
          "total_objects": int,
          "image_result":  str  (base64-encoded annotated PNG),
          "inference_ms":  float (wall-clock inference time in ms),
        }

    Raises:
        ValueError:  if the image bytes cannot be decoded.
        RuntimeError: if the model raises an unexpected error.
    """
    # ------------------------------------------------------------------
    # 1. Decode image → PIL (for YOLO) and OpenCV (for drawing)
    # ------------------------------------------------------------------
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Cannot decode image: {exc}") from exc

    # Convert PIL → BGR numpy array (OpenCV native format)
    bgr_image: np.ndarray = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # ------------------------------------------------------------------
    # 2. Inference
    # ------------------------------------------------------------------
    model = get_model()

    t_start = time.perf_counter()
    # YOLO handles all internal preprocessing (resize, normalise, batch)
    results = model.predict(
        source=pil_image,
        conf=conf_threshold,
        iou=0.45,
        verbose=False,
    )
    inference_ms = (time.perf_counter() - t_start) * 1000

    logger.info(
        "Dental CV inference completed in %.1f ms (conf_threshold=%.2f)",
        inference_ms,
        conf_threshold,
    )

    # ------------------------------------------------------------------
    # 3. Parse raw YOLO output
    # ------------------------------------------------------------------
    detections: List[Dict[str, Any]] = []

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_idx   = int(box.cls.item())
            cls_name  = result.names.get(cls_idx, str(cls_idx))
            conf_val  = float(box.conf.item())
            label_vi  = _resolve_label(cls_name, cls_idx)

            detections.append(
                {
                    "label":       cls_name,
                    "label_vi":    label_vi,
                    "confidence":  round(conf_val, 4),
                    "bbox":        [
                        round(x1, 2), round(y1, 2),
                        round(x2, 2), round(y2, 2),
                    ],
                    "class_idx":   cls_idx,
                }
            )

    # Sort highest confidence first
    detections.sort(key=lambda d: d["confidence"], reverse=True)

    # ------------------------------------------------------------------
    # 4. Visualise – draw boxes on original image
    # ------------------------------------------------------------------
    annotated_bgr = _draw_detections(bgr_image, detections)

    # Encode annotated image to PNG → base64
    success, png_buf = cv2.imencode(".png", annotated_bgr)
    if not success:
        raise RuntimeError("Failed to encode annotated image to PNG.")

    image_b64 = base64.b64encode(png_buf.tobytes()).decode("utf-8")

    # ------------------------------------------------------------------
    # 5. Build clean response (drop internal class_idx from public output)
    # ------------------------------------------------------------------
    public_detections = [
        {
            "label":      d["label_vi"],          # Vietnamese label
            "confidence": d["confidence"],
            "bbox":       d["bbox"],              # [x1, y1, x2, y2]
        }
        for d in detections
    ]

    return {
        "detections":    public_detections,
        "total_objects": len(public_detections),
        "image_result":  image_b64,
        "inference_ms":  round(inference_ms, 2),
    }


def predict_dental_image(
    image_bytes: bytes,
    conf_threshold: float = 0.3,
) -> bytes:
    """
    Full inference pipeline for dental pathology detection.
    Returns a JPEG image with bounding boxes drawn (as raw bytes).

    Args:
        image_bytes:     Raw bytes of the uploaded image (JPG / PNG).
        conf_threshold:  Minimum confidence score to keep a detection (0–1).

    Returns:
        JPEG image bytes with annotated bounding boxes.

    Raises:
        ValueError:   if the image bytes cannot be decoded.
        RuntimeError: if the model raises an unexpected error.
    """
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Cannot decode image: {exc}") from exc

    bgr_image: np.ndarray = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    model = get_model()

    results = model.predict(
        source=pil_image,
        conf=conf_threshold,
        iou=0.45,
        verbose=False,
    )

    detections: list = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_idx  = int(box.cls.item())
            cls_name = result.names.get(cls_idx, str(cls_idx))
            conf_val = float(box.conf.item())
            label_vi = _resolve_label(cls_name, cls_idx)
            detections.append(
                {
                    "label":      cls_name,
                    "label_vi":   label_vi,
                    "confidence": round(conf_val, 4),
                    "bbox":       [x1, y1, x2, y2],
                    "class_idx":  cls_idx,
                }
            )

    detections.sort(key=lambda d: d["confidence"], reverse=True)

    annotated_bgr = _draw_detections(bgr_image, detections)

    success, jpg_buf = cv2.imencode(
        ".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92]
    )
    if not success:
        raise RuntimeError("Failed to encode annotated image to JPEG.")

    return jpg_buf.tobytes()
