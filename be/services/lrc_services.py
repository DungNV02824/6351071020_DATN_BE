import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import io
import math
import timm

# =========================================================================
# CẤU HÌNH THÔNG SỐ (PHẢI GIỐNG LÚC TRAIN)
# =========================================================================
IMG_SIZE = 512
NUM_LANDMARKS = 29 

# MAPPING CHÍNH XÁC THEO DATASET 29 ĐIỂM CỦA BẠN (Đã trừ 1 để làm index mảng 0-28)
LANDMARK_MAP = {
    "S": 10,  # Sella (Điểm 11 trên ảnh)
    "N": 4,   # Nasion (Điểm 5 trên ảnh)
    "A": 0,   # Subspinale (Điểm 1 trên ảnh)
    "B": 23,  # Supramentale (Điểm 24 trên ảnh)
    "Go": 14, # Gonion (Điểm 15 trên ảnh)
    "Pog": 6, # Pogonion (Điểm 7 trên ảnh)
    "Me": 3   # Menton (Điểm 4 trên ảnh)
}
# MAPPING ĐẦY ĐỦ 29 ĐIỂM THEO DANH SÁCH CỦA BẠN
LANDMARK_MAP_2 = {
    "A": 0, "ANS": 1, "B": 2, "Me": 3, "N": 4, "Or": 5, "Pog": 6, 
    "PNS": 7, "Pn": 8, "R": 9, "S": 10, "Ar": 11, "Co": 12, "Gn": 13, 
    "Go": 14, "Po": 15, "LPM": 16, "LIT": 17, "LMT": 18, "UPM": 19, 
    "UIA": 20, "UIT": 21, "UMT": 22, "LIA": 23, "Li": 24, "Ls": 25, 
    "N`": 26, "Pog`": 27, "Sn": 28
}

# =========================================================================
# 1. HÀM CHUYỂN ĐỔI HEATMAP VÀ KIẾN TRÚC MÔ HÌNH
# =========================================================================
def heatmap_to_coord(hm):
    hm = hm.reshape(NUM_LANDMARKS, -1)
    idx = hm.argmax(axis=1)
    y = idx // IMG_SIZE
    x = idx % IMG_SIZE
    return np.stack([x, y], axis=1)

class HRNetW32(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('hrnet_w32', pretrained=False, features_only=True, out_indices=(3,))
        in_ch = 512
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, NUM_LANDMARKS, 1)
        )

    def forward(self, x):
        x = self.backbone(x)[0]
        x = self.head(x)
        return F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)

# =========================================================================
# 2. KHỞI TẠO VÀ LOAD MÔ HÌNH TỪ FILE
# =========================================================================
DEVICE = torch.device("cpu") 
model = HRNetW32().to(DEVICE)
MODEL_PATH = r"C:\NCKH\be\Model\checkpoint_ep70_LRC.pth" 

try:
    print(f"⏳ Đang load mô hình từ: {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("✅ Đã load mô hình HRNetW32 thành công và sẵn sàng dự đoán!")
except Exception as e:
    print(f"❌ Lỗi khi load mô hình: {e}")

image_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
])

# =========================================================================
# 3. CÁC HÀM XỬ LÝ API (JSON & ẢNH ĐIỂM CƠ BẢN)
# =========================================================================
def process_and_predict(image_bytes: bytes):
    """Trả về tọa độ 29 điểm dạng JSON"""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_w, orig_h = image.size

    img_tensor = image_transforms(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred_hm = model(img_tensor)
        
    pred_hm_np = pred_hm.squeeze(0).cpu().numpy()
    pred_pts = heatmap_to_coord(pred_hm_np)

    scale_x = orig_w / IMG_SIZE
    scale_y = orig_h / IMG_SIZE
    points_list = [{"x": float(pt[0]) * scale_x, "y": float(pt[1]) * scale_y} for pt in pred_pts]
    return points_list

def process_and_draw(image_bytes: bytes) -> bytes:
    """Vẽ toàn bộ 29 điểm (kèm số) lên ảnh gốc"""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_w, orig_h = image.size

    img_tensor = image_transforms(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred_hm = model(img_tensor)

    pred_hm_np = pred_hm.squeeze(0).cpu().numpy()
    pred_pts = heatmap_to_coord(pred_hm_np)

    scale_x = orig_w / IMG_SIZE
    scale_y = orig_h / IMG_SIZE

    draw = ImageDraw.Draw(image)
    radius = max(3, int(min(orig_w, orig_h) * 0.008))

    for i, pt in enumerate(pred_pts):
        cx = float(pt[0]) * scale_x
        cy = float(pt[1]) * scale_y
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(255, 0, 0), outline=(255, 255, 0), width=max(1, radius // 3))
        draw.text((cx + radius + 2, cy - radius), str(i + 1), fill=(255, 255, 0))

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()

# =========================================================================
# 4. HÀM PHÂN TÍCH TOÁN HỌC VÀ VẼ HÌNH CHẨN ĐOÁN (GIỐNG ẢNH MẪU YÊU CẦU)
# =========================================================================
def calculate_angle(p1, vertex, p2):
    """Tính góc giữa 3 điểm (p1 - vertex - p2)"""
    v1 = (p1[0] - vertex[0], p1[1] - vertex[1])
    v2 = (p2[0] - vertex[0], p2[1] - vertex[1])
    
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    
    if mag1 * mag2 == 0:
        return 0.0
    cos_angle = max(min(dot_product / (mag1 * mag2), 1.0), -1.0)
    return math.degrees(math.acos(cos_angle))

def process_and_draw_analysis(image_bytes: bytes) -> bytes:
    """Phân tích Cephalometric, vẽ đường nối S-N-A-B và in kết quả chẩn đoán"""
    try:
        # 1. Đọc và dự đoán điểm
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        orig_w, orig_h = image.size

        img_tensor = image_transforms(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred_hm = model(img_tensor)

        pred_hm_np = pred_hm.squeeze(0).cpu().numpy()
        pred_pts = heatmap_to_coord(pred_hm_np)

        # 2. Scale và lọc ra các điểm quan trọng
        scale_x = orig_w / IMG_SIZE
        scale_y = orig_h / IMG_SIZE
        
        pts = {}
        for key, idx in LANDMARK_MAP.items():
            pts[key] = (float(pred_pts[idx][0]) * scale_x, float(pred_pts[idx][1]) * scale_y)

        # 3. Tính toán góc
        sna = calculate_angle(pts["S"], pts["N"], pts["A"])
        snb = calculate_angle(pts["S"], pts["N"], pts["B"])
        anb = sna - snb

        # 4. Logic chẩn đoán
        if anb > 4.0:
            diagnosis = f"Chẩn Đoán Hô (ANB = {anb:.2f}°)"
        elif anb < 0.0:
            diagnosis = f"Chẩn Đoán Móm (ANB = {anb:.2f}°)"
        else:
            diagnosis = f"Chẩn Đoán Bình Thường (ANB = {anb:.2f}°)"

        # 5. Cài đặt vẽ & Font chữ
        draw = ImageDraw.Draw(image)
        font_size_large = max(20, int(orig_w * 0.035))
        font_size_small = max(14, int(orig_w * 0.025))
        
        try:
            # Ưu tiên load font Arial đậm cho dễ nhìn
            font_title = ImageFont.truetype("arialbd.ttf", font_size_large)
            font_info = ImageFont.truetype("arialbd.ttf", font_size_small)
        except IOError:
            # Fallback nếu không tìm thấy font (trên Linux)
            font_title = font_info = ImageFont.load_default()

        # --- 5.1 Vẽ các đường thẳng (Màu xanh lá) ---
        line_color = (0, 255, 0)
        line_width = max(2, int(orig_w * 0.003))
        
        draw.line([pts["S"], pts["N"]], fill=line_color, width=line_width)
        draw.line([pts["N"], pts["A"]], fill=line_color, width=line_width)
        draw.line([pts["N"], pts["B"]], fill=line_color, width=line_width)
        draw.line([pts["N"], pts["Pog"]], fill=line_color, width=line_width)
        draw.line([pts["Go"], pts["Me"]], fill=line_color, width=line_width)

        # --- 5.2 Vẽ các điểm và tên (Màu đỏ/Trắng) ---
        radius = max(4, int(orig_w * 0.007))
        for name, pt in pts.items():
            cx, cy = pt
            draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(255, 0, 0))
            draw.text((cx + radius + 5, cy - radius - 5), name, fill=(255, 255, 255), font=font_info)

        # --- 5.3 In thông số SNA, SNB, ANB (Góc trái trên) ---
        info_text = f"SNA: {sna:.2f} deg\nSNB: {snb:.2f} deg\nANB: {anb:.2f} deg"
        text_x, text_y = int(orig_w * 0.02), int(orig_h * 0.02)
        
        # Viền đen cho text thông số (Màu Cyan)
        for adj_x, adj_y in [(-1,-1), (-1,1), (1,-1), (1,1)]:
            draw.text((text_x+adj_x, text_y+adj_y), info_text, fill=(0,0,0), font=font_info)
        draw.text((text_x, text_y), info_text, fill=(0, 255, 255), font=font_info) 

        # --- 5.4 In Chẩn Đoán (Căn giữa phía trên) ---
        bbox = font_title.getbbox(diagnosis) # (left, top, right, bottom)
        text_w = bbox[2] - bbox[0]
        title_x = (orig_w - text_w) // 2
        title_y = int(orig_h * 0.01)
        
        # Viền trắng, chữ Xanh sẫm (DarkBlue)
        for adj_x, adj_y in [(-2,-2), (-2,2), (2,-2), (2,2)]:
            draw.text((title_x+adj_x, title_y+adj_y), diagnosis, fill=(255,255,255), font=font_title)
        draw.text((title_x, title_y), diagnosis, fill=(0, 0, 139), font=font_title)

        # 6. Trả về bytes
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        return buf.read()

    except Exception as e:
        raise Exception(f"Lỗi trong quá trình phân tích và vẽ: {str(e)}")
###########################################################################
import cv2
import numpy as np
import io
import json
from PIL import Image

def process_morphing_tps(image_bytes, source_pts_dict, target_pts_dict):
    try:
        # 1. Load ảnh
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(img)
        # OpenCV dùng BGR
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        h, w, _ = img_cv.shape

        # 2. Chuẩn bị điểm mốc
        # Lọc lấy các key chung (S, N, A, B, Pog, Go, Me...)
        keys = [k for k in source_pts_dict if k in target_pts_dict]
        
        src_pts = []
        dst_pts = []
        
        for k in keys:
            src_pts.append([source_pts_dict[k]['x'], source_pts_dict[k]['y']])
            dst_pts.append([target_pts_dict[k]['x'], target_pts_dict[k]['y']])

        # THÊM CÁC ĐIỂM CỐ ĐỊNH Ở 4 GÓC để ảnh không bị méo khung ngoài
        # Đây là bước cực kỳ quan trọng để tránh lỗi co rúm ảnh
        border = [
            [0, 0], [w-1, 0], [0, h-1], [w-1, h-1],
            [w//2, 0], [w//2, h-1], [0, h//2], [w-1, h//2]
        ]
        src_pts.extend(border)
        dst_pts.extend(border)

        # Ép kiểu về float32 và định dạng (1, N, 2)
        sources = np.array([src_pts], dtype=np.float32)
        targets = np.array([dst_pts], dtype=np.float32)

        # 3. Thực hiện thuật toán TPS
        tps = cv2.createThinPlateSplineShapeTransformer()
        
        # Tạo danh sách các điểm khớp (1-1, 2-2...)
        matches = [cv2.DMatch(i, i, 0) for i in range(len(src_pts))]
        
        # Ước tính biến dạng
        tps.estimateTransformation(targets, sources, matches)
        
        # Nắn ảnh
        warped_img = tps.warpImage(img_cv)

        # 4. Trả về bytes
        _, buffer = cv2.imencode(".png", warped_img)
        return buffer.tobytes()
    except Exception as e:
        print(f"Lỗi tại process_morphing_tps: {str(e)}")
        raise e

import json

def draw_analysis_from_custom_points(image_bytes: bytes, points_json: str) -> bytes:
    """Vẽ phân tích dựa trên tọa độ người dùng đã chỉnh sửa trên FE"""
    try:
        # 1. Đọc ảnh và tọa độ
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        orig_w, orig_h = image.size
        pts = json.loads(points_json) # Nhận dictionary { 'S': {'x':..,'y':..}, ... }

        # 2. Tính toán lại góc dựa trên điểm mới
        def get_pt(name): return (pts[name]['x'], pts[name]['y'])
        
        sna = calculate_angle(get_pt("S"), get_pt("N"), get_pt("A"))
        snb = calculate_angle(get_pt("S"), get_pt("N"), get_pt("B"))
        anb = sna - snb

        # Logic chẩn đoán
        if anb > 4.0: diagnosis = f"Chuan Doan Ho (ANB = {anb:.2f} deg)"
        elif anb < 0.0: diagnosis = f"Chuan Doan Mom (ANB = {anb:.2f} deg)"
        else: diagnosis = f"Chuan Doan Binh Thuong (ANB = {anb:.2f} deg)"

        # 3. Vẽ lên ảnh
        draw = ImageDraw.Draw(image)
        line_color = (0, 255, 0) # Xanh lá
        line_width = max(3, int(orig_w * 0.004))
        
        # Vẽ các đường nối
        def draw_l(p1, p2): draw.line([get_pt(p1), get_pt(p2)], fill=line_color, width=line_width)
        draw_l("S", "N"); draw_l("N", "A"); draw_l("N", "B"); draw_l("N", "Pog"); draw_l("Go", "Me")

        # Vẽ điểm và tên
        radius = max(5, int(orig_w * 0.008))
        for name in pts:
            curr_pt = get_pt(name)
            draw.ellipse([curr_pt[0]-radius, curr_pt[1]-radius, curr_pt[0]+radius, curr_pt[1]+radius], fill=(255,0,0))
            # Vẽ text (giả định bạn đã có logic font từ code trước)
            # draw.text(...) 

        # 4. Xuất ảnh
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        raise Exception(f"Lỗi khi vẽ ảnh chỉnh sửa: {str(e)}")
    
def process_and_get_analysis_data(image_bytes: bytes):
    """Dự đoán các điểm và trả về tọa độ cùng kích thước ảnh gốc"""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        orig_w, orig_h = image.size

        # Giả sử IMG_SIZE model của bạn là 512
        img_tensor = image_transforms(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred_hm = model(img_tensor)

        pred_hm_np = pred_hm.squeeze(0).cpu().numpy()
        pred_pts = heatmap_to_coord(pred_hm_np)

        scale_x = orig_w / 512 # Thay 512 bằng IMG_SIZE của bạn
        scale_y = orig_h / 512
        
        landmarks = {}
        for key, idx in LANDMARK_MAP.items():
            landmarks[key] = {
                "x": float(pred_pts[idx][0]) * scale_x,
                "y": float(pred_pts[idx][1]) * scale_y
            }

        return {
            "landmarks": landmarks,
            "image_size": {"width": orig_w, "height": orig_h}
        }
    except Exception as e:
        raise Exception(f"Lỗi phân tích: {str(e)}")

def process_and_draw_analysis_2(image_bytes: bytes) -> bytes:
    """Phân tích Cephalometric, vẽ đường nối S-N-A-B và in kết quả chẩn đoán"""
    try:
        # 1. Đọc và dự đoán điểm
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        orig_w, orig_h = image.size

        img_tensor = image_transforms(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred_hm = model(img_tensor)

        pred_hm_np = pred_hm.squeeze(0).cpu().numpy()
        pred_pts = heatmap_to_coord(pred_hm_np)

        # 2. Scale và lọc ra các điểm quan trọng
        scale_x = orig_w / IMG_SIZE
        scale_y = orig_h / IMG_SIZE
        
        pts = {}
        for key, idx in LANDMARK_MAP.items():
            pts[key] = (float(pred_pts[idx][0]) * scale_x, float(pred_pts[idx][1]) * scale_y)

        # 3. Tính toán góc
        sna = calculate_angle(pts["S"], pts["N"], pts["A"])
        snb = calculate_angle(pts["S"], pts["N"], pts["B"])
        anb = sna - snb

        # 4. Logic chẩn đoán
        if anb > 4.0:
            diagnosis = f"Chẩn Đoán Hô (ANB = {anb:.2f}°)"
        elif anb < 0.0:
            diagnosis = f"Chẩn Đoán Móm (ANB = {anb:.2f}°)"
        else:
            diagnosis = f"Chẩn Đoán Bình Thường (ANB = {anb:.2f}°)"

        # 5. Cài đặt vẽ & Font chữ
        draw = ImageDraw.Draw(image)
        font_size_large = max(20, int(orig_w * 0.035))
        font_size_small = max(14, int(orig_w * 0.025))
        
        try:
            # Ưu tiên load font Arial đậm cho dễ nhìn
            font_title = ImageFont.truetype("arialbd.ttf", font_size_large)
            font_info = ImageFont.truetype("arialbd.ttf", font_size_small)
        except IOError:
            # Fallback nếu không tìm thấy font (trên Linux)
            font_title = font_info = ImageFont.load_default()

        # --- 5.1 Vẽ các đường thẳng (Màu xanh lá) ---
        line_color = (0, 255, 0)
        line_width = max(2, int(orig_w * 0.003))
        
        draw.line([pts["S"], pts["N"]], fill=line_color, width=line_width)
        draw.line([pts["N"], pts["A"]], fill=line_color, width=line_width)
        draw.line([pts["N"], pts["B"]], fill=line_color, width=line_width)
        draw.line([pts["N"], pts["Pog"]], fill=line_color, width=line_width)
        draw.line([pts["Go"], pts["Me"]], fill=line_color, width=line_width)

        # --- 5.2 Vẽ các điểm và tên (Màu đỏ/Trắng) ---
        radius = max(4, int(orig_w * 0.007))
        for name, pt in pts.items():
            cx, cy = pt
            draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(255, 0, 0))
            draw.text((cx + radius + 5, cy - radius - 5), name, fill=(255, 255, 255), font=font_info)

        # 6. Trả về bytes
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        return buf.read()

    except Exception as e:
        raise Exception(f"Lỗi trong quá trình phân tích và vẽ: {str(e)}")
##############################################################################


def process_and_get_analysis_data(image_bytes: bytes) -> dict:
    """Trả về tọa độ các điểm mốc, góc SNA/SNB/ANB và chẩn đoán dạng JSON"""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        orig_w, orig_h = image.size

        img_tensor = image_transforms(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred_hm = model(img_tensor)

        pred_hm_np = pred_hm.squeeze(0).cpu().numpy()
        pred_pts = heatmap_to_coord(pred_hm_np)

        scale_x = orig_w / IMG_SIZE
        scale_y = orig_h / IMG_SIZE

        pts = {}
        for key, idx in LANDMARK_MAP.items():
            pts[key] = (float(pred_pts[idx][0]) * scale_x, float(pred_pts[idx][1]) * scale_y)

        sna = calculate_angle(pts["S"], pts["N"], pts["A"])
        snb = calculate_angle(pts["S"], pts["N"], pts["B"])
        anb = sna - snb

        if anb > 4.0:
            diagnosis = "Hô"
        elif anb < 0.0:
            diagnosis = "Móm"
        else:
            diagnosis = "Bình Thường"

        return {
            "status": "success",
            "landmarks": {
                name: {"x": round(coord[0], 2), "y": round(coord[1], 2)}
                for name, coord in pts.items()
            },
            "angles": {
                "SNA": round(sna, 4),
                "SNB": round(snb, 4),
                "ANB": round(anb, 4),
            },
            "diagnosis": diagnosis,
            "image_size": {"width": orig_w, "height": orig_h},
        }

    except Exception as e:
        raise Exception(f"Lỗi trong quá trình phân tích dữ liệu: {str(e)}")


# =========================================================================
# 5. PHÂN TÍCH CEPHALOMETRIC AI CHUYÊN SÂU (GPT-4o)
# =========================================================================

async def get_ceph_ai_analysis_from_image(image_bytes: bytes) -> dict:
    """
    Chạy LRC model trên ảnh ceph → tính góc → gọi GPT-4o phân tích chuyên sâu.
    Trả về dict tương ứng CephAiAnalysis schema.
    """
    import asyncio, json, os
    from openai import AsyncOpenAI

    # Bước 1: lấy góc từ mô hình LRC
    data = process_and_get_analysis_data(image_bytes)
    sna = data["angles"]["SNA"]
    snb = data["angles"]["SNB"]
    anb = data["angles"]["ANB"]
    diagnosis = data["diagnosis"]       # "Hô" | "Móm" | "Bình Thường"

    sna_status = "bình thường" if 80 <= sna <= 84 else ("cao" if sna > 84 else "thấp")
    snb_status = "bình thường" if 78 <= snb <= 82 else ("cao" if snb > 82 else "thấp")
    anb_status = "bình thường" if 0 <= anb <= 4 else ("dương tính tăng" if anb > 4 else "âm tính")

    if anb > 4.0:
        conclusion = "Khớp cắn Loại II (Class II) — Hô"
        conclusion_detail = f"ANB = {anb:.2f}° — Xương hàm trên nhô ra trước so với hàm dưới"
    elif anb < 0.0:
        conclusion = "Khớp cắn Loại III (Class III) — Móm"
        conclusion_detail = f"ANB = {anb:.2f}° — Xương hàm dưới nhô ra trước so với hàm trên"
    else:
        conclusion = "Khớp cắn Loại I (Class I)"
        conclusion_detail = f"ANB = {anb:.2f}° — Không có dấu hiệu hô hoặc móm đáng kể"

    # Bước 2: gọi GPT-4o
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return {
            "skeletal_summary": "Chưa cấu hình OPENAI_API_KEY — không thể tạo phân tích AI.",
            "sna_interpretation": f"SNA = {sna:.2f}° ({sna_status})",
            "snb_interpretation": f"SNB = {snb:.2f}° ({snb_status})",
            "anb_interpretation": f"ANB = {anb:.2f}° ({anb_status}). {conclusion_detail}",
            "clinical_implications": [],
            "treatment_plan": [],
            "severity": "low",
        }

    system_prompt = (
        "Bạn là chuyên gia phân tích Cephalometric X-Ray với hơn 20 năm kinh nghiệm "
        "trong lĩnh vực Chỉnh nha (Orthodontics) và Phẫu thuật hàm mặt (Orthognathic Surgery). "
        "Hãy phân tích kết quả đo đạc Cephalometric bằng tiếng Việt theo chuẩn mực lâm sàng, "
        "trả về CHÍNH XÁC JSON theo schema được yêu cầu, không thêm văn bản nào ngoài JSON."
    )

    user_prompt = (
        "Phân tích kết quả Cephalometric X-Ray sau đây (được đo tự động bởi mô hình AI HRNetW32):\n\n"
        f"• SNA = {sna:.2f}° (Chuẩn: 82 ± 2°) → {sna_status}\n"
        f"• SNB = {snb:.2f}° (Chuẩn: 80 ± 2°) → {snb_status}\n"
        f"• ANB = {anb:.2f}° (Chuẩn: 2 ± 2°) → {anb_status}\n"
        f"• Chẩn đoán phân loại xương: {diagnosis} ({conclusion})\n"
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
        severity = str(parsed.get("severity", "medium")).lower()
        if severity not in ("low", "medium", "high"):
            severity = "medium"

        return {
            "skeletal_summary":    parsed.get("skeletal_summary", ""),
            "sna_interpretation":  parsed.get("sna_interpretation", ""),
            "snb_interpretation":  parsed.get("snb_interpretation", ""),
            "anb_interpretation":  parsed.get("anb_interpretation", ""),
            "clinical_implications": clinical,
            "treatment_plan":      treatment,
            "severity":            severity,
        }

    except json.JSONDecodeError:
        return {
            "skeletal_summary": "Không thể phân tích phản hồi JSON từ AI.",
            "sna_interpretation": f"SNA = {sna:.2f}°",
            "snb_interpretation": f"SNB = {snb:.2f}°",
            "anb_interpretation": f"ANB = {anb:.2f}°. {conclusion_detail}",
            "clinical_implications": [],
            "treatment_plan": [],
            "severity": "medium",
        }
    except Exception as exc:
        return {
            "skeletal_summary": f"Lỗi khi kết nối AI: {exc}",
            "sna_interpretation": f"SNA = {sna:.2f}°",
            "snb_interpretation": f"SNB = {snb:.2f}°",
            "anb_interpretation": f"ANB = {anb:.2f}°. {conclusion_detail}",
            "clinical_implications": [],
            "treatment_plan": [],
            "severity": "medium",
        }