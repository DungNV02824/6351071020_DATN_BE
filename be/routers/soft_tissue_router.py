
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import Response
import json
from services.lrc_services import process_and_predict, process_morphing_tps

from services.lrc_services import process_and_predict
from services.lrc_services import process_and_predict, process_and_draw, process_and_draw_analysis_2, process_and_get_analysis_data, get_ceph_ai_analysis_from_image,draw_analysis_from_custom_points

from fastapi.responses import JSONResponse
import base64
router = APIRouter(prefix="/api/soft-tissue", tags=["Soft Tissue"])


    
@router.post("/predict_analysis_image", summary="Trả về ảnh X-Quang đã phân tích Hô/Móm (Đường nối, Góc SNA, SNB, ANB)")
async def predict_analysis_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file ảnh (JPG, PNG)")

    try:
        image_bytes = await file.read()
        # Gọi hàm AI xử lý và phân tích
        result_image_bytes = process_and_draw_analysis_2(image_bytes)
        
        # Trả về trực tiếp ảnh để xem trên Postman/Swagger
        return Response(content=result_image_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


from services.lrc_services import process_and_get_analysis_data # Import hàm mới

@router.post("/predict_analysis_data", summary="Trả về tọa độ các điểm mốc để tương tác trên FE")
async def predict_analysis_data(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file ảnh")

    try:
        image_bytes = await file.read()
        # Gọi hàm service để lấy tọa độ JSON
        data = process_and_get_analysis_data(image_bytes)
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/render_adjusted_image")
async def render_adjusted_image(file: UploadFile = File(...), points: str = Form(...)):
    """Nhận ảnh và tọa độ đã sửa, trả về ảnh đã vẽ hoàn chỉnh"""
    try:
        image_bytes = await file.read()
        # Gọi hàm vẽ từ points tùy chỉnh
        result_image = draw_analysis_from_custom_points(image_bytes, points)
        return Response(content=result_image, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/simulate_morphing")
async def simulate_morphing(
    file: UploadFile = File(...), 
    initial_points: str = Form(...), 
    target_points: str = Form(...)
):
    try:
        image_bytes = await file.read()
        # Chuyển string JSON từ Frontend thành Dictionary Python
        src_pts = json.loads(initial_points)
        dst_pts = json.loads(target_points)
        
        # Gọi hàm nắn ảnh
        morphed_bytes = process_morphing_tps(image_bytes, src_pts, dst_pts)
        
        return Response(content=morphed_bytes, media_type="image/png")
    except Exception as e:
        print(f"Lỗi API simulate_morphing: {e}")
        raise HTTPException(status_code=500, detail=str(e))