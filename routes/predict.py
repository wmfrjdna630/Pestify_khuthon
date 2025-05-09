from fastapi import APIRouter, UploadFile, File, Form
from models.dummy_model import analyze_image

router = APIRouter(prefix="/predict")

@router.post("")
async def predict_image(
    image: UploadFile = File(...),
    userId: str = Form(...),
    lat: float = Form(...),
    lng: float = Form(...)
):
    result = analyze_image(image)
    return {
        "result": result["label"],
        "confidence": result["confidence"],
        "tips": result["tip"],
        "location": {"lat": lat, "lng": lng},
        "alertSent": True
    }