from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torchvision.transforms as T
import io
from pest_rec.models.insect_pest_module import InsectPestLitModule

model = InsectPestLitModule.load_from_checkpoint("last.ckpt", map_location="cpu")
model.eval()

with open("classes.txt", "r", encoding="utf-8") as f:
    class_names = [line.strip().split("  ")[1] for line in f.readlines()]

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    image_data = await image.read()
    img = Image.open(io.BytesIO(image_data)).convert("RGB")
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        pred = output.argmax(dim=1).item()
        prob = torch.softmax(output, dim=1)[0][pred].item()

    return {
        "result": class_names[pred],
        "confidence": round(prob, 4),
        "tips": "이 병해충은 조기 방제가 중요하며, 정기적인 예찰을 권장합니다."
    }
