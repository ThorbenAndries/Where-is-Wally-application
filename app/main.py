import os
import io
import uuid
import math
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n.pt")
WALLY_CLASS = os.getenv("WALLY_CLASS", "wally")
model = None
try:
    model = YOLO(MODEL_PATH)
except Exception:
    # model may be downloaded at first run; keep None for now
    model = None

# simple in-memory store: image_id -> {'bbox': (x1,y1,x2,y2)}
store = {}

@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    global model

    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    img.save(tmp_path)

    if model is None:
        # lazy-load model
        try:
            model = YOLO(MODEL_PATH)
        except Exception:
            os.remove(tmp_path)
            raise HTTPException(status_code=500, detail="Model loading failed")

    results = model(tmp_path)
    os.remove(tmp_path)

    # parse boxes (pick highest-confidence box)
    boxes = []
    try:
        r = results[0]
        names = getattr(r, "names", {}) or {}
        for b in r.boxes:
            xyxy = b.xyxy[0].tolist()
            conf = float(b.conf[0]) if hasattr(b, 'conf') else 0.0
            cls_idx = int(b.cls[0]) if hasattr(b, "cls") else -1
            cls_name = str(names.get(cls_idx, "")).lower()
            boxes.append((xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf, cls_name))
    except Exception:
        pass

    image_id = uuid.uuid4().hex
    if not boxes:
        store[image_id] = { 'bbox': None }
        return { 'image_id': image_id, 'status': 'no-detection' }

    # If the model has a class named "wally", prefer those detections.
    wally_boxes = [b for b in boxes if b[5] == WALLY_CLASS.lower()]
    selected = wally_boxes if wally_boxes else boxes
    selected.sort(key=lambda x: x[4], reverse=True)
    x1, y1, x2, y2, _, _ = selected[0]
    store[image_id] = { 'bbox': (x1,y1,x2,y2) }
    return { 'image_id': image_id, 'status': 'ok' }

class ClickPayload(BaseModel):
    image_id: str
    x: float
    y: float

@app.post("/click")
def click(payload: ClickPayload):
    entry = store.get(payload.image_id)
    if not entry:
        raise HTTPException(status_code=404, detail="image_id not found")
    bbox = entry.get('bbox')
    if bbox is None:
        return { 'feedback': 'Geen detectie' }

    x1,y1,x2,y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    dist = math.hypot(payload.x - cx, payload.y - cy)
    diag = math.hypot(x2 - x1, y2 - y1)

    if diag == 0:
        feedback = 'Gevonden'
    elif dist <= 0.2 * diag:
        feedback = 'Gevonden'
    elif dist <= 0.5 * diag:
        feedback = 'Dichtbij'
    else:
        feedback = 'Ver weg'

    return { 'feedback': feedback }
