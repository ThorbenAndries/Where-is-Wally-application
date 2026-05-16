import os
import io
import uuid
import math
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL_PATH = os.getenv("MODEL_PATH", "models/mascot.pt" if os.path.exists("models/mascot.pt") else "yolov8n.pt")
TARGET_CLASS = os.getenv("TARGET_CLASS", os.getenv("WALLY_CLASS", "mascot"))
MASCOT_PATH = os.getenv("MASCOT_PATH", "assets/mascot.png")
INFERENCE_IMAGE_SIZE = int(os.getenv("INFERENCE_IMAGE_SIZE", "1280"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.05"))
TILE_SIZE = int(os.getenv("TILE_SIZE", "1280"))
TILE_OVERLAP = int(os.getenv("TILE_OVERLAP", "256"))
FULL_IMAGE_CONFIDENCE = float(os.getenv("FULL_IMAGE_CONFIDENCE", "0.12"))
model = None
try:
    model = YOLO(MODEL_PATH)
except Exception:
    # model may be downloaded at first run; keep None for now
    model = None

# simple in-memory store: image_id -> {'bbox': (x1,y1,x2,y2), 'confidence': float, 'class_name': str}
store = {}


def _result_boxes(
    result,
    offset_x: int = 0,
    offset_y: int = 0,
    source: str = "full",
) -> list[tuple[float, float, float, float, float, str, str]]:
    boxes = []
    names = getattr(result, "names", {}) or {}
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0]) if hasattr(box, "conf") else 0.0
        cls_idx = int(box.cls[0]) if hasattr(box, "cls") else -1
        cls_name = str(names.get(cls_idx, "")).lower()
        boxes.append((x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y, conf, cls_name, source))
    return boxes


def _tile_origins(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]

    stride = max(1, tile_size - overlap)
    origins = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if origins[-1] != last:
        origins.append(last)
    return origins


def _run_detection(image: Image.Image) -> list[tuple[float, float, float, float, float, str, str]]:
    temp_paths: list[str] = []
    tile_offsets: list[tuple[int, int]] = []
    sources: list[str] = []

    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            full_path = tmp.name
        image.save(full_path, quality=92)
        temp_paths.append(full_path)
        tile_offsets.append((0, 0))
        sources.append("full")

        if max(image.size) > TILE_SIZE:
            x_origins = _tile_origins(image.width, TILE_SIZE, TILE_OVERLAP)
            y_origins = _tile_origins(image.height, TILE_SIZE, TILE_OVERLAP)
            for y in y_origins:
                for x in x_origins:
                    tile = image.crop((x, y, min(x + TILE_SIZE, image.width), min(y + TILE_SIZE, image.height)))
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                        tile_path = tmp.name
                    tile.save(tile_path, quality=92)
                    temp_paths.append(tile_path)
                    tile_offsets.append((x, y))
                    sources.append("tile")

        results = model(
            temp_paths,
            imgsz=INFERENCE_IMAGE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
        )

        boxes = []
        for result, (offset_x, offset_y), source in zip(results, tile_offsets, sources):
            boxes.extend(_result_boxes(result, offset_x, offset_y, source))
        return boxes
    finally:
        for path in temp_paths:
            try:
                os.remove(path)
            except OSError:
                pass


@app.get("/mascot")
def download_mascot():
    if not os.path.exists(MASCOT_PATH):
        raise HTTPException(status_code=404, detail="Mascot image not found")
    return FileResponse(
        MASCOT_PATH,
        media_type="image/png",
        filename=os.path.basename(MASCOT_PATH),
    )


def _detection_score(
    box: tuple[float, float, float, float, float, str, str],
    image_width: int,
    image_height: int,
) -> float:
    x1, y1, x2, y2, confidence, _, source = box
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    area_ratio = (width * height) / max(1.0, image_width * image_height)
    aspect_ratio = height / width

    area_bonus = min(area_ratio / 0.004, 1.0) * 0.25
    shape_bonus = 0.12 if aspect_ratio >= 0.9 else -0.2
    source_bonus = 0.15 if source == "full" else 0.0
    return confidence + area_bonus + shape_bonus + source_bonus

@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    global model

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")

    if model is None:
        # lazy-load model
        try:
            model = YOLO(MODEL_PATH)
        except Exception:
            raise HTTPException(status_code=500, detail="Model loading failed")

    try:
        boxes = _run_detection(img)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Detection failed: {exc}")

    image_id = uuid.uuid4().hex
    if not boxes:
        store[image_id] = { 'bbox': None, 'confidence': None, 'class_name': None }
        return { 'image_id': image_id, 'status': 'no-detection', 'detection': None }

    # Prefer detections for the target class, but keep the app usable with generic models.
    target_boxes = [b for b in boxes if b[5] == TARGET_CLASS.lower()]
    selected = target_boxes if target_boxes else boxes
    full_image_boxes = [b for b in selected if b[6] == "full" and b[4] >= FULL_IMAGE_CONFIDENCE]
    candidates = full_image_boxes if full_image_boxes else selected
    candidates.sort(key=lambda x: _detection_score(x, img.width, img.height), reverse=True)
    x1, y1, x2, y2, confidence, class_name, source = candidates[0]
    detection = {
        'bbox': {
            'x1': round(x1, 2),
            'y1': round(y1, 2),
            'x2': round(x2, 2),
            'y2': round(y2, 2),
        },
        'confidence': round(confidence, 4),
        'class_name': class_name or TARGET_CLASS,
        'source': source,
    }
    store[image_id] = {
        'bbox': (x1, y1, x2, y2),
        'confidence': confidence,
        'class_name': class_name,
    }
    return { 'image_id': image_id, 'status': 'ok', 'detection': detection }

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

    inside_bbox = x1 <= payload.x <= x2 and y1 <= payload.y <= y2

    if inside_bbox:
        feedback = 'Gevonden'
    elif diag == 0:
        feedback = 'Gevonden'
    elif dist <= 0.35 * diag:
        feedback = 'Gevonden'
    elif dist <= 0.8 * diag:
        feedback = 'Dichtbij'
    else:
        feedback = 'Ver weg'

    return { 'feedback': feedback }
