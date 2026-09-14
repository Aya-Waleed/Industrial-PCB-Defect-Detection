# Inference Instructions
**Model:** YOLO11n — `best.pt`
**Image size used in training:** 320x320

## 1. Requirements

```bash
pip install ultralytics
```

## 2. Load the model

```python
from ultralytics import YOLO

model = YOLO("model/best.pt")
```

## 3. Run inference on a single image

```python
results = model.predict(
    source="path/to/pcb_image.jpg",
    imgsz=320,
    conf=0.25,   # confidence threshold — adjust as needed
)

result = results[0]

for box in result.boxes:
    cls_id = int(box.cls[0])
    class_name = model.names[cls_id]
    confidence = float(box.conf[0])
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    print(class_name, confidence, (x1, y1, x2, y2))
```

## 4. Notes for integration (Saif)

- `model.names` already contains the class ID → name mapping (also duplicated
  in `model/class_names.txt` for convenience).
- If the input image has no defects, `result.boxes` will simply be empty —
  handle this case explicitly in the app (show a "No defects detected" message
  instead of an error).
- `conf` (confidence threshold) and `iou` (NMS IoU threshold) can be exposed
  as adjustable sliders in the Streamlit UI.
- `result.plot()` returns an image (numpy array, BGR) with boxes already drawn,
  which can be shown directly in Streamlit — or boxes can be drawn manually for
  more control over labels/colors (see `src/inference.py`).
