# models/

Place pre-downloaded model weights here (optional).

## YOLOv8 face model
Download `yolov8n-face.pt` from:
https://github.com/akanametov/yolov8-face/releases

Place it here as: `models/yolov8n-face.pt`
Then update config.json:  "yolo_model": "models/yolov8n-face.pt"

## InsightFace (buffalo_l)
Downloaded automatically on first run to ~/.insightface/
No manual step needed.
