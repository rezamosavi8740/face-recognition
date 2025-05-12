import os, sys

# suppress any prints during the import
_devnull = open(os.devnull, 'w')
_old_stdout, _old_stderr = sys.stdout, sys.stderr
sys.stdout, sys.stderr = _devnull, _devnull

from ultralytics import YOLO
# …
# restore real stdout/stderr
sys.stdout, sys.stderr = _old_stdout, _old_stderr

TRITON_SERVER_URL = "localhost:{port_}".format(port_=8000)
name = f"http://{TRITON_SERVER_URL}"

TRITON_MODEL_NAME =  "face_detection_small"  # Change this to match your deployed model name
face_model_link = f"{name}/{TRITON_MODEL_NAME}"
model = YOLO(face_model_link, task="detect")

def do_face_detect(image):
    results = model.predict(
        image,
        imgsz=(288, 352),
        classes=[0],
        conf=0.30,
        max_det=1,
        verbose=False
    )

    boxes = results[0].boxes.xyxy.cpu().numpy()
    if len(boxes) == 0:        # ← NEW: nothing detected
        return None

    box = boxes[0]             # first (and only) box
    return [int(x) for x in box]
