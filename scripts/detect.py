from ultralytics import YOLO
import cv2
import os

# Loading COCO pretrained YOLOv8 nano
model = YOLO("yolov8n.pt")

image_dir = "../elephant"

for img_name in os.listdir(image_dir):
    img_path = os.path.join(image_dir, img_name)

    img = cv2.imread(img_path)
    if img is None:
        continue

    results = model(img)

    for box in results[0].boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)

        # The COCO elephant class ID = 20
        if cls_id == 20:
            print(f"[{img_name}] Elephant detected | confidence = {conf:.2f}")

    # Saving the output
    results[0].save(filename=f"output_{img_name}")
