import math
import sys
from ultralytics import YOLO
import cv2

MODEL_PATH = "Z:\\coding\\PR project\\card-detection\\runs\\detect\\train13\\weights\\best.pt"
VIDEO_PATH = "Z:\\coding\\PR project\\29-card-game\\29-gameplay-1.mp4"


configuration_dict = {
    "synthetic": {
        "model_path": MODEL_PATH,
        "class_names": [
            "10c",
            "10d",
            "10h",
            "10s",
            "2c",
            "2d",
            "2h",
            "2s",
            "3c",
            "3d",
            "3h",
            "3s",
            "4c",
            "4d",
            "4h",
            "4s",
            "5c",
            "5d",
            "5h",
            "5s",
            "6c",
            "6d",
            "6h",
            "6s",
            "7c",
            "7d",
            "7h",
            "7s",
            "8c",
            "8d",
            "8h",
            "8s",
            "9c",
            "9d",
            "9h",
            "9s",
            "Ac",
            "Ad",
            "Ah",
            "As",
            "Jc",
            "Jd",
            "Jh",
            "Js",
            "Kc",
            "Kd",
            "Kh",
            "Ks",
            "Qc",
            "Qd",
            "Qh",
            "Qs",
        ],
    },
    "tuned": {
        "model_path": MODEL_PATH,
        "class_names": ["10h", "2h", "3h", "4h", "5h", "6h", "7h", "8h", "9h", "Ah", "Jh", "Kh", "Qh"],
    },
}



card_values = {
    "7c": 7,
    "7d": 7,
    "7h": 7,
    "7s": 7,
    "8c": 8,
    "8d": 8,
    "8h": 8,
    "8s": 8,
    "9c": 9,
    "9d": 9,
    "9h": 9,
    "9s": 9,
    "10c": 10,
    "10d": 10,
    "10h": 10,
    "10s": 10,
    "Ac": 1,
    "Ad": 1,
    "Ah": 1,
    "As": 1,
    "Jc": 10,
    "Jd": 10,
    "Jh": 10,
    "Js": 10,
    "Kc": 10,
    "Kd": 10,
    "Kh": 10,
    "Ks": 10,
    "Qc": 10,
    "Qd": 10,
    "Qh": 10,
    "Qs": 10,
}

print("Loading application...")

configuration_model = sys.argv[1] if len(sys.argv) >= 2 else "synthetic"
print(configuration_model)

if configuration_model not in configuration_dict.keys():
    print(f"Allowed parameters for model are {configuration_dict.keys()}. Defaulting to synthetic...")
    configuration_model = "synthetic"

current_config = configuration_dict.get(configuration_model)

model = YOLO(current_config["model_path"])
classNames = current_config["class_names"]
SHOW_CONFIDENCE = True

img_path = "Z:\\coding\\PR project\\dataset\\images\\val\\000026.jpg"
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Could not read image: {img_path}")


results = model(img_path, stream=True)

total_score = 0

    # Coordinates
for r in results:
    boxes = r.boxes

    for box in boxes:
        # Bounding box
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values

            # Put box in cam
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Confidence
        confidence = math.ceil((box.conf[0] * 100)) / 100
        print("Confidence --->", confidence)

            # Class name
        cls = int(box.cls[0])
        class_name = classNames[cls]
        print("Class name -->", class_name)

            # Add card value to total score
        total_score += card_values.get(class_name, 0)

            # Object details
        org = [x1, y1]
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2
        display_text = class_name if not SHOW_CONFIDENCE else f"{class_name} {confidence}"
        cv2.putText(img, display_text, org, font, fontScale, color, thickness)

    # Display total score on the screen
    score_text = f"Total Score: {total_score}"
    cv2.putText(img, score_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if cv2.waitKey(1) == ord("q"):
        break
    if cv2.waitKey(1) == ord("s"):
        SHOW_CONFIDENCE = not SHOW_CONFIDENCE
