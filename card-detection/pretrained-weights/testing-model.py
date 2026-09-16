# Base code provided by Dipankar Medhi article https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993
# Note press Q to stop the demo
# MODEL_PATH = "Z:\\coding\\PR project\\card-detection\\runs\\detect\\train13\\weights\\best.pt"
MODEL_PATH = "Z:\\coding\\PR project\\card-detection\\pretrained-weights\\yolov8m_synthetic.pt"
VIDEO_PATH = "Z:\\coding\\PR project\\29-card-game\\29-gameplay-1.mp4"


import math
import sys
from ultralytics import YOLO
import cv2

# Change to 'tuned' to use it as the default one
DEFAULT_MODEL = "synthetic"
SHOW_CONFIDENCE = False

configuration_dict = {
    "synthetic": {
        "model_path": MODEL_PATH,
        "class_names": [
            "10c",
            "10d",
            "10h",
            "10s",
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
        "class_names": ["10h", "7h", "8h", "9h", "Ah", "Jh", "Kh", "Qh"],
    },
}

print("Loading application...")

configuration_model = sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_MODEL

if configuration_model not in configuration_dict.keys():
    print(f"Allowed parameters for model are {configuration_dict.keys()}. Defaulting to {DEFAULT_MODEL}...")
    configuration_model = DEFAULT_MODEL

current_config = configuration_dict.get(configuration_model)

# Load the model and class names
model = YOLO(current_config["model_path"])
classNames = current_config["class_names"]

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)


# Card values mapping
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

# '7S': 0, '8S': 1, '9S': 2, '10S': 3, 'JS': 4, 'QS': 5, 'KS': 6, 'AS': 7, '7H': 8, '8H': 9, '9H': 10, '10H': 11, 'JH': 12, 'QH': 13, 'KH': 14, 'AH': 15, '7D': 16, '8D': 17, '9D': 18, '10D': 19, 'JD': 20, 'QD': 21, 'KD': 22, 'AD': 23, '7C': 24, '8C': 25, '9C': 26, '10C': 27, 'JC': 28, 'QC': 29, 'KC': 30, 'AC': 31}

window_title = f"Playing Cards Detection - Model: {configuration_model}"

while True:
    success, img = cap.read()
    results = model(img, stream=True)

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

    cv2.imshow(window_title, img)
    if cv2.waitKey(1) == ord("q"):
        break
    if cv2.waitKey(1) == ord("s"):
        SHOW_CONFIDENCE = not SHOW_CONFIDENCE


cap.release()
cv2.destroyAllWindows()