# live_detector.py
import cv2
import time
import numpy as np
from collections import deque
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision import hand_landmarker
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model
import json


from features import extract_hand_features  # your features.py


# -----------------------------
# Load Model
# -----------------------------
model = load_model("sign_model.h5")  


# -----------------------------
# Read label_map.json
# -----------------------------
with open("label_map.json", "r") as f:
    label_map = json.load(f)

idx_to_label = {v: k for k, v in label_map.items()}


# -----------------------------
# Config
# -----------------------------
SEQ_LEN = 30
MOTION_THRESHOLD = 0.015

#Prediction config
PRED_THRESHOLD = 0.75
prediction_history = deque(maxlen=10)
current_prediction = ""
current_confidence = 0.0

# Hand connections (same as MediaPipe)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]


# -----------------------------
# Prediction function
# -----------------------------
def predict_from_buffer(model, buffer):
    if len(buffer) < SEQ_LEN:
        return None, 0.0

    seq = np.expand_dims(np.array(buffer), axis=0)  # (1, 30, 126)
    probs = model.predict(seq, verbose=0)[0]

    pred_idx = np.argmax(probs)
    confidence = probs[pred_idx]

    return pred_idx, confidence

# -----------------------------
# MediaPipe HandLandmarker
# -----------------------------
base = base_options.BaseOptions(model_asset_path="hand_landmarker.task")
options = hand_landmarker.HandLandmarkerOptions(
    base_options=base,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2  # detect both hands
)
detector = hand_landmarker.HandLandmarker.create_from_options(options)

# -----------------------------
# Buffers
# -----------------------------
buffer = deque(maxlen=SEQ_LEN)
prev_landmarks_left = None
prev_landmarks_right = None

# -----------------------------
# Open webcam
# -----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

# Warm-up
for _ in range(10):
    cap.read()

# -----------------------------
# Live loop
# -----------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
    timestamp = int(time.time() * 1000)

    results = detector.detect_for_video(mp_image, timestamp)

    # -----------------------------
    # Draw hands
    # -----------------------------
    if results.hand_landmarks:
        for hand in results.hand_landmarks:
            # Draw points
            for lm in hand:
                x_px = int(lm.x * frame.shape[1])
                y_px = int(lm.y * frame.shape[0])
                cv2.circle(frame, (x_px, y_px), 4, (0, 255, 0), -1)

            # Draw connections
            for start_idx, end_idx in HAND_CONNECTIONS:
                start = hand[start_idx]
                end = hand[end_idx]
                x1, y1 = int(start.x * frame.shape[1]), int(start.y * frame.shape[0])
                x2, y2 = int(end.x * frame.shape[1]), int(end.y * frame.shape[0])
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # -----------------------------
    # Feature extraction per hand
    # -----------------------------
    left_hand = results.hand_landmarks[0] if len(results.hand_landmarks) > 0 else None
    right_hand = results.hand_landmarks[1] if len(results.hand_landmarks) > 1 else None

    # Left hand
    if left_hand:
        if prev_landmarks_left is not None:
            motion_left = np.mean([
                abs(lm.x - plm.x) + abs(lm.y - plm.y) + abs(lm.z - plm.z)
                for lm, plm in zip(left_hand, prev_landmarks_left)
            ])
            is_static_left = motion_left < MOTION_THRESHOLD
        else:
            motion_left = 0.0
            is_static_left = False

        features_left, coords_left = extract_hand_features(
            hand_landmarks=left_hand,
            prev_landmarks=prev_landmarks_left,
            is_static=is_static_left
        )
        prev_landmarks_left = left_hand
        buffer.append(features_left)

        cv2.putText(frame, f"Left Motion: {motion_left:.4f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Left Static: {is_static_left}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Right hand
    if right_hand:
        if prev_landmarks_right is not None:
            motion_right = np.mean([
                abs(lm.x - plm.x) + abs(lm.y - plm.y) + abs(lm.z - plm.z)
                for lm, plm in zip(right_hand, prev_landmarks_right)
            ])
            is_static_right = motion_right < MOTION_THRESHOLD
        else:
            motion_right = 0.0
            is_static_right = False

        features_right, coords_right = extract_hand_features(
            hand_landmarks=right_hand,
            prev_landmarks=prev_landmarks_right,
            is_static=is_static_right
        )
        prev_landmarks_right = right_hand
        buffer.append(features_right)

        cv2.putText(frame, f"Right Motion: {motion_right:.4f}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Right Static: {is_static_right}", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
    
    # -----------------------------
    # Prediction
    # -----------------------------
    pred_idx, confidence = predict_from_buffer(model, buffer)

    if pred_idx is not None and confidence > PRED_THRESHOLD:
        prediction_history.append(pred_idx)

        # Majority vote (anti-flicker)
        if prediction_history.count(pred_idx) > len(prediction_history) // 2:
            current_prediction = idx_to_label[pred_idx]
            current_confidence = confidence

    # -----------------------------
    # Display
    # -----------------------------
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"Sign: {current_prediction} ({current_confidence:.2f})",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )



cap.release()
cv2.destroyAllWindows()
