# live_detector.py

import cv2
import numpy as np
from collections import deque
import mediapipe as mp
from tensorflow.keras.models import load_model
import json

from features import extract_frame_features


# -----------------------------
# Load Model
# -----------------------------
model = load_model("sign_model_best.h5")

with open("label_map.json", "r") as f:
    label_map = json.load(f)

idx_to_label = {v: k for k, v in label_map.items()}


# -----------------------------
# Config
# -----------------------------
SEQ_LEN = 30
PRED_THRESHOLD = 0.75


# -----------------------------
# Buffers
# -----------------------------
buffer = deque(maxlen=SEQ_LEN)
prediction_history = deque(maxlen=10)

current_prediction = ""
current_confidence = 0.0

prev_left = None
prev_right = None
prev_pose = None


# -----------------------------
# Prediction Function
# -----------------------------
def predict_from_buffer(model, buffer):
    if len(buffer) < SEQ_LEN:
        return None, 0.0

    seq = np.expand_dims(np.array(buffer), axis=0)
    probs = model.predict(seq, verbose=0)[0]

    pred_idx = np.argmax(probs)
    confidence = probs[pred_idx]

    return pred_idx, confidence


# -----------------------------
# MediaPipe Holistic Setup
# -----------------------------
mp_holistic = mp.solutions.holistic

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    refine_face_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils


# -----------------------------
# Open Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()


# -----------------------------
# Live Loop
# -----------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)

    # -----------------------------
    # Draw Landmarks
    # -----------------------------
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS
        )

    # -----------------------------
    # Extract Features
    # -----------------------------
    features, prev_left, prev_right, prev_pose = extract_frame_features(
        results,
        prev_left,
        prev_right,
        prev_pose,
        is_static=False
    )

    buffer.append(features)

    # -----------------------------
    # Prediction
    # -----------------------------
    pred_idx, confidence = predict_from_buffer(model, buffer)

    if pred_idx is not None and confidence > PRED_THRESHOLD:
        prediction_history.append(pred_idx)

        # Majority vote smoothing
        if prediction_history.count(pred_idx) > len(prediction_history) // 2:
            current_prediction = idx_to_label[pred_idx]
            current_confidence = confidence

    # -----------------------------
    # Display
    # -----------------------------
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)

    cv2.putText(
        frame,
        f"Sign: {current_prediction} ({current_confidence:.2f})",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Live Sign Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
holistic.close()
cv2.destroyAllWindows()