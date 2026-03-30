# prepare_data.py

# Takes all videos, extract features from every frame and bui;d X.npy and y.npy for trainning 

import os
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import json
from features import extract_frame_features

# =============================
# CONFIG
# =============================
DATASET_PATH = "video_dataset"
SEQ_LEN = 30
STATIC_MOTION_THRESHOLD = 0.01

STATIC_SIGNS = {
    "A","B","C","D","E","F","G","H","I","K",
    "L","M","N","O","P","Q","R","S","T",
    "U","V","W","X","Y"
}

MOTION_SIGNS = {"J", "Love_u", "Hello", "Sorry", "Z"}

# =============================
# MEDIAPIPE
# =============================
mp_holistic = mp.solutions.holistic

# =============================
# STATIC VIDEO PROCESSING
# =============================
def process_static_video(video_path, holistic):
    cap = cv2.VideoCapture(video_path)

    stable_frames = []
    prev_feats = None
    prev_left = None
    prev_right = None
    prev_pose = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        feats, prev_left, prev_right, prev_pose = extract_frame_features(
            results,
            prev_left,
            prev_right,
            prev_pose,
            is_static=False
        )

        # Compare ONLY position part (ignore velocity)
        if prev_feats is not None:
            position_size = 21*3*2 + 6*3  # left + right + pose (xyz only)
            motion = np.mean(np.abs(
                feats[:position_size] - prev_feats[:position_size]
            ))

            if motion < STATIC_MOTION_THRESHOLD:
                stable_frames.append(feats)

        prev_feats = feats

    cap.release()

    if len(stable_frames) == 0 and prev_feats is not None:
        stable_frames.append(prev_feats)

    avg_frame = np.mean(stable_frames, axis=0)

    return np.tile(avg_frame, (SEQ_LEN, 1))


# =============================
# MOTION VIDEO PROCESSING
# =============================
def process_motion_video(video_path, holistic):
    cap = cv2.VideoCapture(video_path)

    frames = []
    prev_left = None
    prev_right = None
    prev_pose = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        feats, prev_left, prev_right, prev_pose = extract_frame_features(
            results,
            prev_left,
            prev_right,
            prev_pose,
            is_static=False
        )

        frames.append(feats)

    cap.release()

    frames = np.array(frames)

    if len(frames) >= SEQ_LEN:
        return frames[:SEQ_LEN]
    else:
        pad = np.zeros((SEQ_LEN - len(frames), frames.shape[1]))
        return np.vstack([frames, pad])


# =============================
# DATASET BUILDER
# =============================
def prepare_dataset():
    actions = sorted([
        f for f in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, f))
    ])

    label_map = {action: idx for idx, action in enumerate(actions)}

    X, y = [], []

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        for action in actions:
            action_path = os.path.join(DATASET_PATH, action)

            videos = [v for v in os.listdir(action_path) if v.endswith(".mp4")]

            if not videos:
                print(f"⚠️ No videos found for sign '{action}', skipping...")
                continue

            for video in tqdm(videos, desc=f"Processing {action}"):
                video_path = os.path.join(action_path, video)

                try:
                    if action in STATIC_SIGNS:
                        sequence = process_static_video(video_path, holistic)
                    else:
                        sequence = process_motion_video(video_path, holistic)

                    X.append(sequence)
                    y.append(label_map[action])

                except Exception as e:
                    print(f"❌ Error processing {video_path}: {e}")
                    continue

    X = np.array(X)
    y = to_categorical(y, num_classes=len(actions))

    return X, y, label_map


# =============================
# MAIN
# =============================
if __name__ == "__main__":
    X, y, label_map = prepare_dataset()

    np.save("X.npy", X)
    np.save("y.npy", y)

    print("✅ Dataset prepared")
    print("X shape:", X.shape)   # Should be (N, 30, 288)
    print("y shape:", y.shape)
    print("Label map:", label_map)

    with open("label_map.json", "w") as f:
        json.dump(label_map, f)

    print("💾 label_map.json saved")