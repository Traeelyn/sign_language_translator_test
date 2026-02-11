# prepare_data.py


# Reads all videos signs.
# Uses MediaPipe to extract hand landmarks from frames.
# Converts each video into a fixed-length sequence of features.
# Handles static and motion signs differently.
# Converts sign labels to one-hot encoding.
# Saves the dataset as .npy files ready for model training.

import os
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import json

# =============================
# CONFIG
# =============================
DATASET_PATH = "video_dataset"   # root folder
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
# FEATURE EXTRACTION
# =============================
def extract_frame_features(results):
    """
    Extract hands-only features (left + right)
    Output shape: (126,)
    """

    def flatten_hand(hand):
        if hand:
            return np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark]).flatten()
        return np.zeros(21 * 3)

    left = flatten_hand(results.left_hand_landmarks)
    right = flatten_hand(results.right_hand_landmarks)

    return np.concatenate([left, right])


def motion_score(curr, prev):
    if prev is None:
        return np.inf
    return np.mean(np.abs(curr - prev))

# =============================
# VIDEO PROCESSING
# =============================
def process_static_video(video_path, holistic):
    """
    Static sign:
    - find stable frames
    - average them
    - repeat to SEQ_LEN
    """
    cap = cv2.VideoCapture(video_path)
    stable_frames = []
    prev_feats = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        feats = extract_frame_features(results)

        if motion_score(feats, prev_feats) < STATIC_MOTION_THRESHOLD:
            stable_frames.append(feats)

        prev_feats = feats

    cap.release()

    # fallback if no stable frames found
    if len(stable_frames) == 0:
        stable_frames.append(prev_feats)

    avg_frame = np.mean(stable_frames, axis=0)
    return np.tile(avg_frame, (SEQ_LEN, 1))


def process_motion_video(video_path, holistic):
    """
    Motion sign:
    - full temporal sequence
    - pad / trim to SEQ_LEN
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        frames.append(extract_frame_features(results))

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
        model_complexity=1,
        refine_face_landmarks=False
    ) as holistic:

        for action in actions:
            action_path = os.path.join(DATASET_PATH, action)
            
            # Only take .mp4 files
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
    print("X shape:", X.shape)   # (29, 30, 126)
    print("y shape:", y.shape)   # (29, 29)
    print("Label map:", label_map)

    with open("label_map.json", "w") as f:
        json.dump(label_map, f)

    print("💾 label_map.json saved")