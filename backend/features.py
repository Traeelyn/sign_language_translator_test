# STRUCTURE:

# [ low-level extractors ]
#   ├─ extract_hand_features
#   ├─ extract_pose_subset
#   ├─ extract_face_subset

# [ mid-level ]
#   ├─ extract_hands_from_holistic

# [ top-level ]
#   ├─ extract_frame_features(results)


import numpy as np

# =============================
# Config
# =============================
HAND_LANDMARKS = 21
HAND_FEATURES_PER_LM = 6   # x, y, z, dx, dy, dz
HAND_FEATURE_DIM = HAND_LANDMARKS * HAND_FEATURES_PER_LM  # 126

SMOOTHING_ALPHA = 0.7


# =============================
# Normalization
# =============================
def normalize_translation(landmarks):
    """
    Center landmarks around wrist (landmark 0).
    """
    wrist = landmarks[0]
    return [
        (
            lm.x - wrist.x,
            lm.y - wrist.y,
            lm.z - wrist.z
        )
        for lm in landmarks
    ]


# =============================
# Temporal smoothing
# =============================
def smooth_landmarks(current, previous, alpha=SMOOTHING_ALPHA):
    """
    Exponential moving average smoothing.
    """
    if previous is None:
        return current

    smoothed = []
    for c, p in zip(current, previous):
        smoothed.append((
            alpha * c[0] + (1 - alpha) * p[0],
            alpha * c[1] + (1 - alpha) * p[1],
            alpha * c[2] + (1 - alpha) * p[2],
        ))
    return smoothed



# =============================
# Hand feature extractor
# =============================
def extract_hand_features(hand_landmarks, prev_landmarks=None, is_static=False):
    """
    Returns:
    - feature vector (126,)
    - smoothed coords for next frame
    """

    # Normalize translation
    coords = normalize_translation(hand_landmarks)

    # Convert previous landmarks to coords (IMPORTANT FIX)
    if prev_landmarks is not None:
        prev_coords = [(lm.x, lm.y, lm.z) for lm in prev_landmarks]
    else:
        prev_coords = None

    # Temporal smoothing
    coords = smooth_landmarks(coords, prev_coords)

    # Motion computation
    if prev_coords is None:
        prev_coords = coords

    features = []
    for (x, y, z), (px, py, pz) in zip(coords, prev_coords):
        dx, dy, dz = x - px, y - py, z - pz
        if is_static:
            features.extend([x, y, z, 0.0, 0.0, 0.0])
        else:
            features.extend([x, y, z, dx, dy, dz])

    return np.array(features, dtype=np.float32), coords




# -----------------------------
#  Pose feature extractor
# -----------------------------
POSE_IDX = [
    11, 12,   # shoulders
    13, 14,   # elbows
    15, 16    # wrists
]
POSE_DIM = len(POSE_IDX) * 3

def extract_pose_subset(pose_landmarks):
    if pose_landmarks is None:
        return np.zeros(POSE_DIM, dtype=np.float32)

    features = []
    for idx in POSE_IDX:
        lm = pose_landmarks.landmark[idx]
        features.extend([lm.x, lm.y, lm.z])

    return np.array(features, dtype=np.float32)

# -----------------------------
# Extract face subset
# -----------------------------
# Minimal mouth + nose subset (safe to expand later)
FACE_IDX = [
    1, 2, 13, 14,        # nose
    78, 95, 88, 178,     # mouth
    308, 324, 318, 402
]

FACE_DIM = len(FACE_IDX) * 3


def extract_face_subset(face_landmarks, use_face=False):
    """
    Returns fixed-size face vector.
    Zeroed if face is unused.
    """
    if not use_face or face_landmarks is None:
        return np.zeros(FACE_DIM, dtype=np.float32)

    features = []
    for idx in FACE_IDX:
        lm = face_landmarks[idx]
        features.extend([lm.x, lm.y, lm.z])

    return np.array(features, dtype=np.float32)




# -----------------------------
#  Extract hands from hollistic
# -----------------------------
def extract_hands_from_holistic(results, prev_left=None, prev_right=None, is_static=False):
    left_feats, left_coords = (
        extract_hand_features(
            results.left_hand_landmarks.landmark,
            prev_left,
            is_static=is_static
        )
        if results.left_hand_landmarks
        else (np.zeros(HAND_FEATURE_DIM, dtype=np.float32), None)
    )

    right_feats, right_coords = (
        extract_hand_features(
            results.right_hand_landmarks.landmark,
            prev_right,
            is_static=is_static
        )
        if results.right_hand_landmarks
        else (np.zeros(HAND_FEATURE_DIM, dtype=np.float32), None)
    )

    return left_feats, right_feats, left_coords, right_coords



# -----------------------------
#  Frame level feature extractor
# -----------------------------
def extract_frame_features(
    results,
    prev_left=None,
    prev_right=None,
    use_face=True,
    is_static=False
):
    left_hand, right_hand, left_coords, right_coords = \
        extract_hands_from_holistic(
            results,
            prev_left=prev_left,
            prev_right=prev_right,
            is_static=is_static
        )

    pose = extract_pose_subset(results.pose_landmarks)

    face = extract_face_subset(
        results.face_landmarks.landmark if results.face_landmarks else None,
        use_face=use_face
    )

    frame_features = np.concatenate([
        left_hand,   # 126
        right_hand,  # 126
        pose,        # 18
        face         # 36
    ])

    return frame_features, left_coords, right_coords



    
