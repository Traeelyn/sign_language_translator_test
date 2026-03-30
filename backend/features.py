#features.py
# Takes raw mediapipe landmarsk and turn them into numeric feature vector for LSTM to use

# RESPONSIBILITIES: 
## Normalise hands
## Temporal smoothing (takes current & previous coordinates and applies exponential moving average to reduce jitter & make LSTM smoother)
## Velocity computation (Calculates (dx, dy, dx) for ach landmark and how much it moved from prev frame...is_static=True (all velocities 0))
## Flatten features (Converts (x, y, z, dx, dy, dz) for 21 hand landmarks into a 1D list of 126 features per hand)
## Frame-level feature extraction
import numpy as np

EMA_ALPHA = 0.5

# -------------------------
# HAND NORMALIZATION
# -------------------------
def normalize_hand(hand_landmarks):
    """
    Normalize hand landmarks relative to wrist (landmark 0).
    Returns list of 21 (x,y,z) tuples.
    """

    if hand_landmarks is None:
        return [(0.0, 0.0, 0.0)] * 21

    wrist = hand_landmarks.landmark[0]
    coords = []

    for lm in hand_landmarks.landmark:
        coords.append((
            lm.x - wrist.x,
            lm.y - wrist.y,
            lm.z - wrist.z
        ))

    return coords


# -------------------------
# POSE NORMALIZATION
# -------------------------
def normalize_pose(pose_landmarks):
    """
    Use shoulders (11,12) as center reference.
    Extracts 6 upper body points: 11–16
    """

    if pose_landmarks is None:
        return [(0.0, 0.0, 0.0)] * 6

    pose_points = [11, 12, 13, 14, 15, 16]

    left_shoulder = pose_landmarks.landmark[11]
    right_shoulder = pose_landmarks.landmark[12]

    center_x = (left_shoulder.x + right_shoulder.x) / 2
    center_y = (left_shoulder.y + right_shoulder.y) / 2
    center_z = (left_shoulder.z + right_shoulder.z) / 2

    coords = []

    for idx in pose_points:
        lm = pose_landmarks.landmark[idx]
        coords.append((
            lm.x - center_x,
            lm.y - center_y,
            lm.z - center_z
        ))

    return coords


# -------------------------
# SMOOTHING
# -------------------------
def smooth_landmarks(coords, prev_coords):
    if prev_coords is None:
        return coords

    smoothed = []
    for curr, prev in zip(coords, prev_coords):
        smoothed.append(tuple(
            EMA_ALPHA * c + (1 - EMA_ALPHA) * p
            for c, p in zip(curr, prev)
        ))

    return smoothed


# -------------------------
# VELOCITY
# -------------------------
def compute_velocity(coords, prev_coords):
    if prev_coords is None:
        return [(0.0, 0.0, 0.0)] * len(coords)

    velocity = []
    for curr, prev in zip(coords, prev_coords):
        velocity.append(tuple(c - p for c, p in zip(curr, prev)))

    return velocity


# -------------------------
# FLATTEN (x,y,z,dx,dy,dz)
# -------------------------
def flatten_features(coords, velocity):
    flat = []
    for (x, y, z), (dx, dy, dz) in zip(coords, velocity):
        flat.extend([x, y, z, dx, dy, dz])
    return flat


# -------------------------
# MAIN FEATURE EXTRACTOR
# -------------------------
def extract_frame_features(results,
                           prev_left,
                           prev_right,
                           prev_pose,
                           is_static=False):
    """
    Extracts full feature vector from MediaPipe Holistic results.

    Output size:
    - Left hand: 21 * 6 = 126
    - Right hand: 21 * 6 = 126
    - Pose (6 points): 6 * 6 = 36
    TOTAL = 288 features
    """

    # LEFT HAND
    left_coords = normalize_hand(results.left_hand_landmarks)
    left_coords = smooth_landmarks(left_coords, prev_left)
    left_vel = compute_velocity(left_coords, prev_left)

    if is_static:
        left_vel = [(0.0, 0.0, 0.0)] * 21

    left_features = flatten_features(left_coords, left_vel)

    # RIGHT HAND
    right_coords = normalize_hand(results.right_hand_landmarks)
    right_coords = smooth_landmarks(right_coords, prev_right)
    right_vel = compute_velocity(right_coords, prev_right)

    if is_static:
        right_vel = [(0.0, 0.0, 0.0)] * 21

    right_features = flatten_features(right_coords, right_vel)

    # POSE
    pose_coords = normalize_pose(results.pose_landmarks)
    pose_coords = smooth_landmarks(pose_coords, prev_pose)
    pose_vel = compute_velocity(pose_coords, prev_pose)

    if is_static:
        pose_vel = [(0.0, 0.0, 0.0)] * 6

    pose_features = flatten_features(pose_coords, pose_vel)

    # FINAL VECTOR
    features = np.array(
        left_features +
        right_features +
        pose_features
    )

    return features, left_coords, right_coords, pose_coords