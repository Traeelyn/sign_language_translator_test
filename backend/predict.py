#predict.py
import numpy as np
from collections import deque

# -----------------------------
# Confidence thresholds per class
# -----------------------------
CONF_THRESH = {
    "A": 0.92, "B": 0.85, "C": 0.90, "D": 0.90, "E": 0.80, "F": 0.92,
    "G": 0.90, "H": 0.90, "I": 0.90, "K": 0.90, "L": 0.90, "M": 0.90,
    "N": 0.90, "O": 0.90, "P": 0.90, "Q": 0.90, "R": 0.90, "S": 0.90,
    "T": 0.90, "U": 0.90, "V": 0.90, "W": 0.92, "X": 0.90, "Y": 0.90,
    "J": 0.90, "Love_u": 0.88, "Hello": 0.80, "Sorry": 0.85, "Z": 0.90
}

# -----------------------------
# Define static vs motion signs
# -----------------------------
STATIC_SIGNS = {
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N",
    "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"
}

MOTION_SIGNS = {"J", "Love_u", "Hello", "Sorry", "Z"}

# -----------------------------
# Temporal smoothing
# -----------------------------
PREDICTION_HISTORY = 5
pred_history = deque(maxlen=PREDICTION_HISTORY)

# -----------------------------
# Motion score calculation
# -----------------------------
def compute_motion_score(landmarks):
    """Compute average movement of hand landmarks per frame."""
    deltas = landmarks[3::6] + landmarks[4::6] + landmarks[5::6]
    return np.mean(np.abs(deltas))

# -----------------------------
# Predict function
# -----------------------------
def predict_sequence(model, buffer, labels, motion_threshold=0.015):
    """
    Input:
        model     : trained keras model
        buffer    : deque of last SEQ_LEN frames (landmark features)
        labels    : list of label names
    Output:
        stable_label : str or None
        confidence   : float
        motion_score : float
    """
    X = np.array(buffer, dtype=np.float32).reshape(1, len(buffer), len(buffer[0]))
    probs = model.predict(X, verbose=0)[0]
    pred_index = np.argmax(probs)
    confidence = probs[pred_index]
    label = labels[pred_index]

    # ---- Motion gating ----
    motion_score = np.mean([compute_motion_score(f) for f in buffer])
    is_motion = motion_score > motion_threshold
    
    if label in STATIC_SIGNS and is_motion:
        return None, confidence, motion_score
    if label in MOTION_SIGNS and not is_motion:
        return None, confidence, motion_score


    # ---- Confidence threshold ----
    threshold = CONF_THRESH.get(label, 0.85)
    if confidence < threshold:
        return None, confidence, motion_score

    # ---- Temporal smoothing ----
    if len(pred_history) == 0 or label == pred_history[-1]:
        pred_history.append(label)
    else:
        pred_history.clear()
        pred_history.append(label)

    if len(pred_history) == PREDICTION_HISTORY:
        stable_label = pred_history[0]
        pred_history.clear()
        return stable_label, confidence, motion_score

    return None, confidence, motion_score
