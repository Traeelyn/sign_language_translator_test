from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, Bidirectional
from tensorflow.keras.optimizers import Adam

SEQ_LEN = 30
NUM_FEATURES = 126
NUM_CLASSES = 29

def build_model():
    model = Sequential([
        Masking(mask_value=0.0, input_shape=(SEQ_LEN, NUM_FEATURES)),

        # 🔥 Strong temporal encoder
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.3),

        Bidirectional(LSTM(64)), #Sees start → end and end → start, Huge boost for motion signs
        Dropout(0.3),

        # 🧠 Classification head
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(NUM_CLASSES, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0003),  # slightly lower for stability
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
