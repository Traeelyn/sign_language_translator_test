# train_model.py


# Trains an LSTM model on prepared sign language data

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# =============================
# LOAD DATA
# =============================
X = np.load("X.npy")   # shape: (samples, 30, 288)
y = np.load("y.npy")   # shape: (samples, num_classes)

print("X shape:", X.shape)
print("y shape:", y.shape)

SEQ_LEN = X.shape[1]
N_FEATURES = X.shape[2]
N_CLASSES = y.shape[1]

# =============================
# TRAIN / VAL SPLIT
# =============================
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=np.argmax(y, axis=1)
)

print("Training samples:", X_train.shape[0])
print("Validation samples:", X_val.shape[0])

# =============================
# MODEL
# =============================
model = Sequential([

    LSTM(128, return_sequences=True, input_shape=(SEQ_LEN, N_FEATURES)),
    BatchNormalization(),
    Dropout(0.4),

    LSTM(64),
    BatchNormalization(),
    Dropout(0.4),

    Dense(128, activation="relu"),
    Dropout(0.3),

    Dense(N_CLASSES, activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =============================
# CALLBACKS
# =============================
callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=12,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        verbose=1
    ),

    ModelCheckpoint(
        "sign_model_best.h5",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
]

# =============================
# TRAIN
# =============================
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=120,
    batch_size=16,   # Increased slightly for stability
    callbacks=callbacks
)

# =============================
# SAVE FINAL MODEL
# =============================
model.save("sign_model_final.h5")

print("✅ Model training complete")