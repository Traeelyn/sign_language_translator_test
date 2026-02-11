# train_model.py


# Trains an LSTM model on prepared sign language data

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# =============================
# LOAD DATA
# =============================
X = np.load("X.npy")   # shape: (samples, 30, 126)
y = np.load("y.npy")   # shape: (samples, num_classes)

print("X shape:", X.shape)
print("y shape:", y.shape)

SEQ_LEN = X.shape[1]
N_FEATURES = X.shape[2]
N_CLASSES = y.shape[1]

# =============================
# TRAIN / VAL SPLIT (splits data into training data and validation data)
# =============================
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=np.argmax(y, axis=1)
)

# =============================
# MODEL
# =============================
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(SEQ_LEN, N_FEATURES)),
    Dropout(0.3),

    LSTM(64),
    Dropout(0.3),

    Dense(64, activation="relu"),
    Dense(N_CLASSES, activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =============================
# CALLBACKS (helpers that watch training and step in when needed)
# =============================
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        "sign_model.h5",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
]

# =============================
# TRAIN
# =============================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=8, #The model learns from 8 videos at a time.
    callbacks=callbacks
)

# Epoch 1 → sees all training samples once
# Epoch 2 → sees them again (but smarter) ...Epoch = 100 means it goes on till 100


# =============================
# SAVE FINAL MODEL
# =============================
model.save("sign_model_final.h5")
print("✅ Model training complete")


#lstm: reccurent neural network 
    # - Processes data step by step
    # - Keeps a memory of what happened earlier
    # - Decides what to remember and what to forget