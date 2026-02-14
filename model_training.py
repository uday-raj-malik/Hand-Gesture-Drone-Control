import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# ---------------- LOAD TRAIN/TEST DATA ----------------
train_df = pd.read_csv("gesture_train.csv")
test_df  = pd.read_csv("gesture_test.csv")

X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values

X_test  = test_df.iloc[:, :-1].values
y_test  = test_df.iloc[:, -1].values

# ---------------- ADD ORIENTATION FEATURES ----------------
def add_orientation(X):
    wrist = X[:, 0:3]
    thumb = X[:, 12:15]
    index = X[:, 24:27]

    thumb_orientation = (thumb - wrist) * 2.0
    index_orientation = index - wrist

    return np.hstack([X, thumb_orientation, index_orientation])

X_train = add_orientation(X_train)
X_test  = add_orientation(X_test)

print("Train features:", X_train.shape[1])  # MUST be 69
print("Test features:", X_test.shape[1])    # MUST be 69

# ---------------- LABEL ENCODING ----------------
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test  = le.transform(y_test)
joblib.dump(le, "label_encoder.pkl")

# ---------------- SCALING ----------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
joblib.dump(scaler, "scaler.pkl")

# ---------------- BUILD MODEL ----------------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(69,)),

    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.35),

    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.30),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Dense(64, activation='relu'),

    tf.keras.layers.Dense(len(np.unique(y_train)), activation='softmax')
])

# ---------------- OPTIMIZER ----------------
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.001,
    decay_steps=3000,
    alpha=0.0001
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ---------------- CALLBACKS ----------------
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=12,
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_gesture_model.keras",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# ---------------- TRAIN ----------------
history = model.fit(
    X_train, y_train,
    epochs=120,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# ---------------- EVALUATE ----------------
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nFinal Test Accuracy: {acc:.4f}")

# ---------------- SAVE FINAL MODEL ----------------
model.save("gesture_model.keras")

print("\nTraining complete!")
print("Final model saved as gesture_model.keras")
