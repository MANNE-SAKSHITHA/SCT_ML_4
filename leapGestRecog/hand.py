import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pickle

# =========================
# CONFIG
# =========================
DATASET = "leapGestRecog"
IMG_SIZE = 64
MODEL_PATH = "gesture_model.h5"
LABEL_ENCODER_PATH = "label_encoder.pkl"

data = []
labels = []

# =========================
# LOAD DATASET
# =========================
print("Loading dataset...")

for subject in os.listdir(DATASET):
    subject_path = os.path.join(DATASET, subject)

    if not os.path.isdir(subject_path):
        continue

    for gesture in os.listdir(subject_path):
        gesture_path = os.path.join(subject_path, gesture)

        if not os.path.isdir(gesture_path):
            continue

        for img_name in os.listdir(gesture_path):
            img_path = os.path.join(gesture_path, img_name)

            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

                data.append(img)
                labels.append(gesture)

            except:
                pass

print("Dataset loaded ✔")

# =========================
# PREPROCESS DATA
# =========================
data = np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
data = data / 255.0

# Label encoding
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels)

print("Classes:", le.classes_)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# =========================
# MODEL DEFINITION
# =========================
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(le.classes_), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# TRAIN ONLY ONCE LOGIC
# =========================
if os.path.exists(MODEL_PATH) and os.path.exists(LABEL_ENCODER_PATH):
    print("✅ Loading saved model and label encoder...")

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(LABEL_ENCODER_PATH, "rb") as f:
        le = pickle.load(f)

else:
    print("🚀 Training model (first time)...")

    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_test, y_test)
    )

    # Evaluate
    loss, acc = model.evaluate(X_test, y_test)
    print("Test Accuracy:", acc)

    # Save model
    model.save(MODEL_PATH)

    # Save label encoder
    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)

    print("💾 Model + Encoder saved!")

    # Plot accuracy
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.legend()
    plt.show()

# =========================
# FINAL TEST EVALUATION
# =========================
loss, acc = model.evaluate(X_test, y_test)
print("Final Test Accuracy:", acc)
# =========================
# REAL-TIME TESTING (WEBCAM)
# =========================

import cv2
import numpy as np

print("📷 Starting webcam... Press Q to exit")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    x1, y1, x2, y2 = 100, 100, 300, 300
    roi = frame[y1:y2, x1:x2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (64, 64))
    img = img.reshape(1, 64, 64, 1)
    img = img / 255.0

    pred = model.predict(img, verbose=0)
    class_id = np.argmax(pred)
    label = le.inverse_transform([class_id])[0]

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()