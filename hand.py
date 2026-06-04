import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# =========================
# 1. DATASET PATH
# =========================
train_path = "train2"
model_path = "model.keras"

# =========================
# 2. GESTURE NAMES (0–19)
# =========================
gesture_names = [
    "Fist (Stop)",            # 0
    "Open Palm (Hello)",      # 1
    "Peace Sign",             # 2
    "Thumbs Up",              # 3
    "Thumbs Down",            # 4
    "OK Sign",                # 5
    "Point Left",            # 6
    "Point Right",           # 7
    "Point Up",              # 8
    "Point Down",            # 9
    "Call Me",               # 10
    "Rock Sign",             # 11
    "Three Fingers",         # 12
    "Four Fingers",          # 13
    "Five Fingers Spread",   # 14
    "L Sign",                # 15
    "Pinch Gesture",        # 16
    "Swipe Left",           # 17
    "Swipe Right",          # 18
    "Custom Gesture"        # 19
]

# =========================
# 3. TRAIN MODEL
# =========================
def train_model():

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_data = datagen.flow_from_directory(
        train_path,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    val_data = datagen.flow_from_directory(
        train_path,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Flatten(),
        Dense(128, activation='relu'),
        Dense(train_data.num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_data, validation_data=val_data, epochs=10)

    model.save(model_path)

    print("✅ Model trained and saved!")

# =========================
# 4. REAL-TIME PREDICTION
# =========================
def run_camera():

    model = load_model(model_path)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        img = cv2.resize(frame, (64, 64))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img, verbose=0)

        class_id = np.argmax(prediction)
        confidence = np.max(prediction)

        # SAFE LABEL MAPPING
        if class_id < len(gesture_names):
            label = gesture_names[class_id]
        else:
            label = f"Unknown ({class_id})"

        cv2.putText(frame, f"Gesture: {label}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        cv2.imshow("Hand Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# =========================
# 5. MAIN MENU
# =========================
if __name__ == "__main__":

    print("\n1. Train Model")
    print("2. Run Webcam Prediction")

    choice = input("Enter choice: ")

    if choice == "1":
        train_model()

    elif choice == "2":
        run_camera()

    else:
        print("Invalid choice")