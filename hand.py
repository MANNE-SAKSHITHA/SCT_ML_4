# Hand Gesture Recognition using ASL Alphabet Dataset
# SkillCraft Technology - Task 04

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt

# ==========================================
# STEP 1: DATASET PATH
# ==========================================

dataset_path = "asl_alphabet_train"   # Change if needed

# ==========================================
# STEP 2: LOAD DATASET
# ==========================================

data_generator = ImageDataGenerator(
    rescale=1/255,
    validation_split=0.2
)

train_data = data_generator.flow_from_directory(
    dataset_path,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_data = data_generator.flow_from_directory(
    dataset_path,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# ==========================================
# STEP 3: BUILD CNN MODEL
# ==========================================

model = Sequential()

# First Convolution Layer
model.add(
    Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(64,64,3)
    )
)

model.add(MaxPooling2D(2,2))

# Second Convolution Layer
model.add(
    Conv2D(
        64,
        (3,3),
        activation='relu'
    )
)

model.add(MaxPooling2D(2,2))

# Flatten Layer
model.add(Flatten())

# Hidden Layer
model.add(Dense(128, activation='relu'))

# Output Layer
model.add(
    Dense(
        train_data.num_classes,
        activation='softmax'
    )
)

# ==========================================
# STEP 4: COMPILE MODEL
# ==========================================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Show Model Structure
model.summary()

# ==========================================
# STEP 5: TRAIN MODEL
# ==========================================

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=5
)

# ==========================================
# STEP 6: EVALUATE MODEL
# ==========================================

loss, accuracy = model.evaluate(validation_data)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

# ==========================================
# STEP 7: SAVE MODEL
# ==========================================

model.save("hand_gesture_model.h5")

print("Model saved successfully!")

# ==========================================
# STEP 8: PLOT ACCURACY GRAPH
# ==========================================

plt.figure(figsize=(10,5))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend([
    "Training Accuracy",
    "Validation Accuracy"
])

plt.show()