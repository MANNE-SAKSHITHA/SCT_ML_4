# Hand Gesture Recognition

A real-time Hand Gesture Recognition system using CNN, TensorFlow, and OpenCV.

## Features

* Train CNN model on LeapGestRecog dataset
* Save/load trained model automatically
* Real-time webcam gesture prediction

## Technologies

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Scikit-learn

## Run Project

Install dependencies:

```bash id="f88z6r"
pip install tensorflow opencv-python numpy matplotlib scikit-learn
```

Run:

```bash id="g7qt6x"
python gesture_recognition.py
```

Press `Q` to exit webcam window.

## Files

* `gesture_recognition.py` → Main code
* `gesture_model.h5` → Saved model
* `label_encoder.pkl` → Labels
