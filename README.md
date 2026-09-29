# Real-Time Hand Gesture Recognition System

A real-time hand gesture recognition system built with Python, OpenCV, MediaPipe, and a machine learning classifier.

The system uses a webcam to detect hand landmarks, extract hand features, and classify five different hand gestures in real time.

## Features

* Real-time hand detection using MediaPipe
* 21 hand landmark detection
* Custom gesture dataset collected using a webcam
* Machine learning classification using Random Forest
* Real-time gesture prediction with confidence scores
* Supports five hand gestures:

  * Fist
  * Open Palm
  * Pointing
  * Peace
  * Thumbs Up

## Technologies

* Python
* OpenCV
* MediaPipe
* Pandas
* Scikit-learn
* Joblib

## How It Works

```text
Webcam
   ↓
OpenCV
   ↓
MediaPipe Hand Detection
   ↓
21 Hand Landmarks
   ↓
Feature Extraction
   ↓
Random Forest Classifier
   ↓
Gesture + Confidence
```

## Dataset

A custom dataset was collected using the laptop's built-in webcam.

The dataset contains **10,951 samples** across five gesture classes:

| Label | Gesture   | Samples |
| ----- | --------- | ------: |
| 0     | Fist      |   2,110 |
| 1     | Open Palm |   1,943 |
| 2     | Pointing  |   2,282 |
| 3     | Peace     |   2,021 |
| 4     | Thumbs Up |   2,595 |

Each sample contains 63 numerical features based on the x, y, and z coordinates of 21 hand landmarks.

The dataset is excluded from the GitHub repository through `.gitignore`.

## Model

A Random Forest classifier was trained using an 80/20 train-test split.

* Training samples: 8,760
* Testing samples: 2,191
* Test accuracy: **99.18%**

The trained model is saved as:

```text
models/gesture_classifier.joblib
```

> Note: The reported accuracy is based on a random train-test split from the collected dataset. Real-world performance may vary when the system is used with different users, backgrounds, lighting conditions, hand positions, and camera angles.

## Installation

Clone the repository and create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment and install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the real-time gesture recognition application:

```bash
python gesture_app.py
```

Press **Q** to exit the application.

## Project Structure

```text
hand-gesture-recognition/
│
├── camera.py
├── hand_tracker.py
├── collect_data.py
├── train_model.py
├── gesture_app.py
│
├── models/
│   ├── hand_landmarker.task
│   └── gesture_classifier.joblib
│
├── data/
│   └── gestures.csv
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Future Improvements

* Improve robustness across different users and environments
* Add an "Unknown" class for low-confidence predictions
* Improve prediction stability with temporal smoothing
* Evaluate the model using data from separate recording sessions
* Improve the visual interface
* Add support for additional gestures

## Project Purpose

This project was developed as a hands-on learning project to understand the complete workflow of a computer vision and machine learning application, from webcam input and hand landmark detection to dataset collection, model training, evaluation, and real-time prediction.
