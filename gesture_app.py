import cv2
import joblib
import mediapipe as mp


# Load trained model
model = joblib.load("models/gesture_classifier.joblib")

# Gesture names
gesture_names = {
    0: "FIST",
    1: "OPEN PALM",
    2: "POINTING",
    3: "PEACE",
    4: "THUMBS UP"
}

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)

# Start webcam
camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()

    if not success:
        print("Failed to read from camera")
        break

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect hand landmarks
    result = landmarker.detect(mp_image)

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        # Extract 21 landmarks × 3 coordinates = 63 features
        features = []

        for landmark in hand:
            features.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        # Predict gesture
        prediction = model.predict([features])[0]

        # Get confidence
        probabilities = model.predict_proba([features])[0]
        confidence = probabilities[int(prediction)]

        gesture_name = gesture_names[int(prediction)]

        # Display prediction
        cv2.putText(
            frame,
            gesture_name,
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3
        )

        cv2.putText(
            frame,
            f"Confidence: {confidence * 100:.1f}%",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # Draw landmarks
        height, width, _ = frame.shape

        for landmark in hand:
            x = int(landmark.x * width)
            y = int(landmark.y * height)

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )

    cv2.imshow("Hand Gesture Recognition", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
landmarker.close()
cv2.destroyAllWindows()

