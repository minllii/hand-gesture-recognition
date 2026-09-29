import cv2
import csv
import os
import mediapipe as mp

DATA_FILE = "data/gestures.csv"

os.makedirs("data", exist_ok=True)

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

if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w", newline="") as file:
        writer = csv.writer(file)

        header = []

        for i in range(21):
            header.extend([
                f"x{i}",
                f"y{i}",
                f"z{i}"
            ])

        header.append("label")

        writer.writerow(header)
        
camera = cv2.VideoCapture(0)

gesture = input(
    "Enter gesture label (0=FIST, 1=OPEN_PALM, 2=POINTING, 3=PEACE, 4=THUMBS_UP): "
)

with open(DATA_FILE, "a", newline="") as file:
    writer = csv.writer(file)

    while True:
        success, frame = camera.read()

        if not success:
            print("Failed to read from camera")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = landmarker.detect(mp_image)

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]

            row = []

            for landmark in hand:
                row.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            row.append(gesture)
            writer.writerow(row)

            cv2.putText(
                frame,
                f"Collecting gesture: {gesture}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Press Q to stop",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        cv2.imshow("Data Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
landmarker.close()
cv2.destroyAllWindows()

print("Data collection finished!")

