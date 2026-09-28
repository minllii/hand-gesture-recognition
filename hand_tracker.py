import cv2
import mediapipe as mp
import math

def distance(point1, point2):
    return math.sqrt(
        (point1.x - point2.x) ** 2 +
        (point1.y - point2.y) ** 2
    )
def angle(point1, point2, point3):
    vector1 = (
        point1.x - point2.x,
        point1.y - point2.y
    )

    vector2 = (
        point3.x - point2.x,
        point3.y - point2.y
    )

    dot_product = (
        vector1[0] * vector2[0] +
        vector1[1] * vector2[1]
    )

    magnitude1 = math.sqrt(
        vector1[0] ** 2 +
        vector1[1] ** 2
    )

    magnitude2 = math.sqrt(
        vector2[0] ** 2 +
        vector2[1] ** 2
    )

    cosine = dot_product / (magnitude1 * magnitude2)

    cosine = max(-1, min(1, cosine))

    return math.degrees(math.acos(cosine))

def is_finger_open(hand, tip, joint):
    return distance(hand[0], hand[tip]) > distance(hand[0], hand[joint])

# Create MediaPipe Hand Landmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2
)

landmarker = HandLandmarker.create_from_options(options)

camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()

    if not success:
        print("Failed to read from camera")
        break

    # OpenCV uses BGR, MediaPipe expects RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert OpenCV image to MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect hands
    result = landmarker.detect(mp_image)

    # Draw detected landmarks
    if result.hand_landmarks:
        for hand in result.hand_landmarks:

            index_open = is_finger_open(hand, 8, 6)
            middle_open = is_finger_open(hand, 12, 10)
            ring_open = is_finger_open(hand, 16, 14)
            pinky_open = is_finger_open(hand, 20, 18)

            thumb_up = hand[4].y < hand[3].y
            
            index_angle = angle(hand[5], hand[6], hand[7])

            print("Index angle:", index_angle)

            print(
                "Index:", index_open,
                "| Middle:", middle_open,
                "| Ring:", ring_open,
                "| Pinky:", pinky_open
            )

            if index_open and middle_open and ring_open and pinky_open:
                print("GESTURE: OPEN PALM")

            for i, landmark in enumerate(hand):
                print(i, landmark.x, landmark.y, landmark.z)
            
            for landmark in hand:
                height, width, _ = frame.shape

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            if index_open and middle_open and ring_open and pinky_open:
                cv2.putText(
                    frame,
                    "OPEN PALM",
                    (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 255, 0),
                    3
                )

            elif index_open and not middle_open and not ring_open and not pinky_open:
                cv2.putText(
                    frame,
                    "POINTING",
                    (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 255, 0),
                    3
                )

            elif index_open and middle_open and not ring_open and not pinky_open:
                cv2.putText(
                    frame,
                    "PEACE",
                    (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 255, 0),
                    3
                )

            elif thumb_up and not index_open and not middle_open and not ring_open and not pinky_open:
                cv2.putText(
                    frame,
                    "THUMBS UP",
                    (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 255, 0),
                    3
                )

            elif not index_open and not middle_open and not ring_open and not pinky_open:
                cv2.putText(
                    frame,
                    "FIST",
                    (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 255, 0),
                    3
                )

    cv2.imshow("Hand Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
landmarker.close()
cv2.destroyAllWindows()