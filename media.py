import mediapipe as mp
import cv2
import time
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import numpy as np

model_path = "hand_landmarker.task"

cont = True


class LandmarkerAndResult:
    def __init__(self):
        self.landmarker: mp.tasks.vision.HandLandmarker | None = None
        self.result: mp.tasks.vision.HandLandmarkerResult | None = None
        self.output_image = None
        self.create_landmarker()

    def create_landmarker(self):
        def update_result(result, output_image, timestamp_ms):
            self.result = result
            self.output_image = output_image

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            result_callback=update_result,
            num_hands=2,
        )
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def detect_async(self, frame_rgb):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self.landmarker.detect_async(mp_image, int(time.time() * 1000))


MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green


def draw_landmarks_on_image(
    rgb_image, detection_result: mp.tasks.vision.HandLandmarkerResult
):
    """Courtesy of https://github.com/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb"""
    try:
        if detection_result.hand_landmarks == []:
            return rgb_image
        else:
            hand_landmarks_list = detection_result.hand_landmarks
            annotated_image = np.copy(rgb_image)

            # Loop through the detected hands to visualize.
            for idx in range(len(hand_landmarks_list)):
                hand_landmarks = hand_landmarks_list[idx]

                # Draw the hand landmarks.
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend(
                    [
                        landmark_pb2.NormalizedLandmark(
                            x=landmark.x, y=landmark.y, z=landmark.z
                        )
                        for landmark in hand_landmarks
                    ]
                )
                mp.solutions.drawing_utils.draw_landmarks(
                    annotated_image,
                    hand_landmarks_proto,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style(),
                )
            return annotated_image
    except:
        return rgb_image


landmarker = LandmarkerAndResult()


cap = cv2.VideoCapture(0)
while cont:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    landmarker.detect_async(frame_rgb)

    if landmarker.result is not None and landmarker.result.handedness:
        annotated_image = draw_landmarks_on_image(frame_rgb, landmarker.result)

        # “Left” / “Right”
        if landmarker.result.handedness[0][0].category_name == "Left":
            print("Left hand detected")
        else:
            print("Right hand detected")
    else:
        annotated_image = frame_rgb
        print("no hand detected yet")

    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    cv2.imshow("annotated_image", annotated_image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cont = False

cap.release()
cv2.destroyAllWindows()


# example = mp.tasks.vision.HandLandmarkerResult(
#     # 1. 2-D image–space landmarks (normalized [0-1] coordinates) ──
#     hand_landmarks=[
#         [  # 21 points for LEFT hand (index 0)
#             landmark_pb2.NormalizedLandmark(x=0.44, y=0.64, z=-0.02),  # 0  WRIST
#             landmark_pb2.NormalizedLandmark(x=0.40, y=0.55, z=-0.02),  # 1  THUMB_CMC
#             landmark_pb2.NormalizedLandmark(x=0.28, y=0.24, z=-0.05),  # 20 PINKY_TIP
#         ],
#         [  # 21 points for RIGHT hand (index 1)
#             landmark_pb2.NormalizedLandmark(x=0.62, y=0.66, z=-0.02),
#             landmark_pb2.NormalizedLandmark(x=0.66, y=0.57, z=-0.02),
#             landmark_pb2.NormalizedLandmark(x=0.62, y=0.57, z=-0.02),
#             landmark_pb2.NormalizedLandmark(x=0.80, y=0.26, z=-0.05),
#         ],
#     ],
#     # 2. 3-D world-space landmarks (in meters, origin ≈ wrist) ──
#     hand_world_landmarks=[
#         [  # LEFT hand
#             landmark_pb2.Landmark(x=-0.037, y=-0.019, z=0.046),
#         ],
#         [  # RIGHT hand
#             landmark_pb2.Landmark(x=0.041, y=-0.021, z=0.048),
#         ],
#     ],
#     # 3. Handedness classification for each detected hand ──
#     handedness=[
#         [  # LEFT hand
#             mp.tasks.vision.HandLandmarkerResult.Category(
#                 index=1,
#                 score=0.9914,
#                 category_name="Left",
#                 display_name="",
#             )
#         ],
#         [  # RIGHT hand
#             mp.tasks.vision.HandLandmarkerResult.Category(
#                 index=0,
#                 score=0.9882,
#                 category_name="Right",
#                 display_name="",
#             )
#         ],
#     ],
# )
