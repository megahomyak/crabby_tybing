import re
from mediapipe.python.solutions import hands
import cv2

keycodes = {}
for k, v in re.findall(r"(\d{7})\s+(\S+)", open("README").read()):
    node = keycodes
    for c in k:
        node = node.setdefault(int(c), {})
    node["keycode"] = v

def distance(landmark1, landmark2):
    return ((landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2) ** 0.5

CAMERA_INDEX = 0
FINGER_PROXIMITY_THRESHOLD = 0.05

video_capture = cv2.VideoCapture(CAMERA_INDEX)
MIN_HAND_MODEL_COMPLEXITY = 0
with hands.Hands(max_num_hands=2, model_complexity=MIN_HAND_MODEL_COMPLEXITY) as hands_recognizer:
    clack_existence = [False, False]
    node = keycodes
    while True:
        is_success, frame = video_capture.read()
        assert is_success
        output = hands_recognizer.process(frame)
        if output.multi_hand_landmarks and len(output.multi_hand_landmarks) == 2:
            left = min(*output.multi_hand_landmarks, key=lambda hand: hand.landmark[hands.HandLandmark.WRIST].x)
            right = max(*output.multi_hand_landmarks, key=lambda hand: hand.landmark[hands.HandLandmark.WRIST].x)
            for hand, bit in zip(
                [left, right],
                [0, 1],
            ):
                hand = hand.landmark
                finger_tips = [
                    hand[hands.HandLandmark.THUMB_TIP],
                    hand[hands.HandLandmark.INDEX_FINGER_TIP],
                    hand[hands.HandLandmark.MIDDLE_FINGER_TIP],
                    hand[hands.HandLandmark.RING_FINGER_TIP],
                    hand[hands.HandLandmark.PINKY_TIP],
                ]
                hand_is_clacked = max(
                    distance(finger_tip, other_finger_tip)
                    for finger_tip in finger_tips
                    for other_finger_tip in finger_tips
                ) < FINGER_PROXIMITY_THRESHOLD
                if hand_is_clacked and not clack_existence[bit]:
                    print(bit, end="", flush=True)
                    message = None
                    try:
                        node = node[bit]
                    except KeyError:
                        message = " NOT FOUND"
                    else:
                        try:
                            keycode = node["keycode"]
                        except KeyError:
                            pass
                        else:
                            message = f" {keycode}"
                            # Exec key code
                    if message is not None:
                        print(message)
                        node = keycodes
                clack_existence[bit] = hand_is_clacked
