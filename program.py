import re
from mediapipe.python.solutions import hands
import cv2
from types import SimpleNamespace as SN
import ait

keys = {}
for k, v in re.findall(r"(\d{7})\s+(\S+)", open("README").read()):
    node = keys
    for c in k:
        node = node.setdefault(int(c), {})
    key_code = {
        "Backspace": "\b",
        "Tab": "\t",
        "New line": "\n",
        "Space": " ",
        "Arrow up": "up",
        "Arrow down": "down",
        "Arrow left": "left",
        "Arrow right": "right",
    }.get(v, v)
    node["key"] = SN(name=v, code=key_code)

def distance(landmark1, landmark2):
    return ((landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2) ** 0.5

CAMERA_INDEX = 0
FINGER_PROXIMITY_THRESHOLD = 0.1

video_capture = cv2.VideoCapture(CAMERA_INDEX)
MIN_HAND_MODEL_COMPLEXITY = 0
with hands.Hands(max_num_hands=2, model_complexity=MIN_HAND_MODEL_COMPLEXITY, static_image_mode=False) as hands_recognizer:
    clack_existence = [False, False]
    node = keys
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
                thumb_tip = hand[hands.HandLandmark.THUMB_TIP]
                index_finger_tip = hand[hands.HandLandmark.INDEX_FINGER_TIP]
                finger_tips = [
                    thumb_tip,
                    index_finger_tip,
                    hand[hands.HandLandmark.MIDDLE_FINGER_TIP],
                    hand[hands.HandLandmark.RING_FINGER_TIP],
                    hand[hands.HandLandmark.PINKY_TIP],
                ]
                is_clack_on = max(
                    distance(finger_tip, other_finger_tip)
                    for finger_tip in finger_tips
                    for other_finger_tip in finger_tips
                ) < FINGER_PROXIMITY_THRESHOLD
                is_clack_off = distance(thumb_tip, index_finger_tip) > FINGER_PROXIMITY_THRESHOLD
                if clack_existence[bit]:
                    if is_clack_off:
                        clack_existence[bit] = False
                else:
                    if is_clack_on:
                        clack_existence[bit] = True
                        print(bit, end="", flush=True)
                        end_message = None
                        try:
                            node = node[bit]
                        except KeyError:
                            end_message = "NOT FOUND"
                        else:
                            try:
                                key = node["key"]
                            except KeyError:
                                pass
                            else:
                                end_message = key.name
                                ait.press(key.code)
                        if end_message is not None:
                            print(f" {end_message}")
                            node = keys
