import re
from mediapipe.python.solutions import hands
import cv2

keycodes = dict(re.findall(r"(\d{7})\s+(\S+)", open("README").read()))

def distance(landmark1, landmark2):
    return ((landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2) ** 0.5

CAMERA_INDEX = 0
FINGER_PROXIMITY_THRESHOLD = 0.1

video_capture = cv2.VideoCapture(CAMERA_INDEX)
with hands.Hands(max_num_hands=2) as hands_recognizer:
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
                print(max(
                    distance(finger_tip, other_finger_tip)
                    for finger_tip in finger_tips
                    for other_finger_tip in finger_tips
                ))
                if max(
                    distance(finger_tip, other_finger_tip)
                    for finger_tip in finger_tips
                    for other_finger_tip in finger_tips
                ) < FINGER_PROXIMITY_THRESHOLD:
                    print(bit)
