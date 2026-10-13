from collections import deque
from pathlib import Path
import math

import cv2
import mediapipe as mp


# Input video filename as attached by the user (spelling is intentional).
BASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = BASE_DIR / "hand_dector_youtube.mp4"
OUTPUT_PATH = BASE_DIR / "hand_dector_youtube_recognized.mp4"

# Tune these thresholds if the gesture is not detected reliably.
TOUCH_RATIO = 0.35          # thumb-ring fingertip distance / palm width
PINKY_MOTION_RATIO = 0.22   # pinky movement over the time window / palm width
MOTION_WINDOW_SECONDS = 0.7


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def detect_target_pose(hand_landmarks, width, height, pinky_history):
    """Heuristically detect: index/middle crossed + thumb/ring touching + pinky moving."""
    pts = [(lm.x * width, lm.y * height) for lm in hand_landmarks.landmark]

    # Palm width is used to make thresholds scale-independent.
    palm_width = distance(pts[5], pts[17])
    if palm_width < 1:
        return False, False, False, 0.0

    # Thumb tip (4) close to ring-finger tip (16).
    thumb_ring_touch = distance(pts[4], pts[16]) / palm_width < TOUCH_RATIO

    # Crossing heuristic: index and middle finger tips swap left-right order
    # relative to their MCP joints. Works best with the palm facing the camera.
    base_dx = pts[5][0] - pts[9][0]
    tip_dx = pts[8][0] - pts[12][0]
    index_middle_crossed = base_dx * tip_dx < 0

    # Track pinky-tip position relative to the wrist so whole-hand translation
    # affects the measurement less. Accumulated path detects a wiggle/flex.
    wrist = pts[0]
    pinky_relative = ((pts[20][0] - wrist[0]) / palm_width,
                      (pts[20][1] - wrist[1]) / palm_width)
    pinky_history.append(pinky_relative)
    pinky_path = sum(
        distance(pinky_history[i - 1], pinky_history[i])
        for i in range(1, len(pinky_history))
    )
    pinky_moving = (len(pinky_history) >= 4 and
                    pinky_path >= PINKY_MOTION_RATIO)

    recognized = index_middle_crossed and thumb_ring_touch and pinky_moving
    return recognized, index_middle_crossed, thumb_ring_touch, pinky_path


def main():
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH}\n"
            "Put hand_dector_youtube.mp4 in the same folder as this .py file."
        )

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open the video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Could not read the video's frame dimensions.")

    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {OUTPUT_PATH}")

    mp_hands = mp.solutions.hands
    drawing = mp.solutions.drawing_utils
    styles = mp.solutions.drawing_styles
    window_frames = max(4, int(fps * MOTION_WINDOW_SECONDS))
    pinky_histories = {}
    frame_number = 0

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands:
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                frame_number += 1
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                if result.multi_hand_landmarks:
                    for hand_index, hand_lms in enumerate(result.multi_hand_landmarks):
                        history = pinky_histories.setdefault(
                            hand_index, deque(maxlen=window_frames)
                        )
                        recognized, crossed, touching, pinky_path = detect_target_pose(
                            hand_lms, width, height, history
                        )

                        drawing.draw_landmarks(
                            image,
                            hand_lms,
                            mp_hands.HAND_CONNECTIONS,
                            styles.get_default_hand_landmarks_style(),
                            styles.get_default_hand_connections_style(),
                        )

                        color = (0, 220, 0) if recognized else (0, 200, 255)
                        status = "TARGET GESTURE DETECTED" if recognized else "Gesture not complete"
                        y = 30 + hand_index * 72
                        cv2.putText(image, status, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.62, color, 2, cv2.LINE_AA)
                        details = f"crossed={crossed}  thumb-ring={touching}  pinky-motion={pinky_path:.2f}"
                        cv2.putText(image, details, (10, y + 25), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.43, color, 1, cv2.LINE_AA)
                else:
                    cv2.putText(image, "No hand detected", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                                cv2.LINE_AA)

                writer.write(image)
                cv2.imshow("Target hand gesture recognition", image)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    print(f"Done. Recognized video saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
