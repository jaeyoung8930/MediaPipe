from pathlib import Path

import cv2
import mediapipe as mp


# 이 파이썬 파일과 같은 폴더의 face.mp4를 엽니다.
video_path = Path(__file__).with_name("face.mp4")
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise SystemExit(f"동영상 파일을 열 수 없습니다: {video_path}")

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

try:
    with mp_face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
    ) as face_detection:

        while cap.isOpened():
            success, frame = cap.read()

            # 동영상 끝에 도달하면 종료합니다.
            if not success:
                break

            # MediaPipe 입력은 RGB 형식으로 변환합니다.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            # 얼굴이 감지되면 경계 상자와 주요 지점을 표시합니다.
            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(frame, detection)

            cv2.imshow("MediaPipe Face Detection", frame)

            # ESC 또는 q를 누르면 종료합니다.
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break
finally:
    cap.release()
    cv2.destroyAllWindows()