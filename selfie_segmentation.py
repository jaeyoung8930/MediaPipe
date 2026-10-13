#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import mediapipe as mp

# =========================================
# 🧩 Mediapipe Selfie Segmentation 초기화
# =========================================
mp_selfie = mp.solutions.selfie_segmentation
segmenter = mp_selfie.SelfieSegmentation(model_selection=1)
# model_selection: 0=기본(빠름, 근거리), 1=고해상도(정확, 배경 분리용)

# =========================================
# 📸 카메라 연결
# =========================================
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("face.mp4")  # 파일로 테스트 시

print("📷 Selfie Segmentation 시작 — ESC를 눌러 종료합니다.")

# =========================================
# 🎞️ 실시간 배경 분리 루프
# =========================================
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("⚠️ 프레임을 읽지 못했습니다. 카메라 연결을 확인하세요.")
        break

    # 좌우 반전 (셀카 모드)
    frame = cv2.flip(frame, 1)

    # BGR → RGB 변환
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 배경 분리 수행
    result = segmenter.process(rgb)

    # segmentation mask 생성
    mask = (result.segmentation_mask > 0.5).astype(np.uint8)

    # 배경 블러 생성
    blur = cv2.GaussianBlur(frame, (31, 31), 0)

    # 전경/배경 합성
    output = frame * mask[..., None] + blur * (1 - mask[..., None])
    output = output.astype(np.uint8)

    # 결과 화면 표시
    cv2.imshow("🪞 Selfie Segmentation (ESC to quit)", output)

    # ESC 키로 종료
    if cv2.waitKey(1) & 0xFF == 27:
        print("👋 종료합니다.")
        break

# =========================================
# 🔚 종료 처리
# =========================================
cap.release()
cv2.destroyAllWindows()
