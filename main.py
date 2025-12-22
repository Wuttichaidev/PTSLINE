import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def check_head_turn(image):
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if not results.multi_face_landmarks:
        return "No Face"

    for face_landmarks in results.multi_face_landmarks:
        # จุดปลายจมูก (Nose tip) คือจุดที่ 1
        # จุดขอบหน้าซ้าย คือจุดที่ 234
        # จุดขอบหน้าขวา คือจุดที่ 454
        
        img_h, img_w, _ = image.shape
        nose = face_landmarks.landmark[1].x
        left_side = face_landmarks.landmark[234].x
        right_side = face_landmarks.landmark[454].x

        # คำนวณ Ratio
        # ถ้าค่าเข้าใกล้ 0 หรือ 1 มากๆ แสดงว่าหันไปด้านนั้นสุดๆ
        ratio = (nose - left_side) / (right_side - left_side)

        if ratio < 0.35:
            return "Turn Left"
        elif ratio > 0.65:
            return "Turn Right"
        else:
            return "Forward"

    return "Unknown"