import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import mediapipe as mp
import numpy as np

# MediaPipe sozlamalari
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Burchakni hisoblash funksiyasi
def calculate_angle(a, b, c):
    a = np.array(a) # Yelka
    b = np.array(b) # Tirsak
    c = np.array(c) # Bilak
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
    return angle

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.counter = 0
        self.stage = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Kerakli nuqtalar (O'ng qo'l misolida)
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            # Burchakni o'lchash
            angle = calculate_angle(shoulder, elbow, wrist)
            
            # Sanash mantiqi
            if angle > 160:
                self.stage = "Pastda"
            if angle < 30 and self.stage == "Pastda":
                self.stage = "Tepada"
                self.counter += 1

            # Natijani ekranga chiqarish
            cv2.putText(img, f"Burchak: {int(angle)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, f"Soni: {self.counter}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            
            # Nuqtalarni chizish
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        return frame.from_ndarray(img, format="bgr24")

st.title("AI Fitness Trainer 🏋️‍♂️")
st.write(f"Harakatni boshlang! Biz sizning mashqlaringizni sanaymiz.")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

webrtc_streamer(key="fitness", video_processor_factory=VideoProcessor, rtc_configuration=RTC_CONFIGURATION)
