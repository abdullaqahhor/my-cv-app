import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import mediapipe as mp
import numpy as np

# Faqat Pose modelini yuklaymiz (Holistic'dan ancha yengil)
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return 360-angle if angle > 180.0 else angle

class OptimizedProcessor(VideoProcessorBase):
    def __init__(self):
        # Modelni yengil rejimda ishga tushiramiz
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0, # 0 - eng tezkor, 1 - o'rtacha, 2 - aniq
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.counter = 0
        self.stage = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        # Tasvir sifatini biroz pasaytirish (tezlikni oshiradi)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Nuqtalar
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            angle = calculate_angle(shoulder, elbow, wrist)
            
            # Mashq sanash
            if angle > 160: self.stage = "Pastda"
            if angle < 30 and self.stage == "Pastda":
                self.stage = "Tepada"
                self.counter += 1

            # Chizmalar
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(img, f"REPS: {self.counter}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        return frame.from_ndarray(img, format="bgr24")

st.set_page_config(page_title="Fast AI Trainer", layout="wide")
st.title("⚡ Tezkor AI Murabbiy")

col1, col2 = st.columns([3, 1])

with col1:
    webrtc_streamer(
        key="fast-vision",
        video_processor_factory=OptimizedProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": {"width": 640, "frameRate": 15}, "audio": False},
        async_processing=True # BU JUDA MUHIM: UI qotib qolmasligi uchun
    )

with col2:
    st.write("### Ko'rsatkichlar")
    st.info("Video qotsa, brauzerni yangilang (Refresh).")
