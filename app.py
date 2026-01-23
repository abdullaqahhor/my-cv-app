import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import mediapipe as mp
import numpy as np

# MediaPipe modullarini to'g'ridan-to'g'ri chaqirish (AttributeError oldini olish uchun)
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

# 1. Burchakni hisoblash funksiyasi
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return 360-angle if angle > 180.0 else angle

# 2. Asosiy Video ishlovchi klass
class OptimizedProcessor(VideoProcessorBase):
    def __init__(self):
        # Eng yengil model sozlamalari (qotib qolmasligi uchun)
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0, # 0 - eng tezkor model
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.counter = 0
        self.stage = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Mirror effekt
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Nuqtalar koordinatalarini olish (Chap qo'l misolida)
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            angle = calculate_angle(shoulder, elbow, wrist)
            
            # Sport sanash mantiqi
            if angle > 160: self.stage = "Pastda"
            if angle < 30 and self.stage == "Pastda":
                self.stage = "Tepada"
                self.counter += 1

            # Ekranga ma'lumotlarni chizish
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Dashboard qismi
            cv2.rectangle(img, (0,0), (250, 70), (0,0,0), -1)
            cv2.putText(img, f"ANGLE: {int(angle)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(img, f"REPS: {self.counter}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

        return frame.from_ndarray(img, format="bgr24")

# --- UI INTERFEYS ---
st.set_page_config(page_title="AI Pro Trainer", layout="wide")
st.title("🏋️‍♂️ AI Professional Fitness Dashboard")

# Tarmoq ulanishini kuchaytirish (SignallingTimeout oldini oladi)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}
    ]}
)

col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### 🎥 Jonli tahlil")
    webrtc_streamer(
        key="pro-trainer",
        video_processor_factory=OptimizedProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": {"width": 640, "height": 480, "frameRate": 15},
            "audio": False
        },
        async_processing=True
    )

with col2:
    st.markdown("### 📊 Ko'rsatkichlar")
    st.info("Kameraga yoningiz bilan turing.")
    st.metric(label="Mashq", value="Bicep Curls")
    if st.button("Hisobni nolga tushirish"):
        st.experimental_rerun()
