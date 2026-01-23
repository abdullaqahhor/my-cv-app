import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import mediapipe as mp
import numpy as np

# MediaPipe sozlamalari
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Burchakni hisoblash funksiyasi
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return 360-angle if angle > 180.0 else angle

class FullAIProcessor(VideoProcessorBase):
    def __init__(self):
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.counter = 0
        self.stage = None
        self.user_name = "Aniqlanmoqda..."

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Mirror effekt
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(img_rgb)

        # 1. Yuzni tahlil qilish (Identifikatsiya simulyatsiyasi)
        if results.face_landmarks:
            self.user_name = "Abdulla (Admin)" # Bu yerda bazaga solishtirish mantiqi bo'ladi
            cv2.putText(img, f"User: {self.user_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            mp_drawing.draw_landmarks(img, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                                     mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1))

        # 2. Sport tahlili (Pose)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            angle = calculate_angle(shoulder, elbow, wrist)
            
            # Sanash mantiqi
            if angle > 160: self.stage = "Pastda"
            if angle < 30 and self.stage == "Pastda":
                self.stage = "Tepada"
                self.counter += 1

            # Vizual ma'lumotlar
            cv2.rectangle(img, (0, 400), (250, 480), (0, 0, 0), -1)
            cv2.putText(img, f"REPS: {self.counter}", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        return frame.from_ndarray(img, format="bgr24")

# --- INTERFEYS (UI) ---
st.set_page_config(page_title="Pro AI Vision Dashboard", layout="wide")

# Sidebar (Yon panel)
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103930.png", width=100)
    st.title("Sozlamalar")
    st.info("Bu tizim AI yordamida yuzni taniydi va mashqlarni nazorat qiladi.")
    mode = st.selectbox("Rejimni tanlang:", ["Sport tahlili", "Xavfsizlik", "Imo-ishora"])
    st.write("---")
    st.metric(label="Tizim holati", value="A'lo", delta="Online")

# Asosiy oyna
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(f"### 🎥 Jonli efir: {mode}")
    RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    webrtc_streamer(key="main-app", video_processor_factory=FullAIProcessor, rtc_configuration=RTC_CONFIGURATION)

with col2:
    st.markdown("### 📊 Statistika")
    st.success("Foydalanuvchi tasdiqlandi!")
    st.write("**Mashq turi:** Bicep Curls")
    st.warning("Eslatma: Tirsak burchagi 90 darajadan kam bo'lishiga e'tibor bering.")
    
    # Progress bar simulyatsiyasi
    st.write("Kunlik reja:")
    st.progress(45) # 45% bajarildi
