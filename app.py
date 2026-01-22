import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import mediapipe as mp

# MediaPipe modullarini yuklash
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        # Yuqori aniqlikdagi tahlil modeli
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )

    def recv(self, frame):
        # Kadrlarni massiv ko'rinishida olish
        img = frame.to_ndarray(format="bgr24")
        
        # Tasvirni RGB ga o'tkazish (MediaPipe uchun)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(img_rgb)

        # 1. Yuzdagi nuqtalarni chizish
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                img, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1)
            )

        # 2. Qo'llarni chizish (O'ng va Chap)
        mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # 3. Tana pozitsiyasini chizish
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # Qayta ishlangan kadrni qaytarish
        return frame.from_ndarray(img, format="bgr24")

# Veb-interfeys sarlavhasi
st.set_page_config(page_title="AI Vision", layout="wide")
st.title("Senior Computer Vision Web App")
st.write("Quyidagi 'Start' tugmasini bosing va kameraga ruxsat bering.")

# Google STUN serverlari - telefon brauzerida ulanish uchun shart
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)

# Video oqimini sozlash
webrtc_streamer(
    key="ai-vision-app",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True # Tezlikni oshirish uchun
)
