import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import mediapipe as mp
import numpy as np

# MediaPipe qo'l moduli
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing

class FinalProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.7)
        self.vol_per = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 1. Kursorni aniqlash (Ko'rsatkich barmog'i)
                idx = hand_landmarks.landmark[8]
                cx, cy = int(idx.x * w), int(idx.y * h)
                
                # Virtual kursorni chizish
                cv2.circle(img, (cx, cy), 20, (255, 0, 255), cv2.FILLED)

                # 2. Ovozni hisoblash (Bosh barmoq va Ko'rsatkich barmoq masofasi)
                thm = hand_landmarks.landmark[4]
                tx, ty = int(thm.x * w), int(thm.y * h)
                
                # Masofani o'lchash
                dist = np.hypot(cx - tx, cy - ty)
                # Ovozni 0% dan 100% gacha foizga aylantirish
                self.vol_per = np.interp(dist, [30, 150], [0, 100])

                # Ma'lumotlarni ekranga chiqarish
                cv2.line(img, (cx, cy), (tx, ty), (0, 255, 0), 2)
                cv2.putText(img, f"Ovoz: {int(self.vol_per)}%", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        return frame.from_ndarray(img, format="bgr24")

# --- INTERFEYS ---
st.set_page_config(page_title="AI Hand Controller")
st.title("🖐️ AI Virtual Controller")
st.write("Ko'rsatkich barmog'i - kursor. Ikki barmoq masofasi - ovoz.")

RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

webrtc_streamer(
    key="controller",
    video_processor_factory=FinalProcessor,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
