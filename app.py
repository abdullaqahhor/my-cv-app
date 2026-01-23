import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import mediapipe as mp

# MediaPipe sozlamalari
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing

class FingerCounterProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, c = img.shape
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        total_fingers = 0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Barmoq uchlari indekslari (MediaPipe bo'yicha)
                # Ko'rsatkich: 8, O'rta: 12, Nomsiz: 16, Kichik: 20
                finger_tips = [8, 12, 16, 20]
                fingers = []

                # 1. Bosh barmoqni tekshirish (Thumb)
                # Bosh barmoq uchi (4) uning asosi (2) dan o'ngda yoki chapda ekanligiga qarab
                if hand_landmarks.landmark[4].x > hand_landmarks.landmark[2].x:
                    fingers.append(1)
                else:
                    fingers.append(0)

                # 2. Qolgan 4 ta barmoqni tekshirish
                # Agar barmoq uchi (8, 12, 16, 20) o'rta bo'g'imdan (6, 10, 14, 18) yuqorida bo'lsa - ochiq
                for tip in finger_tips:
                    if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                total_fingers += fingers.count(1)
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Natijani ekranga chiqarish
        cv2.rectangle(img, (20, 20), (200, 120), (0, 0, 0), -1)
        cv2.putText(img, str(total_fingers), (45, 105), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
        cv2.putText(img, "RAQAM", (45, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame.from_ndarray(img, format="bgr24")

# --- UI ---
st.set_page_config(page_title="AI Finger Counter")
st.title("🔢 Barmoqlar bilan sanash")
st.write("Kameraga barmoqlaringizni ko'rsating (1 dan 10 gacha).")

RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

webrtc_streamer(
    key="finger-counter",
    video_processor_factory=FingerCounterProcessor,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
