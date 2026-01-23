import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import mediapipe as mp
import numpy as np

# MediaPipe qo'l moduli
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing

class GestureProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0, # Tez ishlashi uchun
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.draw_color = (0, 255, 0)
        self.canvas = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        # Kanvas yaratish (chizish uchun)
        if self.canvas is None:
            self.canvas = np.zeros_like(img)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Ko'rsatkich barmog'i uchi (Index finger tip)
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                h, w, c = img.shape
                cx, cy = int(index_tip.x * w), int(index_tip.y * h)

                # Bosh barmoq uchi (Thumb tip) - masofani o'lchash uchun
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)

                # Barmoqlar orasidagi masofa (Click simulyatsiyasi)
                distance = np.sqrt((cx-tx)**2 + (cy-ty)**2)

                if distance < 40:
                    # Agar barmoqlar birlashsa - CHIZISH (qizil rangda)
                    cv2.circle(self.canvas, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
                    cv2.putText(img, "CHIZISH REJIMIDA", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    # Oddiy harakat - KO'RSATISH (yashil nuqta)
                    cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Kanvasni asosiy rasm bilan birlashtirish
        img = cv2.addWeighted(img, 1, self.canvas, 0.5, 0)
        return frame.from_ndarray(img, format="bgr24")

# --- UI ---
st.set_page_config(page_title="AI Virtual Mouse", layout="wide")
st.title("🖐️ AI Virtual Drawing & Gesture Control")

col1, col2 = st.columns([3, 1])

with col1:
    webrtc_streamer(
        key="gesture-control",
        video_processor_factory=GestureProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

with col2:
    st.write("### 🎮 Qanday boshqariladi?")
    st.info("""
    1. **Ko'rsatkich barmog'ingizni** ko'taring - bu sichqoncha kursorini harakatlantiradi.
    2. **Bosh barmoq va ko'rsatkich barmoqni** bir-biriga tekkizing - bu chizishni boshlaydi (Click).
    3. Ekranni tozalash uchun sahifani yangilang (Refresh).
    """)
    if st.button("Rasmni o'chirish"):
        st.rerun()
