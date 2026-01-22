import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import mediapipe as mp

# MediaPipe modullarini sozlash
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Tasvirni qayta ishlash
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(img_rgb)

        # 1. Yuzdagi nuqtalarni chizish (Face Mesh)
        mp_drawing.draw_landmarks(img, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1))

        # 2. Qo'llarni chizish
        mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # 3. Tana pozitsiyasini chizish (Pose)
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        return frame.from_ndarray(img, format="bgr24")

st.title("Senior Computer Vision Web App")
st.write("Telefoningiz kamerasini yoqing va natijani ko'ring!")

webrtc_streamer(key="example", video_processor_factory=VideoProcessor)
