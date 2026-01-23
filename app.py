import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# 1. Rasmni raqamli kodga aylantiruvchi modelni yuklash
@st.cache_resource
def load_model():
    # MobileNetV2 - tasvirlarni tahlil qilish uchun juda yengil va tezkor model
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, pooling='avg')
    return base_model

model = load_model()

class UniversalProcessor(VideoProcessorBase):
    def __init__(self):
        self.database = {} # { 'nomi': [feature_vector] }
        self.last_frame = None

    def extract_features(self, frame):
        # Tasvirni modelga moslab tayyorlash
        img = cv2.resize(frame, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return model.predict(img, verbose=0)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.last_frame = img.copy()
        
        # Agar baza bo'sh bo'lmasa, eng yaqin o'xshashni topish
        label = "Tanimayapman..."
        if self.database:
            current_features = self.extract_features(img)
            best_match = None
            min_dist = float('inf')
            
            for name, saved_features in self.database.items():
                dist = np.linalg.norm(current_features - saved_features)
                if dist < min_dist:
                    min_dist = dist
                    best_match = name
            
            # Agar o'xshashlik chegarasi yetarli bo'lsa (Threshold)
            if min_dist < 10.0: 
                label = f"Bu: {best_match}"

        cv2.putText(img, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame.from_ndarray(img, format="bgr24")

# --- Streamlit Interfeysi ---
st.title("🧠 AI Universal Trainer")
ctx = webrtc_streamer(key="trainer", video_processor_factory=UniversalProcessor)

obj_name = st.text_input("Predmet nomini yozing (masalan: Banan, 1, Ruchka):")

if st.button("Saqlash (Datasetga qo'shish)"):
    if ctx.video_processor and ctx.video_processor.last_frame is not None:
        features = ctx.video_processor.extract_features(ctx.video_processor.last_frame)
        ctx.video_processor.database[obj_name] = features
        st.success(f"'{obj_name}' xususiyatlari saqlandi! Endi kameraga qayta ko'rsating.")
    else:
        st.error("Kamera hali ishga tushmadi!")
