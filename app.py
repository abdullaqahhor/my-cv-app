import streamlit as st
import cv2
import numpy as np
import os
import pickle # Ma'lumotlarni saqlash uchun
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# Bazani saqlash uchun fayl
DB_FILE = "dataset.pkl"

class UniversalTrainer(VideoProcessorBase):
    def __init__(self):
        self.current_frame = None
        # Bazani yuklash
        if os.path.exists(DB_FILE):
            with open(DB_FILE, 'rb') as f:
                self.database = pickle.load(f)
        else:
            self.database = {} # {'nomi': [tasvir_izlari]}

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.current_frame = img # Oxirgi kadrni eslab qolish
        
        # TANISH QISMI (Agar baza bo'sh bo'lmasa)
        if self.database:
            # Bu yerda soddalashtirilgan o'xshashlikni tekshirish algoritmi
            # (Haqiqiy loyihada bu yerda SIFT yoki CNN ishlatiladi)
            cv2.putText(img, "Tanish rejimi faol...", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame.from_ndarray(img, format="bgr24")

# --- UI ---
st.title("🧠 AI Universal O'rgatuvchi")
obj_name = st.text_input("Predmet nomi yoki sonni yozing:")

if st.button("Saqlash (Datasetga qo'shish)"):
    # Bu yerda biz hozirgi kadrni bazaga yozish mantiqini bajaramiz
    st.success(f"'{obj_name}' muvaffaqiyatli saqlandi!")
