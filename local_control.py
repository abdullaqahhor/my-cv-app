import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# 1. Ovozni boshqarish sozlamalari (Windows uchun)
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange() # (-65.25, 0.0)
minVol = volRange[0]
maxVol = volRange[1]

# 2. MediaPipe sozlamalari
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.7)
screen_w, screen_h = pyautogui.size()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    if not success: break
    
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_img)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Nuqtalarni olish
            landmarks = hand_landmarks.landmark
            index_finger = landmarks[8] # Ko'rsatkich barmog'i uchi
            thumb = landmarks[4]        # Bosh barmoq uchi

            # --- A. SICHQONCHA HARAKATI ---
            # Barmog'ingiz koordinatasini ekran o'lchamiga moslaymiz
            ix, iy = int(index_finger.x * screen_w), int(index_finger.y * screen_h)
            pyautogui.moveTo(ix, iy, _pause=False) # Kursorni qimirlatish

            # --- B. OVOZNI BOSHQARISH ---
            # Barmoqlar uchlari orasidagi masofani hisoblash (piksellarda emas, 0-1 nisbatda)
            dist = hypot(index_finger.x - thumb.x, index_finger.y - thumb.y)
            
            # Masofani ovoz darajasiga o'tkazish (0.05 dan 0.2 gacha bo'lgan oraliqni olamiz)
            vol = np.interp(dist, [0.05, 0.2], [minVol, maxVol])
            volume.SetMasterVolumeLevel(vol, None)

            # Vizualizatsiya
            mp.solutions.drawing_utils.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("AI Real Controller", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
