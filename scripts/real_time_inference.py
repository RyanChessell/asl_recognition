import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import joblib
from collections import deque

# Paths
STATIC_MODEL  = os.path.join("models", "asl_letter_mlp.keras")
STATIC_ENCODER= os.path.join("models", "label_encoder.pkl")
DYNAMIC_MODEL = os.path.join("models", "asl_dynamic_ml.keras")

# Load static model + label encoder
static_model = tf.keras.models.load_model(STATIC_MODEL)
static_le    = joblib.load(STATIC_ENCODER)

# Load dynamic model into a 30‐frame buffer
try:
    dynamic_model = tf.keras.models.load_model(DYNAMIC_MODEL)
    motion_buffer = deque(maxlen=30)
    use_dynamic = True
    print("Dynamic J/Z model loaded.")
except Exception:
    use_dynamic = False
    print("No dynamic model found—skipping motion detection.")

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res   = hands.process(rgb)

    if res.multi_hand_landmarks:
        lm_list = res.multi_hand_landmarks[0].landmark

        # --- Static letter prediction ---
        feats = []
        for pt in lm_list:
            feats.extend([pt.x, pt.y, pt.z])
        static_pred = static_model.predict(np.array([feats]), verbose=0)
        letter = static_le.inverse_transform([np.argmax(static_pred)])[0]
        cv2.putText(frame, f"Letter: {letter}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)

        # --- Dynamic motion prediction (J/Z) ---
        if use_dynamic:
            tip = lm_list[8]  # index fingertip
            motion_buffer.append([tip.x, tip.y])

            if len(motion_buffer) == motion_buffer.maxlen:
                seq = np.array(motion_buffer).flatten().reshape(1, -1)
                dyn_pred = dynamic_model.predict(seq, verbose=0)
                idx = np.argmax(dyn_pred)
                conf = dyn_pred[0, idx]
                if conf > 0.7:
                    motion = "J" if idx == 0 else "Z"
                    cv2.putText(frame, f"Motion: {motion}", (10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)

        # Draw hand landmarks
        mp_draw.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

    cv2.imshow("ASL Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
