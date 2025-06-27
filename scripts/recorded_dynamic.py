import os
import cv2
import numpy as np
import mediapipe as mp

LABELS = {"j": "J", "z": "Z"}
MAX_FRAMES = 30
OUT_BASE = os.path.join("data", "raw", "dynamic")

mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam.")

print("Press 'j' to record J, 'z' to record Z, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)
    cv2.imshow("Record Dynamic", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if chr(key) in LABELS:
        label = LABELS[chr(key)]
        traj = []
        print(f"Recording {label}…")

        while len(traj) < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = mp_hands.process(rgb)

            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0].landmark[8]
                traj.append([lm.x, lm.y])

            cv2.putText(frame, f"{label} {len(traj)}/{MAX_FRAMES}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Record Dynamic", frame)
            cv2.waitKey(1)

        out_dir = os.path.join(OUT_BASE, label)
        os.makedirs(out_dir, exist_ok=True)
        idx = len([f for f in os.listdir(out_dir) if f.endswith(".npy")])
        np.save(os.path.join(out_dir, f"{label}_{idx:03d}.npy"), np.array(traj))
        print(f"Saved {label}_{idx:03d}.npy")

cap.release()
cv2.destroyAllWindows()
