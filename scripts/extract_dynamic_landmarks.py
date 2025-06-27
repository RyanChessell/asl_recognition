import os
import cv2
import numpy as np
import mediapipe as mp

# Where the raw videos live
RAW_DIR = os.path.join("data", "raw", "dynamic")
# Fingertip trajectories
OUT_DIR = os.path.join("data", "processed", "dynamic")
# Number of frames per sample
MAX_FRAMES = 30

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Make sure output folders exist
for label in ("J", "Z"):
    os.makedirs(os.path.join(OUT_DIR, label), exist_ok=True)

# Loop over J and Z folders
for label in ("J", "Z"):
    in_folder = os.path.join(RAW_DIR, label)
    out_folder = os.path.join(OUT_DIR, label)

    for fname in sorted(os.listdir(in_folder)):
        in_path = os.path.join(in_folder, fname)
        cap = cv2.VideoCapture(in_path)
        traj = []

        # Read up to MAX_FRAMES frames, extracting the index fingertip
        while len(traj) < MAX_FRAMES and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = mp_hands.process(rgb)
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0].landmark[8]
                traj.append([lm.x, lm.y])
        cap.release()

        # If no hand was detected at all, skip this file
        if not traj:
            print(f"[WARN] No hand detected in {fname}, skipping.")
            continue

        # If the clip was shorter than MAX_FRAMES, pad with the last point
        while len(traj) < MAX_FRAMES:
            traj.append(traj[-1])

        arr = np.array(traj)  # shape (MAX_FRAMES, 2)
        out_path = os.path.join(out_folder, os.path.splitext(fname)[0] + ".npy")
        np.save(out_path, arr)
        print(f"Saved {out_path}")
