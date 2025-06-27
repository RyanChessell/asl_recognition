import os
import csv
import cv2
import mediapipe as mp

# Paths
RAW_DIR = os.path.join(os.getcwd(), "data", "raw", "asl_alphabet_train")
OUT_CSV = os.path.join(os.getcwd(), "data", "processed", "train_landmarks.csv")

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# Ensure output folder exists
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

# Write CSV header
with open(OUT_CSV, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    header = ["label"] + [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
    writer.writerow(header)

    # Loop over each letter folder
    for label in sorted(os.listdir(RAW_DIR)):
        folder = os.path.join(RAW_DIR, label)
        if not os.path.isdir(folder):
            continue
        print(f"Processing {label}...")
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            image = cv2.imread(img_path)
            if image is None:
                continue

            # Detect landmarks
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            if not results.multi_hand_landmarks:
                continue

            lm = results.multi_hand_landmarks[0].landmark
            row = [label]
            for pt in lm:
                row.extend([pt.x, pt.y, pt.z])
            writer.writerow(row)

print("Done! Landmarks saved to", OUT_CSV)
