import cv2
import os

ROOT = "."

GESTURES = {
    "up": "UP",
    "down": "DOWN",
    "left": "LEFT",
    "right": "RIGHT"
}

OUTPUT = "NEW_FRAMES"
FRAME_SKIP = 3

os.makedirs(OUTPUT, exist_ok=True)

for label, folder in GESTURES.items():

    out_dir = os.path.join(OUTPUT, label)
    os.makedirs(out_dir, exist_ok=True)

    count = 0

    for file in os.listdir(folder):

        if not file.endswith(".mp4"):
            continue

        video_path = os.path.join(folder, file)
        cap = cv2.VideoCapture(video_path)

        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % FRAME_SKIP == 0:
                cv2.imwrite(
                    os.path.join(out_dir, f"{label}_{count}.jpg"),
                    frame
                )
                count += 1

            frame_id += 1

        cap.release()

    print(label, "frames extracted:", count)
