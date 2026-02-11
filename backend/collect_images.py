#collect_images.py
import os
import cv2

DATA_DIR = './video_dataset'
CLASS_NAME = 'Y'
DATASET_SIZE = 100
CAMERA_INDEX = 0 
FPS = 20
FRAME_SIZE = (640, 480) 

# -----------------------------
# Open webcam
# -----------------------------
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = None
recording = False
video_count = 0

print("📌 Controls:")
print("Press 'r' to START recording")
print("Press 's' to STOP & save video")
print("Press 'q' to QUIT")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read from camera")
        break

    if recording:
        video_writer.write(frame)
        cv2.putText(frame, "RECORDING...", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Video Capture", frame)
    key = cv2.waitKey(1) & 0xFF

    # Start recording
    if key == ord('r') and not recording:
        video_path = os.path.join(DATA_DIR, CLASS_NAME, f"Y_{video_count}.mp4")
        video_writer = cv2.VideoWriter(
            video_path, fourcc, FPS, FRAME_SIZE
        )
        recording = True
        print(f"🔴 Recording started: {video_path}")

    # Stop recording
    elif key == ord('s') and recording:
        recording = False
        video_writer.release()
        video_count += 1
        print("✅ Recording saved")

    # Quit
    elif key == ord('q'):
        break

cap.release()
if video_writer:
    video_writer.release()
cv2.destroyAllWindows()