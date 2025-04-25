import os
import cv2
from ultralytics import RTDETR


# Load GroundingDINO model
video_path = "demo/toys/toys1.mp4"
output_dir = "demo/toys_output_frames_rtdetr"
os.makedirs(output_dir, exist_ok=True)
rtdetr_model = RTDETR("weights/rtdetr-x.pt").to("cuda")

# Open the video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {video_path}")

frame_idx = 0

# Process each frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = rtdetr_model(frame)
    # Plot the detection results on the image
    annotated_frame = results[0].plot()

    # Save the annotated frame to the output directory
    frame_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg")
    cv2.imwrite(frame_path, annotated_frame)
    print(f"Saved: {frame_path}")

    frame_idx += 1

# Release the video capture object
cap.release()
print("Processing complete. Annotated frames saved.")

