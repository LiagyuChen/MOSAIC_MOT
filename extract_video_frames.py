import cv2
import os

# Define video path and output directory
video_path = "demo/final_results/pens/masa_pen_clip_outputs.mp4"
output_dir = "demo/final_results/pens/masa_frames"
os.makedirs(output_dir, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {video_path}")

frame_idx = 0

# Process each frame
while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit the loop if no more frames

    # Save the frame to the output directory
    frame_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg")
    cv2.imwrite(frame_path, frame)
    print(f"Saved: {frame_path}")

    frame_idx += 1

# Release the video capture object
cap.release()
print("Processing complete. Frames saved.")
