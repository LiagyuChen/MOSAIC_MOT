from ultralytics import RTDETR
import imageio

# Load the YOLOv11-L model
# model = RTDETR("weights/rtdetr-l-sku110k.pt").to("cuda")
model = RTDETR("weights/rtdetr-x.pt").to("cuda")
model.conf = 0.1  # Lower confidence threshold
model.iou = 0.1   # Adjust IoU threshold

# Define input and output video paths
# video_path = 'vids/products2.mp4'
# output_path = 'out_vids/products2_tracked.mp4'
video_path = 'track_demo/VID_20250111_184705.mp4'
output_path = 'track_demo/seg_detect_out/VID_20250111_184705_original2.mp4'

# Open the input video
reader = imageio.get_reader(video_path, format='ffmpeg')
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer(output_path, fps=fps)

for frame in reader:
    # Perform detection and tracking
    results = model.track(source=frame, tracker='bytetrack.yaml')
    
    # Visualize results
    annotated_frame = results[0].plot()  # Annotated frame with bounding boxes
    
    # Write the annotated frame to the output video
    writer.append_data(annotated_frame)

# Close the writer to save the output video
writer.close()
