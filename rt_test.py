from ultralytics import RTDETR
import cv2
import os

# Load a COCO-pretrained RT-DETR model
# model = RTDETR("weights/rtdetr-x.pt").to("cuda")
model = RTDETR("weights/rtdetr-l-sku110k.pt").to("cuda")

# Directory paths
input_dir = "imgs"
output_dir = "rt_outs"
os.makedirs(output_dir, exist_ok=True)

# Set a lower detection threshold for more detections
conf_threshold = 0.3  # Lowering the confidence threshold

# Process all images in the input directory
for img_file in os.listdir(input_dir):
    if img_file.endswith(('.jpg', '.png', '.jpeg')):
        # Full path to the image
        img_path = os.path.join(input_dir, img_file)
        
        # Run inference on the image
        results = model(img_path, conf=conf_threshold)
        
        # Plot the detection results on the image
        result_img = results[0].plot()
        
        # Save the result to disk
        output_path = os.path.join(output_dir, f"ft_detected_{img_file}")
        cv2.imwrite(output_path, result_img)
        print(f"Processed image saved as '{output_path}'")
