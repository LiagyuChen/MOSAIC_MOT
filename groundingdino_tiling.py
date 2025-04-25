import os
import cv2
import torch
import numpy as np
from PIL import Image
import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model, predict


def crop_image_dynamic(image, tile_percentage, overlap_percentage):
    h, w, _ = image.shape
    tile_w, tile_h = int(w * tile_percentage), int(h * tile_percentage)
    overlap_w, overlap_h = int(tile_w * overlap_percentage), int(tile_h * overlap_percentage)

    crops = []
    for y in range(0, h, tile_h - overlap_h):
        for x in range(0, w, tile_w - overlap_w):
            x_end = min(x + tile_w, w)
            y_end = min(y + tile_h, h)
            tile = image[y:y_end, x:x_end]
            crops.append(((x, y, x_end, y_end), tile))
    return crops, (h, w)

def load_image(tile):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = np.asarray(tile)
    image_transformed, _ = transform(tile, None)
    return image, image_transformed

def annotate_tile(image, coords, model, text_prompt="hand", box_threshold=0.45, text_threshold=0.4):
    x_min, y_min, x_max, y_max = coords
    tile = Image.fromarray(cv2.cvtColor(image[y_min:y_max, x_min:x_max], cv2.COLOR_BGR2RGB))
    image_source, image_tensor = load_image(tile)

    # Run GroundingDINO prediction
    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device="cuda"
    )
    return image_source, boxes, logits, phrases

# Function to convert normalized bounding boxes to pixel coordinates
def convert_boxes_to_pixel_coordinates(boxes, image_source):
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h

    return torch.stack((x1, y1, x2, y2), dim=-1).numpy()

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Compute intersection area
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Compute areas of individual boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute union area
    union_area = box1_area + box2_area - inter_area

    # Avoid division by zero
    if union_area == 0:
        return 0

    return inter_area / union_area


def merge_boxes(boxes, crops, original_size):
    adjusted_boxes = []

    for tile_boxes, crop in zip(boxes, crops):
        crop_coords, _ = crop
        x_min_tile, y_min_tile, _, _ = crop_coords

        # Adjust each detection box within the tile to the original image coordinates
        for box in tile_boxes:
            x_min = box[0] + x_min_tile
            y_min = box[1] + y_min_tile
            x_max = box[2] + x_min_tile
            y_max = box[3] + y_min_tile

            adjusted_boxes.append([x_min, y_min, x_max, y_max])

    return adjusted_boxes


def detect_hands(image_folder, output_folder, config_file_path, model_path, tile_percentage, overlap_percentage):
    # Load the GroundingDINO model
    model = load_model(config_file_path, model_path)

    # Process each image in the folder
    for img_file in os.listdir(image_folder):
        if not img_file.endswith(".jpg") or not img_file.startswith("conf"):
            continue

        # Load the image
        image_path = os.path.join(image_folder, img_file)
        image = cv2.imread(image_path)

        # Crop image into tiles dynamically
        crops, original_size = crop_image_dynamic(image, tile_percentage, overlap_percentage)
        print(f"Processing {img_file}: {len(crops)} tiles.")

        all_boxes = []
        all_confidences = []
        for coords, _ in crops:
            # Annotate each tile
            image_source, boxes, logits, _ = annotate_tile(image, coords, model)
            detections = convert_boxes_to_pixel_coordinates(boxes, image_source)
            all_boxes.append(detections)
            all_confidences.extend(logits.cpu().numpy())
        
        # Merge boxes into original image coordinates
        adjusted_boxes = merge_boxes(all_boxes, crops, original_size)

        for i, box in enumerate(adjusted_boxes):
            x_min, y_min, x_max, y_max = map(int, box)
            label = "hand"
            confidence = all_confidences[i]

            # Draw the bounding box and label on the tile
            color = (0, 255, 0)  # Green for matched objects
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
            text = f"{label} ({confidence:.2f})"
            cv2.putText(image, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Save the annotated tile
        file_name = os.path.splitext(img_file)[0]
        output_name = os.path.join(output_folder, f"{file_name}_annotated.jpg")
        cv2.imwrite(output_name, image)
        print(f"Detected Image saved to {output_name}")


if __name__ == "__main__":
    image_folder = "imgs/hands/"
    output_folder = "outs/hand_tiling2/"
    config_file_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"
    model_path = "weights/groundingdino_swinb_cogcoor.pth"
    os.makedirs(output_folder, exist_ok=True)
    tile_percentage = 0.4
    overlap_percentage = 0.3
    detect_hands(image_folder, output_folder, config_file_path, model_path, tile_percentage, overlap_percentage)
