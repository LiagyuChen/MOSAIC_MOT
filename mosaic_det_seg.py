import os
import cv2
import torch
import logging
import numpy as np
from PIL import Image
import multiprocessing
from ultralytics import YOLOE, FastSAM
from few_shot_features import FewShotProcessor
from typing import Dict, List, Tuple, Optional, Union, Any


class MosaicDetSeg:
    """
    MOSAIC Detection and Segmentation class for single-image few-shot detection workflow.
    Implements the complete pipeline from YOLO-E detection to feature extraction and matching.
    """

    def __init__(
        self,
        yoloe_model: str,
        sam_model: str,
        examples_root: str,
        cache_file: str = "mosaic_example_features.pkl",
        dinov2_model: str = "facebook/dinov2-base",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the MosaicDetSeg instance
        
        Args:
            yoloe_model: Path to YOLO-E model
            examples_root: Path to few-shot example images
            cache_file: Path to cache file for computed features
            dinov2_model: DINOv2 model name or path
            config: Configuration dictionary
        """
        self.cache_file = cache_file
        self.examples_root = examples_root
        self.device = "cpu"

        # Default configuration
        self.config = config

        # Load YOLO-E model
        self.yoloe_model = YOLOE(yoloe_model).to(self.device)

        # Load SAM model
        self.sam_model = FastSAM(sam_model).to(self.device)

        # Initialize feature processor
        self.feature_processor = FewShotProcessor(
            cache_file=cache_file,
            few_shot_image_dir=examples_root,
            dinov2_model=dinov2_model,
            config=self.config.get("few_shot_configs", {})
        )

        # Initialize result cache
        self.result_cache = {}

        # Prepare the exemplar features (loads from cache or computes)
        self.example_features = self.feature_processor.prepare_example_features()

        # Get class names from feature processor
        self.label_names = self.feature_processor.label_names
        self.label_mappings = {name: i+1 for i, name in enumerate(self.label_names)}
        
        print(f"Initialized MosaicDetSeg with {len(self.label_names)} classes")

    # Detection and Segmentation
    def _detect_with_yoloe(
        self,
        image: np.ndarray
    ) -> List[Dict]:
        """
        Detect objects using YOLO-E model and return results
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detection dictionaries with bbox, confidence, and mask
        """
        try:
            # Get original image shape
            orig_h, orig_w = image.shape[:2]

            # Run YOLO-E prediction (YOLO-E resizes internally)
            with torch.no_grad():
                results = self.yoloe_model.predict(image, conf=0.2, iou=0.5)

            if not results:
                return []

            result = results[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            masks_xy = result.masks.xy if result.masks else []

            # Create binary masks
            masks = np.zeros((len(masks_xy), orig_h, orig_w), dtype=np.uint8)
            for i, poly in enumerate(masks_xy):
                pts = np.array([poly], dtype=np.int32)
                cv2.fillPoly(masks[i], [pts], 1)

            # Final binary masks
            masks = (masks > 0).astype(np.uint8)

            # Assemble output detections
            detections = [
                {'bbox': box, 'mask': mask, 'confidence': conf}
                for box, mask, conf in zip(boxes, masks, confs)
            ]
            return detections

        except Exception as e:
            logging.error(f"Error in YOLO-E detection: {e}")
            raise e

    def _seg_with_fastsam(
        self,
        image: np.ndarray,
        detection_box: list
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Segment repaired objects with FastSAM"""
        # Get original dimensions
        large_image_threshold = self.config.get("large_image_threshold", 1088)

        # Calculate optimal size (maintain aspect ratio, multiple of 32)
        h, w = image.shape[:2]
        max_dim = max(h, w)
        if max_dim > large_image_threshold:
            scale = large_image_threshold / max_dim
            new_h, new_w = int(h * scale), int(w * scale)
            # Round to nearest multiple of 32
            new_h = ((new_h + 16) // 32) * 32
            new_w = ((new_w + 16) // 32) * 32
            image_small = cv2.resize(image, (new_w, new_h))
            image_rgb = cv2.cvtColor(image_small, cv2.COLOR_BGR2RGB)
            resize_factor = new_h / h  # Store for rescaling masks later
        else:
            # Don't resize smaller images
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            resize_factor = 1.0

        # Run inference with consistent resolution
        detection_box = list(map(int, detection_box))
        results = self.sam_model(image_rgb, device=self.device, retina_masks=True, bboxes=detection_box)

        masks = results[0].masks.data
        if len(masks) == 0:
            return {}
        print(f"Segmented {len(masks)} masks with bbox {detection_box}")

        # Convert masks to NumPy and resize to original image size
        mask_areas = []
        resized_masks = []

        for idx in range(len(masks)):
            mask_tensor = masks[idx].cpu().numpy().astype(np.uint8)
            if resize_factor != 1.0:
                mask_resized = cv2.resize(mask_tensor, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                mask_resized = mask_tensor
            
            print(f"mask_resized.shape: {mask_resized.shape}", "image.shape: ", image.shape)
            area = np.sum(mask_resized)
            mask_areas.append(area)
            resized_masks.append(mask_resized)

        # Find the mask with the largest area
        if not mask_areas:
            return None, None

        largest_idx = np.argmax(mask_areas)
        largest_mask = resized_masks[largest_idx]

        # Get bounding box from the largest mask
        bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(largest_mask)
        bbox = [bbox_x, bbox_y, bbox_x + bbox_w, bbox_y + bbox_h]

        return bbox, largest_mask
    
    # Filtering and Deduplication
    def _remove_background_masks(
        self,
        masks: np.ndarray,
        bbox_stack: np.ndarray,
        bbox_areas: np.ndarray,
        mask_areas: np.ndarray,
        total_area: float
    ) -> np.ndarray:
        """Remove masks that are too large (likely background)"""
        mask_area_thresh = self.config.get("mask_area_thresh", 0.8)
        conflict_thresh = self.config.get("conflict_thresh", 0.1)
        mask_contrain_ratio = self.config.get("mask_contrain_ratio", 0.95)

        to_keep = np.ones(len(masks), dtype=bool)

        # Vectorized computation of area ratios
        mask_area_ratios = mask_areas / total_area
        bbox_area_ratios = bbox_areas / total_area

        intersection_mask_mask = np.logical_and(masks[:, None], masks).sum(axis=(2, 3))
        intersection_bbox_mask = np.logical_and(bbox_stack[:, None], masks).sum(axis=(2, 3))
        mask_areas_eps = mask_areas + 1e-6
        
        # Ratio of intersection over target mask area
        inner_mask_ratio_matrix = intersection_mask_mask / mask_areas_eps[None, :]
        bbox_mask_ratio_matrix = intersection_bbox_mask / mask_areas_eps[None, :]

        # Masks considered "large" by mask or bbox area
        is_large = (mask_area_ratios > mask_area_thresh) | (bbox_area_ratios > mask_area_thresh)

        # Step 1: Remove large masks that almost fully enclose at least one other mask
        self_mask = np.eye(len(masks), dtype=bool)
        inner_mask_condition = (inner_mask_ratio_matrix >= mask_contrain_ratio) & ~self_mask
        has_enclosed_others = inner_mask_condition.any(axis=1)
        invalid_large_by_inner = is_large & has_enclosed_others

        # Step 2: Remove masks conflicting due to bbox shape and coverage
        conflict_condition = (bbox_mask_ratio_matrix >= mask_area_thresh) & (inner_mask_ratio_matrix <= conflict_thresh)
        has_conflict = conflict_condition.any(axis=1)

        # Update to_keep for valid indices
        to_keep &= ~(invalid_large_by_inner | has_conflict)

        # Clear numpy arrays
        del inner_mask_condition, has_enclosed_others, invalid_large_by_inner
        del conflict_condition, has_conflict

        return to_keep

    def _split_multi_object_masks(
        self,
        masks: np.ndarray,
        bboxes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Split masks that contain multiple disconnected objects"""
        new_masks = []
        new_bboxes = []
        bridge_thresh = self.config.get("bridge_thresh", 3)

        for i, bitmap in enumerate(masks):
            # Break thin bridges
            kernel = np.ones((bridge_thresh, bridge_thresh), dtype=np.uint8)
            broken = cv2.morphologyEx(bitmap.astype(np.uint8), cv2.MORPH_OPEN, kernel)

            num_labels, labels_cc = cv2.connectedComponents(broken)
            if num_labels <= 2:
                new_masks.append(bitmap)
                new_bboxes.append(bboxes[i])
                continue

            for label in range(1, num_labels):
                component = (labels_cc == label).astype(np.uint8)
                contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue

                poly = contours[0].squeeze()
                if poly.ndim != 2 or poly.shape[0] < 3:
                    continue

                new_masks.append(component)
                ys, xs = np.where(component)
                x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
                new_bboxes.append([x1, y1, x2, y2])

        # Clear numpy arrays
        del kernel, broken, num_labels, labels_cc, contours, bboxes, masks

        return np.array(new_bboxes), np.array(new_masks)

    def _remove_high_iou_bboxes(
        self,
        masks: np.ndarray,
        boxes: np.ndarray
    ) -> np.ndarray:
        conf_thresh_mask = self.config.get("mask_nms_iou", 0.9)
        conf_thresh_box = self.config.get("box_nms_iou", 0.9)
        n = len(boxes)

        # Recompute IoU matrices after filtering
        mask_inter = np.logical_and(masks[:, None], masks[None, :]).sum(axis=(2, 3)).astype(np.float32)
        mask_union = np.logical_or(masks[:, None], masks[None, :]).sum(axis=(2, 3)).astype(np.float32) + 1e-6
        mask_iou_matrix = mask_inter / mask_union

        xx1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
        yy1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
        xx2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
        yy2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
        iw = np.clip(xx2 - xx1, 0, None)
        ih = np.clip(yy2 - yy1, 0, None)
        inter_area = iw * ih
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = area[:, None] + area[None, :] - inter_area + 1e-6
        box_iou_matrix = inter_area / union_area

        # Apply greedy NMS correctly on filtered list
        to_keep = np.ones(n, dtype=bool)
        for i in range(n):
            if not to_keep[i]:
                continue
            to_keep[i+1:] &= ~(
                (mask_iou_matrix[i, i+1:] > conf_thresh_mask) |
                (box_iou_matrix[i, i+1:] > conf_thresh_box)
        )

        # Clear numpy arrays
        del mask_iou_matrix, box_iou_matrix, xx1, yy1, xx2, yy2, iw, ih, inter_area, area, union_area

        return to_keep

    def _remove_small_and_union_masks(
        self,
        mask_stack,
        mask_areas,
        to_keep,
    ):
        """Remove small masks and masks that are unions of complete others"""
        min_mask_area = self.config.get("min_mask_area", 800)
        area_match_thresh = self.config.get("area_match_thresh", 0.99)
        # Only process masks that are still kept
        valid_indices = np.where(to_keep)[0]
        if len(valid_indices) == 0:
            return to_keep

        # Remove tiny masks
        to_keep[valid_indices] &= mask_areas[valid_indices] > min_mask_area

        # Sort valid indices by area for processing
        sorted_valid_indices = valid_indices[np.argsort(mask_areas[valid_indices])[::-1]]
        
        # Remove masks that are unions of complete others
        for i in sorted_valid_indices:
            if not to_keep[i]:
                continue

            # Get current valid indices (excluding i)
            current_valid = np.where(to_keep)[0]
            current_valid = current_valid[current_valid != i]

            if len(current_valid) == 0:
                continue

            # Compute intersections with current valid masks
            inters = np.logical_and(mask_stack[i][None], mask_stack[current_valid])
            inter_areas = inters.sum(axis=(1, 2))
            contain_ratio = inter_areas / (mask_areas[current_valid] + 1e-6)

            # Find masks that are almost fully contained
            inner_ids = current_valid[contain_ratio >= area_match_thresh]        
            if len(inner_ids) >= 2:
                union_inner = np.any(mask_stack[inner_ids], axis=0)
                if np.sum(np.logical_and(union_inner, mask_stack[i])) / (mask_areas[i] + 1e-6) >= area_match_thresh:
                    to_keep[i] = False

        # Clear numpy arrays
        del valid_indices, sorted_valid_indices, current_valid, inner_ids

        return to_keep

    def _remove_partial_objects(
        self,
        mask_stack,
        bboxes_xyxy,
        bbox_areas,
        image_height,
        image_width,
        to_keep,
    ):
        """Remove partial objects based on bbox overlap"""
        iou_thresh_high = self.config.get("iou_thresh_high", 0.4)
        area_sim_thresh = self.config.get("area_sim_thresh", 0.25)
        inner_mask_containment_thresh = self.config.get("inner_mask_containment_thresh", 0.7)
        # Only process masks that are still kept
        valid_indices = np.where(to_keep)[0]
        if len(valid_indices) == 0:
            return to_keep

        # Precompute pairwise intersection areas for valid masks
        valid_bboxes = bboxes_xyxy[valid_indices]
        valid_bbox_areas = bbox_areas[valid_indices]

        x1 = np.maximum(valid_bboxes[:, None, 0], valid_bboxes[None, :, 0])
        y1 = np.maximum(valid_bboxes[:, None, 1], valid_bboxes[None, :, 1])
        x2 = np.minimum(valid_bboxes[:, None, 2], valid_bboxes[None, :, 2])
        y2 = np.minimum(valid_bboxes[:, None, 3], valid_bboxes[None, :, 3])
        iw = np.clip(x2 - x1, 0, None)
        ih = np.clip(y2 - y1, 0, None)
        inter_area = iw * ih

        # Compute containment ratios
        ratio_i_over_j = inter_area / (valid_bbox_areas[:, None] + 1e-6)
        ratio_j_over_i = inter_area / (valid_bbox_areas[None, :] + 1e-6)

        # Sort valid indices by area for processing
        sorted_valid_indices = valid_indices[np.argsort(valid_bbox_areas)[::-1]]
        for i in sorted_valid_indices:
            if not to_keep[i]:
                continue

            i_idx = np.where(valid_indices == i)[0][0]
            for j in sorted_valid_indices:
                if i == j or not to_keep[j]:
                    continue

                j_idx = np.where(valid_indices == j)[0][0]
                r_i = ratio_i_over_j[i_idx, j_idx]
                r_j = ratio_j_over_i[i_idx, j_idx]
                area_i = bbox_areas[i]
                area_j = bbox_areas[j]

                if r_i > iou_thresh_high or r_j > iou_thresh_high:
                    large_idx, small_idx = (i, j) if area_i > area_j else (j, i)
                    if abs(area_i - area_j) / max(area_i, area_j) < area_sim_thresh:
                        to_keep[small_idx] = False
                        continue

                    # Build exclusion mask
                    x1_l, y1_l, x2_l, y2_l = bboxes_xyxy[large_idx].astype(int)
                    x1_s, y1_s, x2_s, y2_s = bboxes_xyxy[small_idx].astype(int)
                    region = np.zeros((image_height, image_width), dtype=bool)
                    region[y1_l:y2_l, x1_l:x2_l] = True
                    region[y1_s:y2_s, x1_s:x2_s] = False
                    rest_area_mask = mask_stack[large_idx] & ~mask_stack[small_idx]

                    # Check if region contains other masks
                    candidates = [k for k in valid_indices if k not in {i, j} and to_keep[k]]
                    if candidates:
                        overlap_union = np.zeros((image_height, image_width), dtype=bool)
                        for k in candidates:
                            overlap_union |= np.logical_and(rest_area_mask, mask_stack[k])

                        containment_ratio = np.sum(overlap_union) / (np.sum(rest_area_mask) + 1e-6)

                        # Only discard large_idx if inner objects are connected
                        if np.any(containment_ratio > inner_mask_containment_thresh):
                            # connected_num = 0
                            # for k in candidates:
                            #     # mask_overlap = self._compute_mask_overlap(mask_stack[k], mask_stack[small_idx])
                            #     bbox_overlap = self._compute_bbox_overlap(bboxes_xyxy[k], bboxes_xyxy[small_idx])
                            #     # if self._masks_are_connected(mask_stack[k], mask_stack[small_idx]) and 0.15 < bbox_overlap < 0.3:
                            #     if self._masks_are_connected(mask_stack[k], mask_stack[small_idx]):
                            #         connected_num += 1

                            # if connected_num >= 1:
                            #     to_keep[small_idx] = False  # These small parts are part of full object
                            # else:
                            #     to_keep[large_idx] = False  # Truly overlapping other objects
                            to_keep[large_idx] = False
                        else:
                            to_keep[small_idx] = False
                        
                        break

        # Clear numpy arrays
        del valid_bboxes, valid_bbox_areas, x1, y1, x2, y2, iw, ih, inter_area, ratio_i_over_j, ratio_j_over_i, area_i, area_j, r_i, r_j

        return to_keep
    

    # Check if excluded region is spatially connected to included one
    def _masks_are_connected(self, mask1, mask2, kernel_size=1):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated1 = cv2.dilate(mask1.astype(np.uint8), kernel, iterations=1)
        overlap = np.logical_and(dilated1, mask2)
        return np.any(overlap)
    
    def _compute_mask_overlap(
        self,
        mask1: np.ndarray,
        mask2: np.ndarray
    ) -> np.ndarray:
        """Compute IoU between masks (vectorized)"""
        mask1 = mask1.astype(bool)
        mask2 = mask2.astype(bool)
        intersection = np.logical_and(mask1, mask2).sum()
        area1 = mask1.sum()
        area2 = mask2.sum()
        smaller_area = min(area1, area2)

        if smaller_area == 0:
            return 0.0

        return intersection / (smaller_area + 1e-6)

    def _compute_mask_iou(
        self,
        masks1: np.ndarray,
        masks2: np.ndarray
    ) -> np.ndarray:
        """Compute IoU between masks (vectorized)"""
        masks1 = masks1.astype(bool)
        masks2 = masks2.astype(bool)

        # Reshape if needed
        if masks1.ndim == 2:
            masks1 = masks1.reshape(1, *masks1.shape)
        if masks2.ndim == 2:
            masks2 = masks2.reshape(1, *masks2.shape)

        # Compute intersection and union
        intersection = np.logical_and(masks1[:, None], masks2[None, :]).sum(axis=(2, 3))
        union = np.logical_or(masks1[:, None], masks2[None, :]).sum(axis=(2, 3))

        # Compute IoU
        iou = intersection / (union + 1e-6)
        return iou
    
    def _compute_bbox_overlap(
        self,
        bbox1: np.ndarray,
        bbox2: np.ndarray
    ) -> np.ndarray:
        """Compute IoU between bboxes (vectorized)"""
        x1 = np.maximum(bbox1[0], bbox2[0])
        y1 = np.maximum(bbox1[1], bbox2[1])
        x2 = np.minimum(bbox1[2], bbox2[2])
        y2 = np.minimum(bbox1[3], bbox2[3])
        iw = np.clip(x2 - x1, 0, None)
        ih = np.clip(y2 - y1, 0, None)
        inter_area = iw * ih
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        smaller_area = min(area1, area2)
        if smaller_area == 0:
            return 0.0
        return inter_area / (smaller_area + 1e-6)



    def _remove_contained_masks(
        self,
        mask_stack,
        mask_areas,
        to_keep,
    ):
        """Remove masks that are fully contained by larger masks"""
        mask_contained_threshold = self.config.get("mask_contained_threshold", 0.9)
        mask_contained_threshold = 0.75
        # Only process masks that are still kept
        valid_indices = np.where(to_keep)[0]
        if len(valid_indices) == 0:
            return to_keep

        # Sort valid indices by area for processing
        valid_mask_areas = mask_areas[valid_indices]
        sorted_valid_indices = valid_indices[np.argsort(valid_mask_areas)]

        for i in sorted_valid_indices:
            if not to_keep[i]:
                continue

            overlap = np.logical_and(mask_stack[i][None], mask_stack[to_keep])
            overlap_area = overlap.sum(axis=(1, 2))
            ratios = overlap_area / (mask_areas[i] + 1e-6)
            if np.any((ratios >= mask_contained_threshold) & (np.where(to_keep)[0] != i)):
                to_keep[i] = False

        # Clear numpy arrays
        del valid_mask_areas, sorted_valid_indices, overlap, overlap_area, ratios

        return to_keep

    def _filter_and_deduplicate_detections(
        self,
        detections: List[Dict]
    ) -> List[Dict]:
        """
        Combines background mask removal and deduplication using IoU matrices and geometric heuristics.
        """
        if not detections:
            return []

        H, W = self.image_height, self.image_width
        total_area = H * W

        # Extract masks and boxes scores
        masks = np.stack([det['mask'] > 0 for det in detections])
        boxes = np.array([det['bbox'] for det in detections])

        bbox_stack = np.zeros((len(masks), H, W), dtype=np.uint8)
        bbox_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        mask_areas = masks.sum(axis=(1, 2)).astype(np.float32)

        bboxes_int = boxes.astype(int)
        for i in range(len(masks)):
            x1, y1, x2, y2 = bboxes_int[i]
            bbox_stack[i, y1:y2, x1:x2] = 1

        masks = masks.astype(bool)
        bbox_stack = bbox_stack.astype(bool)

        to_keep = self._remove_background_masks(masks, bbox_stack, bbox_areas, mask_areas, total_area)

        filtered_indices = np.where(to_keep)[0]
        order = np.argsort(-bbox_areas[filtered_indices])
        detections = [detections[i] for i in filtered_indices[order]]
        masks = masks[filtered_indices[order]]
        boxes = boxes[filtered_indices[order]]
        mask_areas = mask_areas[filtered_indices[order]]

        # Split masks that contain multiple disconnected objects
        new_bboxes, new_masks = self._split_multi_object_masks(masks, boxes)

        # Remove detections that have high IoU with other detections
        to_keep = self._remove_high_iou_bboxes(new_masks, new_bboxes)
        new_masks = new_masks[to_keep]
        new_bboxes = new_bboxes[to_keep]
        new_mask_areas = new_masks.sum(axis=(1, 2)).astype(np.float32)
        new_bbox_areas = (new_bboxes[:, 2] - new_bboxes[:, 0]) * (new_bboxes[:, 3] - new_bboxes[:, 1])
        to_keep = np.ones(len(new_masks), dtype=bool)

        to_keep = self._remove_small_and_union_masks(new_masks, new_mask_areas, to_keep)
        to_keep = self._remove_partial_objects(new_masks, new_bboxes, new_bbox_areas, H, W, to_keep)
        to_keep = self._remove_contained_masks(new_masks, new_mask_areas, to_keep)

        final_detections = [
            {
                'bbox': new_bboxes[i],
                'mask': new_masks[i],
            }
            for i in range(len(new_bboxes)) if to_keep[i]
        ]
        discarded_detections = [
            {
                'bbox': new_bboxes[i],
                'mask': new_masks[i],
            }
            for i in range(len(new_bboxes)) if not to_keep[i]
        ]

        # Clear numpy arrays
        del masks, boxes, bbox_stack, bbox_areas, mask_areas
        del new_bboxes, new_masks, new_mask_areas, new_bbox_areas

        return final_detections, discarded_detections

    # Feature Extraction
    def _extract_detection_features(
        self,
        image: np.ndarray,
        detections: List[Dict]
    ) -> List[Dict]:
        """Extract features from detected objects by masking and cropping the image."""
        detection_features = []

        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            mask = detection['mask']
            # Convert bbox to integer coordinates
            x1, y1, x2, y2 = map(int, bbox[:4])

            # Convert mask to uint8
            mask_uint8 = mask.astype(np.uint8)

            # Apply mask to the full image to extract the object
            masked_image = cv2.bitwise_and(image, image, mask=mask_uint8)

            # Crop the masked image and mask
            cropped_img = masked_image[y1:y2, x1:x2]

            # Skip if cropped regions are empty
            if cropped_img.size == 0:
                continue

            # Extract features using the few-shot processor
            features = self.feature_processor.extract_full_features(cropped_img)
            if features is None:
                continue

            # Store detection info and features
            detection_features.append({
                'bbox': bbox,
                'mask': mask,
                'embedding': features.get('embedding'),
                'patches': features.get('patches', []),
                'contour': features.get('contour', None)
            })

        return detection_features

    # Object Matching and Selection
    def _compute_embedding_similarities(
        self,
        detection_features: List[Dict]
    ) -> Dict[str, np.ndarray]:
        """Compute cosine similarity between detection embeddings and all example embeddings."""
        embedding_similarities = {}
        detection_embeddings = np.array([det['embedding'] for det in detection_features])
        for class_name in self.label_names:
            ex_embeddings = self.example_features[class_name].get('embeddings', [])
            if ex_embeddings:
                ex_embeddings_np = np.array(ex_embeddings)
                sim_matrix = self.feature_processor.precompute_similarities(detection_embeddings, ex_embeddings_np)
                embedding_similarities[class_name] = np.max(sim_matrix, axis=1)
            else:
                embedding_similarities[class_name] = np.zeros(len(detection_features))
        return embedding_similarities

    def _apply_adaptive_gating(
        self,
        detection_features: List[Dict],
        embedding_similarities: Dict[str, np.ndarray]
    ) -> List[Tuple[int, str, float]]:
        """
        Apply adaptive gating to select detections based on highest class similarity scores.

        Returns:
            List of selected (detection_index, class_name, similarity_score) tuples.
        """
        if not detection_features or not embedding_similarities:
            return []

        similarity_threshold = self.config.get("similarity_threshold", 0.65)

        # Stack all class similarity scores into a 2D matrix: shape (num_classes, num_detections)
        class_names = list(embedding_similarities.keys())
        similarity_matrix = np.stack([embedding_similarities[cls] for cls in class_names])  # (C, D)

        # Get best score and corresponding class index per detection
        best_scores = similarity_matrix.max(axis=0)
        best_class_indices = similarity_matrix.argmax(axis=0)

        # Apply threshold
        valid_mask = best_scores >= similarity_threshold
        selected_indices = np.where(valid_mask)[0]

        # Build selected results
        selected_detections = [
            (int(i), class_names[int(best_class_indices[i])], float(best_scores[i]))
            for i in selected_indices
        ]

        return selected_detections

    # Handle Over-matching and Partial-matching cases
    def _compute_cosine_similarity(
        self,
        query_features: np.ndarray,
        example_features: np.ndarray
    ) -> np.ndarray:
        """Helper to compute cosine similarity, wrapper around precompute_similarities."""
        if isinstance(query_features, list):
            query_features = np.array(query_features)
        if isinstance(example_features, list):
            example_features = np.array(example_features)

        if query_features.ndim == 1:
            query_features = query_features.reshape(1, -1)
        if example_features.ndim == 1:
            example_features = example_features.reshape(1, -1)
        return self.feature_processor.precompute_similarities(query_features, example_features)

    def compute_shape_similarity(
        self,
        features: Dict[str, np.ndarray],
        class_name: str
    ) -> Tuple[float, str]:
        """Compute shape similarity based on contour and patch features."""
        # Get the detection and example contour features
        detection_contour_vector = features.get('contour', None)
        det_contour_shape_vector = np.array(detection_contour_vector[:3])
        det_contour_hu_moments = np.array(detection_contour_vector[3:])
        example_contour_vectors = self.example_features.get(class_name, {}).get('contours', None)
        example_contour_shape_vectors = np.array([vector[:3] for vector in example_contour_vectors])
        example_contour_hu_moments = np.array([vector[3:] for vector in example_contour_vectors])

        # Compute absolute difference between detection and each exemplar contour shapes
        shape_diffs = [np.abs(det_contour_shape_vector - ex_shape_vec) for ex_shape_vec in example_contour_shape_vectors]
        shape_diff_means = np.mean(shape_diffs, axis=0) if shape_diffs else np.zeros(3)
        shape_diff_score = np.mean(shape_diff_means)
        print("Shape deviation score:", shape_diff_score)

        # Calculate L2 distance of hu-moments between contour features
        sigma = 1.0
        hu_moment_sims = []
        det_contour_hu_moments = det_contour_hu_moments[:4]
        for example_hu_moment in example_contour_hu_moments:
            example_hu_moment = example_hu_moment[:4]
            m1_norm = det_contour_hu_moments / (np.linalg.norm(det_contour_hu_moments) + 1e-8)
            m2_norm = example_hu_moment / (np.linalg.norm(example_hu_moment) + 1e-8)
            dist = np.linalg.norm(m1_norm - m2_norm)
            sim = np.exp(-dist / sigma)
            hu_moment_sims.append(sim)

        contour_similarity = float(np.mean(hu_moment_sims)) if hu_moment_sims else 0.0
        print("Contour similarity: ", contour_similarity)

        combined_shape_score = contour_similarity * (1 - shape_diff_score)

        # Pairwise similarity between detection and example patches
        patch_embeddings = np.array([patch.get('patch_embedding', []) for patch in features.get('patches', [])])
        example_patch_embeddings = np.array([patch.get('patch_embedding', []) for patch in self.example_features.get(class_name, {}).get('patches', [])])
        example_embeddings = np.array(self.example_features.get(class_name, {}).get('embeddings', []))

        patch_similarity = self._compute_cosine_similarity(patch_embeddings, example_embeddings)
        patch_pair_similarity = self._compute_cosine_similarity(patch_embeddings, example_patch_embeddings)
        
        # Calculate patch coverage fraction -- Over case
        over_match_thresh = self.config.get('over_match_thresh', 0.45)
        matched_det = np.sum(patch_similarity.mean(axis=1) >= over_match_thresh)
        det_patch_coverage = matched_det / len(patch_embeddings)

        # Calculate patch missing fraction -- Partial case
        patch_match_thresh = self.config.get('patch_match_thresh', 0.45)
        partial_matched_det = np.sum(patch_pair_similarity.mean(axis=0) < patch_match_thresh)
        det_patch_missing = partial_matched_det / len(patch_pair_similarity[0])

        # Calculate patch score -- Averaged best-matched patch similarity
        patch_score = np.mean(patch_pair_similarity.max(axis=1))
        combined_patch_score = patch_score * (1 - det_patch_coverage) * (1 - det_patch_missing)
        print("Patch Coverage: ", det_patch_coverage, "Patch Missing: ", det_patch_missing, "Patch Embedding Score: ", patch_score, "Combined Score: ", combined_patch_score)

        # Determine repair mode
        repair = False
        shape_diff_thresh = self.config.get('shape_diff_thresh', 0.2)
        patch_coverage_thresh = self.config.get('patch_coverage_thresh', 0.3)
        patch_missing_thresh = self.config.get('patch_missing_thresh', 0.05)
        if (shape_diff_score >= shape_diff_thresh and det_patch_coverage <= patch_coverage_thresh and det_patch_missing >= patch_missing_thresh):
            repair = True

        # Forming the weighted shape similarity score
        contour_weight = self.config.get('contour_weight', 0.5)
        patch_weight = self.config.get('patch_weight', 0.5)
        total = contour_weight + patch_weight
        contour_weight /= total
        patch_weight /= total
        shape_score = combined_shape_score * contour_weight + combined_patch_score * patch_weight

        all_scores = {
            'shape_score': shape_score,
            'contour_similarity': contour_similarity,
            'patch_score': patch_score,
            'combined_patch_score': combined_patch_score,
            'combined_shape_score': combined_shape_score,
            'det_patch_coverage': det_patch_coverage,
            'det_patch_missing': det_patch_missing
        }
        return shape_score, repair, all_scores

    def _repair_low_margin_detections(
        self,
        all_detection_features: List[Dict],
        selected_detections: List[Tuple[int, str, float]],
        discarded_detections: List[Dict]
    ) -> List[Dict]:
        """
        Refined repair logic:
        1. Classify all detections as 'exact', 'partial', or 'over'.
        2. For 'over' detections, discard if overlapping with smaller detections.
        3. For 'partial' detections, discard if contained within larger ones.
        4. Expand remaining 'partial' and rerun; rerun 'over' with larger region.
        """
        final_detections = []
        repair_meta = []

        # Step 1: Classify all detections
        for det_idx, cls, score in selected_detections:
            feats = all_detection_features[det_idx]
            shape_score, repair, all_scores = self.compute_shape_similarity(feats, cls)
            updated_score = score * 0.6 + shape_score * 0.4
            print("class label: ", cls, "updated_score: ", updated_score, "score: ", score, "shape_score: ", shape_score, "repair: ", repair)
            print("--------------------------------")
            repair_meta.append({
                'class_name': cls,
                'score': updated_score,
                'all_scores': all_scores,
                'repair': repair,
                'bbox': feats['bbox'],
                'mask': feats['mask'],
                'features': feats,
                'discard': False,
                'discarded_detection': discarded_detections if det_idx == 0 else None
            })

        # Step 2: Precompute pairwise mask overlaps and bounding box overlaps using vectorized operations
        masks = np.array([meta['mask'] for meta in repair_meta])
        bboxes = np.array([meta['bbox'] for meta in repair_meta])

        # Vectorized mask overlap computation
        mask_intersections = np.logical_and(masks[:, None], masks).sum(axis=(2, 3))
        mask_areas = masks.sum(axis=(1, 2))
        mask_overlaps = mask_intersections / (mask_areas[:, None] + 1e-6)

        # Vectorized bounding box overlap computation
        xx1 = np.maximum(bboxes[:, None, 0], bboxes[None, :, 0])
        yy1 = np.maximum(bboxes[:, None, 1], bboxes[None, :, 1])
        xx2 = np.minimum(bboxes[:, None, 2], bboxes[None, :, 2])
        yy2 = np.minimum(bboxes[:, None, 3], bboxes[None, :, 3])
        iw = np.clip(xx2 - xx1, 0, None)
        ih = np.clip(yy2 - yy1, 0, None)
        inter_area = iw * ih
        bbox_areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        bbox_overlaps = inter_area / (bbox_areas[:, None] + 1e-6)

        # Step 3: Overlap and containment analysis with merging logic
        inner_mask_over_thresh = self.config.get("inner_mask_over_thresh", 0.7)
        for i, meta_i in enumerate(repair_meta):
            if not meta_i['repair']:
                continue
            for j, meta_j in enumerate(repair_meta):
                if i == j:
                    continue
                if mask_overlaps[i, j] > inner_mask_over_thresh:
                    print("Overlap detected, discard the over-matched detection")
                    meta_i['discard'] = True
                    break
                # Check if bounding boxes overlap and belong to the same class
                if meta_i['class_name'] == meta_j['class_name'] and bbox_overlaps[i, j] > 0.05:
                    # Merge masks and bounding boxes
                    meta_i['mask'] = np.logical_or(meta_i['mask'], meta_j['mask'])
                    x1 = min(meta_i['bbox'][0], meta_j['bbox'][0])
                    y1 = min(meta_i['bbox'][1], meta_j['bbox'][1])
                    x2 = max(meta_i['bbox'][2], meta_j['bbox'][2])
                    y2 = max(meta_i['bbox'][3], meta_j['bbox'][3])
                    meta_i['bbox'] = [x1, y1, x2, y2]
                    meta_j['discard'] = True  # Mark the second detection for discard

        # Step 4: Apply repair logic
        for meta in repair_meta:
            if meta['discard']:
                continue

            detection_data_features = meta['features']
            detection_data_features.pop('bbox')
            detection_data_features.pop('mask')
            detection_data = {
                'bbox': meta['bbox'],
                'mask': meta['mask'],
                'score': meta['score'],
                'class_name': meta['class_name'],
                'features': detection_data_features,
                'discarded_detection': meta['discarded_detection']
            }
            final_detections.append(detection_data)

        return final_detections

    # Mosaic Workflow
    def detect_and_segment(
        self,
        image: Union[np.ndarray, Image.Image]
    ) -> List[Dict]:
        """
        Detect and segment objects following the described pipeline:
        YOLOe -> Deduplicate -> Extract Features -> Filter by Embedding Sim -> Repair -> Finalize
        """
        # Prepare Input Image
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Keep original frame for potential use in repair steps
        self.frame = image.copy() 
        self.image_height, self.image_width = image.shape[:2]

        # Ensure image is 3-channel RGB
        if image.ndim == 2:
             image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = image[:, :, :3]

        # Initial Detection (YOLO-E)
        initial_detections = self._detect_with_yoloe(image)
        print(f"Initial YOLO-E detections: {len(initial_detections)}")
        if not initial_detections:
            return []

        # Deduplicate detections
        deduplicated_detections, discarded_detections = self._filter_and_deduplicate_detections(initial_detections)
        print(f"Detections after deduplication: {len(deduplicated_detections)}")
        print(f"Discarded detections: {len(discarded_detections)}")
        if not deduplicated_detections:
            return []

        # Extract comprehensive features from detections
        all_detection_features = self._extract_detection_features(
            image,
            deduplicated_detections
        )
        print(f"Features extracted for {len(all_detection_features)} detections.")
        if not all_detection_features:
            return [] 

        # Compute embedding similarities to match detections to examples
        embedding_similarities = self._compute_embedding_similarities(all_detection_features)
        selected_detections_tuples = self._apply_adaptive_gating(
            all_detection_features,
            embedding_similarities
        )
        print(f"Detections after initial similarity gating: {len(selected_detections_tuples)}")
        if not selected_detections_tuples:
            return []

        # Patch-Level Repair
        final_detections = self._repair_low_margin_detections(
            all_detection_features,
            selected_detections_tuples,
            discarded_detections
        )
        print(f"Detections after repair stage: {len(final_detections)}")

        # Clear all used memory and variables, including numpy arrays, keeping only final_detections
        del initial_detections
        del deduplicated_detections
        del discarded_detections
        del all_detection_features
        del embedding_similarities
        del selected_detections_tuples

        # Optionally clear any other large variables or caches
        torch.cuda.empty_cache()  # Clear CUDA memory if using GPU
        
        return final_detections

    # Visualize the annotations
    def _get_color(
        self,
        class_name: str
    ) -> Tuple[int, int, int]:
        """Get consistent color for a class"""
        # Hash the class name to get a consistent color
        hash_val = hash(class_name if class_name else "unknown")
        r = (hash_val & 0xFF0000) >> 16
        g = (hash_val & 0x00FF00) >> 8
        b = hash_val & 0x0000FF
        return (int(r), int(g), int(b))

    def visualize_results(
        self,
        image: Union[np.ndarray, Image.Image],
        results: List[Dict],
        output_path: str = None
    ) -> np.ndarray:
        """Visualize detection and segmentation results using final output format."""
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Create a copy of the image for visualization
        vis_image = image.copy()
        H, W = vis_image.shape[:2]

        # Draw each detection
        alpha = self.config.get("visualization_alpha", 0.6)
        for det in results:
            try:
                # Get detection info
                bbox = det.get('bbox')
                mask = det.get('mask')
                score = det.get('score', 0.0)
                class_name = det.get('class_name', 'unknown')

                # Get color for this class
                color = self._get_color(class_name)

                # Draw mask if available
                if mask is not None and mask.size > 0 and mask.ndim == 2:
                    # Ensure mask has same dimensions as image
                    if mask.shape[:2] != (H, W):
                        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

                    # Ensure mask is binary (already should be uint8, but check > 0)
                    binary_mask = (mask > 0)

                    # Create colored mask overlay
                    mask_color = np.zeros_like(vis_image)
                    mask_color[binary_mask] = color

                    # Blend with original image
                    vis_image = cv2.addWeighted(vis_image, 1, mask_color, alpha, 0)

                # Draw bbox if available
                if bbox is not None and len(bbox) >= 4:
                    x1, y1, x2, y2 = map(int, bbox[:4])
                    # Clip coordinates to image bounds
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(W - 1, x2), min(H - 1, y2)
                    # Draw only if valid box
                    if x1 < x2 and y1 < y2:
                         cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                         # Draw label
                         label = f"{class_name} {score:.2f}"

                         label_height = y1 - 10 if y1 - 10 > 10 else y1 + 20
                         cv2.putText(vis_image, label, (x1, label_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            except Exception as e:
                logging.error(f"Error visualizing detection: {det}. Error: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Save output if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            try:
                cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            except Exception as e:
                 logging.error(f"Failed to save visualization to {output_path}: {e}")

        return vis_image



if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    config = {
        # Few-Shot Selector configuration
        "few_shot_configs": {
            # DINOv2 layers
            "early_layer": 4,
            "mid_layer": 8,
            "late_layer": 12,
            "early_layer_weight": 0.1,
            "mid_layer_weight": 0.3,
            "late_layer_weight": 0.6,
            # Performance settings
            "max_workers": min(16, multiprocessing.cpu_count()),
            "batch_size": 32,
            # Image processing
            "min_image_size": 20,
            "aug_angles": [30, 60, 90, 150, 180, 270],
            "aug_scales": [0.5, 0.75],
            # SLIC settings
            "slic_n_segments": 16,
            "slic_compactness": 15,
            "slic_sigma": 3.0,
        },

        # Mosaic Detection and Segmentation configuration
        "large_image_threshold": 1088,
        "mask_area_thresh": 0.8,
        "conflict_thresh": 0.1,
        "mask_contrain_ratio": 0.95,
        "bridge_thresh": 3,
        "mask_nms_iou": 0.9,
        "box_nms_iou": 0.9,
        "similarity_threshold": 0.6,
        "over_match_thresh": 0.45,
        "patch_match_thresh": 0.45,
        "contour_weight": 0.5,
        "patch_weight": 0.5,
        "inner_mask_over_thresh": 0.7,
        "inner_mask_partial_thresh": 0.95,
        "shape_diff_thresh": 0.2,
        "patch_coverage_thresh": 0.3,
        "patch_missing_thresh": 0.05,
        "min_mask_area": 100,
        "visualization_alpha": 0.6,
    }
    detector = MosaicDetSeg(
        yoloe_model="yoloe-11l-seg-pf.pt",
        sam_model="weights/FastSAM-x.pt",
        examples_root="demo/toys_label",
        cache_file="demo/toys_label/mosaic_example_features.pkl",
        dinov2_model="facebook/dinov2-base",
        config=config
    )
    # target_image_path = "demo/toy5_frames/frame_0101.jpg"
    # image = Image.open(target_image_path)
    # output_image_path = "demo/mosaic1_det_seg_outs2/frame_0101_out.jpg"
    # os.makedirs("demo/mosaic1_det_seg_outs2", exist_ok=True)
    # results = detector.detect_and_segment(image)
    # output_image = detector.visualize_results(image, results, output_image_path)

    input_dir = "demo/toy1_frames"
    output_dir = "demo/final_results/mosaic1_det_seg_outs2"
    # input_dir = "demo/toy1_frames_special"
    # output_dir = "demo/mosaic_det_seg_outs_special"
    os.makedirs(output_dir, exist_ok=True)
    for image_path in os.listdir(input_dir):
        image = Image.open(os.path.join(input_dir, image_path))
        results = detector.detect_and_segment(image)
        output_image = detector.visualize_results(image, results, os.path.join(output_dir, image_path))
