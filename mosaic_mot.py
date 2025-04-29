import os
import cv2
import time
import random
import imageio
import logging
import numpy as np
from tqdm import tqdm
import multiprocessing
from ultralytics.cfg import get_cfg
from typing import Dict, List, Tuple, Optional, Any
from ultralytics.trackers.byte_tracker import BYTETracker
from mosaic_det_seg import MosaicDetSeg


class Result:
    def __init__(self, conf, xywh, cls):
        self.conf = conf
        self.xywh = xywh
        self.cls = cls


class MosaicMOT(MosaicDetSeg):
    def __init__(
        self,
        yoloe_model: str,
        sam_model: str,
        examples_root: str,
        cache_file: str = "mosaic_example_features.pkl",
        dinov2_model: str = "facebook/dinov2-base",
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(yoloe_model, sam_model, examples_root, cache_file, dinov2_model, config)
        self.cache_file=cache_file
        self.device = "cpu"
        self.config = config
        self.frame_groups = {obj_name: [] for obj_name in self.label_names}
        self.frame_info = {}
        self.object_feature_bank = {}

        # Global Tracklet Info Bank
        self.tracked_objects_cache = {}  # Cache for tracked objects with unique IDs
        self.tracked_detection_cache = {}  # Cache for previous frames' detections
        self.max_frames_missing = self.config.get("max_frames_missing", 30)
        self.frames_to_cache = self.config.get("frames_to_cache", 3)
        self.current_frame_idx = 0  # Track current frame index
        self.previous_tracks = None # Initialize previous frame's tracking results

        # Initialize class colors if not already done
        if not hasattr(self, 'class_colors'):
            self.class_colors = {}
            random.seed(42)  # For reproducibility
            for class_name in self.label_names:
                self.class_colors[class_name] = tuple(random.randint(0, 255) for _ in range(3))


    def _xyxy2cxywh(
        self, 
        xyxy: np.ndarray
    ) -> np.ndarray:
        """
        Convert bounding boxes from xyxy (x1, y1, x2, y2) to cxywh (cx, cy, w, h)

        Args:
            xyxy: Bounding boxes in xyxy format

        Returns:
            Bounding boxes in cxywh format
        """

        if xyxy.size == 0:
            return np.zeros((0, 4), dtype=xyxy.dtype)

        cxywh = np.zeros_like(xyxy)
        cxywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2
        cxywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2
        cxywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
        cxywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]

        return cxywh

    def _iou_vectorized(
        self,
        boxes1: np.ndarray,
        boxes2: np.ndarray
    ) -> np.ndarray:
        """
        Vectorized IoU calculation between boxes1 and boxes2, in xyxy format

        Args:
            boxes1: First set of bounding boxes
            boxes2: Second set of bounding boxes

        Returns:
            IoU matrix between boxes1 and boxes2
        """

        x1 = np.maximum(boxes1[:, None, 0], boxes2[:, 0])
        y1 = np.maximum(boxes1[:, None, 1], boxes2[:, 1])
        x2 = np.minimum(boxes1[:, None, 2], boxes2[:, 2])
        y2 = np.minimum(boxes1[:, None, 3], boxes2[:, 3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1[:, None] + area2 - intersection

        return intersection / union

    def _is_mask_contained(
        self,
        mask1: np.ndarray,
        mask2: np.ndarray,
        threshold: float = 0.9
    ) -> bool:
        """Check if mask1 is contained within mask2 based on overlap ratio.

        Args:
            mask1: First binary mask (numpy array)
            mask2: Second binary mask (numpy array)
            threshold: Threshold for containment (default: 0.9)
  
        Returns:
            True if mask1 is contained in mask2, False otherwise
        """

        # Ensure masks are binary (0 or 1)
        mask1_binary = mask1.astype(bool)
        mask2_binary = mask2.astype(bool)

        intersection = np.logical_and(mask1_binary, mask2_binary).sum()
        area1 = mask1_binary.sum()
        if area1 == 0:
            return False

        # Calculate containment ratio (what portion of mask1 is within mask2)
        containment_ratio = intersection / area1
        
        # Return True if containment ratio is above threshold
        return containment_ratio > threshold

    def _compute_mask_iou(
        self,
        masks1: np.ndarray,
        masks2: np.ndarray
    ) -> np.ndarray:
        """
        Compute IoU between masks.

        Args:
            masks1: First set of masks
            masks2: Second set of masks

        Returns:
            IoU matrix between masks1 and masks2
        """

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

    def _compute_hu_moment_distance(
        self,
        contour1: np.ndarray,
        contour2: np.ndarray
    ) -> float:
        """
        Compute L2 distance between log Hu moments.

        Args:
            contour1: First contour
            contour2: Second contour

        Returns:
            L2 distance between log Hu moments
        """

        # Extract Hu moments
        hu1 = contour1[3:7]
        hu2 = contour2[3:7]

        # Normalize
        hu1_norm = hu1 / (np.linalg.norm(hu1) + 1e-8)
        hu2_norm = hu2 / (np.linalg.norm(hu2) + 1e-8)

        # Compute L2 distance
        distance = np.linalg.norm(hu1_norm - hu2_norm)
        return distance

    def _bi_softmax_cosine_similarity(
        self,
        A: np.ndarray,
        B: np.ndarray,
        temperature: float = 0.3
    ) -> np.ndarray:
        """
        Bi-directional softmax similarity with temperature scaling.

        Args:
            A: First set of features
            B: Second set of features
            temperature: Temperature scaling factor

        Returns:
            Bi-directional softmax similarity matrix
        """

        sim = self._compute_cosine_similarity(A, B)
        # Numerically stable softmax with temperature scaling
        max_sim = np.max(sim, axis=1, keepdims=True)
        row_softmax = np.exp((sim - max_sim) / temperature) / (np.sum(np.exp((sim - max_sim) / temperature), axis=1, keepdims=True) + 1e-8)

        max_sim = np.max(sim, axis=0, keepdims=True)
        col_softmax = np.exp((sim - max_sim) / temperature) / (np.sum(np.exp((sim - max_sim) / temperature), axis=0, keepdims=True) + 1e-8)
        return 0.5 * (row_softmax + col_softmax)

    def _intersection_over_area_vectorized(
        self,
        boxes1: np.ndarray,
        boxes2: np.ndarray
    ) -> np.ndarray:
        """
        Calculates intersection over area of second box for each pair of boxes.
        This measures what portion of box2 is covered by box1, useful for detecting contained boxes.

        Args:
            boxes1: First set of bounding boxes in xyxy format
            boxes2: Second set of bounding boxes in xyxy format

        Returns:
            Matrix where each element [i,j] represents the intersection area divided by area of box2[j]
        """
        # Compute intersection
        x1 = np.maximum(boxes1[:, None, 0], boxes2[:, 0])
        y1 = np.maximum(boxes1[:, None, 1], boxes2[:, 1])
        x2 = np.minimum(boxes1[:, None, 2], boxes2[:, 2])
        y2 = np.minimum(boxes1[:, None, 3], boxes2[:, 3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # Calculate area of second boxes
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        # Return intersection over area2
        # Add small epsilon to avoid division by zero
        return intersection / (area2[:, None].T + 1e-6)

    def _match_detections_with_tracks(
        self,
        bboxes: np.ndarray,
        masks: np.ndarray,
        detection_features: List[Dict],
        tracks: np.ndarray,
        labels: np.ndarray,
        original_labels: np.ndarray,
        scores: np.ndarray
    ) -> Tuple[List[Dict], List[int], np.ndarray]:
        """
        Match detections with tracks based on IoU, mask overlap, and embedding similarity

        Args:
            bboxes: Detection bounding boxes
            masks: Detection masks
            detection_features: Detection features
            tracks: ByteTrack tracks
            labels: Detection labels
            original_labels: Original labels
            scores: Detection scores

        Returns:
            Matched results, unmatched track IDs, unmatched detection indices
        """

        iou_threshold = self.config.get("iou_threshold", 0.5)
        mask_iou_threshold = self.config.get("mask_iou_threshold", 0.5)
        embedding_threshold = self.config.get("embedding_threshold", 0.55)
        
        # Handle empty cases
        if len(tracks) == 0:
            return [], [], np.ones(len(bboxes), dtype=bool)
        if len(bboxes) == 0:
            return [], [int(track[4]) for track in tracks], np.array([], dtype=bool)
        
        # Determine the larger dimension for padding
        n_tracks = len(tracks)
        n_detections = len(bboxes)
        max_dim = max(n_tracks, n_detections)

        # Compute IoU between predicted boxes and detection boxes
        predicted_bbox = tracks[:, :4]
        predicted_bbox_reshaped = predicted_bbox.reshape(-1, 4)

        # Calculate raw IoU matrix
        raw_bbox_iou = self._iou_vectorized(predicted_bbox_reshaped, bboxes)
        
        # Create padded IoU matrix
        bbox_iou_matrix = np.zeros((max_dim, max_dim))
        bbox_iou_matrix[:n_tracks, :n_detections] = raw_bbox_iou

        # Compute mask IoU
        track_masks = np.array([self.tracked_objects_cache.get(int(track[4]), {}).get('last_mask', np.zeros_like(masks[0])) 
                              for track in tracks])

        # Calculate raw mask IoU matrix
        raw_mask_iou = self._compute_mask_iou(track_masks, masks)
        
        # Create padded mask IoU matrix
        mask_iou_matrix = np.zeros((max_dim, max_dim))
        mask_iou_matrix[:n_tracks, :n_detections] = raw_mask_iou

        # Extract embeddings
        det_embeddings = np.array([feat.get('embedding', None) for feat in detection_features])

        # Get track embeddings with shape checking
        track_embeddings = []
        for track in tracks:
            track_id = int(track[4])
            track_embed = self.tracked_objects_cache.get(track_id, {}).get('features', {}).get('embedding', None)
            track_embeddings.append(track_embed or np.zeros(2304))

        track_embeddings = np.array(track_embeddings)

        # Calculate embedding similarity with error handling
        raw_bi_softmax = self._bi_softmax_cosine_similarity(track_embeddings, det_embeddings)

        # Create padded embedding similarity matrix
        bi_softmax_sim = np.zeros((max_dim, max_dim))
        bi_softmax_sim[:len(track_embeddings), :len(det_embeddings)] = raw_bi_softmax

        # Handle missing tracks (new detections)
        if n_detections > n_tracks:
            for i in range(n_tracks, n_detections):
                bi_softmax_sim[i, i] = 0.0

        # Combine metrics
        # combined_matrix = (bbox_iou_matrix >= iou_threshold).astype(float) + \
        #                  (mask_iou_matrix >= mask_iou_threshold).astype(float) + \
        #                  (bi_softmax_sim >= embedding_threshold).astype(float)
        combined_matrix = (bbox_iou_matrix >= iou_threshold).astype(float) + \
                         (mask_iou_matrix >= mask_iou_threshold).astype(float)
 

        # For each track, find the best matching detection
        best_match_idx = np.argmax(combined_matrix, axis=1)
        best_match_count = np.max(combined_matrix, axis=1)

        # Valid matches have at least 2 of the 3 criteria met
        valid_matches = best_match_count >= 1

        # Create matched results
        matched_results = []
        matched_track_indices = np.zeros(max_dim, dtype=bool)
        matched_det_indices = np.zeros(n_detections, dtype=bool)

        # Process real tracks (not padding)
        for i in range(n_tracks):
            valid = valid_matches[i]
            match_idx = best_match_idx[i]
            
            if valid and match_idx < n_detections and not matched_det_indices[match_idx]:
                matched_track_indices[i] = True
                matched_det_indices[match_idx] = True
                result = {
                    "bbox": bboxes[match_idx].astype(np.int32),
                    "mask": masks[match_idx],
                    "class_name": original_labels[match_idx],
                    "score": scores[match_idx],
                    "label": labels[match_idx],
                    "features": detection_features[match_idx],
                    "track_id": int(tracks[i, 4]),
                    "bbox_prediction": predicted_bbox[i]
                }
                matched_results.append(result)

        # Process "virtual" tracks (padding for new detections)
        for i in range(n_tracks, max_dim):
            match_idx = best_match_idx[i]
            if match_idx < n_detections and not matched_det_indices[match_idx]:
                matched_det_indices[match_idx] = True

        # Return unmatched track IDs instead of indices
        unmatched_track_ids = [int(tracks[i, 4]) for i in range(n_tracks) if not matched_track_indices[i]]

        return matched_results, unmatched_track_ids, ~matched_det_indices

    def _recover_with_discarded_detections(
        self,
        unmatched_track_ids: List[int],
        tracks: np.ndarray,
        discarded_detections: List[Dict],
    ) -> List[Dict]:
        """
        Recover unmatched tracks using discarded detections
        
        Args:
            unmatched_track_ids: Unmatched track IDs
            tracks: ByteTrack tracks
            discarded_detections: Discarded detections

        Returns:
            List of recovered results
        """

        if not discarded_detections or len(unmatched_track_ids) == 0:
            return []

        area_threshold = self.config.get("size_prior_threshold", 0.3)
        recovered_inputs = []
        recovered_results = []

        # Prepare arrays for vectorized calculations
        discarded_bboxes = np.array([det.get('bbox', np.zeros(4)) for det in discarded_detections])
        discarded_masks = np.array([det.get('mask', np.zeros_like(self.frame)) for det in discarded_detections])
        discarded_areas = (discarded_bboxes[:, 2] - discarded_bboxes[:, 0]) * (discarded_bboxes[:, 3] - discarded_bboxes[:, 1])

        # Iterate over unmatched tracks
        for track_id in unmatched_track_ids:
            track = tracks[tracks[:, 4] == track_id]
            if track.size == 0:
                continue

            cached = self.tracked_objects_cache.get(track_id, {})
            cached_bbox = cached.get('last_bbox')
            if cached_bbox is None:
                continue

            cached_bbox = np.array(cached_bbox, dtype=np.int32)
            cached_area = (cached_bbox[2] - cached_bbox[0]) * (cached_bbox[3] - cached_bbox[1])
            area_ratios = np.abs(discarded_areas - cached_area) / (cached_area + 1e-6)
            size_valid = area_ratios < area_threshold

            ious = self._intersection_over_area_vectorized(cached_bbox.reshape(1, 4), discarded_bboxes)[0]
            valid_indices = np.where(size_valid)[0]
            if len(valid_indices) == 0:
                continue

            best_idx = valid_indices[np.argmax(ious[valid_indices])]
            if ious[best_idx] <= 0.4:
                continue

            # Store detection info to extract features later
            recovered_inputs.append({
                "bbox": discarded_bboxes[best_idx],
                "mask": discarded_masks[best_idx],
            })
            recovered_results.append({
                "track_id": track_id,
                "bbox_prediction": track[:4],
                "recovered": True,
            })

        # Extract features in a batch
        if recovered_inputs:
            extracted_features = self._extract_detection_features(self.frame, recovered_inputs)
            embedding_sims = self._compute_embedding_similarities(extracted_features)
            selected = self._apply_adaptive_gating(extracted_features, embedding_sims)
            for (i, class_name, similarity_score) in selected:
                recovered_results[i]["bbox"] = recovered_inputs[i]["bbox"].astype(np.int32)
                recovered_results[i]["mask"] = recovered_inputs[i]["mask"]
                recovered_results[i]["features"] = extracted_features[i]
                recovered_results[i]["class_name"] = class_name
                recovered_results[i]["score"] = similarity_score

        return recovered_results

    def _rescue_with_cache(
        self,
        unmatched_det_indices: np.ndarray,
        bboxes: np.ndarray,
        masks: np.ndarray,
        detection_features: List[Dict],
    ) -> List[Dict]:
        """
        Match remaining unmatched detections with cached track data
        
        Args:
            unmatched_det_indices: Unmatched detection indices
            bboxes: Detection bounding boxes
            masks: Detection masks
            detection_features: Detection features

        Returns:
            List of rescued results
        """

        hu_weight = self.config.get("hu_weight", 0.3)
        embedding_weight = self.config.get("embedding_weight", 0.7)
        similarity_threshold = self.config.get("similarity_threshold", 0.35)
        if unmatched_det_indices is None or len(unmatched_det_indices) == 0:
            return []

        unmatched_bboxes = bboxes[unmatched_det_indices]
        unmatched_features = [detection_features[i] for i in np.where(unmatched_det_indices)[0]]

        # Track IDs seen recently but missing now
        candidate_ids = [
            track_id for track_id, info in self.tracked_objects_cache.items()
            if 0 < info.get("frames_missing", 0) <= self.frames_to_cache
        ]
        if not candidate_ids:
            return []

        rescued_info = []
        refined_detections = []

        for bbox, features in zip(unmatched_bboxes, unmatched_features):
            best_score, best_track_id = float("inf"), None
            for track_id in candidate_ids:
                cached = self.tracked_objects_cache.get(track_id, {})
                cached_feat = cached.get("features", {})

                # Hu moment distance
                hu_dist = self._compute_hu_moment_distance(
                    features.get("contour"), cached_feat.get("contour")
                ) if features.get("contour") is not None and cached_feat.get("contour") is not None else 1.0

                # Embedding similarity
                emb_sim = self._compute_cosine_similarity(
                    features.get("embedding", np.zeros(128)).reshape(1, -1),
                    cached_feat.get("embedding", np.zeros(128)).reshape(1, -1)
                )[0, 0]

                # Combined score
                score = hu_weight * hu_dist + embedding_weight * (1 - emb_sim)
                if score < best_score:
                    best_score, best_track_id = score, track_id

            if best_score < similarity_threshold and best_track_id is not None:
                cached_bbox = self.tracked_objects_cache[best_track_id].get("last_bbox")
                if cached_bbox is None:
                    continue

                # Union + padding
                x1 = min(bbox[0], cached_bbox[0])
                y1 = min(bbox[1], cached_bbox[1])
                x2 = max(bbox[2], cached_bbox[2])
                y2 = max(bbox[3], cached_bbox[3])
                pad_x, pad_y = 50, 50
                padded_bbox = [
                    max(0, x1 - pad_x),
                    max(0, y1 - pad_y),
                    min(self.image_width, x2 + pad_x),
                    min(self.image_height, y2 + pad_y)
                ]

                # FastSAM refinement
                refined_bbox, refined_mask = self._seg_with_fastsam(self.frame, padded_bbox)
                if refined_bbox is not None and refined_mask is not None:
                    refined_detections.append({
                        "bbox": np.array(refined_bbox, dtype=np.int32),
                        "mask": refined_mask
                    })
                    rescued_info.append({
                        "track_id": best_track_id,
                        "bbox": np.array(refined_bbox, dtype=np.int32),
                        "mask": refined_mask
                    })

        # Extract all features in one call
        if refined_detections:
            refined_features = self._extract_detection_features(self.frame, refined_detections)
            embedding_sims = self._compute_embedding_similarities(refined_features)
            selected_detections = self._apply_adaptive_gating(refined_features, embedding_sims)
            return [
                {
                    "bbox": refined_detections[idx]["bbox"],
                    "mask": refined_detections[idx]["mask"],
                    "features": refined_features[idx],
                    "track_id": rescued_info[idx]["track_id"],
                    "class_name": class_name,
                    "score": score,
                    "rescued": True
                }
                for idx, class_name, score in selected_detections
            ]
        return []

    def _recover_missed_detections(
        self,
        unmatched_track_ids: List[int],
        tracks: np.ndarray,
    ) -> List[Dict]:
        """
        Recover missed detections using ByteTrack predictions + FastSAM + embedding verification

        Args:
            unmatched_track_ids: Unmatched track IDs
            tracks: ByteTrack tracks

        Returns:
            List of recovered results
        """

        min_area_ratio = self.config.get("min_area_ratio", 0.4)
        class_similarity_threshold = self.config.get("class_similarity_threshold", 0.6)
        if len(unmatched_track_ids) == 0:
            return []

        recovered_results = []
        refined_detections = []
        recovery_metadata = []

        for track in tracks[np.isin(tracks[:, 4], unmatched_track_ids)]:
            track_id = int(track[4])
            cached = self.tracked_objects_cache.get(track_id, {})
            cached_bbox = cached.get("last_bbox")
            cached_mask = cached.get("last_mask")
            cached_class = cached.get("class_name")
            if cached_bbox is None or cached_mask is None or cached_class is None:
                continue

            predicted_bbox = track[:4]
            refined_bbox, refined_mask = self._seg_with_fastsam(self.frame, predicted_bbox)
            if refined_bbox is None or refined_mask is None:
                continue

            # Check area match
            cached_area = np.sum(cached_mask)
            refined_area = np.sum(refined_mask)
            if refined_area < min_area_ratio * cached_area:
                continue

            # Queue up for feature extraction
            refined_detections.append({
                "bbox": np.array(refined_bbox, dtype=np.int32),
                "mask": refined_mask
            })
            recovery_metadata.append({
                "track_id": track_id,
                "bbox": np.array(refined_bbox, dtype=np.int32),
                "mask": refined_mask,
                "cached_class": cached_class
            })

        if not refined_detections:
            return []

        # Batch feature extraction
        extracted_features = self._extract_detection_features(self.frame, refined_detections)
        embedding_sims = self._compute_embedding_similarities(extracted_features)
        selected = self._apply_adaptive_gating(extracted_features, embedding_sims)
        for (i, class_name, similarity_score) in selected:
            meta = recovery_metadata[i]
            if class_name == meta["cached_class"] and similarity_score >= class_similarity_threshold:
                recovered_results.append({
                    "bbox": meta["bbox"],
                    "mask": meta["mask"],
                    "features": extracted_features[i],
                    "track_id": meta["track_id"],
                    "class_name": class_name,
                    "score": similarity_score,
                    "recovered_missed": True
                })

        return recovered_results

    def _handle_overmatched_detections(
        self,
        matched_results: List[Dict]
    ) -> List[Dict]:
        """
        Handle cases where multiple objects are merged into one detection.
        
        Args:
            matched_results: Matched results

        Returns:
            List of refined results
        """

        if len(matched_results) <= 1:
            return matched_results

        embedding_sim_thresh = self.config.get("embedding_similarity_threshold", 0.6)
        mask_iou_thresh = self.config.get("mask_iou_threshold", 0.5)
        embeddings = [r.get("features", {}).get("embedding", np.zeros(2304)) for r in matched_results]
        masks_array = np.array([r.get("mask") for r in matched_results])
        bboxes_array = np.array([r.get("bbox") for r in matched_results])

        # Create a boolean matrix to check if mask i is contained within mask j
        containment_matrix = np.array([
            [
                np.sum(np.logical_and(masks_array[i], masks_array[j])) / (np.sum(masks_array[i]) + 1e-6) > 0.9
                for j in range(len(masks_array)) if i != j
            ]
            for i in range(len(masks_array))
        ])

        # Identify contained masks (a mask is contained if it's inside any other mask)
        contained_indices = np.where(np.any(containment_matrix, axis=0))[0]
        non_contained_indices = np.where(~np.any(containment_matrix, axis=0))[0]
        if len(contained_indices) == 0:
            return matched_results

        # Exclude inner masks and send to FastSAM
        new_detections = []
        refined_results = []
        for idx in contained_indices:
            x1, y1, x2, y2 = bboxes_array[idx]
            # Exclude inner contained mask
            excluded_mask = masks_array[idx] & ~np.any(masks_array[non_contained_indices], axis=0)
            roi = self.frame[y1:y2, x1:x2] * excluded_mask[y1:y2, x1:x2, None]
            refined_bbox, refined_mask = self._seg_with_fastsam(roi, bboxes_array[idx])
            if refined_bbox is not None and refined_mask is not None:
                # Map the refined bbox back to the original frame
                refined_bbox[0] += x1
                refined_bbox[1] += y1
                refined_bbox[2] += x1
                refined_bbox[3] += y1

                # Create a full-size mask and place the refined mask in the correct location
                full_size_mask = np.zeros_like(self.frame[:, :, 0], dtype=bool)
                full_size_mask[y1:y2, x1:x2] = refined_mask

                new_detections.append({
                    'bbox': refined_bbox,
                    'mask': full_size_mask
                })

        # Extract features for new detections
        if new_detections:
            new_features = self._extract_detection_features(self.frame, new_detections)
            new_embeddings = np.array([feat.get("embedding") for feat in new_features])

            # Compute cosine similarity and mask IoU in a vectorized manner
            sim_matrix = self._compute_cosine_similarity(new_embeddings, embeddings)
            iou_matrix = np.array([self._compute_mask_iou(det["mask"].reshape(1, *det["mask"].shape), masks_array)[0] for det in new_detections])

            # Filter based on thresholds
            valid_indices = np.where((sim_matrix < embedding_sim_thresh) & (np.max(iou_matrix, axis=1) < mask_iou_thresh))[0]

            embedding_similarities = self._compute_embedding_similarities(new_features)
            selected_detections_tuples = self._apply_adaptive_gating(
                new_features,
                embedding_similarities
            )

            # Append valid detections to refined_results with track_ids
            for idx, cls_name, score in selected_detections_tuples:
                if idx in valid_indices:
                    refined_results.append({
                        'bbox': new_detections[idx]['bbox'],
                        'mask': new_detections[idx]['mask'],
                        'features': new_features[idx],
                        'class_name': cls_name,
                        'score': score,
                        'track_id': matched_results[contained_indices[idx]]['track_id'],
                        'refined': True
                    })

        # Remove original contained masks and update return list
        refined_results = [matched_results[i] for i in non_contained_indices] + refined_results

        return refined_results

    def _update_tracking_cache(
        self,
        matched_results: List[Dict],
        unmatched_track_indices: np.ndarray,
        tracks: np.ndarray
    ) -> None:
        """
        Update tracking caches for matched and unmatched objects.

        Args:
            matched_results: Matched results
            unmatched_track_indices: Unmatched track indices
            tracks: ByteTrack tracks

        Returns:
            None
        """

        for result in matched_results:
            track_id = int(result["track_id"])
            if track_id not in self.tracked_objects_cache:
                self.tracked_objects_cache[track_id] = {
                    "frames_missing": 0,
                    "last_bbox": None,
                    "last_mask": None,
                    "features": None,
                    "class_name": None,
                }
            bbox = result.get("bbox")
            mask = result.get("mask")
            features = result.get("features", {})
            class_name = result.get("class_name")
            if bbox is not None:
                self.tracked_objects_cache[track_id]["last_bbox"] = bbox
            if mask is not None:
                self.tracked_objects_cache[track_id]["last_mask"] = mask
            if features:
                self.tracked_objects_cache[track_id]["features"] = features
            if class_name is not None:
                self.tracked_objects_cache[track_id]["class_name"] = class_name

        # Update for unmatched tracks
        unmatched_track_indices = np.where(unmatched_track_indices)[0]
        if len(unmatched_track_indices) > 0:
            unmatched_tracks = tracks[unmatched_track_indices]
            for track in unmatched_tracks:
                track_id = int(track[4])

                # Skip if track not in cache
                if track_id not in self.tracked_objects_cache:
                    continue

                # Increment missing frames counter
                self.tracked_objects_cache[track_id]["frames_missing"] += 1

        # Remove outdated cache entries
        for track_id in list(self.tracked_objects_cache.keys()):
            info = self.tracked_objects_cache[track_id]
            if info.get("frames_missing", 0) > self.max_frames_missing:
                del self.tracked_objects_cache[track_id]

        # Update frame detection cache
        self.tracked_detection_cache[self.current_frame_idx % self.frames_to_cache] = matched_results
        self.current_frame_idx += 1

    def _match_detection_objects(
        self,
        bboxes: np.ndarray,
        masks: np.ndarray,
        labels: np.ndarray,
        original_labels: np.ndarray,
        scores: np.ndarray,
        detection_features: List[Dict],
        discarded_detections: List[Dict],
        tracks: np.ndarray,
        frame_rgb: np.ndarray = None
    ) -> List[Dict]:
        """
        Implement the complete MOSAIC-MOT tracking procedure for a frame.

        Args:
            bboxes: Detection bounding boxes (x1, y1, x2, y2)
            masks: Detection masks
            labels: Detection labels (class indices)
            original_labels: Original class names
            scores: Detection confidence scores
            detection_features: Detection features
            discarded_detections: Discarded detections
            tracks: ByteTrack tracks
            frame_rgb: Current frame

        Returns:
            List of matched detection results
        """

        # Store current frame for segmentation refinement
        self.frame = frame_rgb
        self.image_height, self.image_width = frame_rgb.shape[:2]

        # STEP 1: Match detections with tracks using IoU, mask overlap, and embedding similarity
        matched_results, unmatched_track_ids, unmatched_det_indices = self._match_detections_with_tracks(
            bboxes=bboxes,
            masks=masks,
            detection_features=detection_features,
            tracks=tracks,
            labels=labels,
            original_labels=original_labels,
            scores=scores
        )
        logging.info(f"Step 1: Initial matching - {len(matched_results)} matched, {len(unmatched_track_ids)} unmatched tracks, {np.sum(unmatched_det_indices)} unmatched detections")

        # # STEP 2: Recover unmatched tracks using discarded detections
        # recovered_results = self._recover_with_discarded_detections(
        #     unmatched_track_ids=unmatched_track_ids,
        #     tracks=tracks,
        #     discarded_detections=discarded_detections
        # )
        # logging.info(f"Step 2: Recovered with discarded detections - {len(recovered_results)} objects")

        # # Update unmatched track IDs
        # unmatched_track_ids = [track_id for track_id in unmatched_track_ids if track_id not in [res['track_id'] for res in recovered_results]]

        # # STEP 3: Match remaining problematic detections with cache-based rescue
        # rescued_results = self._rescue_with_cache(
        #     unmatched_det_indices=unmatched_det_indices,
        #     bboxes=bboxes,
        #     masks=masks,
        #     detection_features=detection_features
        # )
        # logging.info(f"Step 3: Rescued with cache - {len(rescued_results)} objects")

        # # STEP 4: Recover completely missed detections using ByteTrack predictions
        # missed_results = self._recover_missed_detections(
        #     unmatched_track_ids=unmatched_track_ids,
        #     tracks=tracks
        # )
        # logging.info(f"Step 4: Recovered missed detections - {len(missed_results)} objects")

        # # STEP 5:  Handle over-matched detections (merged objects)
        # all_results = matched_results + recovered_results + rescued_results + missed_results
        # refined_results = self._handle_overmatched_detections(matched_results=all_results)
        # logging.info(f"Step 5: After handling over-matched - {len(refined_results)} objects")

        refined_results = matched_results

        # STEP 6: Update tracking cache
        self._update_tracking_cache(
            matched_results=refined_results,
            unmatched_track_indices=np.array([i for i, track in enumerate(tracks) if int(track[4]) in unmatched_track_ids]),
            tracks=tracks
        )
        logging.info(f"Step 6: Updated tracking cache - {len(self.tracked_objects_cache)} tracked objects")

        return refined_results

    def _process_frame(
        self,
        detection_results: List[Dict],
        tracker: Any,
        frame_rgb: np.ndarray,
        frame_idx: int
    ) -> List[Dict]:
        """
        Process a single frame with tracking and key frame optimization

        Args:
            detection_results: Detection results
            tracker: ByteTrack tracker
            frame_rgb: Current frame
            frame_idx: Frame index

        Returns:
            List of matched detection results
        """

        # Ensure bboxes is a numpy array
        bboxes = np.array([result.get("bbox", None) for result in detection_results])
        masks = np.array([result.get("mask", None) for result in detection_results])
        scores = np.array([result.get("score", None) for result in detection_results])
        original_labels = np.array([result.get("class_name", None) for result in detection_results])
        labels = np.array([self.label_mappings.get(result.get("class_name", None), 1) for result in detection_results])
        detection_features = [result.get("features", None) for result in detection_results]
        discarded_detections = detection_results[0]["discarded_detection"]

        # Update tracker with current detections
        centered_bboxes = self._xyxy2cxywh(bboxes)
        if frame_idx == 0 or frame_idx == 190:
            tracks = tracker.update(Result(scores, centered_bboxes, labels))
            matched_results = []
            for i, res in enumerate(detection_results):
                detection_result = res
                detection_result["track_id"] = tracks[i][4]
                matched_results.append(detection_result)

            self._update_tracking_cache(
                matched_results=matched_results,
                unmatched_track_indices=np.zeros(len(tracks), dtype=bool),
                tracks=tracks
            )
            self.previous_tracks = tracks
            return detection_results

        matched_results = self._match_detection_objects(
            bboxes=bboxes,
            masks=masks,
            labels=labels,
            original_labels=original_labels,
            scores=scores,
            detection_features=detection_features,
            discarded_detections=discarded_detections,
            tracks=self.previous_tracks,
            frame_rgb=frame_rgb
        )

        # Update tracker with matched results for the next frame
        self.previous_tracks = tracker.update(Result(scores, centered_bboxes, labels))

        return matched_results

    def process_video_batch(
        self,
        video_path: str,
        output_path: str,
        tracker: Any,
        config: Dict[str, Any],
        save_annotated_frame_path: str = None
    ) -> int:
        """
        Process a video with comprehensive concurrent execution at each stage

        Args:
            video_path: Path to the input video
            output_path: Path to save the output video
            tracker: ByteTrack tracker
            config: Configuration dictionary
            save_annotated_frame_path: Path to save annotated frames

        Returns:
            Number of frames processed
        """
        # Open the input video
        reader = imageio.get_reader(video_path, format='ffmpeg')
        fps = reader.get_meta_data()['fps']
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        writer = imageio.get_writer(output_path, fps=fps)

        # Get total frames
        total_frames = reader.count_frames()
        progress_bar = tqdm(total=total_frames, desc="Processing frames", unit="frame")

        # Initialize frame cache
        frame_count = 0

        # Create debug directory for saving annotated frames
        if save_annotated_frame_path is not None:
            os.makedirs(save_annotated_frame_path, exist_ok=True)
            logging.info(f"Debug frames will be saved to: {save_annotated_frame_path}")

        # Process video frames
        text_color = config.get("text_color", (0, 0, 255))
        try:
            for i, frame in enumerate(reader):
                if i < 190:
                    continue

                # Get the detection and segmentation results
                # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = frame
                detection_results = self.detect_and_segment(frame_rgb)
                logging.info(f"Frame {i}: Detection results - {len(detection_results)}")

                matched_results = []
                if len(detection_results) > 0:
                    matched_results = self._process_frame(
                        detection_results=detection_results,
                        tracker=tracker,
                        frame_rgb=frame_rgb,
                        frame_idx=i
                    )
                logging.info(f"Frame {i}: Mosaic MOT results - {len(matched_results)} objects")

                # Store results for next frame
                for result in matched_results:
                    self.object_feature_bank[result["track_id"]] = result

                # Annotate the results
                if save_annotated_frame_path is not None:
                    frame_path = os.path.join(save_annotated_frame_path, f"frame_{i:04d}.jpg")
                    self.visualize_results(frame_rgb, matched_results, frame_path)

                # Save tracked objects
                vis_image = frame_rgb.copy()
                for result in matched_results:
                    # Get bounding box in xyxy format and convert to integer coordinates
                    x1, y1, x2, y2 = map(int, result["bbox"])
                    track_id = int(result["track_id"])
                    score = float(result.get("score", 0.0))
                    class_name = result.get("class_name", "unknown")

                    # Use the pre-defined color for each class
                    color = self.class_colors.get(class_name, (255, 0, 0))

                    # Draw mask if available
                    mask = result.get("mask", None)
                    # Ensure mask has same dimensions as image
                    if mask.shape[:2] != (frame.shape[0], frame.shape[1]):
                        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

                    # Ensure mask is binary (already should be uint8, but check > 0)
                    binary_mask = (mask > 0)
                    mask_color = np.zeros_like(vis_image)
                    mask_color[binary_mask] = color
                    vis_image = cv2.addWeighted(vis_image, 1, mask_color, 0.6, 0)

                    # Check if box is valid
                    h, w = frame.shape[:2]
                    if x1 < 0 or y1 < 0 or x2 >= w or y2 >= h:
                        logging.warning(f"WARNING: Box coordinates out of bounds! Image size: {w}x{h}")
                        # Clip the coordinates to frame boundaries
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w-1, x2), min(h-1, y2)

                    # Draw on final output
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

                    # Add recovery/rescue status indicator to text
                    status = ""
                    if result.get("recovered", False):
                        status = " [R]"
                    elif result.get("rescued", False):
                        status = " [S]"
                    elif result.get("recovered_missed", False):
                        status = " [M]"
                    elif result.get("refined", False):
                        status = " [O]"

                    text = f"{class_name} #{track_id} ({score:.2f}){status}"
                    cv2.putText(
                        vis_image, text, (x1, max(y1 - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2
                    )

                # Convert to BGR format for saving
                vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
                writer.append_data(vis_image_bgr)

                frame_count += 1
                progress_bar.update(1)

        except Exception as e:
            logging.error(f"Error processing video: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Ensure writer is closed properly
            writer.close()
            progress_bar.close()

        return frame_count


def mosaic_mot(
    bytetrack_cfg_path: str,
    video_path: str,
    output_path: str,
    yoloe_model: str,
    sam_model: str,
    examples_root: str,
    cache_file: str = "mosaic_example_features.pkl",
    dinov2_model: str = "facebook/dinov2-base",
    save_annotated_frame_path: str = None,
    config: Dict[str, Any] | None = None, 
) -> None:
    logging.info(f"Starting tracking with similarity threshold: {config.get('similarity_threshold')}")
    t_start = time.time()

    # Load ByteTrack configuration
    tracker_cfg = get_cfg(bytetrack_cfg_path)
    logging.info(f"ByteTrack configuration: {tracker_cfg}")
    tracker = BYTETracker(tracker_cfg)

    # Initialize FewShotSelector with configurable parameters
    mosaic = MosaicMOT(
        yoloe_model=yoloe_model,
        sam_model=sam_model,
        examples_root=examples_root,
        cache_file=cache_file,
        dinov2_model=dinov2_model,
        config=config
    )
    frame_count = mosaic.process_video_batch(
        video_path=video_path,
        output_path=output_path,
        tracker=tracker,
        config=config,
        save_annotated_frame_path=save_annotated_frame_path
    )

    t_end = time.time()
    processing_time = t_end - t_start
    logging.info(f"Processed {frame_count} frames in {processing_time:.2f}s ({frame_count / processing_time:.2f} FPS)")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    config = {
        # Few-Shot Selector configurations
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

        # MOSAIC-DET-SEG configurations
        "mosaic_mode": "accurate",
        "large_image_threshold": 1088,
        "mask_area_thresh": 0.8,
        "conflict_thresh": 0.1,
        "mask_contrain_ratio": 0.95,
        "bridge_thresh": 3,
        "mask_nms_iou": 0.9,
        "box_nms_iou": 0.9,
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
        "similarity_threshold": 0.6,

        # MOSAIC-MOT specific parameters
        "max_frames_missing": 30,
        "frames_to_cache": 3,
        "bbox_iou_threshold": 0.4,
        "mask_iou_threshold": 0.3,
        "embedding_similarity_threshold": 0.6,
        "size_prior_threshold": 0.3,
        "hu_weight": 0.3,
        "embedding_weight": 0.7,
        "min_area_ratio": 0.4,
        "class_similarity_threshold": 0.6,

        # Visualization configuration
        "draw_color": (255, 0, 0),
        "line_thickness": 2,
        "font": cv2.FONT_HERSHEY_SIMPLEX,
        "font_scale": 0.5,
        "text_color": (255, 0, 0),
        "visualization_alpha": 0.6,
    }
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    mosaic_mot(
        bytetrack_cfg_path="bytetrack.yaml",
        video_path="demo/toy1.mp4",
        output_path="demo/final_results/toy1_mosaic_mot_out.mp4",
        yoloe_model="weights/yoloe-11l-seg-pf.pt",
        sam_model="weights/FastSAM-s.pt",
        examples_root="demo/toys_label",
        cache_file="demo/toys_label/mosaic_example_features.pkl",
        dinov2_model="facebook/dinov2-base",
        save_annotated_frame_path="demo/final_results/debug_frames",
        config=config
    )

