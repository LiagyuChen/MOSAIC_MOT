import os
import cv2
import multiprocessing

from mosaic_mot import mosaic_mot



adjustable_configs = {
    "mode": "accurate",
    "similarity_threshold": 0.6,
    "max_frames_missing": 30,
    "frames_to_cache": 3,
    "bbox_iou_threshold": 0.4,
    "mask_iou_threshold": 0.3,
    "embedding_similarity_threshold": 0.6,
    "size_prior_threshold": 0.3,
    "hu_weight": 0.3,
    "embedding_weight": 0.7,
    "min_area_ratio": 0.4,
    "class_similarity_threshold": 0.6
}
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
    "mode": "accurate",

    # Visualization configuration
    "text_color": (255, 0, 0),
}
mosaic_mot(
    bytetrack_cfg_path="bytetrack.yaml",
    video_path="demo/toy1.mp4",
    output_path="demo/final_results/toy1_mosaic_mot_out.mp4",
    yoloe_model="yoloe-11l-seg-pf.pt",
    sam_model="FastSAM-s.pt",
    examples_root="demo/toys_label",
    cache_file="demo/toys_label/mosaic_example_features.pkl",
    dinov2_model="facebook/dinov2-base",
    save_annotated_frame_path="demo/final_results/debug_frames",
    config=config
)
